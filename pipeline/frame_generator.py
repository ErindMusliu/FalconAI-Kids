import gc
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    AnimateDiffPipeline,
    MotionAdapter,
    DDIMScheduler,
)

from config.settings import DIFFUSION_CONFIG, ANIMATOR_CONFIG, DEVICE
from utils.logger import get_logger
from utils.exceptions import FrameGenerationError, ModelLoadError

logger = get_logger(__name__)


class FrameGenerator:
    """
    Frame generation engine for FalconAI Kids.

    GPU mode:
        - Stable Diffusion
        - AnimateDiff
        - IP-Adapter

    CPU mode:
        - Stable Diffusion only
        - No AnimateDiff
        - No IP-Adapter
        - Conservative inference settings
        - Aggressive memory cleanup

    The CPU path intentionally generates one base image per scene and
    duplicates it into frames. Motion can be added later by the video
    pipeline using zoom/pan effects.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        use_cpu_offload: bool = True,
    ):
        self.seed = seed

        self.sd_pipe = None
        self.anim_pipe = None

        self.ip_adapter_loaded = False

        self.is_cuda = DEVICE == "cuda" and torch.cuda.is_available()
        self.is_cpu = not self.is_cuda

        # CPU offload only makes sense when CUDA exists.
        self.use_cpu_offload = use_cpu_offload and self.is_cuda

        if self.is_cpu:
            logger.info(
                "FalconAI Kids FrameGenerator initialized in CPU-only mode."
            )
        else:
            logger.info(
                f"FalconAI Kids FrameGenerator initialized on CUDA device: {DEVICE}"
            )

        self._load_models()

    # ------------------------------------------------------------------
    # MODEL LOADING
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        try:
            self._load_stable_diffusion()

            # IP-Adapter is intentionally disabled on CPU.
            self._load_ip_adapter()

            # AnimateDiff is intentionally disabled on CPU.
            if self.is_cuda:
                self._load_animatediff()
            else:
                logger.info(
                    "CPU mode: AnimateDiff disabled to avoid excessive RAM usage."
                )
                self.anim_pipe = None

        except FrameGenerationError:
            raise

        except Exception as exc:
            logger.exception("Unexpected FrameGenerator initialization failure.")
            raise ModelLoadError(
                "FrameGenerator",
                str(exc),
            )

    def _load_stable_diffusion(self) -> None:
        model_name = DIFFUSION_CONFIG["model_name"]
        cache_dir = DIFFUSION_CONFIG["model_cache_dir"]

        logger.info(
            f"Loading Stable Diffusion model: {model_name}"
        )

        dtype = (
            torch.float16
            if self.is_cuda
            else torch.float32
        )

        try:
            self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )

            self.sd_pipe.scheduler = DDIMScheduler.from_config(
                self.sd_pipe.scheduler.config
            )

            self._place_pipe_on_device(self.sd_pipe)

            # GPU optimizations.
            if self.is_cuda:
                self.sd_pipe.enable_attention_slicing()
                self.sd_pipe.enable_vae_slicing()

                try:
                    self.sd_pipe.enable_xformers_memory_efficient_attention()

                    logger.info(
                        "xFormers memory-efficient attention enabled."
                    )

                except Exception:
                    logger.debug(
                        "xFormers unavailable. Using native attention."
                    )

            # CPU optimizations.
            else:
                logger.info(
                    "CPU mode: Stable Diffusion loaded with float32."
                )

                # Attention slicing can reduce peak RAM at the cost of speed.
                try:
                    self.sd_pipe.enable_attention_slicing()
                except Exception:
                    logger.debug(
                        "CPU attention slicing unavailable."
                    )

                try:
                    self.sd_pipe.enable_vae_slicing()
                except Exception:
                    logger.debug(
                        "CPU VAE slicing unavailable."
                    )

            logger.success(
                "Stable Diffusion pipeline loaded successfully."
            )

        except Exception as exc:
            raise FrameGenerationError(
                f"Failed to initialize Stable Diffusion: {exc}"
            )

    def _load_ip_adapter(self) -> None:
        """
        IP-Adapter is disabled in CPU mode.

        It can consume significant additional memory and is not required
        for the CPU fallback pipeline.
        """

        if not self.is_cuda:
            logger.info(
                "CPU mode: IP-Adapter disabled."
            )

            self.ip_adapter_loaded = False
            return

        ip_model = DIFFUSION_CONFIG["ip_adapter_model"]

        logger.info(
            f"Loading IP-Adapter: {ip_model}"
        )

        try:
            self.sd_pipe.load_ip_adapter(
                ip_model,
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
                cache_dir=DIFFUSION_CONFIG["model_cache_dir"],
            )

            self.sd_pipe.set_ip_adapter_scale(
                DIFFUSION_CONFIG["ip_adapter_scale"]
            )

            self.ip_adapter_loaded = True

            logger.success(
                "IP-Adapter loaded successfully."
            )

        except Exception as exc:
            logger.warning(
                f"IP-Adapter failed to load: {exc}. "
                "Continuing without IP-Adapter."
            )

            self.ip_adapter_loaded = False

    def _load_animatediff(self) -> None:
        """
        AnimateDiff is GPU-only.

        Loading AnimateDiff on CPU is intentionally avoided because the
        temporal model can consume a very large amount of RAM.
        """

        if not self.is_cuda:
            self.anim_pipe = None
            return

        model_name = ANIMATOR_CONFIG["model_name"]
        cache_dir = ANIMATOR_CONFIG["model_cache_dir"]

        logger.info(
            f"Loading AnimateDiff motion adapter: {model_name}"
        )

        try:
            dtype = torch.float16

            adapter = MotionAdapter.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=dtype,
            )

            self.anim_pipe = AnimateDiffPipeline.from_pretrained(
                DIFFUSION_CONFIG["model_name"],
                motion_adapter=adapter,
                cache_dir=DIFFUSION_CONFIG["model_cache_dir"],
                torch_dtype=dtype,
            )

            self.anim_pipe.scheduler = DDIMScheduler.from_config(
                self.anim_pipe.scheduler.config,
                beta_schedule="sqrt_linear",
                clip_sample=False,
                timestep_spacing="linspace",
                steps_offset=1,
            )

            if self.ip_adapter_loaded:
                try:
                    self.anim_pipe.load_ip_adapter(
                        DIFFUSION_CONFIG["ip_adapter_model"],
                        subfolder="models",
                        weight_name="ip-adapter_sd15.bin",
                        cache_dir=DIFFUSION_CONFIG["model_cache_dir"],
                    )

                    self.anim_pipe.set_ip_adapter_scale(
                        DIFFUSION_CONFIG["ip_adapter_scale"]
                    )

                except Exception as exc:
                    logger.warning(
                        f"AnimateDiff IP-Adapter unavailable: {exc}"
                    )

            self._place_pipe_on_device(self.anim_pipe)

            self.anim_pipe.enable_attention_slicing()
            self.anim_pipe.enable_vae_slicing()

            logger.success(
                "AnimateDiff pipeline loaded successfully."
            )

        except Exception as exc:
            logger.warning(
                f"AnimateDiff initialization failed: {exc}. "
                "Falling back to static frame generation."
            )

            self.anim_pipe = None

    # ------------------------------------------------------------------
    # DEVICE MANAGEMENT
    # ------------------------------------------------------------------

    def _place_pipe_on_device(self, pipe) -> None:
        if self.is_cpu:
            pipe.to("cpu")
            return

        if self.use_cpu_offload:
            try:
                pipe.enable_model_cpu_offload()

                logger.info(
                    "GPU mode: model CPU offload enabled."
                )

                return

            except Exception as exc:
                logger.warning(
                    f"CPU offload unavailable: {exc}. "
                    "Loading pipeline directly onto GPU."
                )

        pipe.to("cuda")

    # ------------------------------------------------------------------
    # MAIN GENERATION
    # ------------------------------------------------------------------

    def generate(
        self,
        scenes: list[dict],
        face_embedding: Optional[np.ndarray],
        face_image_path: Optional[Path],
        output_dir: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Path:

        total_scenes = len(scenes)

        logger.info(
            f"Generating frames for {total_scenes} scenes."
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        face_image = self._load_face_image(
            face_image_path
        )

        for scene_idx, scene in enumerate(scenes):

            scene_num = scene.get(
                "index",
                scene_idx + 1,
            )

            scene_dir = (
                output_dir /
                f"scene_{scene_num:02d}"
            )

            scene_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            logger.info(
                f"Processing scene "
                f"{scene_idx + 1}/{total_scenes}: "
                f"{scene.get('title', 'Untitled')}"
            )

            try:
                self._generate_scene_frames(
                    scene=scene,
                    scene_dir=scene_dir,
                    face_image=face_image,
                    scene_idx=scene_idx,
                    progress_callback=progress_callback,
                    total_scenes=total_scenes,
                )

            except FrameGenerationError:
                raise

            except Exception as exc:
                raise FrameGenerationError(
                    str(exc),
                    scene_index=scene_idx,
                )

            self._free_memory()

        logger.success(
            f"All scene frames generated successfully: "
            f"{output_dir}"
        )

        return output_dir

    # ------------------------------------------------------------------
    # SCENE GENERATION
    # ------------------------------------------------------------------

    def _generate_scene_frames(
        self,
        scene: dict,
        scene_dir: Path,
        face_image: Optional[Image.Image],
        scene_idx: int,
        progress_callback: Optional[Callable],
        total_scenes: int,
    ) -> None:

        prompt = scene.get(
            "visual_prompt",
            "",
        )

        negative_prompt = scene.get(
            "negative_prompt",
            "ugly, blurry, low quality, distorted face",
        )

        num_frames = int(
            ANIMATOR_CONFIG.get(
                "num_frames",
                16,
            )
        )

        mood = scene.get(
            "mood",
            "happy",
        )

        prompt = self._enrich_prompt(
            prompt,
            mood,
        )

        logger.debug(
            f"Scene prompt: {prompt[:120]}..."
        )

        # --------------------------------------------------------------
        # GPU PATH
        # --------------------------------------------------------------

        if (
            self.is_cuda
            and self.anim_pipe is not None
        ):
            logger.info(
                f"Scene {scene_idx + 1}: "
                "using AnimateDiff."
            )

            frames = self._generate_animated_frames(
                prompt=prompt,
                negative_prompt=negative_prompt,
                face_image=face_image,
                num_frames=num_frames,
            )

        elif (
            self.is_cuda
            and self.ip_adapter_loaded
            and face_image is not None
        ):
            logger.info(
                f"Scene {scene_idx + 1}: "
                "using Stable Diffusion + IP-Adapter."
            )

            frames = self._generate_static_frames_with_face(
                prompt=prompt,
                negative_prompt=negative_prompt,
                face_image=face_image,
                num_frames=num_frames,
            )

        # --------------------------------------------------------------
        # CPU PATH
        # --------------------------------------------------------------

        else:
            logger.info(
                f"Scene {scene_idx + 1}: "
                "using CPU static-frame generation."
            )

            frames = self._generate_static_frames(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
            )

        self._save_frames(
            frames,
            scene_dir,
        )

        if progress_callback:
            progress_callback(
                scene_idx + 1,
                total_scenes,
                f"Scene {scene_idx + 1}/{total_scenes} Render Complete",
            )

    # ------------------------------------------------------------------
    # ANIMATEDIFF
    # ------------------------------------------------------------------

    def _generate_animated_frames(
        self,
        prompt: str,
        negative_prompt: str,
        face_image: Optional[Image.Image],
        num_frames: int,
    ) -> list[Image.Image]:

        if not self.is_cuda or self.anim_pipe is None:
            return self._generate_static_frames(
                prompt,
                negative_prompt,
                num_frames,
            )

        generator = self._get_generator()

        try:
            kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "num_inference_steps": DIFFUSION_CONFIG[
                    "num_inference_steps"
                ],
                "guidance_scale": DIFFUSION_CONFIG[
                    "guidance_scale"
                ],
                "width": DIFFUSION_CONFIG["width"],
                "height": DIFFUSION_CONFIG["height"],
                "generator": generator,
            }

            if (
                self.ip_adapter_loaded
                and face_image is not None
            ):
                kwargs["ip_adapter_image"] = face_image

            output = self.anim_pipe(
                **kwargs
            )

            return list(
                output.frames[0]
            )

        except torch.cuda.OutOfMemoryError:

            logger.warning(
                "CUDA OOM during AnimateDiff. "
                "Retrying with fewer frames."
            )

            self._free_memory()

            return self._generate_animated_frames_reduced(
                prompt,
                negative_prompt,
                face_image,
                num_frames,
            )

        except Exception as exc:

            logger.warning(
                f"AnimateDiff failed: {exc}. "
                "Falling back to static frames."
            )

            return self._generate_static_frames(
                prompt,
                negative_prompt,
                num_frames,
            )

    def _generate_animated_frames_reduced(
        self,
        prompt: str,
        negative_prompt: str,
        face_image: Optional[Image.Image],
        num_frames: int,
    ) -> list[Image.Image]:

        if not self.is_cuda or self.anim_pipe is None:
            return self._generate_static_frames(
                prompt,
                negative_prompt,
                num_frames,
            )

        reduced = max(
            8,
            num_frames // 2,
        )

        try:

            generator = self._get_generator()

            kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": reduced,
                "num_inference_steps": max(
                    15,
                    DIFFUSION_CONFIG[
                        "num_inference_steps"
                    ] - 5,
                ),
                "guidance_scale": DIFFUSION_CONFIG[
                    "guidance_scale"
                ],
                "width": DIFFUSION_CONFIG["width"],
                "height": DIFFUSION_CONFIG["height"],
                "generator": generator,
            }

            if (
                self.ip_adapter_loaded
                and face_image is not None
            ):
                kwargs["ip_adapter_image"] = face_image

            output = self.anim_pipe(
                **kwargs
            )

            frames = list(
                output.frames[0]
            )

            if not frames:
                raise RuntimeError(
                    "AnimateDiff returned zero frames."
                )

            while len(frames) < num_frames:
                frames.append(
                    frames[-1].copy()
                )

            return frames[:num_frames]

        except Exception as exc:

            logger.warning(
                f"Reduced AnimateDiff generation failed: "
                f"{exc}"
            )

            return self._generate_static_frames(
                prompt,
                negative_prompt,
                num_frames,
            )

    # ------------------------------------------------------------------
    # STATIC + IP ADAPTER
    # ------------------------------------------------------------------

    def _generate_static_frames_with_face(
        self,
        prompt: str,
        negative_prompt: str,
        face_image: Image.Image,
        num_frames: int,
    ) -> list[Image.Image]:

        if (
            not self.is_cuda
            or not self.ip_adapter_loaded
            or self.sd_pipe is None
        ):
            return self._generate_static_frames(
                prompt,
                negative_prompt,
                num_frames,
            )

        frames = []

        generator = self._get_generator()

        variations = self._get_prompt_variations(
            num_frames
        )

        for i, variation in enumerate(variations):

            frame_prompt = (
                f"{prompt}, {variation}"
            )

            try:

                output = self.sd_pipe(
                    prompt=frame_prompt,
                    negative_prompt=negative_prompt,
                    ip_adapter_image=face_image,
                    num_inference_steps=max(
                        15,
                        DIFFUSION_CONFIG[
                            "num_inference_steps"
                        ] - 5,
                    ),
                    guidance_scale=DIFFUSION_CONFIG[
                        "guidance_scale"
                    ],
                    width=DIFFUSION_CONFIG["width"],
                    height=DIFFUSION_CONFIG["height"],
                    generator=generator,
                )

                frames.append(
                    output.images[0]
                )

            except Exception as exc:

                logger.warning(
                    f"IP-Adapter frame {i + 1} failed: "
                    f"{exc}"
                )

                if frames:
                    frames.append(
                        frames[-1].copy()
                    )

            self._free_memory()

        if not frames:
            return self._generate_static_frames(
                prompt,
                negative_prompt,
                num_frames,
            )

        return frames[:num_frames]

    # ------------------------------------------------------------------
    # CPU / STATIC GENERATION
    # ------------------------------------------------------------------

    def _generate_static_frames(
        self,
        prompt: str,
        negative_prompt: str,
        num_frames: int,
    ) -> list[Image.Image]:

        if self.sd_pipe is None:
            raise FrameGenerationError(
                "Stable Diffusion pipeline is not initialized."
            )

        generator = self._get_generator()

        # CPU gets fewer inference steps.
        if self.is_cpu:

            steps = int(
                DIFFUSION_CONFIG.get(
                    "cpu_inference_steps",
                    15,
                )
            )

            guidance_scale = float(
                DIFFUSION_CONFIG.get(
                    "cpu_guidance_scale",
                    7.0,
                )
            )

        else:

            steps = int(
                DIFFUSION_CONFIG[
                    "num_inference_steps"
                ]
            )

            guidance_scale = float(
                DIFFUSION_CONFIG[
                    "guidance_scale"
                ]
            )

        width = int(
            DIFFUSION_CONFIG["width"]
        )

        height = int(
            DIFFUSION_CONFIG["height"]
        )

        # --------------------------------------------------------------
        # CPU MEMORY PROTECTION
        # --------------------------------------------------------------

        if self.is_cpu:

            max_cpu_width = int(
                DIFFUSION_CONFIG.get(
                    "cpu_max_width",
                    512,
                )
            )

            max_cpu_height = int(
                DIFFUSION_CONFIG.get(
                    "cpu_max_height",
                    512,
                )
            )

            width = min(
                width,
                max_cpu_width,
            )

            height = min(
                height,
                max_cpu_height,
            )

            # Stable Diffusion dimensions should normally be divisible
            # by 8.
            width = max(
                256,
                (width // 8) * 8,
            )

            height = max(
                256,
                (height // 8) * 8,
            )

            logger.debug(
                f"CPU generation resolution: "
                f"{width}x{height}, "
                f"{steps} steps."
            )

        try:

            with torch.inference_mode():

                output = self.sd_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                )

            base_image = output.images[0]

            # ----------------------------------------------------------
            # IMPORTANT:
            # Generate ONE image only.
            #
            # We don't run Stable Diffusion num_frames times.
            # This dramatically reduces CPU processing time.
            # ----------------------------------------------------------

            frames = [
                base_image.copy()
                for _ in range(num_frames)
            ]

            return frames

        except Exception as exc:

            logger.error(
                f"Static frame generation failed: {exc}"
            )

            raise FrameGenerationError(
                "Stable Diffusion failed to generate "
                "the scene image."
            )

        finally:
            self._free_memory()

    # ------------------------------------------------------------------
    # IMAGE SAVING
    # ------------------------------------------------------------------

    def _save_frames(
        self,
        frames: list[Image.Image],
        output_dir: Path,
    ) -> None:

        if not frames:
            raise FrameGenerationError(
                "No frames were generated."
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        target_w = int(
            DIFFUSION_CONFIG["width"]
        )

        target_h = int(
            DIFFUSION_CONFIG["height"]
        )

        for i, frame in enumerate(
            frames,
            start=1,
        ):

            frame_path = (
                output_dir /
                f"frame_{i:04d}.png"
            )

            if frame.mode != "RGB":
                frame = frame.convert("RGB")

            if frame.size != (
                target_w,
                target_h,
            ):
                frame = frame.resize(
                    (target_w, target_h),
                    Image.LANCZOS,
                )

            frame.save(
                frame_path,
                format="PNG",
                optimize=False,
            )

    # ------------------------------------------------------------------
    # FACE IMAGE
    # ------------------------------------------------------------------

    def _load_face_image(
        self,
        face_image_path: Optional[Path],
    ) -> Optional[Image.Image]:

        if not face_image_path:
            return None

        path = Path(
            face_image_path
        )

        if not path.exists():
            return None

        try:

            with Image.open(path) as img:

                image = img.convert(
                    "RGB"
                )

                return image.resize(
                    (224, 224),
                    Image.LANCZOS,
                )

        except Exception as exc:

            logger.warning(
                f"Unable to load face reference: {exc}"
            )

            return None

    # ------------------------------------------------------------------
    # RANDOM GENERATOR
    # ------------------------------------------------------------------

    def _get_generator(
        self,
    ) -> torch.Generator:

        generator = torch.Generator(
            device="cpu"
        )

        if self.seed is not None:
            generator.manual_seed(
                self.seed
            )
        else:
            generator.seed()

        return generator

    # ------------------------------------------------------------------
    # PROMPTS
    # ------------------------------------------------------------------

    def _truncate_prompt(
        self,
        prompt: str,
        max_words: int = 50,
    ) -> str:

        words = prompt.split()

        return " ".join(
            words[:max_words]
        )

    def _enrich_prompt(
        self,
        prompt: str,
        mood: str,
    ) -> str:

        quality_suffix = (
            "masterpiece, best quality, "
            "highly detailed, sharp focus"
        )

        mood_keywords = {
            "happy": (
                "bright cinematic lighting, "
                "warm vibrant tones, "
                "cheerful landscape"
            ),

            "adventure": (
                "dynamic composition, "
                "sweeping cinematic views, "
                "adventurous theme"
            ),

            "magical": (
                "magical sparkles, "
                "glowing elements, "
                "ethereal soft light, "
                "wonderland vibe"
            ),

            "exciting": (
                "dramatic composition, "
                "high contrast action tones, "
                "intense colors"
            ),

            "mysterious": (
                "atmospheric deep shadows, "
                "mysterious soft fog"
            ),

            "heroic": (
                "epic lighting profile, "
                "dramatic golden hour, "
                "triumphant composition"
            ),
        }

        mood_kw = mood_keywords.get(
            mood,
            "vibrant colors, whimsical setup",
        )

        final_prompt = (
            f"{prompt}, "
            f"{mood_kw}, "
            f"{quality_suffix}"
        )

        return self._truncate_prompt(
            final_prompt,
            50,
        )

    def _get_prompt_variations(
        self,
        n: int,
    ) -> list[str]:

        variations = [
            "slightly left angle cinematic pan",
            "slightly right angle frame pan",
            "close-up detail layout",
            "wide cinematic viewing angle",
        ]

        return [
            variations[i % len(variations)]
            for i in range(n)
        ]

    # ------------------------------------------------------------------
    # MEMORY MANAGEMENT
    # ------------------------------------------------------------------

    def _free_memory(self) -> None:

        gc.collect()

        if (
            self.is_cuda
            and torch.cuda.is_available()
        ):
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """
        Explicitly unload pipelines when the generator is no longer needed.
        """

        logger.info(
            "Releasing FrameGenerator resources..."
        )

        self.anim_pipe = None
        self.sd_pipe = None

        self.ip_adapter_loaded = False

        self._free_memory()

        logger.info(
            "FrameGenerator resources released."
        )

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass
