import gc
import time
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
    def __init__(self, seed: Optional[int] = None, use_cpu_offload: bool = True):
        self.seed = seed
        self.sd_pipe = None
        self.anim_pipe = None
        self.ip_adapter_loaded = False

        self.use_cpu_offload = use_cpu_offload and DEVICE == "cuda"

        self._load_models()

    def _load_models(self) -> None:
        try:
            self._load_stable_diffusion()
            self._load_ip_adapter()

            if DEVICE == "cuda":
                self._load_animatediff()
            else:
                logger.info("Current execution target is CPU. Skipping AnimateDiff pipeline to prevent RAM exhaustion.")
                self.anim_pipe = None
        except FrameGenerationError:
            raise
        except Exception as e:
            raise ModelLoadError("FrameGenerator", str(e))

    def _load_stable_diffusion(self) -> None:
        model_name = DIFFUSION_CONFIG["model_name"]
        cache_dir = DIFFUSION_CONFIG["model_cache_dir"]

        logger.debug(f"Loading Base Stable Diffusion Engine: {model_name}")
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

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

            if DEVICE == "cuda":
                self.sd_pipe.enable_attention_slicing()
                self.sd_pipe.enable_vae_slicing()
                try:
                    self.sd_pipe.enable_xformers_memory_efficient_attention()
                    logger.debug("xFormers memory-efficient attention activated for SD pipeline")
                except Exception:
                    logger.debug("xFormers optimization library unavailable, continuing with native attention handlers")

            logger.success("Stable Diffusion base pipeline successfully loaded!")

        except Exception as e:
            raise FrameGenerationError(f"Critical exception raised during base SD instantiation: {e}")

    def _load_ip_adapter(self) -> None:
        if DEVICE != "cuda":
            logger.info("Skipping IP-Adapter architecture setup under standard CPU layout to prevent system stalls.")
            self.ip_adapter_loaded = False
            return

        ip_model = DIFFUSION_CONFIG["ip_adapter_model"]
        logger.debug(f"Loading IP-Adapter reference models: {ip_model}")

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
            logger.success("IP-Adapter reference mapping engine successfully bound to SD pipeline")

        except Exception as e:
            logger.warning(f"IP-Adapter engine failed initialization: {e}. Falling back to standard generative text modes.")
            self.ip_adapter_loaded = False

    def _load_animatediff(self) -> None:
        model_name = ANIMATOR_CONFIG["model_name"]
        cache_dir = ANIMATOR_CONFIG["model_cache_dir"]

        logger.debug(f"Loading AnimateDiff Temporal Layers: {model_name}")

        try:
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32

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
                self.anim_pipe.load_ip_adapter(
                    DIFFUSION_CONFIG["ip_adapter_model"],
                    subfolder="models",
                    weight_name="ip-adapter_sd15.bin",
                    cache_dir=DIFFUSION_CONFIG["model_cache_dir"],
                )
                self.anim_pipe.set_ip_adapter_scale(
                    DIFFUSION_CONFIG["ip_adapter_scale"]
                )

            self._place_pipe_on_device(self.anim_pipe)
            self.anim_pipe.enable_attention_slicing()
            self.anim_pipe.enable_vae_slicing()

            logger.success("AnimateDiff temporal sequencing rendering layers initialized successfully")

        except Exception as e:
            logger.warning(f"AnimateDiff core subsystem mapping failed: {e}. Automatically reverting pipeline to static render states.")
            self.anim_pipe = None

    def _place_pipe_on_device(self, pipe) -> None:
        if DEVICE != "cuda":
            pipe.to("cpu")
            return

        if self.use_cpu_offload:
            try:
                pipe.enable_model_cpu_offload()
                logger.debug("CPU offload enabled for pipeline (lower VRAM usage, slightly slower).")
                return
            except Exception as e:
                logger.warning(f"CPU offload unavailable ({e}); loading pipeline fully onto GPU instead.")

        pipe.to("cuda")

    def generate(
        self,
        scenes: list[dict],
        face_embedding: Optional[np.ndarray],
        face_image_path: Optional[Path],
        output_dir: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Path:
        total_scenes = len(scenes)
        logger.debug(f"Generating image frame sequences for {total_scenes} distinct narration scenes")

        face_image = self._load_face_image(face_image_path)

        for scene_idx, scene in enumerate(scenes):
            scene_num = scene.get("index", scene_idx + 1)
            scene_dir = output_dir / f"scene_{scene_num:02d}"
            scene_dir.mkdir(parents=True, exist_ok=True)

            logger.step(f"Scene processing index [{scene_num}/{total_scenes}]: '{scene.get('title', 'Untitled')}'")

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
            except Exception as e:
                raise FrameGenerationError(str(e), scene_index=scene_idx)

            self._free_memory()

        logger.success(f"All structural scene frame pipelines fully compiled at location: {output_dir}")
        return output_dir

    def _generate_scene_frames(
        self,
        scene: dict,
        scene_dir: Path,
        face_image: Optional[Image.Image],
        scene_idx: int,
        progress_callback: Optional[Callable],
        total_scenes: int,
    ) -> None:
        prompt = scene.get("visual_prompt", "")
        negative_prompt = scene.get("negative_prompt", "ugly, blurry, low quality, distorted face")
        num_frames = ANIMATOR_CONFIG["num_frames"]
        mood = scene.get("mood", "happy")

        prompt = self._enrich_prompt(prompt, mood)
        logger.debug(f"Compiled Prompt: {prompt[:80]}... | Targeted Frame Length: {num_frames}")

        if self.anim_pipe is not None and self.anim_pipe.unet is not None and DEVICE == "cuda":
            logger.debug(f"Scene {scene_idx + 1}: using AnimateDiff (real motion, {num_frames} frames)")
            frames = self._generate_animated_frames(
                prompt=prompt,
                negative_prompt=negative_prompt,
                face_image=face_image,
                num_frames=num_frames,
            )
        elif self.ip_adapter_loaded and face_image is not None and DEVICE == "cuda":
            logger.debug(f"Scene {scene_idx + 1}: AnimateDiff unavailable, using varied static frames with face reference")
            frames = self._generate_static_frames_with_face(
                prompt=prompt,
                negative_prompt=negative_prompt,
                face_image=face_image,
                num_frames=num_frames,
            )
        else:
            logger.debug(f"Scene {scene_idx + 1}: falling back to fully static frame (no GPU/AnimateDiff available)")
            frames = self._generate_static_frames(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
            )

        self._save_frames(frames, scene_dir)

        if progress_callback:
            progress_callback(
                scene_idx + 1,
                total_scenes,
                f"Scene {scene_idx + 1}/{total_scenes} Render Complete"
            )

    def _generate_animated_frames(
        self,
        prompt: str,
        negative_prompt: str,
        face_image: Optional[Image.Image],
        num_frames: int,
    ) -> list[Image.Image]:
        generator = self._get_generator()
        try:
            kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "num_inference_steps": DIFFUSION_CONFIG["num_inference_steps"],
                "guidance_scale": DIFFUSION_CONFIG["guidance_scale"],
                "width": DIFFUSION_CONFIG["width"],
                "height": DIFFUSION_CONFIG["height"],
                "generator": generator,
            }

            if self.ip_adapter_loaded and face_image:
                kwargs["ip_adapter_image"] = face_image

            output = self.anim_pipe(**kwargs)
            return list(output.frames[0])

        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA out of memory during AnimateDiff generation. Retrying with fewer frames...")
            self._free_memory()
            return self._generate_animated_frames_reduced(prompt, negative_prompt, face_image, num_frames)
        except Exception as e:
            logger.warning(f"AnimateDiff pipeline execution failure encountered: {e}. Automatically falling back to static structures.")
            return self._generate_static_frames(prompt, negative_prompt, num_frames)

    def _generate_animated_frames_reduced(
        self,
        prompt: str,
        negative_prompt: str,
        face_image: Optional[Image.Image],
        num_frames: int,
    ) -> list[Image.Image]:
        reduced = max(8, num_frames // 2)
        try:
            generator = self._get_generator()
            kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": reduced,
                "num_inference_steps": max(15, DIFFUSION_CONFIG["num_inference_steps"] - 5),
                "guidance_scale": DIFFUSION_CONFIG["guidance_scale"],
                "width": DIFFUSION_CONFIG["width"],
                "height": DIFFUSION_CONFIG["height"],
                "generator": generator,
            }
            if self.ip_adapter_loaded and face_image:
                kwargs["ip_adapter_image"] = face_image

            output = self.anim_pipe(**kwargs)
            frames = list(output.frames[0])

            while len(frames) < num_frames:
                frames.append(frames[-1])
            return frames

        except Exception as e:
            logger.warning(f"Reduced-frame AnimateDiff retry also failed: {e}. Falling back to static frame.")
            return self._generate_static_frames(prompt, negative_prompt, num_frames)

    def _generate_static_frames_with_face(
        self,
        prompt: str,
        negative_prompt: str,
        face_image: Image.Image,
        num_frames: int,
    ) -> list[Image.Image]:
        frames = []
        generator = self._get_generator()
        variations = self._get_prompt_variations(num_frames)

        for i, variation in enumerate(variations):
            frame_prompt = f"{prompt}, {variation}"
            try:
                output = self.sd_pipe(
                    prompt=frame_prompt,
                    negative_prompt=negative_prompt,
                    ip_adapter_image=face_image,
                    num_inference_steps=max(15, DIFFUSION_CONFIG["num_inference_steps"] - 5),
                    guidance_scale=DIFFUSION_CONFIG["guidance_scale"],
                    width=DIFFUSION_CONFIG["width"],
                    height=DIFFUSION_CONFIG["height"],
                    generator=generator,
                )
                frames.append(output.images[0])
            except Exception as e:
                logger.warning(f"Frame {i+1} structural generation failed: {e}")
                if frames:
                    frames.append(frames[-1])

        return frames

    def _generate_static_frames(
        self,
        prompt: str,
        negative_prompt: str,
        num_frames: int,
    ) -> list[Image.Image]:
        frames = []
        generator = self._get_generator()

        steps = 15 if DEVICE != "cuda" else DIFFUSION_CONFIG["num_inference_steps"]

        try:
            output = self.sd_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=7.0,
                width=DIFFUSION_CONFIG["width"],
                height=DIFFUSION_CONFIG["height"],
                generator=generator,
            )
            base_image = output.images[0]

            frames = [base_image for _ in range(num_frames)]

        except Exception as e:
            logger.error(f"Critical inference pipeline failure occurred during standard frame processing operations: {e}")

        if not frames:
            raise FrameGenerationError("Active pipeline configuration layout was unable to generate structural image matrices.")

        return frames

    def _save_frames(self, frames: list[Image.Image], output_dir: Path) -> None:
        for i, frame in enumerate(frames, 1):
            frame_path = output_dir / f"frame_{i:04d}.png"
            if frame.mode != "RGB":
                frame = frame.convert("RGB")

            target_w, target_h = DIFFUSION_CONFIG["width"], DIFFUSION_CONFIG["height"]
            if frame.size != (target_w, target_h):
                frame = frame.resize((target_w, target_h), Image.LANCZOS)

            frame.save(frame_path, format="PNG", optimize=False)

    def _load_face_image(self, face_image_path: Optional[Path]) -> Optional[Image.Image]:
        if not face_image_path or not Path(face_image_path).exists():
            return None
        try:
            img = Image.open(face_image_path).convert("RGB")
            return img.resize((224, 224), Image.LANCZOS)
        except Exception:
            return None

    def _get_generator(self) -> torch.Generator:
        generator_device = "cpu" if self.use_cpu_offload else DEVICE
        generator = torch.Generator(device=generator_device)
        if self.seed is not None:
            generator.manual_seed(self.seed)
        else:
            generator.seed()
        return generator

    def _truncate_prompt(self, prompt: str, max_words: int = 50) -> str:
        words = prompt.split()
        return " ".join(words[:max_words])

    def _enrich_prompt(self, prompt: str, mood: str) -> str:
        quality_suffix = "masterpiece, best quality, highly detailed, sharp focus, 8k resolution"
        mood_keywords = {
            "happy": "bright cinematic lighting, warm vibrant tones, cheerful landscape",
            "adventure": "dynamic composition, sweeping cinematic views, adventurous theme",
            "magical": "magical sparkles, glowing elements, ethereal soft light, wonderland vibe",
            "exciting": "dramatic composition, high contrast action tones, intense colors",
            "mysterious": "atmospheric deep shadow effects, mysterious soft fog rolling in",
            "heroic": "epic lighting profile, dramatic golden hour, triumphant composition",
        }
        mood_kw = mood_keywords.get(mood, "vibrant colors, whimsical setup")
        final_prompt = f"{prompt}, {mood_kw}, {quality_suffix}"
        return self._truncate_prompt(final_prompt, 50)

    def _get_prompt_variations(self, n: int) -> list[str]:
        variations = ["slightly left angle cinematic pan", "slightly right angle frame pan", "close-up detail layout", "wide cinematic viewing angle"]
        return [variations[i % len(variations)] for i in range(n)]

    def _free_memory(self) -> None:
        gc.collect()
        if DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
