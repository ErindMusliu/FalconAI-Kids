import gc
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from config.settings import DEVICE, PIPELINE_CONFIG, OUTPUT_DIR
from utils.exceptions import (
    AudioGenerationError,
    FalconAIError,
    FalconAIException,
    FrameGenerationError,
    ModelLoadError,
    StoryGenerationError,
    VideoAssemblyError,
)
from utils.logger import get_logger

from pipeline.audio_generator import AudioGenerator
from pipeline.story_generator import StoryGenerator
from pipeline.video_assembler import VideoAssembler


logger = get_logger(__name__)


class PipelineOrchestrator:
    """
    Main FalconAI-Kids 3D Animation Generation Pipeline.

    Default execution graph (Headless 3D Film Engine):

        1. story_generator   (LLM Storyboard Generation)
        2. audio_generator   (TTS & Narration Synthesis)
        3. blender_renderer  (Headless 3D Scene Assembly & Frame Rendering)
        4. video_assembler   (FFmpeg Audio/Video Sync & Rendering)

    The legacy 2D frame generator and 2D character animators are seamlessly
    replaced by the high-performance Headless Blender 3D Engine.
    """

    REQUIRED_CONTEXT_KEYS = (
        "name",
        "birthday",
    )

    DEFAULT_STEPS = (
        "story_generator",
        "audio_generator",
        "blender_renderer",
        "video_assembler",
    )

    def __init__(self, context: Dict[str, Any]):
        self._validate_context(context)

        self.context = dict(context)

        # Safe defaults
        self.context.setdefault("language", "Albanian")
        self.context.setdefault("preferences", {})
        self.context.setdefault("seed", None)

        # Output directory setup
        self.output_dir = Path(
            self.context.get("output_dir", OUTPUT_DIR)
        ).resolve()
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Temporary working directory
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.cleanup_temp = PIPELINE_CONFIG.get(
            "cleanup_temp",
            True,
        )

        logger.info(
            f"Pipeline Orchestrator initialized | "
            f"device={DEVICE.upper()} | "
            f"language={self.context['language']} | "
            f"output={self.output_dir}"
        )

    # ------------------------------------------------------------------
    # CONTEXT VALIDATION
    # ------------------------------------------------------------------

    def _validate_context(
        self,
        context: Dict[str, Any],
    ) -> None:
        if not isinstance(context, dict):
            raise ValueError(
                "Pipeline context must be a dictionary."
            )

        missing = [
            key
            for key in self.REQUIRED_CONTEXT_KEYS
            if not context.get(key)
        ]

        if missing:
            raise ValueError(
                "Missing required pipeline context key(s): "
                + ", ".join(missing)
            )

    # ------------------------------------------------------------------
    # DYNAMIC PROCESSOR LOADING
    # ------------------------------------------------------------------

    def _load_single_processor(
        self,
        step_name: str,
    ) -> Any:
        """
        Lazy-load processor instances per pipeline step to conserve RAM/VRAM.
        """
        logger.debug(
            f"Loading processor for pipeline step: {step_name}"
        )

        language = self.context.get(
            "language",
            "Albanian",
        )

        try:
            if step_name == "story_generator":
                return StoryGenerator(
                    language=language,
                )

            if step_name == "audio_generator":
                return AudioGenerator(
                    language=language,
                )

            if step_name == "blender_renderer":
                from pipeline.blender_renderer import BlenderRenderer

                return BlenderRenderer(context=self.context)

            if step_name == "video_assembler":
                return VideoAssembler()

            # Backward-compatible 2D fallback generator
            if step_name == "frame_generator":
                from pipeline.frame_generator import FrameGenerator

                return FrameGenerator(
                    seed=self.context.get("seed"),
                )

            # Optional Legacy Face Processor
            if step_name == "face_processor":
                from pipeline.face_processor import FaceProcessor

                return FaceProcessor()

            raise ValueError(
                f"Unrecognized pipeline step: {step_name}"
            )

        except FalconAIError as e:
            raise ModelLoadError(
                f"Failed to initialize processor '{step_name}'.",
                str(e),
            )

        except Exception as e:
            raise ModelLoadError(
                f"Unexpected error while initializing processor '{step_name}'.",
                str(e),
            )

    # ------------------------------------------------------------------
    # STEP EXECUTION GRAPH
    # ------------------------------------------------------------------

    def _run_step(
        self,
        step_name: str,
        progress_callback: Optional[Callable] = None,
    ) -> Any:
        self._cleanup_memory()

        processor = self._load_single_processor(step_name)

        try:
            # ==========================================================
            # 1. STORY GENERATION
            # ==========================================================
            if step_name == "story_generator":
                story_data = processor.generate(
                    name=self.context["name"],
                    birthday=self.context["birthday"],
                    gender=self.context.get("gender"),
                    preferences=self.context.get(
                        "preferences",
                        {},
                    ),
                )

                if not story_data or not isinstance(story_data, dict):
                    raise StoryGenerationError(
                        "Story generator returned invalid or empty story data."
                    )

                if not story_data.get("scenes"):
                    raise StoryGenerationError(
                        "Generated story contains no valid scenes."
                    )

                self.context["story"] = story_data
                self.context["story_scenes"] = story_data.get("scenes", [])

                logger.success(
                    f"Story generation completed | "
                    f"title='{story_data.get('title', 'Untitled')}' | "
                    f"scenes={len(story_data.get('scenes', []))}"
                )
                return story_data

            # ==========================================================
            # 2. AUDIO GENERATION
            # ==========================================================
            if step_name == "audio_generator":
                if "story" not in self.context:
                    raise AudioGenerationError(
                        "Audio generation requires a completed story context."
                    )

                audio_dir = self.output_dir / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)

                audio_paths = processor.generate(
                    story=self.context["story"],
                    output_dir=audio_dir,
                    language=self.context.get("language", "Albanian"),
                )

                if not audio_paths:
                    raise AudioGenerationError(
                        "Audio generator produced no output files."
                    )

                self.context["audio_paths"] = audio_paths

                logger.success("Audio generation completed successfully.")
                return audio_paths

            # ==========================================================
            # 3. HEADLESS 3D BLENDER RENDERING
            # ==========================================================
            if step_name == "blender_renderer":
                if "story" not in self.context:
                    raise FrameGenerationError(
                        "3D Blender rendering requires a completed story."
                    )

                rendered_frames = processor.render(
                    progress_callback=progress_callback
                )

                if not rendered_frames:
                    raise FrameGenerationError(
                        "Blender 3D Engine failed to render output frames."
                    )

                self.context["rendered_frames"] = rendered_frames
                self.context["frames_dir"] = processor.frames_dir

                logger.success(
                    f"3D Blender rendering completed | "
                    f"total_frames={len(rendered_frames)}"
                )
                return rendered_frames

            # ==========================================================
            # 4. VIDEO ASSEMBLY & COMPOSITING
            # ==========================================================
            if step_name == "video_assembler":
                if "rendered_frames" not in self.context and "frames_dir" not in self.context:
                    raise VideoAssemblyError(
                        "Video assembly requires rendered 3D frames."
                    )

                if "audio_paths" not in self.context:
                    raise VideoAssemblyError(
                        "Video assembly requires audio soundtrack paths."
                    )

                video_path = self.output_dir / "final_storybook.mp4"

                final_video = processor.assemble(
                    scenes=self.context["story"]["scenes"],
                    frames_dir=self.context.get("frames_dir"),
                    audio_paths=self.context["audio_paths"],
                    output_path=video_path,
                )

                if not final_video:
                    raise VideoAssemblyError(
                        "Video assembler returned no final video file."
                    )

                self.context["final_video"] = final_video

                logger.success(
                    f"Video assembly completed | output={final_video}"
                )
                return final_video

            # ==========================================================
            # LEGACY / FALLBACK STEPS
            # ==========================================================
            if step_name == "face_processor":
                photo_input = self.context.get("photo")
                if not photo_input:
                    logger.warning("No photo supplied. Face processing skipped.")
                    self.context["face_texture_path"] = None
                    return None

                face_result = processor.process(
                    photo_path=Path(photo_input),
                    temp_dir=self.temp_dir,
                )
                self.context["face_texture_path"] = face_result.get("face_image_path")
                return face_result

            raise ValueError(f"Unsupported pipeline step: {step_name}")

        finally:
            del processor
            self._cleanup_memory()

    # ------------------------------------------------------------------
    # MAIN EXECUTION ROUTINE
    # ------------------------------------------------------------------

    def run(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> Path:
        configured_steps = PIPELINE_CONFIG.get("steps")
        steps = list(configured_steps) if configured_steps else list(self.DEFAULT_STEPS)

        if not steps:
            raise FalconAIException(
                "Pipeline contains no execution steps.",
                code="EMPTY_PIPELINE",
            )

        total_steps = len(steps)
        logger.info(
            f"Launching FalconAI-Kids 3D Pipeline | {total_steps} step(s)"
        )

        try:
            for index, step_name in enumerate(steps, start=1):
                logger.info(
                    f"--- [STEP {index}/{total_steps}] {step_name.upper()} ---"
                )
                self._run_step(step_name, progress_callback)

            # Final Output Verification
            final_video = self.context.get("final_video")
            expected_video = self.output_dir / "final_storybook.mp4"
            final_video_path = Path(final_video) if final_video else expected_video

            if not final_video_path.exists():
                raise VideoAssemblyError(
                    "Pipeline completed all steps, but final_storybook.mp4 was not found."
                )

            logger.success(
                f"FalconAI-Kids 3D Pipeline finished successfully! "
                f"Output: {final_video_path}"
            )
            return final_video_path

        except FalconAIException:
            raise

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise FalconAIException(str(e), code="PIPELINE_FAILURE")

        finally:
            if self.cleanup_temp:
                self._cleanup_temp_dir()
            self._cleanup_memory()

    # ------------------------------------------------------------------
    # RESOURCE & MEMORY CLEANUP
    # ------------------------------------------------------------------

    def _cleanup_memory(self) -> None:
        """
        Release Python GC and CUDA VRAM allocations between processing steps.
        """
        try:
            gc.collect()
        except Exception:
            pass

        if DEVICE == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception as e:
                logger.debug(f"CUDA memory cleanup skipped: {e}")

    def _cleanup_temp_dir(self) -> None:
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.debug(f"Temporary directory removed: {self.temp_dir}")
        except Exception as e:
            logger.debug(f"Temporary directory cleanup non-fatal warning: {e}")


# Backward Compatibility Alias
Orchestrator = PipelineOrchestrator
