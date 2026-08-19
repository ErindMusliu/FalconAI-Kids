import gc
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from config.settings import OUTPUT_DIR, DEVICE, PIPELINE_CONFIG

from utils.exceptions import (
    FalconAIException,
    FalconAIError,
    ModelLoadError,
    FaceProcessingError,
    StoryGenerationError,
    AudioGenerationError,
    FrameGenerationError,
    CharacterAnimationError,
    VideoAssemblyError,
)

from utils.logger import get_logger

from pipeline.story_generator import StoryGenerator
from pipeline.audio_generator import AudioGenerator
from pipeline.frame_generator import FrameGenerator
from pipeline.video_assembler import VideoAssembler


logger = get_logger(__name__)


class PipelineOrchestrator:
    """
    Main controller for the FalconAI Kids generation pipeline.

    Default pipeline:

        story_generator
            ↓
        audio_generator
            ↓
        frame_generator
            ↓
        video_assembler

    Optional legacy stages:

        face_processor
        character_animator

    The optional stages are imported lazily so that the application can run
    in lightweight CPU-only environments without requiring heavy dependencies
    such as OpenCV, InsightFace, rembg, or SadTalker.

    The default pipeline does NOT require:
        - a real child photo
        - face recognition
        - InsightFace
        - SadTalker
        - rembg
        - CUDA
        - GPU hardware

    The pipeline is designed to degrade cleanly where possible and to keep
    all stage-specific state inside self.context.
    """

    REQUIRED_CONTEXT_KEYS = (
        "name",
        "birthday",
    )

    # ------------------------------------------------------------------
    # DEFAULT PIPELINE
    # ------------------------------------------------------------------

    DEFAULT_STEPS = (
        "story_generator",
        "audio_generator",
        "frame_generator",
        "video_assembler",
    )

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def __init__(self, context: Dict[str, Any]):
        self._validate_context(context)

        self.context = dict(context)

        # Sensible defaults
        self.context.setdefault("language", "Albanian")
        self.context.setdefault("preferences", {})
        self.context.setdefault("seed", None)

        # Output directory
        self.output_dir = Path(
            self.context.get("output_dir", OUTPUT_DIR)
        )
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
            "Pipeline Orchestrator initialized | "
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
    # PROCESSOR LOADING
    # ------------------------------------------------------------------

    def _load_single_processor(
        self,
        step_name: str,
    ) -> Any:
        """
        Creates the processor associated with one pipeline step.

        Heavy optional processors are imported lazily. This is important for
        CPU-only/cloud environments where cv2, InsightFace, rembg or SadTalker
        may not be installed.
        """

        logger.debug(
            f"Loading processor for step: {step_name}"
        )

        language = self.context.get(
            "language",
            "Albanian",
        )

        try:

            # ----------------------------------------------------------
            # OPTIONAL: FACE PROCESSOR
            # ----------------------------------------------------------

            if step_name == "face_processor":
                from pipeline.face_processor import FaceProcessor

                return FaceProcessor()

            # ----------------------------------------------------------
            # STORY
            # ----------------------------------------------------------

            if step_name == "story_generator":
                return StoryGenerator(
                    language=language,
                )

            # ----------------------------------------------------------
            # AUDIO
            # ----------------------------------------------------------

            if step_name == "audio_generator":
                return AudioGenerator(
                    language=language,
                )

            # ----------------------------------------------------------
            # FRAMES
            # ----------------------------------------------------------

            if step_name == "frame_generator":
                return FrameGenerator(
                    seed=self.context.get("seed"),
                )

            # ----------------------------------------------------------
            # OPTIONAL: CHARACTER ANIMATION
            # ----------------------------------------------------------

            if step_name == "character_animator":
                from pipeline.character_animator import CharacterAnimator

                return CharacterAnimator()

            # ----------------------------------------------------------
            # VIDEO
            # ----------------------------------------------------------

            if step_name == "video_assembler":
                return VideoAssembler()

            raise ValueError(
                f"Unrecognized pipeline step: {step_name}"
            )

        except FalconAIError as e:
            raise ModelLoadError(
                f"Failed to load processor for '{step_name}'.",
                str(e),
            )

        except Exception as e:
            raise ModelLoadError(
                f"Unexpected error while loading '{step_name}'.",
                str(e),
            )

    # ------------------------------------------------------------------
    # STEP EXECUTION
    # ------------------------------------------------------------------

    def _run_step(
        self,
        step_name: str,
        progress_callback: Optional[Callable] = None,
    ) -> Any:

        self._cleanup_memory()

        processor = self._load_single_processor(
            step_name
        )

        try:

            # ==========================================================
            # FACE PROCESSOR
            # ==========================================================

            if step_name == "face_processor":

                photo_input = self.context.get(
                    "photo"
                )

                if not photo_input:
                    logger.warning(
                        "No portrait reference photo supplied. "
                        "Skipping face processing."
                    )

                    self.context["face_embedding"] = None
                    self.context["face_image_path"] = None

                    return None

                photo_path = Path(
                    photo_input
                )

                if not photo_path.exists():
                    raise FaceProcessingError(
                        f"Portrait image does not exist: {photo_path}"
                    )

                face_result = processor.process(
                    photo_path=photo_path,
                    temp_dir=self.temp_dir,
                )

                self.context["face_embedding"] = (
                    face_result.get("embedding")
                )

                self.context["face_image_path"] = (
                    face_result.get("face_image_path")
                )

                self.context["face_result"] = face_result

                return face_result

            # ==========================================================
            # STORY GENERATOR
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

                if not story_data:
                    raise StoryGenerationError(
                        "Story generator returned no story data."
                    )

                if not isinstance(story_data, dict):
                    raise StoryGenerationError(
                        "Story generator returned an invalid data structure."
                    )

                if "scenes" not in story_data:
                    raise StoryGenerationError(
                        "Generated story does not contain 'scenes'."
                    )

                if not story_data["scenes"]:
                    raise StoryGenerationError(
                        "Generated story contains no scenes."
                    )

                self.context["story"] = story_data

                logger.success(
                    "Story generation completed | "
                    f"scenes={len(story_data['scenes'])}"
                )

                return story_data

            # ==========================================================
            # AUDIO GENERATOR
            # ==========================================================

            if step_name == "audio_generator":

                if "story" not in self.context:
                    raise AudioGenerationError(
                        "Audio generation requires a completed story."
                    )

                audio_dir = self.output_dir / "audio"

                audio_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                final_audio_path = processor.generate(
                    story=self.context["story"],
                    output_dir=audio_dir,
                    language=self.context.get(
                        "language",
                        "Albanian",
                    ),
                )

                if not final_audio_path:
                    raise AudioGenerationError(
                        "Audio generator returned no output."
                    )

                self.context["audio_paths"] = (
                    final_audio_path
                )

                logger.success(
                    f"Audio generation completed | output={final_audio_path}"
                )

                return final_audio_path

            # ==========================================================
            # FRAME GENERATOR
            # ==========================================================

            if step_name == "frame_generator":

                if "story" not in self.context:
                    raise FrameGenerationError(
                        "Frame generation requires a completed story."
                    )

                frames_dir = self.output_dir / "frames"

                frames_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                frames_paths = processor.generate(
                    scenes=self.context["story"]["scenes"],
                    face_embedding=self.context.get(
                        "face_embedding"
                    ),
                    face_image_path=self.context.get(
                        "face_image_path"
                    ),
                    output_dir=frames_dir,
                    progress_callback=progress_callback,
                )

                if not frames_paths:
                    raise FrameGenerationError(
                        "Frame generator returned no frames."
                    )

                self.context["frames_dir"] = (
                    frames_paths
                )

                logger.success(
                    f"Frame generation completed | output={frames_paths}"
                )

                return frames_paths

            # ==========================================================
            # CHARACTER ANIMATOR
            # ==========================================================

            if step_name == "character_animator":

                if "frames_dir" not in self.context:
                    raise CharacterAnimationError(
                        "Character animation requires completed frames."
                    )

                if "story" not in self.context:
                    raise CharacterAnimationError(
                        "Character animation requires completed story."
                    )

                animated_dir = (
                    self.output_dir
                    / "frames_animated"
                )

                animated_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                animated_frames_dir = processor.animate(
                    scenes=self.context["story"]["scenes"],
                    frames_dir=self.context["frames_dir"],
                    face_image_path=self.context.get(
                        "face_image_path"
                    ),
                    output_dir=animated_dir,
                    progress_callback=progress_callback,
                )

                if not animated_frames_dir:
                    raise CharacterAnimationError(
                        "Character animator returned no output."
                    )

                self.context["frames_dir"] = (
                    animated_frames_dir
                )

                logger.success(
                    "Character animation completed | "
                    f"output={animated_frames_dir}"
                )

                return animated_frames_dir

            # ==========================================================
            # VIDEO ASSEMBLER
            # ==========================================================

            if step_name == "video_assembler":

                if "frames_dir" not in self.context:
                    raise VideoAssemblyError(
                        "Video assembly requires completed frames."
                    )

                if "audio_paths" not in self.context:
                    raise VideoAssemblyError(
                        "Video assembly requires generated audio."
                    )

                if "story" not in self.context:
                    raise VideoAssemblyError(
                        "Video assembly requires generated story."
                    )

                video_path = (
                    self.output_dir
                    / "final_storybook.mp4"
                )

                final_video = processor.assemble(
                    scenes=self.context["story"]["scenes"],
                    frames_dir=self.context["frames_dir"],
                    audio_paths=self.context["audio_paths"],
                    output_path=video_path,
                )

                if not final_video:
                    raise VideoAssemblyError(
                        "Video assembler returned no output."
                    )

                self.context["final_video"] = (
                    final_video
                )

                logger.success(
                    f"Video assembly completed | output={final_video}"
                )

                return final_video

            # ==========================================================
            # SAFETY
            # ==========================================================

            raise ValueError(
                f"Step '{step_name}' has no execution handler."
            )

        finally:
            # Release processor references as early as possible.
            try:
                del processor
            except Exception:
                pass

            self._cleanup_memory()

    # ------------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------------

    def run(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> Path:

        configured_steps = PIPELINE_CONFIG.get(
            "steps"
        )

        if configured_steps:
            steps = list(configured_steps)
        else:
            steps = list(self.DEFAULT_STEPS)

        if not steps:
            raise FalconAIException(
                "Pipeline contains no execution steps.",
                code="EMPTY_PIPELINE",
            )

        total_steps = len(steps)

        logger.info(
            f"Launching FalconAI Kids pipeline | "
            f"{total_steps} step(s)"
        )

        logger.info(
            "Execution graph: "
            + " -> ".join(steps)
        )

        try:

            for idx, step_name in enumerate(
                steps,
                start=1,
            ):

                logger.info(
                    f"--- [STEP {idx}/{total_steps}] "
                    f"{step_name.upper()} ---"
                )

                try:
                    self._run_step(
                        step_name=step_name,
                        progress_callback=progress_callback,
                    )

                except FalconAIException:
                    raise

                except Exception as e:
                    logger.error(
                        f"Pipeline step '{step_name}' failed: {e}"
                    )

                    logger.debug(
                        traceback.format_exc()
                    )

                    raise

            # ----------------------------------------------------------
            # FINAL VALIDATION
            # ----------------------------------------------------------

            final_video_path = (
                self.context.get("final_video")
                or self.output_dir
                / "final_storybook.mp4"
            )

            final_video_path = Path(
                final_video_path
            )

            if not final_video_path.exists():
                raise VideoAssemblyError(
                    "Pipeline completed but final video "
                    "was not found."
                )

            if final_video_path.stat().st_size <= 0:
                raise VideoAssemblyError(
                    "Final video exists but is empty."
                )

            self.context["final_video"] = (
                final_video_path
            )

            logger.success(
                "=================================================="
            )

            logger.success(
                "FalconAI Kids pipeline completed successfully!"
            )

            logger.success(
                f"Final video: {final_video_path}"
            )

            logger.success(
                "=================================================="
            )

            return final_video_path

        except Exception as e:

            logger.error(
                f"Pipeline execution failed: {e}"
            )

            logger.debug(
                traceback.format_exc()
            )

            if isinstance(
                e,
                FalconAIException,
            ):
                raise

            raise FalconAIException(
                str(e),
                code="PIPELINE_FAILURE",
            )

        finally:

            if self.cleanup_temp:
                self._cleanup_temp_dir()

            self._cleanup_memory()

    # ------------------------------------------------------------------
    # MEMORY MANAGEMENT
    # ------------------------------------------------------------------

    def _cleanup_memory(self) -> None:
        """
        Performs lightweight memory cleanup after every stage.

        CUDA cleanup is conditional. Therefore the default CPU pipeline does
        not require CUDA to be installed.
        """

        try:
            gc.collect()
        except Exception:
            pass

        if DEVICE == "cuda":

            try:
                if cuda_hardware_available():

                    import torch

                    torch.cuda.empty_cache()

                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass

            except Exception as e:
                logger.debug(
                    f"Non-fatal CUDA memory cleanup error: {e}"
                )

    # ------------------------------------------------------------------
    # TEMPORARY FILE CLEANUP
    # ------------------------------------------------------------------

    def _cleanup_temp_dir(self) -> None:

        try:

            if self.temp_dir.exists():

                shutil.rmtree(
                    self.temp_dir,
                    ignore_errors=True,
                )

                logger.debug(
                    f"Temporary pipeline directory cleaned: "
                    f"{self.temp_dir}"
                )

        except Exception as e:

            logger.debug(
                f"Non-fatal temporary directory cleanup error: {e}"
            )


# ----------------------------------------------------------------------
# CUDA DETECTION
# ----------------------------------------------------------------------

def cuda_hardware_available() -> bool:
    """
    Returns True only when PyTorch is installed and CUDA is actually
    available.

    ImportError is intentionally swallowed so CPU-only deployments remain
    functional.
    """

    try:

        import torch

        return bool(
            torch.cuda.is_available()
        )

    except ImportError:
        return False

    except Exception:
        return False


# ----------------------------------------------------------------------
# BACKWARD-COMPATIBILITY ALIAS
# ----------------------------------------------------------------------

Orchestrator = PipelineOrchestrator
