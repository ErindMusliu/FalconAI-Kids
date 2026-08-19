import gc
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from config.settings import OUTPUT_DIR, DEVICE, PIPELINE_CONFIG
from utils.exceptions import (
    FalconAIException,
    FalconAIError,
    ModelLoadError,
    StoryGenerationError,
    AudioGenerationError,
    FrameGenerationError,
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
    Main FalconAI Kids generation pipeline.

    Default execution graph:

        1. story_generator
        2. audio_generator
        3. frame_generator
        4. video_assembler

    The old face-processing and character-animation stages are intentionally
    excluded from the default pipeline.

    This means the default pipeline does NOT require:

        - InsightFace
        - OpenCV face processing
        - SadTalker
        - rembg
        - character_animator.py
        - compositor.py
        - mouth_animator.py

    Characters are generated directly from the story/visual prompts.

    The orchestrator is deliberately modular: individual processors are
    loaded only when their corresponding pipeline step is executed.
    """

    REQUIRED_CONTEXT_KEYS = (
        "name",
        "birthday",
    )

    DEFAULT_STEPS = (
        "story_generator",
        "audio_generator",
        "frame_generator",
        "video_assembler",
    )

    def __init__(self, context: Dict[str, Any]):
        self._validate_context(context)

        self.context = dict(context)

        # Safe defaults
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
            f"Pipeline Orchestrator initialized | "
            f"device={DEVICE.upper()} | "
            f"language={self.context['language']} | "
            f"output={self.output_dir}"
        )

    # ------------------------------------------------------------------
    # CONTEXT
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
        Load only the processor required for the current step.

        Heavy optional modules are intentionally not imported here unless
        explicitly requested by a custom pipeline configuration.
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

            if step_name == "frame_generator":
                return FrameGenerator(
                    seed=self.context.get("seed"),
                )

            if step_name == "video_assembler":
                return VideoAssembler()

            # Optional legacy processors.
            #
            # These are intentionally lazy-loaded so their heavy dependencies
            # cannot crash the default FalconAI Kids pipeline.

            if step_name == "face_processor":
                from pipeline.face_processor import FaceProcessor

                return FaceProcessor()

            if step_name == "character_animator":
                from pipeline.character_animator import CharacterAnimator

                return CharacterAnimator()

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
                f"Unexpected error while initializing "
                f"processor '{step_name}'.",
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
            # STORY GENERATION
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
                        "Story generator returned empty story data."
                    )

                if not isinstance(story_data, dict):
                    raise StoryGenerationError(
                        "Story generator returned invalid story structure."
                    )

                if not story_data.get("scenes"):
                    raise StoryGenerationError(
                        "Generated story contains no scenes."
                    )

                self.context["story"] = story_data

                logger.success(
                    f"Story generation completed | "
                    f"title='{story_data.get('title', 'Untitled')}' | "
                    f"scenes={len(story_data.get('scenes', []))}"
                )

                return story_data

            # ==========================================================
            # AUDIO GENERATION
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

                audio_paths = processor.generate(
                    story=self.context["story"],
                    output_dir=audio_dir,
                    language=self.context.get(
                        "language",
                        "Albanian",
                    ),
                )

                if not audio_paths:
                    raise AudioGenerationError(
                        "Audio generator returned no audio output."
                    )

                self.context["audio_paths"] = audio_paths

                logger.success(
                    "Audio generation completed successfully."
                )

                return audio_paths

            # ==========================================================
            # FRAME GENERATION
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
                    face_embedding=None,
                    face_image_path=None,
                    output_dir=frames_dir,
                    progress_callback=progress_callback,
                )

                if not frames_paths:
                    raise FrameGenerationError(
                        "Frame generator returned no frame output."
                    )

                self.context["frames_dir"] = frames_paths

                logger.success(
                    f"Frame generation completed | "
                    f"output={frames_paths}"
                )

                return frames_paths

            # ==========================================================
            # VIDEO ASSEMBLY
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

                self.context["final_video"] = final_video

                logger.success(
                    f"Video assembly completed | "
                    f"output={final_video}"
                )

                return final_video

            # ==========================================================
            # LEGACY FACE PROCESSOR
            # ==========================================================

            if step_name == "face_processor":

                photo_input = self.context.get("photo")

                if not photo_input:
                    logger.warning(
                        "No portrait photo supplied. "
                        "Face processing skipped."
                    )

                    self.context["face_embedding"] = None
                    self.context["face_image_path"] = None

                    return None

                photo_path = Path(photo_input)

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

                return face_result

            # ==========================================================
            # LEGACY CHARACTER ANIMATOR
            # ==========================================================

            if step_name == "character_animator":

                if "frames_dir" not in self.context:
                    raise ValueError(
                        "Character animation requires completed frames."
                    )

                animated_dir = (
                    self.output_dir
                    / "frames_animated"
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

                self.context["frames_dir"] = (
                    animated_frames_dir
                )

                return animated_frames_dir

            raise ValueError(
                f"Unsupported pipeline step: {step_name}"
            )

        finally:
            del processor
            self._cleanup_memory()

    # ------------------------------------------------------------------
    # MAIN PIPELINE
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

        try:

            for index, step_name in enumerate(
                steps,
                start=1,
            ):

                logger.info(
                    f"--- "
                    f"[STEP {index}/{total_steps}] "
                    f"{step_name.upper()} "
                    f"---"
                )

                self._run_step(
                    step_name,
                    progress_callback,
                )

            # ----------------------------------------------------------
            # FINAL VALIDATION
            # ----------------------------------------------------------

            final_video = self.context.get(
                "final_video"
            )

            expected_video = (
                self.output_dir
                / "final_storybook.mp4"
            )

            if final_video:
                final_video_path = Path(
                    final_video
                )
            else:
                final_video_path = expected_video

            if not final_video_path.exists():
                raise VideoAssemblyError(
                    "Pipeline completed its steps, "
                    "but final_storybook.mp4 was not found."
                )

            logger.success(
                f"FalconAI Kids pipeline completed successfully! "
                f"Output: {final_video_path}"
            )

            return final_video_path

        except FalconAIException:
            raise

        except Exception as e:

            logger.error(
                f"Pipeline execution failed: {e}"
            )

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
        Release Python and CUDA memory between pipeline stages.
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
                logger.debug(
                    f"CUDA memory cleanup skipped: {e}"
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
                    f"Temporary pipeline directory removed: "
                    f"{self.temp_dir}"
                )

        except Exception as e:

            logger.debug(
                f"Temporary directory cleanup failed "
                f"non-fatally: {e}"
            )


# ----------------------------------------------------------------------
# BACKWARD COMPATIBILITY ALIAS
# ----------------------------------------------------------------------

Orchestrator = PipelineOrchestrator
