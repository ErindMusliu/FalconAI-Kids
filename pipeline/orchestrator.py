import gc
import shutil
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from config.settings import OUTPUT_DIR, DEVICE, PIPELINE_CONFIG
from utils.exceptions import FalconAIException
from utils.logger import get_logger
from utils.exceptions import (
    FalconAIError,
    ModelLoadError,
    FaceProcessingError,
    StoryGenerationError,
    AudioGenerationError,
    FrameGenerationError,
    CharacterAnimationError,
    VideoAssemblyError,
)

from pipeline.face_processor import FaceProcessor
from pipeline.story_generator import StoryGenerator
from pipeline.audio_generator import AudioGenerator
from pipeline.frame_generator import FrameGenerator
from pipeline.character_animator import CharacterAnimator
from pipeline.video_assembler import VideoAssembler

logger = get_logger(__name__)


class PipelineOrchestrator:
    REQUIRED_CONTEXT_KEYS = ("name", "birthday")

    DEFAULT_STEPS = (
        "face_processor",
        "story_generator",
        "audio_generator",
        "frame_generator",
        "character_animator",
        "video_assembler",
    )

    def __init__(self, context: Dict[str, Any]):
        self._validate_context(context)

        self.context = context
        self.context.setdefault("language", "Albanian")
        self.context.setdefault("preferences", {})
        self.context.setdefault("seed", None)

        self.output_dir = Path(context.get("output_dir", OUTPUT_DIR))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.cleanup_temp = PIPELINE_CONFIG.get("cleanup_temp", True)

        logger.info(
            f"Pipeline Orchestrator successfully initialized targeting system compute device: {DEVICE.upper()} "
            f"| language: {self.context['language']}"
        )

    def _validate_context(self, context: Dict[str, Any]) -> None:
        missing = [k for k in self.REQUIRED_CONTEXT_KEYS if not context.get(k)]
        if missing:
            raise ValueError(f"Missing required pipeline context key(s): {', '.join(missing)}")

    def _load_single_processor(self, step_name: str) -> Any:
        logger.debug(f"Loading dedicated atomic inference processor module for step: {step_name}")
        language = self.context.get("language", "Albanian")

        try:
            if step_name == "face_processor":
                return FaceProcessor()
            elif step_name == "story_generator":
                return StoryGenerator(language=language)
            elif step_name == "audio_generator":
                return AudioGenerator(language=language)
            elif step_name == "frame_generator":
                return FrameGenerator(seed=self.context.get("seed"))
            elif step_name == "character_animator":
                return CharacterAnimator()
            elif step_name == "video_assembler":
                return VideoAssembler()
            else:
                raise ValueError(f"Unrecognized step assignment: {step_name}")
        except FalconAIError as e:
            raise ModelLoadError(f"Model weight compilation failed for {step_name}", str(e))
        except Exception as e:
            raise ModelLoadError(f"Unexpected structural fault initialization for {step_name}", str(e))

    def _run_step(self, step_name: str, progress_callback: Optional[Callable] = None) -> Any:
        self._cleanup_memory()
        
        processor = self._load_single_processor(step_name)

        try:
            if step_name == "face_processor":
                photo_input = self.context.get("photo")
                if not photo_input:
                    logger.warning("No portrait reference photo detected. Bypassing face structure extractions.")
                    self.context["face_embedding"] = None
                    self.context["face_image_path"] = None
                    return None

                photo_path = Path(photo_input)
                face_result = processor.process(photo_path=photo_path, temp_dir=self.temp_dir)
                self.context["face_embedding"] = face_result.get("embedding")
                self.context["face_image_path"] = face_result.get("face_image_path")
                return face_result

            elif step_name == "story_generator":
                story_data = processor.generate(
                    name=self.context["name"],
                    birthday=self.context["birthday"],
                    gender=self.context.get("gender"),
                    preferences=self.context.get("preferences", {}),
                )
                self.context["story"] = story_data
                return story_data

            elif step_name == "audio_generator":
                if "story" not in self.context:
                    raise AudioGenerationError("Audio generation requires a completed story.")
                audio_dir = self.output_dir / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)
                final_audio_path = processor.generate(
                    story=self.context["story"],
                    output_dir=audio_dir,
                    language=self.context.get("language", "Albanian"),
                )
                self.context["audio_paths"] = final_audio_path
                return final_audio_path

            elif step_name == "frame_generator":
                if "story" not in self.context:
                    raise FrameGenerationError("Frame generation requires a completed story.")
                frames_dir = self.output_dir / "frames"
                frames_paths = processor.generate(
                    scenes=self.context["story"]["scenes"],
                    face_embedding=self.context.get("face_embedding"),
                    face_image_path=self.context.get("face_image_path"),
                    output_dir=frames_dir,
                    progress_callback=progress_callback,
                )
                self.context["frames_dir"] = frames_paths
                return frames_paths

            elif step_name == "character_animator":
                if "frames_dir" not in self.context:
                    raise CharacterAnimationError("Character animation requires completed frames.")
                animated_dir = self.output_dir / "frames_animated"
                animated_frames_dir = processor.animate(
                    scenes=self.context["story"]["scenes"],
                    frames_dir=self.context["frames_dir"],
                    face_image_path=self.context.get("face_image_path"),
                    output_dir=animated_dir,
                    progress_callback=progress_callback,
                )
                self.context["frames_dir"] = animated_frames_dir
                return animated_frames_dir

            elif step_name == "video_assembler":
                if "frames_dir" not in self.context or "audio_paths" not in self.context:
                    raise VideoAssemblyError("Video assembly requires completed frames and audio.")
                video_path = self.output_dir / "final_storybook.mp4"
                final_video = processor.assemble(
                    scenes=self.context["story"]["scenes"],
                    frames_dir=self.context["frames_dir"],
                    audio_paths=self.context["audio_paths"],
                    output_path=video_path,
                )
                self.context["final_video"] = final_video
                return final_video

        finally:
            del processor
            self._cleanup_memory()

    def run(self, progress_callback: Optional[Callable] = None) -> Path:
        steps = PIPELINE_CONFIG.get("steps", list(self.DEFAULT_STEPS))
        total_steps = len(steps)
        logger.info(f"Launching pipeline execution graph with {total_steps} steps.")

        try:
            for idx, step_name in enumerate(steps):
                logger.info(f"--- [STEP {idx + 1}/{total_steps}] Executing: {step_name.upper()} ---")
                self._run_step(step_name, progress_callback)
            
            final_video_path = self.output_dir / "final_storybook.mp4"
            if not final_video_path.exists():
                raise VideoAssemblyError("Generation failed: final video not found.")

            logger.success(f"Pipeline executed successfully! Output: {final_video_path}")
            return final_video_path

        except Exception as e:
            logger.error(f"Operational error at runtime: {str(e)}")
            raise FalconAIException(str(e), code="PIPELINE_FAILURE")
        finally:
            if self.cleanup_temp:
                self._cleanup_temp_dir()

    def _cleanup_memory(self) -> None:
        gc.collect()
        if DEVICE == "cuda" and cuda_hardware_available():
            import torch
            torch.cuda.empty_cache()

    def _cleanup_temp_dir(self) -> None:
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logger.debug(f"Cleanup non-fatal error: {e}")

def cuda_hardware_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

Orchestrator = PipelineOrchestrator
