import gc
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from config.settings import OUTPUT_DIR, DEVICE
from utils.logger import get_logger
from utils.exceptions import (
    FalconAIError,
    ModelLoadError,
    FaceProcessingError,
    StoryGenerationError,
    AudioGenerationError,
    FrameGenerationError,
    VideoAssemblyError,
)

# Importimi i procesorëve individualë nga pipeline
from pipeline.face_processor import FaceProcessor
from pipeline.story_generator import StoryGenerator
from pipeline.audio_generator import AudioGenerator
from pipeline.frame_generator import FrameGenerator
from pipeline.video_assembler import VideoAssembler

logger = get_logger(__name__)


class PipelineOrchestrator:
    def __init__(self, context: Dict[str, Any]):
        self.context = context
        self.output_dir = Path(context.get("output_dir", OUTPUT_DIR))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ruajtja e instancave të procesorëve
        self.processors: Dict[str, Any] = {}
        logger.info(f"Orchestrator u iniciua për mjedisin: {DEVICE.upper()}")

    def _load_single_processor(self, step_name: str) -> Any:
        """Ngarkon në mënyrë dinamike vetëm procesorin që duhet për hapin aktual."""
        logger.debug(f"Duke ngarkuar procesorin për hapin: {step_name}")
        try:
            if step_name == "face_processor":
                return FaceProcessor()
            elif step_name == "story_generator":
                return StoryGenerator()
            elif step_name == "audio_generator":
                return AudioGenerator()
            elif step_name == "frame_generator":
                # Kalojmë seed nëse është i pranishëm në kontekst
                return FrameGenerator(seed=self.context.get("seed"))
            elif step_name == "video_assembler":
                return VideoAssembler()
            else:
                raise ValueError(f"Hapi i panjohur i pipeline: {step_name}")
        except FalconAIError as e:
            # Nëse është gabim i njohur i yni, e risjellim si ModelLoadError me strukturën e saktë pozicionale
            raise ModelLoadError(f"Dështoi ngarkimi i modelit për {step_name}", str(e))
        except Exception as e:
            raise ModelLoadError(f"Gabim i papritur gjatë ngarkimit të {step_name}", str(e))

    def _run_step(self, step_name: str, progress_callback: Optional[Callable] = None) -> Any:
        """Ekzekuton një hap të vetëm të pipeline dhe liron memorinë pas përfundimit."""
        processor = self._load_single_processor(step_name)
        
        try:
            if step_name == "face_processor":
                photo_path = Path(self.context["photo"])
                # Proceson foton për të nxjerrë embedding dhe imazhin e pastruar
                embedding, aligned_face = processor.process(photo_path)
                self.context["face_embedding"] = embedding
                self.context["face_image_path"] = photo_path
                return photo_path

            elif step_name == "story_generator":
                story_data = processor.generate(
                    name=self.context["name"],
                    birthday=self.context["birthday"],
                    gender=self.context.get("gender"),
                    preferences=self.context.get("preferences", {})
                )
                self.context["story"] = story_data
                return story_data

            elif step_name == "audio_generator":
                audio_dir = self.output_dir / "audio"
                audio_paths = processor.generate(
                    scenes=self.context["story"]["scenes"],
                    output_dir=audio_dir
                )
                self.context["audio_paths"] = audio_paths
                return audio_paths

            elif step_name == "frame_generator":
                frames_dir = self.output_dir / "frames"
                frames_paths = processor.generate(
                    scenes=self.context["story"]["scenes"],
                    face_embedding=self.context.get("face_embedding"),
                    face_image_path=self.context.get("face_image_path"),
                    output_dir=frames_dir,
                    progress_callback=progress_callback
                )
                self.context["frames_dir"] = frames_paths
                return frames_paths

            elif step_name == "video_assembler":
                video_path = self.output_dir / "final_storybook.mp4"
                final_video = processor.assemble(
                    scenes=self.context["story"]["scenes"],
                    frames_dir=self.context["frames_dir"],
                    audio_paths=self.context["audio_paths"],
                    output_path=video_path
                )
                return final_video

        finally:
            # Sigurohemi që të fshijmë referencën e procesorit dhe të pastrojmë VRAM/RAM
            del processor
            self._cleanup_memory()

    def run(self, progress_callback: Optional[Callable] = None) -> Path:
        """Orkestron të gjithë rrjedhën e gjenerimit nga fillimi në fund."""
        steps = [
            "face_processor",
            "story_generator",
            "audio_generator",
            "frame_generator",
            "video_assembler"
        ]
        
        total_steps = len(steps)
        logger.info(f"Nisja e pipeline me {total_steps} hapa kryesorë.")

        try:
            for idx, step_name in enumerate(steps):
                logger.info(f"--- [HAPI {idx+1}/{total_steps}] Duke ekzekutuar {step_name.upper()} ---")
                
                self._run_step(step_name, progress_callback)
                
                if progress_callback:
                    # Njoftojmë mjedisin prind për progresin e përgjithshëm të pipeline
                    progress_callback(idx + 1, total_steps, f"Përfundoi {step_name}")

            final_video_path = self.output_dir / "final_storybook.mp4"
            if not final_video_path.exists():
                raise VideoAssemblyError("Pipeline përfundoi por skedari final MP4 nuk u gjet.", "")
                
            logger.success(f"Pipeline u përmbush me sukses të plotë! Video: {final_video_path}")
            return final_video_path

        except FalconAIError as e:
            logger.error(f"Pipeline dështoi në hapin specifik. Duke pastruar mjetet...")
            raise e
        except Exception as e:
            logger.error(f"Gabim i papritur u kap në orchestrator: {str(e)}")
            logger.debug(traceback.format_exc())
            # Kthimi i gabimeve gjenerike të sistemit në strukturën tonë standarde UNEXPECTED_ERROR
            raise FalconAIError(f"Gabim i papritur (TypeError/ValueError): {str(e)}", code="UNEXPECTED_ERROR")

    def _cleanup_memory(self) -> None:
        """Pastron mbetjet e alokuara në RAM dhe VRAM pas çdo hapi të rëndë AI."""
        gc.collect()
        if DEVICE == "cuda" and import_util_cuda_available():
            import torch
            torch.cuda.empty_cache()
            logger.debug("VRAM e pastruar me sukses pas hapit të pipeline.")


def import_util_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
