import gc
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# Provon të importojë torch për të pastruar VRAM-in nëse është i disponueshëm
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from config.settings import (
    TEMP_DIR,
    OUTPUT_DIR,
    PIPELINE_CONFIG,
    DIFFUSION_CONFIG,
    LLM_CONFIG,
    VIDEO_CONFIG,
)
from utils.logger import get_logger, get_pipeline_formatter
from utils.exceptions import (
    FalconAIException,
    ModelLoadError,
    PipelineError,
    FaceProcessingError,
    StoryGenerationError,
    FrameGenerationError,
    AudioGenerationError,
    VideoAssemblyError,
    UpscalingError,
    handle_exception,
)

logger = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        pipeline_config: dict,
        output_dir: Path = OUTPUT_DIR,
        language: str = "Albanian",
        seed: Optional[int] = None,
    ):
        self.pipeline_config = pipeline_config
        self.output_dir      = Path(output_dir)
        self.language        = language
        self.seed            = seed
        self.steps           = pipeline_config.get("steps", [])
        self.cleanup_temp    = pipeline_config.get("cleanup_temp", True)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._context: dict = {}
        self._temp_dirs: list[Path] = []

        self._progress = get_pipeline_formatter(total_steps=len(self.steps))
        
        # Nuk i ngarkojmë modelet këtu për të kursyer RAM-in fillestar
        logger.info("Orchestrator u iniciua me sukses (Lazy Loading i aktivizuar)")

    def _load_single_processor(self, step_name: str):
        """Ngarkon dinamikisht vetëm procesorin që duhet për hapin aktual."""
        try:
            if step_name == "face_processor":
                from pipeline.face_processor import FaceProcessor
                logger.step("Duke ngarkuar FaceProcessor (InsightFace) në memorie...")
                return FaceProcessor()

            elif step_name == "story_generator":
                from pipeline.story_generator import StoryGenerator
                logger.step(f"Duke ngarkuar StoryGenerator ({LLM_CONFIG['model_name']}) në memorie...")
                return StoryGenerator(language=self.language)

            elif step_name == "frame_generator":
                from pipeline.frame_generator import FrameGenerator
                logger.step(f"Duke ngarkuar FrameGenerator ({DIFFUSION_CONFIG['model_name']}) në memorie...")
                return FrameGenerator(seed=self.seed)

            elif step_name == "audio_generator":
                from pipeline.audio_generator import AudioGenerator
                logger.step("Duke ngarkuar AudioGenerator (TTS) në memorie...")
                return AudioGenerator(language=self.language)

            elif step_name == "video_assembler":
                from pipeline.video_assembler import VideoAssembler
                logger.step("Duke ngarkuar VideoAssembler (FFmpeg)...")
                return VideoAssembler()

            elif step_name == "upscaler":
                from pipeline.postprocessing.upscaler import Upscaler
                logger.step("Duke ngarkuar Upscaler (RealESRGAN) në memorie...")
                return Upscaler()
            
        except Exception as e:
            raise ModelLoadError(f"Dështoi ngarkimi i modelit për {step_name}: {str(e)}")
        
        return None

    def run(
        self,
        photo_path: Path,
        name: str,
        birthday: datetime,
        age: int,
    ) -> Path:
        pipeline_start = time.time()

        logger.info(
            f"Pipeline filloi | fëmijë: {name} | moshë: {age} | "
            f"gjuhë: {self.language} | hapa: {len(self.steps)}"
        )

        run_id   = _generate_run_id(name)
        temp_dir = TEMP_DIR / run_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dirs.append(temp_dir)

        self._context = {
            "run_id"    : run_id,
            "name"      : name,
            "birthday"  : birthday,
            "age"       : age,
            "language"  : self.language,
            "photo_path": photo_path,
            "temp_dir"  : temp_dir,
            "output_dir": self.output_dir,
        }

        output_path: Optional[Path] = None

        try:
            for step_name in self.steps:
                output_path = self._run_step(step_name)

        except FalconAIException:
            self._on_failure()
            raise
        except Exception as e:
            self._on_failure()
            raise handle_exception(e, logger)

        if self.cleanup_temp:
            self._cleanup(temp_dir)

        elapsed = time.time() - pipeline_start
        self._progress.finish()

        logger.success(
            f"Pipeline perfundoi | {elapsed:.1f}s | output: {output_path}"
        )

        return output_path

    def _run_step(self, step_name: str) -> Optional[Path]:
        self._progress.start_step(step_name)
        step_start = time.time()

        logger.step(f"Duke filluar hapin: {step_name}")

        processor = self._load_single_processor(step_name)
        if processor is None:
            logger.warning(f"Hapi '{step_name}' nuk u gjet ose nuk ka nevojë për model, kapërcehet")
            self._progress.end_step(step_name, success=False)
            return self._context.get("output_path")

        try:
            result = self._dispatch(step_name, processor)

            if result is not None:
                self._context[f"{step_name}_result"] = result
                self._context["output_path"] = result

            elapsed = time.time() - step_start
            self._progress.end_step(step_name, success=True)
            logger.success(f"{step_name} perfundoi ne {elapsed:.1f}s")
            
            return result

        except FalconAIException as e:
            self._progress.end_step(step_name, success=False)
            logger.error(f"{step_name} deshtoi: {e}")
            raise
        except Exception as e:
            self._progress.end_step(step_name, success=False)
            logger.error(f"{step_name} deshtoi me gabim te papritur: {e}")
            logger.debug(traceback.format_exc())
            raise handle_exception(e, logger)
        
        finally:
            # MEMORY MANAGEMENT: Shkatërrojmë procesorin aktual dhe lirojmë RAM/VRAM menjëherë
            del processor
            gc.collect()
            if HAS_TORCH:
                torch.cuda.empty_cache()
            logger.info(f"Kujtesa u pastrua pas përfundimit të hapit: {step_name}")

    def _dispatch(self, step_name: str, processor) -> Optional[Path]:
        ctx = self._context

        if step_name == "face_processor":
            return self._step_face_processor(processor, ctx)

        elif step_name == "story_generator":
            return self._step_story_generator(processor, ctx)

        elif step_name == "frame_generator":
            return self._step_frame_generator(processor, ctx)

        elif step_name == "audio_generator":
            return self._step_audio_generator(processor, ctx)

        elif step_name == "video_assembler":
            return self._step_video_assembler(processor, ctx)

        elif step_name == "upscaler":
            return self._step_upscaler(processor, ctx)

        else:
            logger.warning(f"Hap i panjohur: '{step_name}', kapërcehet")
            return None

    def _step_face_processor(self, processor, ctx: dict) -> dict:
        logger.debug(f"Duke procesuar foton: {ctx['photo_path']}")

        try:
            result = processor.process(
                photo_path=ctx["photo_path"],
                temp_dir=ctx["temp_dir"],
            )
        except FaceProcessingError:
            raise
        except Exception as e:
            raise FaceProcessingError(str(e))

        logger.debug(
            f"Fytyra u gjet | pozicion: {result.get('bbox')} | "
            f"confidence: {result.get('det_score', 0):.2f}"
        )

        ctx["face_result"] = result
        return result

    def _step_story_generator(self, processor, ctx: dict) -> dict:
        logger.debug(
            f"Duke gjeneruar histori | emër: {ctx['name']} | "
            f"moshë: {ctx['age']} | gjuhë: {ctx['language']}"
        )

        try:
            result = processor.generate(
                name=ctx["name"],
                age=ctx["age"],
                birthday=ctx["birthday"],
                language=ctx["language"],
            )
        except StoryGenerationError:
            raise
        except Exception as e:
            raise StoryGenerationError(str(e))

        scenes = result.get("scenes", [])
        logger.debug(f"Historia u gjenerua | {len(scenes)} skena")

        for i, scene in enumerate(scenes, 1):
            logger.debug(f"   Skena {i}: {scene.get('title', 'pa titull')}")

        ctx["story_result"] = result
        return result

    def _step_frame_generator(self, processor, ctx: dict) -> Path:
        story  = ctx.get("story_result", {})
        face   = ctx.get("face_result", {})
        scenes = story.get("scenes", [])

        if not scenes:
            raise FrameGenerationError("Asnjë skenë nuk u gjet në histori")

        frames_dir = ctx["temp_dir"] / "frames"
        frames_dir.mkdir(exist_ok=True)

        logger.debug(f"Duke gjeneruar frames per {len(scenes)} skena")

        try:
            result_dir = processor.generate(
                scenes=scenes,
                face_embedding=face.get("embedding"),
                face_image_path=face.get("face_image_path"),
                output_dir=frames_dir,
                progress_callback=self._progress.update_progress,
            )
        except FrameGenerationError:
            raise
        except Exception as e:
            raise FrameGenerationError(str(e))

        frame_count = len(list(result_dir.glob("*.png")))
        logger.debug(f"Frames u gjeneruan | total: {frame_count}")

        ctx["frames_dir"] = result_dir
        return result_dir

    def _step_audio_generator(self, processor, ctx: dict) -> Path:
        story = ctx.get("story_result", {})

        if not story:
            raise AudioGenerationError("Historia nuk ekziston, audio nuk mund të gjenerohet")

        audio_dir = ctx["temp_dir"] / "audio"
        audio_dir.mkdir(exist_ok=True)

        try:
            audio_path = processor.generate(
                story=story,
                output_dir=audio_dir,
                language=ctx["language"],
            )
        except AudioGenerationError:
            raise
        except Exception as e:
            raise AudioGenerationError(str(e))

        logger.debug(f"Audio u gjenerua: {audio_path}")

        ctx["audio_path"] = audio_path
        return audio_path

    def _step_video_assembler(self, processor, ctx: dict) -> Path:
        frames_dir = ctx.get("frames_dir")
        audio_path = ctx.get("audio_path")

        if not frames_dir:
            raise VideoAssemblyError("Frames nuk ekzistojnë")

        safe_name    = ctx["name"].lower().replace(" ", "_")
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name  = f"{safe_name}_{timestamp}.{VIDEO_CONFIG['output_format']}"
        output_path  = ctx["output_dir"] / output_name

        try:
            final_path = processor.assemble(
                frames_dir=frames_dir,
                audio_path=audio_path,
                output_path=output_path,
                story=ctx.get("story_result", {}),
            )
        except VideoAssemblyError:
            raise
        except Exception as e:
            raise VideoAssemblyError(str(e))

        logger.debug(f"Video u bashkua: {final_path}")

        ctx["video_path"] = final_path
        return final_path

    def _step_upscaler(self, processor, ctx: dict) -> Path:
        video_path = ctx.get("video_path")

        if not video_path:
            raise UpscalingError("Video nuk ekziston për upscaling")

        try:
            upscaled_path = processor.upscale(
                video_path=video_path,
                output_dir=ctx["output_dir"],
            )
        except UpscalingError:
            raise
        except Exception as e:
            logger.warning(
                f"Upscaling deshtoi, duke kthyer videon origjinale: {e}"
            )
            return video_path

        logger.debug(f"Video u upscalua: {upscaled_path}")

        if upscaled_path != video_path and video_path.exists():
            video_path.unlink()
            logger.debug(f"Video e vjeter u fshi: {video_path}")

        ctx["video_path"] = upscaled_path
        return upscaled_path

    def _cleanup(self, temp_dir: Path) -> None:
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.debug(f"Temp folder u fshi: {temp_dir}")
        except Exception as e:
            open_err = f"Nuk u fshi temp folder {temp_dir}: {e}"
            logger.warning(open_err)

    def _on_failure(self) -> None:
        logger.error("Pipeline deshtoi. Duke pastruar...")
        logger.debug(
            f"Temp files u ruajten per debugging: "
            + ", ".join(str(d) for d in self._temp_dirs)
        )

    def get_context(self) -> dict:
        return self._context.copy()

    def get_active_steps(self) -> list[str]:
        return self.steps.copy()


def _generate_run_id(name: str) -> str:
    safe_name = name.lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_name}_{timestamp}"
