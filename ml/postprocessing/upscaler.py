import gc
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from config.settings import UPSCALER_CONFIG, VIDEO_CONFIG, DEVICE
from utils.logger import get_logger
from utils.exceptions import UpscalingError

logger = get_logger(__name__)

class Upscaler:
    def __init__(self):
        self.scale      = UPSCALER_CONFIG["scale"]
        self.enabled    = UPSCALER_CONFIG["enabled"]
        self.model_name = UPSCALER_CONFIG["model_name"]
        self.upsampler  = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.enabled:
            logger.debug("Upscaler çaktivizuar në settings")
            return
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            cache_dir  = Path(UPSCALER_CONFIG["model_cache_dir"])
            model_path = next(cache_dir.rglob("*.pth"), None)

            if not model_path:
                raise FileNotFoundError(
                    f"RealESRGAN model nuk u gjet në: {cache_dir}\n"
                    "Ekzekuto: python scripts/download_models.py --model esrgan"
                )

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=6, num_grow_ch=32, scale=4
            )

            self.upsampler = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=512,
                tile_pad=10,
                pre_pad=0,
                half=True if DEVICE == "cuda" else False,
                gpu_id=0 if DEVICE == "cuda" else None,
            )
            logger.success(f"RealESRGAN u ngarkua: {model_path.name}")

        except ImportError:
            logger.warning("realesrgan nuk instaluar — upscaling çaktivizuar")
            self.upsampler = None
        except Exception as e:
            logger.warning(f"Upscaler nuk u ngarkua: {e}")
            self.upsampler = None

    def upscale(self, video_path: Path, output_dir: Path) -> Path:
        if self.upsampler is None:
            logger.warning("Upscaler nuk disponueshëm, kthehet video origjinale")
            return video_path

        logger.step("Duke upscaluar videon...")

        suffix       = f"_x{self.scale}upscale"
        output_name  = video_path.stem + suffix + video_path.suffix
        output_path  = output_dir / output_name

        temp_dir = output_dir / "_upscale_temp"
        temp_dir.mkdir(exist_ok=True)

        try:
            frames_dir = temp_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            self._extract_frames(video_path, frames_dir)

            frame_paths = sorted(frames_dir.glob("frame_*.png"))
            if not frame_paths:
                raise UpscalingError("Asnjë frame nuk u ekstraktua")

            logger.debug(f"{len(frame_paths)} frames për upscaling")

            upscaled_dir = temp_dir / "upscaled"
            upscaled_dir.mkdir(exist_ok=True)

            for i, fp in enumerate(frame_paths, 1):
                self._upscale_frame(fp, upscaled_dir / fp.name)
                if i % 10 == 0 or i == len(frame_paths):
                    logger.debug(f"Upscaling: {i}/{len(frame_paths)} frames")

            fps = VIDEO_CONFIG["fps"]
            w, h = VIDEO_CONFIG["resolution"]
            new_w, new_h = w * self.scale, h * self.scale

            self._frames_to_video(upscaled_dir, output_path, fps, new_w, new_h)

            audio_added = self._transfer_audio(video_path, output_path)
            if audio_added:
                logger.debug("Audio u transferua me sukses")

            logger.success(
                f"Upscaling perfundoi | "
                f"{w}x{h} → {new_w}x{new_h} | "
                f"{output_path.name}"
            )
            return output_path

        except Exception as e:
            logger.warning(f"Upscaling deshtoi: {e}, duke kthyer origjinalin")
            return video_path
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            gc.collect()

    def _upscale_frame(self, input_path: Path, output_path: Path) -> None:
        img = np.array(Image.open(input_path).convert("RGB"))
        upscaled, _ = self.upsampler.enhance(img, outscale=self.scale)
        Image.fromarray(upscaled).save(output_path)

    def _extract_frames(self, video_path: Path, frames_dir: Path) -> None:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            str(frames_dir / "frame_%06d.png")
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    def _frames_to_video(
        self, frames_dir: Path, output_path: Path,
        fps: int, width: int, height: int
    ) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%06d.png"),
            "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={width}:{height}",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    def _transfer_audio(self, source_video: Path, target_video: Path) -> bool:
        temp = target_video.parent / f"_tmp_{target_video.name}"
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(target_video),
                "-i", str(source_video),
                "-c:v", "copy", "-c:a", "copy",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest", str(temp),
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0 and temp.exists():
                shutil.move(str(temp), str(target_video))
                return True
        except Exception:
            pass
        temp.unlink(missing_ok=True)
        return False