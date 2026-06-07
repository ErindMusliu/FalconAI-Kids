import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from config.settings import VIDEO_CONFIG, DIFFUSION_CONFIG, LLM_CONFIG
from utils.logger import get_logger
from utils.exceptions import VideoAssemblyError

logger = get_logger(__name__)


class VideoAssembler:
    def __init__(self):
        self.fps        = VIDEO_CONFIG["fps"]
        self.resolution = VIDEO_CONFIG["resolution"]
        self.codec      = VIDEO_CONFIG["codec"]
        self.quality    = VIDEO_CONFIG["quality"]
        self._check_ffmpeg()

    def _check_ffmpeg(self) -> None:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                version_line = result.stdout.split("\n")[0]
                logger.debug(f"FFmpeg: {version_line}")
            else:
                raise VideoAssemblyError(
                    "FFmpeg nuk funksionon. "
                    "Instalo nga: https://ffmpeg.org/download.html"
                )
        except FileNotFoundError:
            raise VideoAssemblyError(
                "FFmpeg nuk është instaluar. "
                "Instalo nga: https://ffmpeg.org/download.html\n"
                "Ubuntu/Debian: sudo apt install ffmpeg\n"
                "macOS: brew install ffmpeg\n"
                "Windows: https://ffmpeg.org/download.html"
            )
        except subprocess.TimeoutExpired:
            raise VideoAssemblyError("FFmpeg nuk u përgjigj brenda 10 sekondave")

    def assemble(
        self,
        frames_dir: Path,
        audio_path: Optional[Path],
        output_path: Path,
        story: dict,
    ) -> Path:
        logger.debug(f"Duke bashkuar video | output: {output_path.name}")

        scene_dirs = self._collect_scene_dirs(frames_dir)
        logger.debug(f"Skenat e gjetura: {len(scene_dirs)}")

        if not scene_dirs:
            raise VideoAssemblyError(
                f"Asnjë skenë nuk u gjet në: {frames_dir}"
            )

        normalized_dirs = self._normalize_frames(scene_dirs, story)

        temp_dir      = frames_dir.parent / "transitions"
        temp_dir.mkdir(exist_ok=True)
        final_frames_dir = self._apply_transitions(
            scene_dirs=normalized_dirs,
            output_dir=temp_dir,
            story=story,
        )

        raw_video_path = output_path.parent / f"_raw_{output_path.name}"
        self._frames_to_video(
            frames_dir=final_frames_dir,
            output_path=raw_video_path,
        )
        logger.debug(f"Raw video u krijua: {raw_video_path.name}")

        if audio_path and audio_path.exists():
            self._add_audio(
                video_path=raw_video_path,
                audio_path=audio_path,
                output_path=output_path,
            )
            raw_video_path.unlink(missing_ok=True)
            logger.debug("Audio u shtua me sukses")
        else:
            shutil.move(str(raw_video_path), str(output_path))
            logger.debug("Video pa audio (audio mungonte)")

        self._add_metadata(output_path, story)

        self._validate_output(output_path)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.success(
            f"Video u bashkua me sukses | "
            f"{size_mb:.1f}MB | "
            f"{output_path.name}"
        )

        return output_path

    def _collect_scene_dirs(self, frames_dir: Path) -> list[Path]:
        scene_dirs = sorted([
            d for d in frames_dir.iterdir()
            if d.is_dir() and d.name.startswith("scene_")
        ])

        if not scene_dirs:
            frames = list(frames_dir.glob("frame_*.png"))
            if frames:
                logger.debug("Frames direkt në folder, jo në nën-folders")
                return [frames_dir]

        return scene_dirs

    def _get_scene_frames(self, scene_dir: Path) -> list[Path]:
        frames = sorted(scene_dir.glob("frame_*.png"))
        if not frames:
            frames = sorted(scene_dir.glob("*.png"))
        return frames

    def _normalize_frames(
        self,
        scene_dirs: list[Path],
        story: dict,
    ) -> list[Path]:
        scenes       = story.get("scenes", [])
        default_dur  = LLM_CONFIG["scene_duration_sec"]

        for i, scene_dir in enumerate(scene_dirs):
            frames      = self._get_scene_frames(scene_dir)
            if not frames:
                logger.warning(f"Skena {i+1} nuk ka frames!")
                continue

            dur_sec     = default_dur
            if i < len(scenes):
                dur_sec = scenes[i].get("duration_sec", default_dur)

            target_n    = int(dur_sec * self.fps)
            current_n   = len(frames)

            if current_n == target_n:
                continue
            elif current_n > target_n:
                indices = np.linspace(0, current_n - 1, target_n, dtype=int)
                keep    = [frames[j] for j in indices]
                to_delete = set(frames) - set(keep)
                for f in to_delete:
                    f.unlink(missing_ok=True)
            else:
                last_frame = frames[-1]
                for j in range(target_n - current_n):
                    new_name = scene_dir / f"frame_{current_n + j + 1:04d}.png"
                    shutil.copy(str(last_frame), str(new_name))

        logger.debug("Normalizimi i frames perfundoi")
        return scene_dirs

    def _apply_transitions(
        self,
        scene_dirs: list[Path],
        output_dir: Path,
        story: dict,
        transition_frames: int = 8,
    ) -> Path:
        logger.debug(
            f"Duke aplikuar tranzicione | "
            f"{len(scene_dirs)} skena | "
            f"{transition_frames} frames tranzicion"
        )

        all_frames_dir = output_dir / "all_frames"
        all_frames_dir.mkdir(exist_ok=True)

        global_idx = 1

        for scene_idx, scene_dir in enumerate(scene_dirs):
            frames = self._get_scene_frames(scene_dir)
            if not frames:
                continue

            is_last_scene = (scene_idx == len(scene_dirs) - 1)

            frames_to_copy = (
                frames if is_last_scene
                else frames[:-transition_frames] if len(frames) > transition_frames * 2
                else frames
            )

            for frame_path in frames_to_copy:
                dst = all_frames_dir / f"frame_{global_idx:06d}.png"
                shutil.copy(str(frame_path), str(dst))
                global_idx += 1

            if not is_last_scene and len(frames) > transition_frames:
                next_dir    = scene_dirs[scene_idx + 1]
                next_frames = self._get_scene_frames(next_dir)

                if next_frames:
                    end_frames   = frames[-transition_frames:]
                    start_frames = next_frames[:transition_frames]

                    crossfade_frames = self._generate_crossfade(
                        from_frames=end_frames,
                        to_frames=start_frames,
                        n_frames=transition_frames,
                    )

                    for cf_frame in crossfade_frames:
                        dst = all_frames_dir / f"frame_{global_idx:06d}.png"
                        cf_frame.save(dst)
                        global_idx += 1

        total_frames = global_idx - 1
        duration_est = total_frames / self.fps
        logger.debug(
            f"Tranzicionet aplikuar | "
            f"{total_frames} frames totale | "
            f"~{duration_est:.1f}s video"
        )

        return all_frames_dir

    def _generate_crossfade(
        self,
        from_frames: list[Path],
        to_frames: list[Path],
        n_frames: int,
    ) -> list[Image.Image]:
        target_w, target_h = DIFFUSION_CONFIG["width"], DIFFUSION_CONFIG["height"]
        result = []

        for i in range(n_frames):
            alpha = i / max(n_frames - 1, 1)

            from_idx = min(i, len(from_frames) - 1)
            to_idx   = min(i, len(to_frames) - 1)

            try:
                img_a = Image.open(from_frames[from_idx]).convert("RGB")
                img_b = Image.open(to_frames[to_idx]).convert("RGB")

                if img_a.size != (target_w, target_h):
                    img_a = img_a.resize((target_w, target_h), Image.LANCZOS)
                if img_b.size != (target_w, target_h):
                    img_b = img_b.resize((target_w, target_h), Image.LANCZOS)

                arr_a   = np.array(img_a, dtype=np.float32)
                arr_b   = np.array(img_b, dtype=np.float32)

                smooth_alpha = 0.5 * (1 - np.cos(np.pi * alpha))
                blended = arr_a * (1 - smooth_alpha) + arr_b * smooth_alpha
                blended = np.clip(blended, 0, 255).astype(np.uint8)

                result.append(Image.fromarray(blended))

            except Exception as e:
                logger.debug(f"Crossfade frame {i} deshtoi: {e}, duke kopjuar")
                try:
                    result.append(Image.open(from_frames[-1]).convert("RGB"))
                except Exception:
                    result.append(Image.new("RGB", (target_w, target_h), (0, 0, 0)))

        return result

    def _frames_to_video(
        self,
        frames_dir: Path,
        output_path: Path,
    ) -> None:
        target_w, target_h = self.resolution

        frame_pattern = str(frames_dir / "frame_%06d.png")

        sample_frames = list(frames_dir.glob("frame_*.png"))
        if not sample_frames:
            raise VideoAssemblyError(f"Asnjë frame nuk u gjet në: {frames_dir}")

        first_frame = sorted(sample_frames)[0].name
        digits = len(first_frame.replace("frame_", "").replace(".png", ""))
        frame_pattern = str(frames_dir / f"frame_%0{digits}d.png")

        cmd = [
            "ffmpeg",
            "-y",                           
            "-framerate", str(self.fps),    
            "-i", frame_pattern,            
            "-c:v", self.codec,             
            "-crf", str(self.quality),      
            "-preset", "medium",            
            "-pix_fmt", "yuv420p",          
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
            "-movflags", "+faststart",      
            str(output_path),
        ]

        logger.debug(f"FFmpeg cmd: {' '.join(cmd)}")
        self._run_ffmpeg(cmd, "frames → video")

    def _add_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),      
            "-i", str(audio_path),      
            "-c:v", "copy",             
            "-c:a", VIDEO_CONFIG["audio_codec"],
            "-b:a", "192k",             
            "-shortest",                
            "-map", "0:v:0",            
            "-map", "1:a:0",            
            str(output_path),
        ]

        self._run_ffmpeg(cmd, "video + audio merge")

    def _add_metadata(self, video_path: Path, story: dict) -> None:
        title   = story.get("title", "FalconAI Kids Movie")
        theme   = story.get("theme", "")
        lang    = story.get("language", "")
        year    = __import__("datetime").datetime.now().year

        temp_out = video_path.parent / f"_meta_{video_path.name}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-c", "copy",
            "-metadata", f"title={title}",
            "-metadata", f"artist=FalconAI Kids",
            "-metadata", f"album=Personalized Movie",
            "-metadata", f"genre=Children",
            "-metadata", f"comment=theme={theme}, language={lang}",
            "-metadata", f"date={year}",
            str(temp_out),
        ]

        try:
            self._run_ffmpeg(cmd, "metadata")
            if temp_out.exists():
                shutil.move(str(temp_out), str(video_path))
        except Exception as e:
            logger.debug(f"Metadata nuk u shtua (jo kritike): {e}")
            temp_out.unlink(missing_ok=True)

    def _validate_output(self, video_path: Path) -> None:
        if not video_path.exists():
            raise VideoAssemblyError(f"Video output nuk u krijua: {video_path}")

        size_bytes = video_path.stat().st_size
        if size_bytes < 1024:
            raise VideoAssemblyError(
                f"Video output është shumë e vogël ({size_bytes} bytes), "
                f"ndoshta u dëmtua gjatë procesimit"
            )

        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                str(video_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15
            )

            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                streams    = probe_data.get("streams", [])
                has_video  = any(s["codec_type"] == "video" for s in streams)

                if not has_video:
                    raise VideoAssemblyError("Video nuk ka stream video")

                video_stream = next(s for s in streams if s["codec_type"] == "video")
                w = video_stream.get("width", 0)
                h = video_stream.get("height", 0)
                logger.debug(f"Video validuar | {w}x{h} | {size_bytes/1024/1024:.1f}MB")

        except VideoAssemblyError:
            raise
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            logger.debug("ffprobe validim u shmang")

    def _run_ffmpeg(self, cmd: list[str], operation: str) -> None:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                error_msg = result.stderr[-500:] if result.stderr else "gabim i panjohur"
                raise VideoAssemblyError(
                    f"FFmpeg deshtoi gjatë '{operation}': {error_msg}"
                )

            logger.debug(f"FFmpeg '{operation}' u ekzekutua me sukses")

        except subprocess.TimeoutExpired:
            raise VideoAssemblyError(
                f"FFmpeg kaloi limtin kohor (5 min) gjatë '{operation}'"
            )
        except VideoAssemblyError:
            raise
        except Exception as e:
            raise VideoAssemblyError(f"Gabim duke ekzekutuar FFmpeg: {e}")

    def get_video_info(self, video_path: Path) -> dict:
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                str(video_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

            if result.returncode != 0:
                return {}

            data       = json.loads(result.stdout)
            streams    = data.get("streams", [])
            fmt        = data.get("format", {})
            video_s    = next((s for s in streams if s["codec_type"] == "video"), {})
            audio_s    = next((s for s in streams if s["codec_type"] == "audio"), None)

            fps_str    = video_s.get("r_frame_rate", "24/1")
            fps_num, fps_den = map(int, fps_str.split("/"))
            fps        = fps_num / fps_den if fps_den else 0

            return {
                "duration"  : float(fmt.get("duration", 0)),
                "width"     : video_s.get("width", 0),
                "height"    : video_s.get("height", 0),
                "fps"       : fps,
                "size_mb"   : video_path.stat().st_size / 1024 / 1024,
                "has_audio" : audio_s is not None,
                "codec"     : video_s.get("codec_name", ""),
            }

        except Exception as e:
            logger.debug(f"get_video_info deshtoi: {e}")
            return {}