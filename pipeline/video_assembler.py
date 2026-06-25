import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
from PIL import Image

from config.settings import VIDEO_CONFIG, DIFFUSION_CONFIG
from utils.logger import get_logger
from utils.exceptions import VideoAssemblyError

logger = get_logger(__name__)


class VideoAssembler:
    def __init__(self):
        self.fps = VIDEO_CONFIG.get("fps", 24)
        self.resolution = VIDEO_CONFIG.get("resolution", (1024, 1024))
        self.codec = VIDEO_CONFIG.get("codec", "libx264")
        self.quality = VIDEO_CONFIG.get("quality", 23)
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
                logger.debug(f"System FFmpeg validation verified: {version_line}")
            else:
                raise VideoAssemblyError(
                    "FFmpeg binary diagnostic reported an unhealthy execution status. "
                    "Please download and re-link configurations from: https://ffmpeg.org/download.html"
                )
        except FileNotFoundError:
            raise VideoAssemblyError(
                "FFmpeg executable binary could not be located on system environmental path contexts.\n"
                "To resolve, please run:\n"
                "  Ubuntu/Debian: sudo apt install ffmpeg\n"
                "  macOS: brew install ffmpeg\n"
                "  Windows: Install via winget or set standard binary PATH locations."
            )
        except subprocess.TimeoutExpired:
            raise VideoAssemblyError("FFmpeg operational interface failed to respond within a standard 10-second window.")

    def assemble(
        self,
        scenes: List[Dict[str, Any]],
        frames_dir: Path,
        audio_paths: Dict[str, Any],
        output_path: Path
    ) -> Path:
        logger.info(f"Commencing video container serialization layer targeting: {output_path.name}")

        frames_dir_path = Path(frames_dir)
        scene_dirs = self._collect_scene_dirs(frames_dir_path)
        logger.debug(f"Located active sequential rendering directories: {len(scene_dirs)}")

        if not scene_dirs:
            raise VideoAssemblyError(f"No valid image sequence directory schemas found matching target: {frames_dir_path}")

        normalized_dirs = self._normalize_frames(scene_dirs, scenes)

        transitions_workspace = frames_dir_path.parent / "transitions_workspace"
        transitions_workspace.mkdir(parents=True, exist_ok=True)
        
        final_frames_dir = self._apply_transitions(
            scene_dirs=normalized_dirs,
            output_dir=transitions_workspace,
            transition_frames=int(self.fps * 0.35)
        )

        raw_video_path = output_path.parent / f"_raw_{output_path.name}"
        self._frames_to_video(
            frames_dir=final_frames_dir,
            output_path=raw_video_path,
        )
        logger.debug(f"Intermediate raw video stream serialized safely: {raw_video_path.name}")

        resolved_audio = self._resolve_audio_context(audio_paths, output_path.parent)

        if resolved_audio and resolved_audio.exists():
            try:
                self._add_audio(
                    video_path=raw_video_path,
                    audio_path=resolved_audio,
                    output_path=output_path,
                )
                logger.debug("Synchronized audio track mixed into video stream wrapper container successfully.")
            finally:
                raw_video_path.unlink(missing_ok=True)
                if "combined_audio" in resolved_audio.name:
                    resolved_audio.unlink(missing_ok=True)
        else:
            logger.warning("Audio path track resolution empty or missing. Defaulting container to silent stream format.")
            shutil.move(str(raw_video_path), str(output_path))

        self._add_metadata(output_path, scenes)
        self._validate_output(output_path)

        try:
            shutil.rmtree(transitions_workspace, ignore_errors=True)
        except Exception:
            pass

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.success(f"Video pipeline asset completely compiled! File Size: {size_mb:.2f} MB | Saved at: {output_path}")

        return output_path

    def _collect_scene_dirs(self, frames_dir: Path) -> List[Path]:
        scene_dirs = sorted([
            d for d in frames_dir.iterdir()
            if d.is_dir() and (d.name.startswith("scene_") or d.name.startswith("scenes_"))
        ])

        if not scene_dirs:
            frames = list(frames_dir.glob("frame_*.png")) or list(frames_dir.glob("*.png"))
            if frames:
                logger.debug("Visual sequence matrices identified directly within principal directory structure framework.")
                return [frames_dir]

        return scene_dirs

    def _get_scene_frames(self, scene_dir: Path) -> List[Path]:
        frames = sorted(scene_dir.glob("frame_*.png"))
        if not frames:
            frames = sorted(scene_dir.glob("*.png"))
        return frames

    def _normalize_frames(self, scene_dirs: List[Path], scenes: List[Dict[str, Any]]) -> List[Path]:
        default_dur = 6.0

        for idx, scene_dir in enumerate(scene_dirs):
            frames = self._get_scene_frames(scene_dir)
            if not frames:
                logger.warning(f"Target process sequence node segment folder contains no valid image data arrays: {scene_dir}")
                continue

            dur_sec = default_dur
            if idx < len(scenes):
                dur_sec = scenes[idx].get("duration_sec", default_dur)

            target_count = int(dur_sec * self.fps)
            current_count = len(frames)

            if current_count == target_count:
                continue
            elif current_count > target_count:
                indices = np.linspace(0, current_count - 1, target_count, dtype=int)
                keep_set = {frames[j] for j in indices}
                for f in frames:
                    if f not in keep_set:
                        f.unlink(missing_ok=True)
            else:
                last_frame = frames[-1]
                for j in range(target_count - current_count):
                    new_frame_path = scene_dir / f"frame_{current_count + j + 1:04d}.png"
                    shutil.copy(str(last_frame), str(new_frame_path))

        logger.debug("Internal temporal structural frame index padding operations completed successfully.")
        return scene_dirs

    def _apply_transitions(self, scene_dirs: List[Path], output_dir: Path, transition_frames: int = 8) -> Path:
        all_frames_dir = output_dir / "compiled_sequence"
        all_frames_dir.mkdir(parents=True, exist_ok=True)

        global_idx = 1

        for idx, scene_dir in enumerate(scene_dirs):
            frames = self._get_scene_frames(scene_dir)
            if not frames:
                continue

            is_last_scene = (idx == len(scene_dirs) - 1)

            if not is_last_scene and len(frames) > (transition_frames * 2):
                frames_to_copy = frames[:-transition_frames]
            else:
                frames_to_copy = frames

            for frame_path in frames_to_copy:
                dst = all_frames_dir / f"frame_{global_idx:06d}.png"
                shutil.copy(str(frame_path), str(dst))
                global_idx += 1

            if not is_last_scene and len(frames) > transition_frames:
                next_scene_dir = scene_dirs[idx + 1]
                next_scene_frames = self._get_scene_frames(next_scene_dir)

                if next_scene_frames:
                    tail_segment = frames[-transition_frames:]
                    head_segment = next_scene_frames[:transition_frames]

                    blended_frames = self._generate_crossfade(
                        from_frames=tail_segment,
                        to_frames=head_segment,
                        n_frames=transition_frames
                    )

                    for blend_img in blended_frames:
                        dst = all_frames_dir / f"frame_{global_idx:06d}.png"
                        blend_img.save(dst)
                        global_idx += 1

        return all_frames_dir

    def _generate_crossfade(self, from_frames: List[Path], to_frames: List[Path], n_frames: int) -> List[Image.Image]:
        target_w = DIFFUSION_CONFIG.get("width", 1024)
        target_h = DIFFUSION_CONFIG.get("height", 1024)
        blended_sequence: List[Image.Image] = []

        for i in range(n_frames):
            alpha = i / max(n_frames - 1, 1)

            from_idx = min(i, len(from_frames) - 1)
            to_idx = min(i, len(to_frames) - 1)

            try:
                img_a = Image.open(from_frames[from_idx]).convert("RGB")
                img_b = Image.open(to_frames[to_idx]).convert("RGB")

                if img_a.size != (target_w, target_h):
                    img_a = img_a.resize((target_w, target_h), Image.Resampling.LANCZOS)
                if img_b.size != (target_w, target_h):
                    img_b = img_b.resize((target_w, target_h), Image.Resampling.LANCZOS)

                arr_a = np.array(img_a, dtype=np.float32)
                arr_b = np.array(img_b, dtype=np.float32)

                smooth_alpha = 0.5 * (1.0 - np.cos(np.pi * alpha))
                blended_matrix = arr_a * (1.0 - smooth_alpha) + arr_b * smooth_alpha
                blended_matrix = np.clip(blended_matrix, 0, 255).astype(np.uint8)

                blended_sequence.append(Image.fromarray(blended_matrix))

            except Exception as blend_err:
                logger.debug(f"Bypassing custom pixel blend state frame computation due to operational faults: {blend_err}")
                try:
                    blended_sequence.append(Image.open(from_frames[-1]).convert("RGB"))
                except Exception:
                    blended_sequence.append(Image.new("RGB", (target_w, target_h), (0, 0, 0)))

        return blended_sequence

    def _frames_to_video(self, frames_dir: Path, output_path: Path) -> None:
        target_w, target_h = self.resolution
        sample_frames = list(frames_dir.glob("frame_*.png"))
        
        if not sample_frames:
            raise VideoAssemblyError(f"Inference sequence engine aborting; visual matrix directory fields are unpopulated: {frames_dir}")

        first_frame_name = sorted(sample_frames)[0].name
        digits_count = len(first_frame_name.replace("frame_", "").replace(".png", ""))
        input_pattern = str(frames_dir / f"frame_%0{digits_count}d.png")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(self.fps),
            "-i", input_pattern,
            "-c:v", self.codec,
            "-crf", str(self.quality),
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
            "-movflags", "+faststart",
            str(output_path),
        ]

        self._run_ffmpeg(cmd, "Visual Stream Frame Raster Encoding Sequence")

    def _resolve_audio_context(self, audio_paths: Any, workspace_dir: Path) -> Optional[Path]:
        if not audio_paths:
            return None

        if isinstance(audio_paths, (str, Path)):
            p = Path(audio_paths)
            return p if p.exists() else None

        if isinstance(audio_paths, dict):
            paths_list = list(audio_paths.values())
            if not paths_list:
                return None
            if len(paths_list) == 1:
                p = Path(paths_list[0])
                return p if p.exists() else None

            try:
                combined_audio_path = workspace_dir / "combined_audio_timeline.mp3"
                txt_list_path = workspace_dir / "audio_tracks.txt"

                with open(txt_list_path, "w", encoding="utf-8") as f:
                    for track in paths_list:
                        track_path = Path(track).resolve()
                        if track_path.exists():
                            f.write(f"file '{track_path}'\n")

                cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(txt_list_path),
                    "-c:a", "libmp3lame",
                    "-b:a", "192k",
                    str(combined_audio_path)
                ]
                
                self._run_ffmpeg(cmd, "Multi-Track Audio Concat Aggregation")
                txt_list_path.unlink(missing_ok=True)
                return combined_audio_path
            except Exception as concat_err:
                logger.error(f"Failed to join component background narration audio sequences safely: {concat_err}")
                return None

        if isinstance(audio_paths, list) and len(audio_paths) > 0:
            p = Path(audio_paths[0])
            return p if p.exists() else None

        return None

    def _add_audio(self, video_path: Path, audio_path: Path, output_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", VIDEO_CONFIG.get("audio_codec", "aac"),
            "-b:a", "192k",
            "-shortest",
            "-map", "0:v:0",
            "-map", "1:a:0",
            str(output_path),
        ]

        self._run_ffmpeg(cmd, "A/V Multiplexing Stream Serialization")

    def _add_metadata(self, video_path: Path, scenes: Any) -> None:
        title = "FalconAI Kids Personalized Movie"
        year = __import__("datetime").datetime.now().year
        temp_meta_out = video_path.parent / f"_meta_assigned_{video_path.name}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-c", "copy",
            "-metadata", f"title={title}",
            "-metadata", "artist=FalconAI Kids Engine System",
            "-metadata", "album=Personalized Children Book Volume Space",
            "-metadata", "genre=Cinematic Children Animation Media",
            "-metadata", f"date={year}",
            str(temp_meta_out),
        ]

        try:
            self._run_ffmpeg(cmd, "Container Parameter Metadata Injection")
            if temp_meta_out.exists():
                shutil.move(str(temp_meta_out), str(video_path))
        except Exception as meta_err:
            logger.debug(f"Optional metadata encapsulation skipped due to non-fatal platform warnings: {meta_err}")
            temp_meta_out.unlink(missing_ok=True)

    def _validate_output(self, video_path: Path) -> None:
        if not video_path.exists():
            raise VideoAssemblyError(f"Pipeline verification failed; targeted video asset not constructed on disk: {video_path}")

        size_bytes = video_path.stat().st_size
        if size_bytes < 2048:
            raise VideoAssemblyError(
                f"Generated container asset displays standard data corruption signatures (size: {size_bytes} bytes)."
            )

        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                str(video_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                streams = probe_data.get("streams", [])
                has_video = any(s.get("codec_type") == "video" for s in streams)

                if not has_video:
                    raise VideoAssemblyError("Compiled production package contains no actionable or parsable visual stream arrays.")

                video_stream = next(s for s in streams if s.get("codec_type") == "video")
                w = video_stream.get("width", 0)
                h = video_stream.get("height", 0)
                logger.debug(f"Media structural validations completed cleanly: Output Resolution {w}x{h} | File Size: {size_bytes / (1024 * 1024):.2f} MB")

        except VideoAssemblyError:
            raise
        except Exception as probe_err:
            logger.warning(f"Secondary ffprobe tracking evaluation skipped due to operating level restrictions: {probe_err}")

    def _run_ffmpeg(self, cmd: List[str], operation: str) -> None:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=450,
            )

            if result.returncode != 0:
                error_msg = result.stderr[-600:] if result.stderr else "Unknown underlying platform exception error trapped."
                raise VideoAssemblyError(
                    f"FFmpeg pipeline integration component dropped trace states during '{operation}': {error_msg}"
                )

            logger.debug(f"FFmpeg tracking process node successfully completed execution: '{operation}'")

        except subprocess.TimeoutExpired:
            raise VideoAssemblyError(f"FFmpeg task execution threshold exceeded target processing windows during active step: '{operation}'")
        except VideoAssemblyError:
            raise
        except Exception as e:
            raise VideoAssemblyError(f"Unexpected operational pipeline failure communicating via host subprocess layers: {e}")

    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
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

            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            fmt = data.get("format", {})
            
            video_s = next((s for s in streams if s.get("codec_type") == "video"), {})
            audio_s = next((s for s in streams if s.get("codec_type") == "audio"), None)

            fps_str = video_s.get("r_frame_rate", "24/1")
            if "/" in fps_str:
                fps_num, fps_den = map(int, fps_str.split("/"))
                fps = fps_num / fps_den if fps_den else 0.0
            else:
                fps = float(fps_str) if fps_str else 0.0

            return {
                "duration": float(fmt.get("duration", 0.0)),
                "width": int(video_s.get("width", 0)),
                "height": int(video_s.get("height", 0)),
                "fps": fps,
                "size_mb": video_path.stat().st_size / (1024 * 1024),
                "has_audio": audio_s is not None,
                "codec": video_s.get("codec_name", ""),
            }

        except Exception as info_err:
            logger.debug(f"Unable to parse platform metadata parameters using ffprobe tools: {info_err}")
            return {}
