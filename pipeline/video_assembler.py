import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Any, Iterable

import numpy as np
from PIL import Image

from config.settings import VIDEO_CONFIG
from utils.logger import get_logger
from utils.exceptions import VideoAssemblyError

logger = get_logger(__name__)


class VideoAssembler:
    """
    Final video assembly layer for FalconAI Kids.

    Responsibilities:
      1. Discover scene frame directories.
      2. Normalize every scene to its intended duration/FPS.
      3. Preserve animation using ping-pong looping when necessary.
      4. Add smooth scene transitions.
      5. Encode the complete frame sequence with FFmpeg.
      6. Resolve and concatenate scene audio.
      7. Mux audio + video.
      8. Inject metadata.
      9. Validate the final MP4 with ffprobe.

    The class intentionally relies on FFmpeg/ffprobe rather than loading
    heavyweight video frameworks into Python.
    """

    def __init__(self):
        self.fps = max(1, int(VIDEO_CONFIG.get("fps", 24)))

        resolution = VIDEO_CONFIG.get("resolution", (512, 512))
        if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
            resolution = (512, 512)

        self.resolution = (
            max(2, int(resolution[0])),
            max(2, int(resolution[1])),
        )

        self.codec = VIDEO_CONFIG.get("codec", "libx264")
        self.audio_codec = VIDEO_CONFIG.get("audio_codec", "aac")
        self.quality = int(VIDEO_CONFIG.get("quality", 23))

        self.preset = VIDEO_CONFIG.get("preset", "medium")
        self.audio_bitrate = VIDEO_CONFIG.get("audio_bitrate", "192k")

        self.transition_duration = float(
            VIDEO_CONFIG.get("transition_duration", 0.35)
        )

        self.default_scene_duration = float(
            VIDEO_CONFIG.get("default_scene_duration", 6.0)
        )

        self.ffmpeg_timeout = int(
            VIDEO_CONFIG.get("ffmpeg_timeout_sec", 450)
        )

        self.ffprobe_timeout = int(
            VIDEO_CONFIG.get("ffprobe_timeout_sec", 20)
        )

        self._check_ffmpeg()
        self._check_ffprobe()

    # ------------------------------------------------------------------
    # External binary validation
    # ------------------------------------------------------------------

    def _check_ffmpeg(self) -> None:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise VideoAssemblyError(
                    "FFmpeg is installed but returned an unhealthy execution status."
                )

            version_line = (result.stdout or "").splitlines()
            version = version_line[0] if version_line else "unknown version"

            logger.debug(f"FFmpeg validation successful: {version}")

        except FileNotFoundError:
            raise VideoAssemblyError(
                "FFmpeg executable was not found on PATH.\n"
                "Install FFmpeg and ensure the executable is available."
            )
        except subprocess.TimeoutExpired:
            raise VideoAssemblyError(
                "FFmpeg did not respond to the version check within 10 seconds."
            )
        except VideoAssemblyError:
            raise
        except Exception as e:
            raise VideoAssemblyError(
                f"Unable to validate FFmpeg installation: {e}"
            )

    def _check_ffprobe(self) -> None:
        try:
            result = subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise VideoAssemblyError(
                    "ffprobe is installed but returned an unhealthy execution status."
                )

            logger.debug("ffprobe validation successful.")

        except FileNotFoundError:
            raise VideoAssemblyError(
                "ffprobe executable was not found on PATH. "
                "ffprobe normally ships with FFmpeg."
            )
        except subprocess.TimeoutExpired:
            raise VideoAssemblyError(
                "ffprobe did not respond to the version check within 10 seconds."
            )
        except VideoAssemblyError:
            raise
        except Exception as e:
            raise VideoAssemblyError(
                f"Unable to validate ffprobe installation: {e}"
            )

    # ------------------------------------------------------------------
    # Main assembly
    # ------------------------------------------------------------------

    def assemble(
        self,
        scenes: List[Dict[str, Any]],
        frames_dir: Path,
        audio_paths: Any,
        output_path: Path,
    ) -> Path:

        output_path = Path(output_path)
        frames_dir = Path(frames_dir)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Starting final video assembly | "
            f"frames={frames_dir} | output={output_path}"
        )

        scene_dirs = self._collect_scene_dirs(frames_dir)

        if not scene_dirs:
            raise VideoAssemblyError(
                f"No usable scene frame directories found in '{frames_dir}'."
            )

        logger.debug(
            f"Discovered {len(scene_dirs)} scene frame sequence(s)."
        )

        normalized_dirs = self._normalize_frames(
            scene_dirs=scene_dirs,
            scenes=scenes or [],
        )

        valid_dirs = [
            d for d in normalized_dirs
            if self._get_scene_frames(d)
        ]

        if not valid_dirs:
            raise VideoAssemblyError(
                "No valid frames remained after frame normalization."
            )

        workspace = frames_dir.parent / "_video_assembly_workspace"

        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)

        workspace.mkdir(parents=True, exist_ok=True)

        raw_video_path = workspace / "video_without_audio.mp4"

        try:
            transition_frames = self._calculate_transition_frames()

            final_frames_dir = self._apply_transitions(
                scene_dirs=valid_dirs,
                output_dir=workspace / "transitions",
                transition_frames=transition_frames,
            )

            self._frames_to_video(
                frames_dir=final_frames_dir,
                output_path=raw_video_path,
            )

            resolved_audio = self._resolve_audio_context(
                audio_paths=audio_paths,
                workspace_dir=workspace,
            )

            if resolved_audio and resolved_audio.exists():
                self._add_audio(
                    video_path=raw_video_path,
                    audio_path=resolved_audio,
                    output_path=output_path,
                )
            else:
                logger.warning(
                    "No valid audio track was resolved. "
                    "Producing a silent video."
                )
                shutil.copy2(raw_video_path, output_path)

            self._add_metadata(output_path, scenes)
            self._validate_output(output_path)

        except VideoAssemblyError:
            raise

        except Exception as e:
            raise VideoAssemblyError(
                f"Unexpected video assembly failure: {e}"
            )

        finally:
            shutil.rmtree(workspace, ignore_errors=True)

        if not output_path.exists():
            raise VideoAssemblyError(
                "Video assembly completed without producing the expected output file."
            )

        size_mb = output_path.stat().st_size / (1024 * 1024)

        logger.success(
            f"Video successfully assembled | "
            f"size={size_mb:.2f} MB | output={output_path}"
        )

        return output_path

    # ------------------------------------------------------------------
    # Scene discovery
    # ------------------------------------------------------------------

    def _collect_scene_dirs(self, frames_dir: Path) -> List[Path]:
        if not frames_dir.exists():
            return []

        if not frames_dir.is_dir():
            return []

        scene_dirs = [
            d for d in frames_dir.iterdir()
            if d.is_dir()
            and (
                d.name.startswith("scene_")
                or d.name.startswith("scenes_")
            )
        ]

        scene_dirs.sort(key=self._natural_sort_key)

        if scene_dirs:
            return scene_dirs

        direct_frames = self._get_scene_frames(frames_dir)

        if direct_frames:
            logger.debug(
                "Frames found directly in the root frame directory."
            )
            return [frames_dir]

        return []

    @staticmethod
    def _natural_sort_key(path: Path):
        import re

        parts = re.split(r"(\d+)", path.name)

        return [
            int(part) if part.isdigit() else part.lower()
            for part in parts
        ]

    def _get_scene_frames(self, scene_dir: Path) -> List[Path]:
        if not scene_dir.exists():
            return []

        frames = sorted(
            scene_dir.glob("frame_*.png"),
            key=self._natural_sort_key,
        )

        if not frames:
            frames = sorted(
                scene_dir.glob("*.png"),
                key=self._natural_sort_key,
            )

        return frames

    # ------------------------------------------------------------------
    # Frame normalization
    # ------------------------------------------------------------------

    def _normalize_frames(
        self,
        scene_dirs: List[Path],
        scenes: List[Dict[str, Any]],
    ) -> List[Path]:

        normalized = []

        for idx, scene_dir in enumerate(scene_dirs):
            frames = self._get_scene_frames(scene_dir)

            if not frames:
                logger.warning(
                    f"Scene {idx + 1} contains no PNG frames: {scene_dir}"
                )
                continue

            duration = self.default_scene_duration

            if idx < len(scenes):
                try:
                    duration = float(
                        scenes[idx].get(
                            "duration_sec",
                            self.default_scene_duration,
                        )
                    )
                except (TypeError, ValueError):
                    duration = self.default_scene_duration

            duration = max(0.1, duration)

            target_count = max(
                1,
                int(round(duration * self.fps)),
            )

            current_count = len(frames)

            logger.debug(
                f"Scene {idx + 1}: "
                f"{current_count} frame(s) -> "
                f"{target_count} frame(s) | "
                f"{duration:.2f}s @ {self.fps}fps"
            )

            if current_count != target_count:
                if current_count > target_count:
                    indices = np.linspace(
                        0,
                        current_count - 1,
                        target_count,
                        dtype=int,
                    )
                else:
                    indices = self._pingpong_index_sequence(
                        current_count,
                        target_count,
                    )

                self._rewrite_scene_frames(
                    scene_dir,
                    frames,
                    indices,
                )

            normalized.append(scene_dir)

        return normalized

    def _pingpong_index_sequence(
        self,
        n: int,
        length: int,
    ) -> np.ndarray:

        if n <= 0:
            raise ValueError("Cannot generate frame indices from zero frames.")

        if n == 1:
            return np.zeros(length, dtype=int)

        cycle = (
            list(range(n))
            + list(range(n - 2, 0, -1))
        )

        return np.array(
            [cycle[i % len(cycle)] for i in range(length)],
            dtype=int,
        )

    def _rewrite_scene_frames(
        self,
        scene_dir: Path,
        frames: List[Path],
        indices: np.ndarray,
    ):

        temp_dir = scene_dir / "_normalize_tmp"

        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            for new_index, source_index in enumerate(
                indices,
                start=1,
            ):
                source = frames[int(source_index)]

                if not source.exists():
                    raise VideoAssemblyError(
                        f"Source frame disappeared during normalization: {source}"
                    )

                destination = (
                    temp_dir / f"frame_{new_index:04d}.png"
                )

                shutil.copy2(source, destination)

            for frame in frames:
                try:
                    frame.unlink(missing_ok=True)
                except Exception:
                    pass

            generated = sorted(
                temp_dir.glob("frame_*.png"),
                key=self._natural_sort_key,
            )

            for generated_frame in generated:
                generated_frame.replace(
                    scene_dir / generated_frame.name
                )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Scene transitions
    # ------------------------------------------------------------------

    def _calculate_transition_frames(self) -> int:
        return max(
            0,
            int(round(self.transition_duration * self.fps)),
        )

    def _apply_transitions(
        self,
        scene_dirs: List[Path],
        output_dir: Path,
        transition_frames: int = 8,
    ) -> Path:

        all_frames_dir = output_dir / "compiled_sequence"

        if all_frames_dir.exists():
            shutil.rmtree(all_frames_dir, ignore_errors=True)

        all_frames_dir.mkdir(parents=True, exist_ok=True)

        global_index = 1

        for scene_index, scene_dir in enumerate(scene_dirs):

            current_frames = self._get_scene_frames(scene_dir)

            if not current_frames:
                continue

            is_last = scene_index == len(scene_dirs) - 1

            if (
                not is_last
                and transition_frames > 0
            ):
                next_frames = self._get_scene_frames(
                    scene_dirs[scene_index + 1]
                )

                if (
                    len(current_frames) > transition_frames
                    and len(next_frames) >= transition_frames
                ):
                    stable_frames = current_frames[
                        :-transition_frames
                    ]

                    for frame in stable_frames:
                        destination = (
                            all_frames_dir
                            / f"frame_{global_index:06d}.png"
                        )

                        shutil.copy2(frame, destination)
                        global_index += 1

                    transition = self._generate_crossfade(
                        from_frames=current_frames[-transition_frames:],
                        to_frames=next_frames[:transition_frames],
                        n_frames=transition_frames,
                    )

                    for image in transition:
                        destination = (
                            all_frames_dir
                            / f"frame_{global_index:06d}.png"
                        )

                        image.save(
                            destination,
                            format="PNG",
                            optimize=False,
                        )

                        global_index += 1

                    continue

            for frame in current_frames:
                destination = (
                    all_frames_dir
                    / f"frame_{global_index:06d}.png"
                )

                shutil.copy2(frame, destination)
                global_index += 1

        if not any(all_frames_dir.glob("frame_*.png")):
            raise VideoAssemblyError(
                "Transition compiler produced no output frames."
            )

        return all_frames_dir

    def _generate_crossfade(
        self,
        from_frames: List[Path],
        to_frames: List[Path],
        n_frames: int,
    ) -> List[Image.Image]:

        target_size = self.resolution

        if not from_frames or not to_frames:
            return []

        result = []

        for i in range(n_frames):
            alpha = (
                i / max(n_frames - 1, 1)
            )

            smooth_alpha = (
                0.5
                * (1.0 - np.cos(np.pi * alpha))
            )

            from_path = from_frames[
                min(i, len(from_frames) - 1)
            ]

            to_path = to_frames[
                min(i, len(to_frames) - 1)
            ]

            try:
                with Image.open(from_path) as source_a:
                    image_a = source_a.convert("RGB")

                with Image.open(to_path) as source_b:
                    image_b = source_b.convert("RGB")

                if image_a.size != target_size:
                    image_a = image_a.resize(
                        target_size,
                        Image.Resampling.LANCZOS,
                    )

                if image_b.size != target_size:
                    image_b = image_b.resize(
                        target_size,
                        Image.Resampling.LANCZOS,
                    )

                array_a = np.asarray(
                    image_a,
                    dtype=np.float32,
                )

                array_b = np.asarray(
                    image_b,
                    dtype=np.float32,
                )

                blended = (
                    array_a * (1.0 - smooth_alpha)
                    + array_b * smooth_alpha
                )

                blended = np.clip(
                    blended,
                    0,
                    255,
                ).astype(np.uint8)

                result.append(
                    Image.fromarray(blended, mode="RGB")
                )

            except Exception as e:
                logger.debug(
                    f"Crossfade frame failed; using source frame instead: {e}"
                )

                try:
                    with Image.open(from_path) as fallback:
                        result.append(
                            fallback.convert("RGB").copy()
                        )
                except Exception:
                    result.append(
                        Image.new(
                            "RGB",
                            target_size,
                        )
                    )

        return result

    # ------------------------------------------------------------------
    # Video encoding
    # ------------------------------------------------------------------

    def _frames_to_video(
        self,
        frames_dir: Path,
        output_path: Path,
    ) -> None:

        frames = sorted(
            frames_dir.glob("frame_*.png"),
            key=self._natural_sort_key,
        )

        if not frames:
            raise VideoAssemblyError(
                f"No frames available for video encoding: {frames_dir}"
            )

        first_name = frames[0].name

        try:
            digits = len(
                first_name.split("frame_", 1)[1]
                .rsplit(".png", 1)[0]
            )
        except Exception:
            digits = 6

        digits = max(1, digits)

        input_pattern = str(
            frames_dir / f"frame_%0{digits}d.png"
        )

        width, height = self.resolution

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(self.fps),
            "-start_number",
            "1",
            "-i",
            input_pattern,
            "-vf",
            (
                f"scale={width}:{height}:"
                "force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
                ":color=black"
            ),
            "-c:v",
            self.codec,
            "-preset",
            self.preset,
            "-crf",
            str(self.quality),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

        self._run_ffmpeg(
            cmd,
            "PNG frame sequence -> H.264 video encoding",
        )

    # ------------------------------------------------------------------
    # Audio resolution
    # ------------------------------------------------------------------

    def _resolve_audio_context(
        self,
        audio_paths: Any,
        workspace_dir: Path,
    ) -> Optional[Path]:

        if not audio_paths:
            return None

        if isinstance(audio_paths, (str, Path)):
            path = Path(audio_paths)

            return path if path.exists() else None

        if isinstance(audio_paths, dict):
            ordered_items = self._ordered_audio_values(audio_paths)

            valid_paths = [
                Path(value)
                for value in ordered_items
                if value and Path(value).exists()
            ]

            if not valid_paths:
                return None

            if len(valid_paths) == 1:
                return valid_paths[0]

            return self._concat_audio(
                valid_paths,
                workspace_dir,
            )

        if isinstance(audio_paths, (list, tuple)):
            valid_paths = [
                Path(value)
                for value in audio_paths
                if value and Path(value).exists()
            ]

            if not valid_paths:
                return None

            if len(valid_paths) == 1:
                return valid_paths[0]

            return self._concat_audio(
                valid_paths,
                workspace_dir,
            )

        return None

    def _ordered_audio_values(
        self,
        audio_paths: Dict[Any, Any],
    ) -> List[Any]:

        def key(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return str(value)

        return [
            audio_paths[key]
            for key in sorted(
                audio_paths.keys(),
                key=key,
            )
        ]

    def _concat_audio(
        self,
        audio_files: List[Path],
        workspace_dir: Path,
    ) -> Optional[Path]:

        list_file = workspace_dir / "audio_concat.txt"
        output_file = workspace_dir / "combined_audio_timeline.m4a"

        try:
            with list_file.open(
                "w",
                encoding="utf-8",
            ) as f:

                for audio_file in audio_files:
                    escaped = (
                        str(audio_file.resolve())
                        .replace("'", "'\\''")
                    )

                    f.write(
                        f"file '{escaped}'\n"
                    )

            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c:a",
                "aac",
                "-b:a",
                self.audio_bitrate,
                str(output_file),
            ]

            self._run_ffmpeg(
                cmd,
                "Scene narration audio concatenation",
            )

            return output_file if output_file.exists() else None

        except Exception as e:
            logger.warning(
                f"Unable to concatenate scene audio tracks: {e}"
            )
            return None

        finally:
            list_file.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # A/V muxing
    # ------------------------------------------------------------------

    def _add_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> None:

        temp_output = output_path.parent / (
            f".{output_path.stem}_muxing{output_path.suffix}"
        )

        temp_output.unlink(missing_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            self.audio_codec,
            "-b:a",
            self.audio_bitrate,
            "-shortest",
            "-movflags",
            "+faststart",
            str(temp_output),
        ]

        try:
            self._run_ffmpeg(
                cmd,
                "Video + narration audio multiplexing",
            )

            if not temp_output.exists():
                raise VideoAssemblyError(
                    "FFmpeg reported successful audio muxing, "
                    "but the temporary output file does not exist."
                )

            temp_output.replace(output_path)

        finally:
            temp_output.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _add_metadata(
        self,
        video_path: Path,
        scenes: Any,
    ) -> None:

        if not video_path.exists():
            return

        from datetime import datetime

        temp_output = video_path.parent / (
            f".{video_path.stem}_metadata{video_path.suffix}"
        )

        temp_output.unlink(missing_ok=True)

        title = "FalconAI Kids Personalized Story"
        scene_count = len(scenes) if isinstance(scenes, list) else 0

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0",
            "-c",
            "copy",
            "-metadata",
            f"title={title}",
            "-metadata",
            "artist=FalconAI Kids",
            "-metadata",
            "album=Personalized Children's Story",
            "-metadata",
            "genre=Children Animation",
            "-metadata",
            f"comment=Generated by FalconAI Kids | scenes={scene_count}",
            "-metadata",
            f"date={datetime.now().year}",
            str(temp_output),
        ]

        try:
            self._run_ffmpeg(
                cmd,
                "MP4 metadata injection",
            )

            if temp_output.exists():
                temp_output.replace(video_path)

        except Exception as e:
            logger.debug(
                f"Metadata injection skipped: {e}"
            )

        finally:
            temp_output.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    def _validate_output(
        self,
        video_path: Path,
    ) -> None:

        if not video_path.exists():
            raise VideoAssemblyError(
                f"Final video was not created: {video_path}"
            )

        size_bytes = video_path.stat().st_size

        if size_bytes < 2048:
            raise VideoAssemblyError(
                f"Final video appears corrupted or empty "
                f"(size={size_bytes} bytes)."
            )

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.ffprobe_timeout,
            )

            if result.returncode != 0:
                raise VideoAssemblyError(
                    "ffprobe could not validate the final video: "
                    f"{(result.stderr or '')[-500:]}"
                )

            data = json.loads(result.stdout)

            streams = data.get("streams", [])

            video_stream = next(
                (
                    stream
                    for stream in streams
                    if stream.get("codec_type") == "video"
                ),
                None,
            )

            if video_stream is None:
                raise VideoAssemblyError(
                    "Final MP4 contains no video stream."
                )

            width = int(
                video_stream.get("width") or 0
            )

            height = int(
                video_stream.get("height") or 0
            )

            if width <= 0 or height <= 0:
                raise VideoAssemblyError(
                    "Final video stream has an invalid resolution."
                )

            duration = self._safe_float(
                data.get("format", {}).get("duration")
            )

            has_audio = any(
                stream.get("codec_type") == "audio"
                for stream in streams
            )

            logger.debug(
                f"Final video validation successful | "
                f"{width}x{height} | "
                f"duration={duration:.2f}s | "
                f"audio={has_audio} | "
                f"size={size_bytes / (1024 * 1024):.2f} MB"
            )

        except VideoAssemblyError:
            raise

        except subprocess.TimeoutExpired:
            raise VideoAssemblyError(
                "ffprobe timed out while validating the final video."
            )

        except json.JSONDecodeError as e:
            raise VideoAssemblyError(
                f"ffprobe returned invalid JSON during validation: {e}"
            )

        except Exception as e:
            raise VideoAssemblyError(
                f"Unable to validate final video: {e}"
            )

    # ------------------------------------------------------------------
    # FFmpeg execution
    # ------------------------------------------------------------------

    def _run_ffmpeg(
        self,
        cmd: List[str],
        operation: str,
    ) -> None:

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.ffmpeg_timeout,
            )

        except subprocess.TimeoutExpired:
            raise VideoAssemblyError(
                f"FFmpeg operation timed out during '{operation}' "
                f"after {self.ffmpeg_timeout}s."
            )

        except FileNotFoundError:
            raise VideoAssemblyError(
                "FFmpeg executable is no longer available on PATH."
            )

        except Exception as e:
            raise VideoAssemblyError(
                f"Unable to execute FFmpeg during '{operation}': {e}"
            )

        if result.returncode != 0:
            stderr = (
                result.stderr
                or result.stdout
                or "Unknown FFmpeg failure."
            )

            raise VideoAssemblyError(
                f"FFmpeg failed during '{operation}': "
                f"{stderr[-1200:]}"
            )

        logger.debug(
            f"FFmpeg operation completed successfully: {operation}"
        )

    # ------------------------------------------------------------------
    # Public media information API
    # ------------------------------------------------------------------

    def get_video_info(
        self,
        video_path: Path,
    ) -> Dict[str, Any]:

        video_path = Path(video_path)

        if not video_path.exists():
            return {}

        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.ffprobe_timeout,
            )

            if result.returncode != 0:
                return {}

            data = json.loads(result.stdout)

            streams = data.get("streams", [])
            fmt = data.get("format", {})

            video_stream = next(
                (
                    stream
                    for stream in streams
                    if stream.get("codec_type") == "video"
                ),
                {},
            )

            audio_stream = next(
                (
                    stream
                    for stream in streams
                    if stream.get("codec_type") == "audio"
                ),
                None,
            )

            fps = self._parse_fps(
                video_stream.get("r_frame_rate")
            )

            duration = self._safe_float(
                fmt.get("duration")
            )

            return {
                "duration": duration,
                "width": int(
                    video_stream.get("width") or 0
                ),
                "height": int(
                    video_stream.get("height") or 0
                ),
                "fps": fps,
                "size_mb": (
                    video_path.stat().st_size
                    / (1024 * 1024)
                ),
                "has_audio": audio_stream is not None,
                "codec": video_stream.get(
                    "codec_name",
                    "",
                ),
                "audio_codec": (
                    audio_stream.get("codec_name", "")
                    if audio_stream
                    else ""
                ),
                "format": fmt.get(
                    "format_name",
                    "",
                ),
            }

        except Exception as e:
            logger.debug(
                f"Unable to retrieve video information via ffprobe: {e}"
            )
            return {}

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _parse_fps(value: Any) -> float:
        if not value:
            return 0.0

        try:
            value = str(value)

            if "/" in value:
                numerator, denominator = value.split(
                    "/",
                    1,
                )

                denominator = float(denominator)

                if denominator == 0:
                    return 0.0

                return float(numerator) / denominator

            return float(value)

        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0


Orchestrator = VideoAssembler
