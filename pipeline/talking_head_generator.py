import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from config.settings import SADTALKER_CONFIG, DEVICE
from utils.logger import get_logger
from utils.exceptions import TalkingHeadGenerationError

logger = get_logger(__name__)


class TalkingHeadGenerator:
    """
    Robust SadTalker wrapper for generating a talking-head sequence from:

        reference face image + scene narration audio -> PNG frames

    Design goals:
    - SadTalker remains completely optional.
    - No external API calls.
    - Uses SadTalker's CLI rather than importing unstable internals.
    - Validates input/output aggressively.
    - Handles different SadTalker CLI variants where possible.
    - Extracts frames with ffmpeg.
    - Keeps output naming compatible with the compositor.
    - Cleans temporary SadTalker output after every run.
    - Avoids crashing the entire pipeline when SadTalker is unavailable.
    """

    VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

    def __init__(self, fps: Optional[int] = None):
        self.repo_dir = Path(SADTALKER_CONFIG["repo_dir"]).expanduser()
        self.checkpoint_dir = Path(
            SADTALKER_CONFIG.get(
                "checkpoint_dir",
                self.repo_dir / "checkpoints",
            )
        ).expanduser()

        self.size = int(SADTALKER_CONFIG.get("size", 256))
        self.preprocess = SADTALKER_CONFIG.get("preprocess", "crop")
        self.still_mode = bool(SADTALKER_CONFIG.get("still_mode", False))
        self.use_enhancer = bool(SADTALKER_CONFIG.get("use_enhancer", False))
        self.expression_scale = float(
            SADTALKER_CONFIG.get("expression_scale", 1.0)
        )
        self.pose_style = int(SADTALKER_CONFIG.get("pose_style", 0))

        self.inference_timeout_sec = int(
            SADTALKER_CONFIG.get("inference_timeout_sec", 600)
        )
        self.ffmpeg_timeout_sec = int(
            SADTALKER_CONFIG.get("ffmpeg_timeout_sec", 180)
        )

        self.fps = int(fps or SADTALKER_CONFIG.get("fps", 24))
        self.python_executable = str(
            SADTALKER_CONFIG.get("python_executable", sys.executable)
        )

        self.available = False

        self._verify_installation()

    # ------------------------------------------------------------------
    # Installation / availability
    # ------------------------------------------------------------------

    def _verify_installation(self) -> None:
        """
        Verify that the minimum SadTalker installation exists.

        This intentionally does not attempt model inference during startup.
        A missing optional dependency should not prevent FalconAI Kids from
        generating the rest of the story.
        """
        inference_script = self.repo_dir / "inference.py"

        if not self.repo_dir.exists():
            logger.warning(
                f"SadTalker repository not found: '{self.repo_dir}'. "
                "Talking-head generation will be disabled."
            )
            self.available = False
            return

        if not inference_script.exists():
            logger.warning(
                f"SadTalker inference script not found: '{inference_script}'. "
                "Talking-head generation will be disabled."
            )
            self.available = False
            return

        if not self.checkpoint_dir.exists():
            logger.warning(
                f"SadTalker checkpoint directory does not exist: "
                f"'{self.checkpoint_dir}'. Talking-head generation will be disabled."
            )
            self.available = False
            return

        try:
            has_checkpoint_files = any(
                p.is_file() for p in self.checkpoint_dir.rglob("*")
            )
        except Exception as e:
            logger.warning(
                f"Unable to inspect SadTalker checkpoints at "
                f"'{self.checkpoint_dir}': {e}"
            )
            self.available = False
            return

        if not has_checkpoint_files:
            logger.warning(
                f"SadTalker checkpoint directory is empty: "
                f"'{self.checkpoint_dir}'. Talking-head generation will be disabled."
            )
            self.available = False
            return

        if not self._command_available("ffmpeg"):
            logger.warning(
                "ffmpeg is not available in PATH. "
                "SadTalker frame extraction will not work."
            )
            self.available = False
            return

        logger.success(
            f"SadTalker installation detected | "
            f"repo='{self.repo_dir}' | "
            f"checkpoints='{self.checkpoint_dir}' | "
            f"fps={self.fps}"
        )

        self.available = True

    def _command_available(self, command: str) -> bool:
        try:
            result = subprocess.run(
                [command, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def is_available(self) -> bool:
        return self.available

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate(
        self,
        face_image_path: Optional[Path],
        audio_path: Optional[Path],
        output_dir: Path,
        scene_index: Optional[int] = None,
    ) -> Optional[Path]:
        """
        Generate talking-head PNG frames for one scene.

        Returns:
            Path to extracted PNG frame directory, or None when SadTalker
            cannot/should not be used.

        Raises:
            TalkingHeadGenerationError only after the method has committed
            to running SadTalker and an actual processing failure occurs.
        """

        if not self.available:
            logger.debug(
                f"[scene {scene_index}] SadTalker unavailable; "
                "skipping talking-head generation."
            )
            return None

        face_path = self._validate_input_file(
            face_image_path,
            "reference face image",
            scene_index,
        )
        if face_path is None:
            return None

        audio = self._validate_input_file(
            audio_path,
            "narration audio",
            scene_index,
        )
        if audio is None:
            return None

        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_dir = output_dir / "sadtalker_raw"
        frames_dir = output_dir / "frames"

        self._safe_prepare_directory(raw_dir)
        self._safe_prepare_directory(frames_dir)

        try:
            raw_video_path = self._run_sadtalker(
                face_image_path=face_path,
                audio_path=audio,
                result_dir=raw_dir,
                scene_index=scene_index,
            )

            self._extract_frames(
                video_path=raw_video_path,
                frames_dir=frames_dir,
                scene_index=scene_index,
            )

            frame_count = self._count_frames(frames_dir)

            if frame_count <= 0:
                raise TalkingHeadGenerationError(
                    "SadTalker completed, but no frames were produced.",
                    scene_index=scene_index,
                )

            logger.success(
                f"[scene {scene_index}] Talking-head generation completed | "
                f"frames={frame_count} | fps={self.fps}"
            )

            return frames_dir

        except TalkingHeadGenerationError:
            raise

        except Exception as e:
            raise TalkingHeadGenerationError(
                f"Unexpected talking-head generation failure: {e}",
                scene_index=scene_index,
            )

        finally:
            self._cleanup_directory(raw_dir)

    # ------------------------------------------------------------------
    # SadTalker execution
    # ------------------------------------------------------------------

    def _build_command(
        self,
        face_image_path: Path,
        audio_path: Path,
        result_dir: Path,
    ) -> list[str]:
        """
        Build the SadTalker CLI command.

        The common SadTalker interface is intentionally used instead of
        importing its internal Python modules.
        """

        cmd = [
            self.python_executable,
            str(self.repo_dir / "inference.py"),
            "--driven_audio",
            str(audio_path),
            "--source_image",
            str(face_image_path),
            "--result_dir",
            str(result_dir),
            "--checkpoint_dir",
            str(self.checkpoint_dir),
            "--size",
            str(self.size),
            "--preprocess",
            str(self.preprocess),
            "--expression_scale",
            str(self.expression_scale),
            "--pose_style",
            str(self.pose_style),
        ]

        if self.still_mode:
            cmd.append("--still")

        if self.use_enhancer:
            cmd.extend(
                [
                    "--enhancer",
                    "gfpgan",
                ]
            )

        # SadTalker versions/forks differ in how CPU mode is exposed.
        # Keep the existing explicit flag because this codebase supports
        # CPU fallback as well.
        if DEVICE != "cuda":
            cmd.append("--cpu")

        return cmd

    def _run_sadtalker(
        self,
        face_image_path: Path,
        audio_path: Path,
        result_dir: Path,
        scene_index: Optional[int],
    ) -> Path:
        cmd = self._build_command(
            face_image_path=face_image_path,
            audio_path=audio_path,
            result_dir=result_dir,
        )

        scene_label = (
            str(scene_index)
            if scene_index is not None
            else "?"
        )

        logger.step(
            f"[scene {scene_label}] Starting SadTalker inference..."
        )

        logger.debug(
            f"[scene {scene_label}] SadTalker command prepared | "
            f"device={DEVICE} | size={self.size} | "
            f"preprocess={self.preprocess} | "
            f"still={self.still_mode} | "
            f"enhancer={self.use_enhancer}"
        )

        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.inference_timeout_sec,
                cwd=str(self.repo_dir),
            )

        except subprocess.TimeoutExpired:
            raise TalkingHeadGenerationError(
                f"SadTalker inference exceeded the configured timeout "
                f"of {self.inference_timeout_sec} seconds.",
                scene_index=scene_index,
            )

        except FileNotFoundError as e:
            raise TalkingHeadGenerationError(
                f"Unable to launch SadTalker using Python executable "
                f"'{self.python_executable}': {e}",
                scene_index=scene_index,
            )

        except Exception as e:
            raise TalkingHeadGenerationError(
                f"Failed to launch SadTalker subprocess: {e}",
                scene_index=scene_index,
            )

        stdout = process.stdout or ""
        stderr = process.stderr or ""

        if process.returncode != 0:
            error_output = self._tail_output(
                stderr if stderr.strip() else stdout,
                max_chars=1500,
            )

            raise TalkingHeadGenerationError(
                "SadTalker inference failed "
                f"(exit code {process.returncode}). "
                f"Output: {error_output}",
                scene_index=scene_index,
            )

        output_video = self._locate_output_video(result_dir)

        if output_video is None:
            diagnostic = self._describe_result_directory(result_dir)

            raise TalkingHeadGenerationError(
                "SadTalker reported successful execution, but no output "
                f"video was found inside '{result_dir}'. {diagnostic}",
                scene_index=scene_index,
            )

        logger.success(
            f"[scene {scene_label}] SadTalker output generated: "
            f"{output_video}"
        )

        return output_video

    # ------------------------------------------------------------------
    # Output discovery
    # ------------------------------------------------------------------

    def _locate_output_video(
        self,
        result_dir: Path,
    ) -> Optional[Path]:
        """
        Locate SadTalker's generated video.

        SadTalker commonly writes into a timestamped directory, therefore
        the path cannot safely be hardcoded.
        """

        if not result_dir.exists():
            return None

        candidates = []

        try:
            for path in result_dir.rglob("*"):
                if not path.is_file():
                    continue

                if path.suffix.lower() not in self.VIDEO_EXTENSIONS:
                    continue

                try:
                    stat = path.stat()
                    candidates.append(
                        (
                            stat.st_mtime,
                            stat.st_size,
                            path,
                        )
                    )
                except OSError:
                    continue

        except Exception as e:
            logger.debug(
                f"Unable to recursively inspect SadTalker output "
                f"directory: {e}"
            )
            return None

        if not candidates:
            return None

        # Prefer the newest non-empty video.
        candidates.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        for _, size, path in candidates:
            if size > 0:
                return path

        return candidates[0][2]

    def _describe_result_directory(self, result_dir: Path) -> str:
        if not result_dir.exists():
            return "Result directory does not exist."

        try:
            files = [
                str(p.relative_to(result_dir))
                for p in result_dir.rglob("*")
                if p.is_file()
            ]

            if not files:
                return "Result directory is empty."

            preview = files[:20]
            return (
                "Files found in result directory: "
                + ", ".join(preview)
            )

        except Exception:
            return "Unable to inspect the SadTalker result directory."

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(
        self,
        video_path: Path,
        frames_dir: Path,
        scene_index: Optional[int],
    ) -> None:
        if not video_path.exists():
            raise TalkingHeadGenerationError(
                f"SadTalker output video does not exist: '{video_path}'.",
                scene_index=scene_index,
            )

        if video_path.stat().st_size <= 0:
            raise TalkingHeadGenerationError(
                f"SadTalker output video is empty: '{video_path}'.",
                scene_index=scene_index,
            )

        frames_dir.mkdir(parents=True, exist_ok=True)

        output_pattern = frames_dir / "frame_%04d.png"

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={self.fps}",
            "-start_number",
            "1",
            str(output_pattern),
        ]

        logger.debug(
            f"[scene {scene_index}] Extracting SadTalker video "
            f"into PNG frames at {self.fps} FPS..."
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.ffmpeg_timeout_sec,
            )

        except subprocess.TimeoutExpired:
            raise TalkingHeadGenerationError(
                "Timed out while extracting SadTalker frames with ffmpeg.",
                scene_index=scene_index,
            )

        except FileNotFoundError:
            raise TalkingHeadGenerationError(
                "ffmpeg executable was not found. "
                "Install ffmpeg and make sure it is available in PATH.",
                scene_index=scene_index,
            )

        except Exception as e:
            raise TalkingHeadGenerationError(
                f"Failed to launch ffmpeg for frame extraction: {e}",
                scene_index=scene_index,
            )

        if result.returncode != 0:
            error_output = self._tail_output(
                result.stderr or result.stdout,
                max_chars=1200,
            )

            raise TalkingHeadGenerationError(
                f"ffmpeg failed while extracting SadTalker frames: "
                f"{error_output}",
                scene_index=scene_index,
            )

        frame_count = self._count_frames(frames_dir)

        if frame_count == 0:
            raise TalkingHeadGenerationError(
                f"ffmpeg completed without producing PNG frames in "
                f"'{frames_dir}'.",
                scene_index=scene_index,
            )

        logger.success(
            f"[scene {scene_index}] Extracted {frame_count} "
            f"talking-head frame(s)."
        )

    # ------------------------------------------------------------------
    # Validation / filesystem helpers
    # ------------------------------------------------------------------

    def _validate_input_file(
        self,
        path: Optional[Path],
        description: str,
        scene_index: Optional[int],
    ) -> Optional[Path]:
        if not path:
            logger.debug(
                f"[scene {scene_index}] No {description} supplied; "
                "skipping talking-head generation."
            )
            return None

        resolved = Path(path).expanduser()

        if not resolved.exists():
            logger.debug(
                f"[scene {scene_index}] {description.capitalize()} "
                f"does not exist: '{resolved}'. "
                "Skipping talking-head generation."
            )
            return None

        if not resolved.is_file():
            logger.debug(
                f"[scene {scene_index}] {description.capitalize()} "
                f"is not a regular file: '{resolved}'. "
                "Skipping talking-head generation."
            )
            return None

        try:
            if resolved.stat().st_size <= 0:
                logger.debug(
                    f"[scene {scene_index}] {description.capitalize()} "
                    f"is empty: '{resolved}'. "
                    "Skipping talking-head generation."
                )
                return None
        except OSError:
            logger.debug(
                f"[scene {scene_index}] Unable to inspect {description}: "
                f"'{resolved}'. Skipping."
            )
            return None

        return resolved

    def _safe_prepare_directory(self, directory: Path) -> None:
        """
        Remove stale generated files before a new scene run.

        This prevents an old SadTalker output or old PNG sequence from being
        mistaken for the current scene's output.
        """
        try:
            if directory.exists():
                shutil.rmtree(directory)

            directory.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            raise TalkingHeadGenerationError(
                f"Unable to prepare temporary output directory "
                f"'{directory}': {e}"
            )

    def _cleanup_directory(self, directory: Path) -> None:
        try:
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=True)
        except Exception as e:
            logger.debug(
                f"Non-fatal SadTalker cleanup error for "
                f"'{directory}': {e}"
            )

    def _count_frames(self, frames_dir: Path) -> int:
        try:
            return len(
                list(frames_dir.glob("frame_*.png"))
            )
        except Exception:
            return 0

    @staticmethod
    def _tail_output(text: str, max_chars: int = 1200) -> str:
        text = (text or "").strip()

        if not text:
            return "No diagnostic output was provided."

        if len(text) <= max_chars:
            return text

        return "... " + text[-max_chars:]

    # ------------------------------------------------------------------
    # Optional maintenance helpers
    # ------------------------------------------------------------------

    def cleanup(self, output_dir: Path) -> None:
        """
        Explicitly remove temporary SadTalker artifacts from an output
        directory.

        Useful for callers that want to clean intermediate files before
        generating another scene.
        """
        output_dir = Path(output_dir)

        self._cleanup_directory(
            output_dir / "sadtalker_raw"
        )

    def refresh_availability(self) -> bool:
        """
        Re-check the SadTalker installation after startup.

        Useful when checkpoints are downloaded after the application has
        already initialized the pipeline.
        """
        self.available = False
        self._verify_installation()
        return self.available
