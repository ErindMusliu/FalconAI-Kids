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
    Wraps SadTalker (https://github.com/OpenTalker/SadTalker) to turn a single
    reference photo of the child plus one scene's narration audio into a short
    talking-head clip — natural head motion synchronized with real lip
    movement — generated entirely locally on the GPU already available to
    this pipeline. No external API calls are made anywhere in this class.

    SadTalker isn't published as a stable, importable pip package; its public
    interface is really its own `inference.py` CLI script plus a checkpoints
    folder that its own download script populates. Consistent with how this
    codebase already shells out to `ffmpeg` in video_assembler.py rather than
    binding to an unstable internal library API, this wrapper drives
    SadTalker's `inference.py` as a subprocess and then re-extracts the
    resulting video into PNG frames with ffmpeg, instead of importing
    SadTalker's internal modules directly (these differ across forks/versions
    and change without notice).
    """

    def __init__(self, fps: Optional[int] = None):
        self.repo_dir = Path(SADTALKER_CONFIG["repo_dir"])
        self.checkpoint_dir = Path(SADTALKER_CONFIG.get("checkpoint_dir", self.repo_dir / "checkpoints"))
        self.size = SADTALKER_CONFIG.get("size", 256)
        self.preprocess = SADTALKER_CONFIG.get("preprocess", "crop")
        self.still_mode = SADTALKER_CONFIG.get("still_mode", False)
        self.use_enhancer = SADTALKER_CONFIG.get("use_enhancer", False)
        self.expression_scale = SADTALKER_CONFIG.get("expression_scale", 1.0)
        self.pose_style = SADTALKER_CONFIG.get("pose_style", 0)
        self.inference_timeout_sec = SADTALKER_CONFIG.get("inference_timeout_sec", 600)
        self.fps = fps or SADTALKER_CONFIG.get("fps", 24)
        self.python_executable = SADTALKER_CONFIG.get("python_executable", sys.executable)

        self.available = False
        self._verify_installation()

    def _verify_installation(self) -> None:
        """Checks that SadTalker is actually present before we ever try to use
        it, and degrades gracefully (available=False) instead of crashing the
        whole pipeline — the character_animator step is expected to fall back
        to the plain animated background frames for the child whenever this
        returns False, rather than failing the entire video generation."""
        inference_script = self.repo_dir / "inference.py"

        if not self.repo_dir.exists() or not inference_script.exists():
            logger.warning(
                f"SadTalker installation not found at '{self.repo_dir}'. "
                f"The talking-head (child lip-sync) path will be skipped for this run; "
                f"scenes will fall back to the animated-but-silent frames from frame_generator.\n"
                f"To enable it:\n"
                f"  git clone https://github.com/OpenTalker/SadTalker '{self.repo_dir}'\n"
                f"  bash '{self.repo_dir}/scripts/download_models.sh'"
            )
            self.available = False
            return

        if not self.checkpoint_dir.exists() or not any(self.checkpoint_dir.iterdir()):
            logger.warning(
                f"SadTalker checkpoints directory is missing or empty: '{self.checkpoint_dir}'. "
                f"Run SadTalker's own checkpoint download script before this feature can work."
            )
            self.available = False
            return

        logger.success(f"SadTalker installation detected at '{self.repo_dir}' (checkpoints present).")
        self.available = True

    def is_available(self) -> bool:
        return self.available

    def generate(
        self,
        face_image_path: Optional[Path],
        audio_path: Optional[Path],
        output_dir: Path,
        scene_index: Optional[int] = None,
    ) -> Optional[Path]:
        """
        Runs SadTalker for exactly one scene and returns a directory of
        extracted PNG frames (frame_0001.png, frame_0002.png, ... at
        self.fps) — the same naming convention frame_generator.py already
        uses, so downstream compositing code doesn't need two different frame
        readers.

        Returns None (does NOT raise) when the talking-head path simply isn't
        applicable to this scene — SadTalker not installed, no reference
        photo, or no narration audio (e.g. a silent establishing scene). The
        caller (character_animator) is expected to treat None as "keep the
        original animated background frames for this scene, unchanged."

        Raises TalkingHeadGenerationError only for genuine failures once we've
        already committed to running SadTalker (subprocess crash, timeout,
        missing output, failed frame extraction).
        """
        if not self.available:
            logger.debug(f"[scene {scene_index}] SadTalker unavailable; skipping talking-head generation.")
            return None

        if not face_image_path or not Path(face_image_path).exists():
            logger.debug(f"[scene {scene_index}] No reference face image supplied; skipping talking-head generation.")
            return None

        if not audio_path or not Path(audio_path).exists():
            logger.debug(f"[scene {scene_index}] No narration audio supplied; skipping talking-head generation.")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sadtalker_result_dir = output_dir / "sadtalker_raw"
        sadtalker_result_dir.mkdir(parents=True, exist_ok=True)

        try:
            raw_video_path = self._run_sadtalker(
                face_image_path=Path(face_image_path),
                audio_path=Path(audio_path),
                result_dir=sadtalker_result_dir,
                scene_index=scene_index,
            )

            frames_dir = output_dir / "frames"
            self._extract_frames(raw_video_path, frames_dir, scene_index=scene_index)

        finally:
            try:
                shutil.rmtree(sadtalker_result_dir, ignore_errors=True)
            except Exception:
                pass

        return frames_dir

    def _run_sadtalker(
        self,
        face_image_path: Path,
        audio_path: Path,
        result_dir: Path,
        scene_index: Optional[int],
    ) -> Path:
        cmd = [
            self.python_executable,
            str(self.repo_dir / "inference.py"),
            "--driven_audio", str(audio_path),
            "--source_image", str(face_image_path),
            "--result_dir", str(result_dir),
            "--checkpoint_dir", str(self.checkpoint_dir),
            "--size", str(self.size),
            "--preprocess", self.preprocess,
            "--expression_scale", str(self.expression_scale),
            "--pose_style", str(self.pose_style),
        ]

        if self.still_mode:
            # "--still" reduces head motion to keep the identity/likeness more
            # stable — useful since our source photo is a real child's face
            # and we'd rather under-animate than warp it unrecognizably.
            cmd.append("--still")

        if self.use_enhancer:
            cmd.extend(["--enhancer", "gfpgan"])

        if DEVICE != "cuda":
            cmd.append("--cpu")

        scene_label = scene_index if scene_index is not None else "?"
        logger.step(f"Running SadTalker inference for scene {scene_label}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.inference_timeout_sec,
                cwd=str(self.repo_dir),
            )
        except subprocess.TimeoutExpired:
            raise TalkingHeadGenerationError(
                f"SadTalker inference exceeded the configured timeout of {self.inference_timeout_sec}s.",
                scene_index=scene_index,
            )
        except Exception as e:
            raise TalkingHeadGenerationError(
                f"Failed to launch SadTalker subprocess: {e}",
                scene_index=scene_index,
            )

        if result.returncode != 0:
            error_tail = (result.stderr or result.stdout or "Unknown SadTalker subprocess failure.")[-800:]
            raise TalkingHeadGenerationError(
                f"SadTalker inference.py exited with a non-zero status: {error_tail}",
                scene_index=scene_index,
            )

        output_video = self._locate_output_video(result_dir)
        if output_video is None:
            raise TalkingHeadGenerationError(
                f"SadTalker reported success but no output video could be located inside '{result_dir}'.",
                scene_index=scene_index,
            )

        logger.success(f"SadTalker talking-head clip generated: {output_video.name}")
        return output_video

    def _locate_output_video(self, result_dir: Path) -> Optional[Path]:
        """SadTalker writes its result mp4 somewhere inside a self-named
        timestamped subfolder of result_dir rather than at a fixed path, so we
        search for it and take the most recently modified match."""
        candidates = sorted(
            result_dir.rglob("*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _extract_frames(self, video_path: Path, frames_dir: Path, scene_index: Optional[int]) -> None:
        frames_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vf", f"fps={self.fps}",
            "-start_number", "1",
            str(frames_dir / "frame_%04d.png"),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            raise TalkingHeadGenerationError(
                "Timed out while extracting frames from the SadTalker output video via ffmpeg.",
                scene_index=scene_index,
            )

        if result.returncode != 0:
            error_tail = (result.stderr or "Unknown ffmpeg extraction failure.")[-500:]
            raise TalkingHeadGenerationError(
                f"Failed to extract frames from the SadTalker output video: {error_tail}",
                scene_index=scene_index,
            )

        if not any(frames_dir.glob("frame_*.png")):
            raise TalkingHeadGenerationError(
                f"Frame extraction from the SadTalker output produced no PNG frames in '{frames_dir}'.",
                scene_index=scene_index,
            )
