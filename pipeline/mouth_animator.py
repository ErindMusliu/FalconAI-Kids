import wave
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from config.settings import MOUTH_ANIMATION_CONFIG
from utils.logger import get_logger
from utils.exceptions import MouthAnimationError

logger = get_logger(__name__)


class MouthAnimator:
    """
    Procedural, audio-driven "mouth-flap" animation for illustrated
    creatures/animals produced by Stable Diffusion.

    IMPORTANT HONEST LIMITATION: unlike SadTalker (used for the child, which
    is trained on real human faces via 3D face reconstruction), there is no
    reliable, general-purpose landmark/keypoint detector for arbitrary
    cartoon creatures. This class does NOT perform true phonetic lip-sync.
    It instead:
      1. Analyzes the narration audio's loudness (RMS envelope) over time.
      2. Buckets each output frame into a mouth state (closed / half / open)
         based on that envelope.
      3. Draws a simple stylized mouth shape at a *heuristically located*
         region of the frame, and composites it in.

    The mouth *position* is the weakest link: by default it uses a fixed,
    configurable relative bounding box (MOUTH_ANIMATION_CONFIG["default_mouth_region"]),
    because Stable Diffusion doesn't guarantee where a creature's face lands
    in frame. `_locate_mouth_region()` is written as a single, isolated method
    specifically so a real detector (e.g. an anime/cartoon face-keypoint
    model) can be dropped in later without touching the rest of the class —
    see the docstring on that method.
    """

    def __init__(self):
        self.default_region = MOUTH_ANIMATION_CONFIG.get("default_mouth_region", (0.38, 0.52, 0.24, 0.20))
        self.silence_rms_threshold = MOUTH_ANIMATION_CONFIG.get("silence_rms_threshold", 0.02)
        self.half_open_rms_threshold = MOUTH_ANIMATION_CONFIG.get("half_open_rms_threshold", 0.09)
        self.smoothing_window_frames = MOUTH_ANIMATION_CONFIG.get("smoothing_window_frames", 2)
        self.mouth_color_closed = MOUTH_ANIMATION_CONFIG.get("mouth_color_closed", (90, 40, 40, 200))
        self.mouth_color_open = MOUTH_ANIMATION_CONFIG.get("mouth_color_open", (60, 15, 15, 220))
        self.fps = MOUTH_ANIMATION_CONFIG.get("fps", 24)

        self._detector = self._try_load_optional_detector()

        logger.success(
            f"MouthAnimator initialized (procedural mode) | "
            f"detector: {'available' if self._detector else 'heuristic fallback only'}"
        )

    def _try_load_optional_detector(self):
        """Attempts to load an optional cartoon/anime face-keypoint detector
        if one is installed and configured. This is intentionally soft —
        the whole feature works without it, just with a cruder fixed mouth
        region. Wire in a real model here later (e.g. an anime-face-detector
        package) without changing any other method in this class."""
        model_name = MOUTH_ANIMATION_CONFIG.get("cartoon_face_detector")
        if not model_name:
            return None

        try:
            # Placeholder import hook: intentionally not hardcoding a specific
            # third-party package here since none is a stable, widely-adopted
            # standard yet for arbitrary illustrated creatures. When one is
            # chosen, import and construct it here and return the instance.
            logger.debug(f"cartoon_face_detector='{model_name}' configured but no loader is wired up yet; using heuristic fallback.")
            return None
        except Exception as e:
            logger.debug(f"Optional cartoon face detector failed to load ({e}); using heuristic fallback.")
            return None

    def is_procedural_only(self) -> bool:
        return self._detector is None

    def generate(
        self,
        frames_dir: Path,
        audio_path: Optional[Path],
        output_dir: Path,
        scene_index: Optional[int] = None,
    ) -> Optional[Path]:
        """
        Reads the existing scene frames (already rendered by frame_generator /
        AnimateDiff) plus that scene's narration audio, and writes a new frame
        sequence — same filenames, same folder convention — with a
        procedural mouth overlay composited on top.

        Returns None (does not raise) when there is no audio to sync against
        (e.g. a narrator_only scene) — the caller should keep the original
        frames unchanged in that case, since there is nothing to animate the
        mouth to.

        Raises MouthAnimationError for genuine failures (unreadable frames,
        corrupt audio, write failures).
        """
        frames_dir = Path(frames_dir)

        if not audio_path or not Path(audio_path).exists():
            logger.debug(f"[scene {scene_index}] No narration audio supplied; skipping mouth-flap animation.")
            return None

        frame_paths = self._collect_frames(frames_dir)
        if not frame_paths:
            raise MouthAnimationError(f"No frames found in '{frames_dir}' to animate.", scene_index=scene_index)

        try:
            samples, sample_rate = self._load_wav_mono(Path(audio_path))
        except Exception as e:
            raise MouthAnimationError(f"Failed to read narration audio for mouth-flap sync: {e}", scene_index=scene_index)

        mouth_states = self._compute_mouth_states(samples, sample_rate, num_frames=len(frame_paths))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        region_box = None  # resolved once per scene; see _locate_mouth_region docstring

        try:
            for idx, (frame_path, state) in enumerate(zip(frame_paths, mouth_states), start=1):
                with Image.open(frame_path) as img:
                    img = img.convert("RGBA")

                    if region_box is None:
                        region_box = self._locate_mouth_region(img)

                    animated = self._draw_mouth(img, region_box, state)
                    animated = animated.convert("RGB")

                    out_path = output_dir / frame_path.name
                    animated.save(out_path, format="PNG", optimize=False)

        except Exception as e:
            raise MouthAnimationError(f"Failed while compositing mouth overlay onto frame {idx}: {e}", scene_index=scene_index)

        logger.success(f"[scene {scene_index}] Procedural mouth-flap animation applied to {len(frame_paths)} frame(s).")
        return output_dir

    def _collect_frames(self, frames_dir: Path) -> List[Path]:
        frames = sorted(frames_dir.glob("frame_*.png"))
        if not frames:
            frames = sorted(frames_dir.glob("*.png"))
        return frames

    def _locate_mouth_region(self, frame: Image.Image) -> Tuple[int, int, int, int]:
        """
        Returns a (x, y, w, h) pixel bounding box for where to draw the mouth
        on this scene's frames.

        Current behavior: if an optional cartoon face detector is configured
        and loaded (see _try_load_optional_detector), it would be used here
        to find the creature's actual face/mouth per scene. Since no such
        detector is wired in by default, this falls back to a fixed relative
        region (MOUTH_ANIMATION_CONFIG["default_mouth_region"]) applied to
        every scene — a deliberate, documented approximation, not a
        real detection. Because Stable Diffusion framing varies between
        generations, this fixed region will sometimes miss the creature's
        actual face; that trade-off was accepted explicitly in favor of
        avoiding an unreliable dependency, per prior discussion.
        """
        width, height = frame.size

        if self._detector is not None:
            try:
                box = self._detector.locate_mouth(frame)
                if box:
                    return box
            except Exception as e:
                logger.debug(f"Cartoon face detector failed at runtime ({e}); falling back to heuristic region.")

        rel_x, rel_y, rel_w, rel_h = self.default_region
        x = int(rel_x * width)
        y = int(rel_y * height)
        w = int(rel_w * width)
        h = int(rel_h * height)
        return (x, y, w, h)

    def _draw_mouth(self, frame: Image.Image, region_box: Tuple[int, int, int, int], state: str) -> Image.Image:
        x, y, w, h = region_box
        result = frame.copy()

        if state == "closed":
            return result

        overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        if state == "half":
            mouth_h = max(2, int(h * 0.35))
            color = self.mouth_color_closed
        else:  # "open"
            mouth_h = max(4, int(h * 0.75))
            color = self.mouth_color_open

        mouth_w = int(w * 0.8)
        cx = x + w // 2
        cy = y + h // 2

        ellipse_box = [
            cx - mouth_w // 2,
            cy - mouth_h // 2,
            cx + mouth_w // 2,
            cy + mouth_h // 2,
        ]
        draw.ellipse(ellipse_box, fill=color)

        return Image.alpha_composite(result, overlay)

    def _compute_mouth_states(self, samples: np.ndarray, sample_rate: int, num_frames: int) -> List[str]:
        if num_frames <= 0:
            return []

        if len(samples) == 0:
            return ["closed"] * num_frames

        samples_per_frame = max(1, len(samples) // num_frames)

        raw_rms = []
        for i in range(num_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame if i < num_frames - 1 else len(samples)
            chunk = samples[start:end]
            rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
            raw_rms.append(rms)

        smoothed_rms = self._smooth(raw_rms, window=self.smoothing_window_frames)

        states = []
        for rms in smoothed_rms:
            if rms < self.silence_rms_threshold:
                states.append("closed")
            elif rms < self.half_open_rms_threshold:
                states.append("half")
            else:
                states.append("open")

        return states

    def _smooth(self, values: List[float], window: int) -> List[float]:
        if window <= 1 or len(values) <= 1:
            return values

        arr = np.array(values, dtype=np.float32)
        kernel = np.ones(window) / window
        padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed.tolist()

    def _load_wav_mono(self, path: Path) -> Tuple[np.ndarray, int]:
        with wave.open(str(path), 'r') as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            n_ch = wf.getnchannels()
            sampw = wf.getsampwidth()

        if sampw == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        elif sampw == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483647.0
        else:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0

        if n_ch == 2:
            data = data.reshape(-1, 2).mean(axis=1)

        return data, sr
