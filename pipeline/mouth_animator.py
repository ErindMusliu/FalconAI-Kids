import wave
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config.settings import MOUTH_ANIMATION_CONFIG
from utils.logger import get_logger
from utils.exceptions import MouthAnimationError

logger = get_logger(__name__)


class MouthAnimator:
    """
    Lightweight CPU-only mouth animation system.

    This module does not require a neural network or GPU.

    Pipeline:
        1. Load scene narration WAV.
        2. Convert audio to mono float32.
        3. Calculate a time-based RMS envelope.
        4. Normalize the envelope adaptively.
        5. Convert loudness into mouth states:
              closed -> half -> open
        6. Smooth state transitions.
        7. Draw a stylized mouth overlay onto each frame.

    This is intentionally procedural. It is NOT phoneme-level lip-sync.

    The mouth location is configurable through:

        MOUTH_ANIMATION_CONFIG["default_mouth_region"]

    Format:

        (relative_x, relative_y, relative_width, relative_height)

    Example:

        (0.38, 0.52, 0.24, 0.20)
    """

    STATES = ("closed", "half", "open")

    def __init__(self):
        self.default_region = MOUTH_ANIMATION_CONFIG.get(
            "default_mouth_region",
            (0.38, 0.52, 0.24, 0.20),
        )

        self.silence_rms_threshold = float(
            MOUTH_ANIMATION_CONFIG.get(
                "silence_rms_threshold",
                0.02,
            )
        )

        self.half_open_rms_threshold = float(
            MOUTH_ANIMATION_CONFIG.get(
                "half_open_rms_threshold",
                0.09,
            )
        )

        self.smoothing_window_frames = max(
            1,
            int(
                MOUTH_ANIMATION_CONFIG.get(
                    "smoothing_window_frames",
                    2,
                )
            ),
        )

        self.fps = max(
            1,
            int(
                MOUTH_ANIMATION_CONFIG.get(
                    "fps",
                    24,
                )
            ),
        )

        self.mouth_color_closed = self._normalize_color(
            MOUTH_ANIMATION_CONFIG.get(
                "mouth_color_closed",
                (90, 40, 40, 200),
            )
        )

        self.mouth_color_half = self._normalize_color(
            MOUTH_ANIMATION_CONFIG.get(
                "mouth_color_half",
                (70, 25, 25, 215),
            )
        )

        self.mouth_color_open = self._normalize_color(
            MOUTH_ANIMATION_CONFIG.get(
                "mouth_color_open",
                (45, 10, 10, 230),
            )
        )

        self.feather_px = max(
            0,
            int(
                MOUTH_ANIMATION_CONFIG.get(
                    "feather_px",
                    1,
                )
            ),
        )

        self._detector = self._try_load_optional_detector()

        logger.success(
            "MouthAnimator initialized | "
            f"mode={'detector-assisted' if self._detector else 'procedural CPU'} | "
            f"fps={self.fps}"
        )

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def _try_load_optional_detector(self):
        """
        Optional detector hook.

        No detector is required for the default CPU pipeline.
        """

        model_name = MOUTH_ANIMATION_CONFIG.get(
            "cartoon_face_detector"
        )

        if not model_name:
            return None

        logger.debug(
            f"Cartoon face detector '{model_name}' configured, "
            "but no external detector dependency is enabled. "
            "Using procedural fallback."
        )

        return None

    def is_procedural_only(self) -> bool:
        return self._detector is None

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------

    def generate(
        self,
        frames_dir: Path,
        audio_path: Optional[Path],
        output_dir: Path,
        scene_index: Optional[int] = None,
    ) -> Optional[Path]:

        frames_dir = Path(frames_dir)

        if not audio_path:
            logger.debug(
                f"[scene {scene_index}] No narration audio supplied. "
                "Skipping mouth animation."
            )
            return None

        audio_path = Path(audio_path)

        if not audio_path.exists():
            logger.debug(
                f"[scene {scene_index}] Narration audio does not exist: "
                f"{audio_path}"
            )
            return None

        frame_paths = self._collect_frames(frames_dir)

        if not frame_paths:
            raise MouthAnimationError(
                f"No frames found in '{frames_dir}' to animate.",
                scene_index=scene_index,
            )

        try:
            samples, sample_rate = self._load_wav_mono(audio_path)

            if sample_rate <= 0:
                raise ValueError(
                    f"Invalid audio sample rate: {sample_rate}"
                )

            if len(samples) == 0:
                logger.warning(
                    f"[scene {scene_index}] Audio contains no samples. "
                    "Generating unchanged frames."
                )
                self._copy_frames(
                    frame_paths,
                    Path(output_dir),
                )
                return Path(output_dir)

        except Exception as e:
            raise MouthAnimationError(
                f"Failed to read narration audio for mouth animation: {e}",
                scene_index=scene_index,
            )

        mouth_states = self._compute_mouth_states(
            samples=samples,
            sample_rate=sample_rate,
            num_frames=len(frame_paths),
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            region_box = None

            for idx, (frame_path, state) in enumerate(
                zip(frame_paths, mouth_states),
                start=1,
            ):
                with Image.open(frame_path) as source:
                    frame = source.convert("RGBA")

                if region_box is None:
                    region_box = self._locate_mouth_region(frame)

                animated = self._draw_mouth(
                    frame=frame,
                    region_box=region_box,
                    state=state,
                )

                out_path = output_dir / frame_path.name

                animated.convert("RGB").save(
                    out_path,
                    format="PNG",
                    optimize=False,
                )

        except Exception as e:
            raise MouthAnimationError(
                f"Failed while rendering mouth animation at frame "
                f"{idx}: {e}",
                scene_index=scene_index,
            )

        logger.success(
            f"[scene {scene_index}] Procedural mouth animation "
            f"rendered successfully: {len(frame_paths)} frame(s)."
        )

        return output_dir

    # ------------------------------------------------------------------
    # FRAME HANDLING
    # ------------------------------------------------------------------

    def _collect_frames(
        self,
        frames_dir: Path,
    ) -> List[Path]:

        frames = sorted(
            frames_dir.glob("frame_*.png")
        )

        if not frames:
            frames = sorted(
                frames_dir.glob("*.png")
            )

        return frames

    def _copy_frames(
        self,
        frames: List[Path],
        output_dir: Path,
    ) -> None:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for frame in frames:
            with Image.open(frame) as img:
                img.convert("RGB").save(
                    output_dir / frame.name,
                    format="PNG",
                    optimize=False,
                )

    # ------------------------------------------------------------------
    # MOUTH LOCATION
    # ------------------------------------------------------------------

    def _locate_mouth_region(
        self,
        frame: Image.Image,
    ) -> Tuple[int, int, int, int]:

        width, height = frame.size

        if self._detector is not None:
            try:
                box = self._detector.locate_mouth(frame)

                if box:
                    x, y, w, h = box

                    return self._clamp_region(
                        x,
                        y,
                        w,
                        h,
                        width,
                        height,
                    )

            except Exception as e:
                logger.debug(
                    f"Optional mouth detector failed: {e}. "
                    "Using configured heuristic region."
                )

        rel_x, rel_y, rel_w, rel_h = self.default_region

        rel_x = float(np.clip(rel_x, 0.0, 1.0))
        rel_y = float(np.clip(rel_y, 0.0, 1.0))
        rel_w = float(np.clip(rel_w, 0.01, 1.0))
        rel_h = float(np.clip(rel_h, 0.01, 1.0))

        x = int(rel_x * width)
        y = int(rel_y * height)
        w = int(rel_w * width)
        h = int(rel_h * height)

        return self._clamp_region(
            x,
            y,
            w,
            h,
            width,
            height,
        )

    def _clamp_region(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        image_w: int,
        image_h: int,
    ) -> Tuple[int, int, int, int]:

        x = max(0, min(int(x), image_w - 1))
        y = max(0, min(int(y), image_h - 1))

        w = max(
            1,
            min(
                int(w),
                image_w - x,
            ),
        )

        h = max(
            1,
            min(
                int(h),
                image_h - y,
            ),
        )

        return x, y, w, h

    # ------------------------------------------------------------------
    # MOUTH DRAWING
    # ------------------------------------------------------------------

    def _draw_mouth(
        self,
        frame: Image.Image,
        region_box: Tuple[int, int, int, int],
        state: str,
    ) -> Image.Image:

        if state not in self.STATES:
            state = "closed"

        if state == "closed":
            return frame.copy()

        x, y, w, h = region_box

        result = frame.copy()

        overlay = Image.new(
            "RGBA",
            result.size,
            (0, 0, 0, 0),
        )

        draw = ImageDraw.Draw(overlay)

        cx = x + w // 2
        cy = y + h // 2

        if state == "half":
            mouth_w = max(
                3,
                int(w * 0.62),
            )
            mouth_h = max(
                2,
                int(h * 0.30),
            )
            color = self.mouth_color_half

        else:
            mouth_w = max(
                4,
                int(w * 0.76),
            )
            mouth_h = max(
                4,
                int(h * 0.62),
            )
            color = self.mouth_color_open

        ellipse_box = (
            cx - mouth_w // 2,
            cy - mouth_h // 2,
            cx + mouth_w // 2,
            cy + mouth_h // 2,
        )

        draw.ellipse(
            ellipse_box,
            fill=color,
        )

        # Small inner highlight/tongue for open state.
        if state == "open":

            inner_w = max(
                2,
                int(mouth_w * 0.55),
            )

            inner_h = max(
                1,
                int(mouth_h * 0.30),
            )

            inner_box = (
                cx - inner_w // 2,
                cy + int(mouth_h * 0.08),
                cx + inner_w // 2,
                cy + int(mouth_h * 0.08) + inner_h,
            )

            draw.ellipse(
                inner_box,
                fill=(120, 45, 45, 160),
            )

        if self.feather_px > 0:
            alpha = overlay.getchannel("A")
            alpha = alpha.filter(
                ImageFilter.GaussianBlur(
                    self.feather_px
                )
            )
            overlay.putalpha(alpha)

        return Image.alpha_composite(
            result,
            overlay,
        )

    # ------------------------------------------------------------------
    # AUDIO ANALYSIS
    # ------------------------------------------------------------------

    def _compute_mouth_states(
        self,
        samples: np.ndarray,
        sample_rate: int,
        num_frames: int,
    ) -> List[str]:

        if num_frames <= 0:
            return []

        if len(samples) == 0:
            return ["closed"] * num_frames

        duration_sec = len(samples) / float(sample_rate)

        frame_duration = duration_sec / float(
            num_frames
        )

        raw_rms = []

        for frame_index in range(num_frames):

            start_time = (
                frame_index * frame_duration
            )

            end_time = (
                (frame_index + 1)
                * frame_duration
            )

            start_sample = int(
                start_time * sample_rate
            )

            end_sample = int(
                end_time * sample_rate
            )

            start_sample = max(
                0,
                min(
                    start_sample,
                    len(samples),
                ),
            )

            end_sample = max(
                start_sample + 1,
                min(
                    end_sample,
                    len(samples),
                ),
            )

            chunk = samples[
                start_sample:end_sample
            ]

            if len(chunk) == 0:
                rms = 0.0
            else:
                rms = float(
                    np.sqrt(
                        np.mean(
                            np.square(chunk)
                        )
                    )
                )

            raw_rms.append(rms)

        smoothed = self._smooth(
            raw_rms,
            window=self.smoothing_window_frames,
        )

        normalized = self._adaptive_normalize(
            smoothed
        )

        states = self._rms_to_states(
            normalized
        )

        states = self._remove_single_frame_jitter(
            states
        )

        return states

    def _adaptive_normalize(
        self,
        values: List[float],
    ) -> List[float]:

        if not values:
            return []

        arr = np.asarray(
            values,
            dtype=np.float32,
        )

        positive = arr[arr > 1e-5]

        if len(positive) == 0:
            return [0.0] * len(arr)

        low = float(
            np.percentile(
                positive,
                10,
            )
        )

        high = float(
            np.percentile(
                positive,
                90,
            )
        )

        if high <= low + 1e-6:
            maximum = float(
                np.max(arr)
            )

            if maximum <= 1e-6:
                return [0.0] * len(arr)

            normalized = arr / maximum

        else:
            normalized = (
                arr - low
            ) / (
                high - low
            )

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

        return normalized.tolist()

    def _rms_to_states(
        self,
        rms_values: List[float],
    ) -> List[str]:

        states = []

        silence = self._normalized_threshold(
            self.silence_rms_threshold
        )

        half_open = self._normalized_threshold(
            self.half_open_rms_threshold
        )

        # Prevent broken configurations.
        silence = min(
            silence,
            0.25,
        )

        half_open = max(
            half_open,
            silence + 0.05,
        )

        for rms in rms_values:

            if rms <= silence:
                states.append("closed")

            elif rms <= half_open:
                states.append("half")

            else:
                states.append("open")

        return states

    def _normalized_threshold(
        self,
        threshold: float,
    ) -> float:

        # Original thresholds are intended for raw normalized audio.
        # Convert them into a conservative normalized-envelope threshold.
        if threshold <= 0:
            return 0.0

        if threshold >= 1:
            return 1.0

        return float(
            np.clip(
                threshold / 0.25,
                0.0,
                1.0,
            )
        )

    # ------------------------------------------------------------------
    # SMOOTHING
    # ------------------------------------------------------------------

    def _smooth(
        self,
        values: List[float],
        window: int,
    ) -> List[float]:

        if not values:
            return []

        if window <= 1:
            return values

        if len(values) <= 2:
            return values

        window = min(
            window,
            len(values),
        )

        arr = np.asarray(
            values,
            dtype=np.float32,
        )

        kernel = (
            np.ones(
                window,
                dtype=np.float32,
            )
            / window
        )

        left = window // 2
        right = window - 1 - left

        padded = np.pad(
            arr,
            (left, right),
            mode="edge",
        )

        smoothed = np.convolve(
            padded,
            kernel,
            mode="valid",
        )

        return smoothed.tolist()

    def _remove_single_frame_jitter(
        self,
        states: List[str],
    ) -> List[str]:

        if len(states) < 3:
            return states

        result = states.copy()

        for i in range(
            1,
            len(states) - 1,
        ):
            previous = states[i - 1]
            current = states[i]
            next_state = states[i + 1]

            if (
                previous == next_state
                and current != previous
            ):
                result[i] = previous

        return result

    # ------------------------------------------------------------------
    # WAV LOADING
    # ------------------------------------------------------------------

    def _load_wav_mono(
        self,
        path: Path,
    ) -> Tuple[np.ndarray, int]:

        with wave.open(
            str(path),
            "rb",
        ) as wf:

            sample_rate = wf.getframerate()
            frame_count = wf.getnframes()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()

            if sample_rate <= 0:
                raise ValueError(
                    "WAV contains an invalid sample rate."
                )

            if frame_count <= 0:
                return (
                    np.zeros(
                        0,
                        dtype=np.float32,
                    ),
                    sample_rate,
                )

            raw = wf.readframes(
                frame_count
            )

        if sample_width == 1:

            data = (
                np.frombuffer(
                    raw,
                    dtype=np.uint8,
                ).astype(
                    np.float32
                )
                - 128.0
            ) / 128.0

        elif sample_width == 2:

            data = (
                np.frombuffer(
                    raw,
                    dtype=np.int16,
                ).astype(
                    np.float32
                )
                / 32768.0
            )

        elif sample_width == 4:

            data = (
                np.frombuffer(
                    raw,
                    dtype=np.int32,
                ).astype(
                    np.float32
                )
                / 2147483648.0
            )

        else:
            raise ValueError(
                f"Unsupported WAV sample width: "
                f"{sample_width} bytes."
            )

        if channels > 1:

            expected = len(data) // channels

            data = data[
                : expected * channels
            ]

            data = data.reshape(
                -1,
                channels,
            ).mean(
                axis=1
            )

        data = np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        data = np.clip(
            data,
            -1.0,
            1.0,
        ).astype(
            np.float32
        )

        return data, sample_rate

    # ------------------------------------------------------------------
    # COLOR HELPERS
    # ------------------------------------------------------------------

    def _normalize_color(
        self,
        color,
    ) -> Tuple[int, int, int, int]:

        if not isinstance(
            color,
            (tuple, list),
        ):
            return (
                60,
                15,
                15,
                220,
            )

        values = list(color)

        if len(values) == 3:
            values.append(255)

        if len(values) != 4:
            return (
                60,
                15,
                15,
                220,
            )

        return tuple(
            int(
                np.clip(
                    value,
                    0,
                    255,
                )
            )
            for value in values
        )
