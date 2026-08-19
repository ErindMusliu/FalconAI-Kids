"""
Audio analysis utilities for FalconAI-Kids.

CPU-only audio analysis with no ML/GPU dependencies.

Features:
- Audio metadata extraction
- Duration and sample information
- RMS / average volume estimation
- Peak amplitude
- Silence detection
- Basic audio quality checks
- Safe handling of invalid/missing files
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from utils.logger import get_logger
from utils.exceptions import StorageError

logger = get_logger(__name__)


SUPPORTED_EXTENSIONS = {
    ".wav",
}


@dataclass
class AudioAnalysis:
    """Result of an audio analysis."""

    path: str
    exists: bool
    valid: bool

    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    sample_width: int = 0
    frame_count: int = 0

    rms: float = 0.0
    peak: float = 0.0
    average_db: float = -float("inf")
    peak_db: float = -float("inf")

    is_silent: bool = False
    is_empty: bool = False
    quality_ok: bool = False

    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert analysis result to a dictionary."""
        return asdict(self)


class AudioAnalyzer:
    """
    Lightweight CPU-only audio analyzer.

    The analyzer intentionally uses Python's standard library for WAV
    processing, avoiding heavyweight audio/ML dependencies.
    """

    def __init__(
        self,
        silence_threshold: float = 0.01,
        min_duration: float = 0.05,
        max_duration: Optional[float] = None,
    ) -> None:
        if silence_threshold < 0:
            raise ValueError("silence_threshold must be >= 0")

        if min_duration < 0:
            raise ValueError("min_duration must be >= 0")

        if max_duration is not None and max_duration <= 0:
            raise ValueError("max_duration must be > 0")

        self.silence_threshold = silence_threshold
        self.min_duration = min_duration
        self.max_duration = max_duration

    def analyze(self, audio_path: str | Path) -> AudioAnalysis:
        """
        Analyze an audio file.

        Currently WAV files are supported because they can be processed
        without external dependencies.
        """
        path = Path(audio_path)

        if not path.exists():
            return AudioAnalysis(
                path=str(path),
                exists=False,
                valid=False,
                error="Audio file does not exist.",
            )

        if not path.is_file():
            return AudioAnalysis(
                path=str(path),
                exists=True,
                valid=False,
                error="Audio path is not a file.",
            )

        if path.stat().st_size == 0:
            return AudioAnalysis(
                path=str(path),
                exists=True,
                valid=False,
                is_empty=True,
                error="Audio file is empty.",
            )

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            return AudioAnalysis(
                path=str(path),
                exists=True,
                valid=False,
                error=(
                    f"Unsupported audio format: {extension}. "
                    f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                ),
            )

        try:
            return self._analyze_wav(path)

        except (wave.Error, EOFError, OSError, ValueError) as exc:
            logger.warning("Failed to analyze audio %s: %s", path, exc)

            return AudioAnalysis(
                path=str(path),
                exists=True,
                valid=False,
                error=str(exc),
            )

        except Exception as exc:
            logger.exception("Unexpected audio analysis error: %s", path)

            return AudioAnalysis(
                path=str(path),
                exists=True,
                valid=False,
                error=str(exc),
            )

    def _analyze_wav(self, path: Path) -> AudioAnalysis:
        """Analyze a WAV file."""
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()

            if channels <= 0:
                raise ValueError("Invalid channel count.")

            if sample_width <= 0:
                raise ValueError("Invalid sample width.")

            if sample_rate <= 0:
                raise ValueError("Invalid sample rate.")

            if frame_count <= 0:
                return AudioAnalysis(
                    path=str(path),
                    exists=True,
                    valid=True,
                    sample_rate=sample_rate,
                    channels=channels,
                    sample_width=sample_width,
                    frame_count=frame_count,
                    is_empty=True,
                    is_silent=True,
                    quality_ok=False,
                    error="Audio contains no frames.",
                )

            duration = frame_count / sample_rate

            if self.max_duration is not None and duration > self.max_duration:
                quality_ok = False
            else:
                quality_ok = duration >= self.min_duration

            raw_data = audio.readframes(frame_count)

        rms, peak = self._calculate_amplitude(
            raw_data,
            sample_width,
        )

        average_db = self._amplitude_to_db(rms)
        peak_db = self._amplitude_to_db(peak)

        is_silent = peak <= self.silence_threshold

        if is_silent:
            quality_ok = False

        return AudioAnalysis(
            path=str(path),
            exists=True,
            valid=True,
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            frame_count=frame_count,
            rms=rms,
            peak=peak,
            average_db=average_db,
            peak_db=peak_db,
            is_silent=is_silent,
            is_empty=False,
            quality_ok=quality_ok,
        )

    @staticmethod
    def _calculate_amplitude(
        raw_data: bytes,
        sample_width: int,
    ) -> tuple[float, float]:
        """
        Calculate RMS and peak amplitude.

        Values are normalized between 0.0 and 1.0.
        """

        if not raw_data:
            return 0.0, 0.0

        if sample_width == 1:
            # 8-bit PCM WAV is unsigned.
            samples = (
                (sample - 128) / 128.0
                for sample in raw_data
            )

        elif sample_width == 2:
            # 16-bit PCM.
            sample_count = len(raw_data) // 2

            values = (
                int.from_bytes(
                    raw_data[index:index + 2],
                    byteorder="little",
                    signed=True,
                )
                / 32768.0
                for index in range(0, sample_count * 2, 2)
            )

            samples = values

        elif sample_width == 3:
            # 24-bit PCM.
            sample_count = len(raw_data) // 3

            def read_24bit(index: int) -> float:
                value = int.from_bytes(
                    raw_data[index:index + 3],
                    byteorder="little",
                    signed=False,
                )

                if value & 0x800000:
                    value -= 0x1000000

                return value / 8388608.0

            samples = (
                read_24bit(index)
                for index in range(0, sample_count * 3, 3)
            )

        elif sample_width == 4:
            # 32-bit PCM.
            sample_count = len(raw_data) // 4

            samples = (
                int.from_bytes(
                    raw_data[index:index + 4],
                    byteorder="little",
                    signed=True,
                )
                / 2147483648.0
                for index in range(0, sample_count * 4, 4)
            )

        else:
            raise ValueError(
                f"Unsupported PCM sample width: {sample_width} bytes."
            )

        squared_sum = 0.0
        peak = 0.0
        count = 0

        for sample in samples:
            amplitude = abs(sample)

            if amplitude > peak:
                peak = amplitude

            squared_sum += sample * sample
            count += 1

        if count == 0:
            return 0.0, 0.0

        rms = math.sqrt(squared_sum / count)

        return min(rms, 1.0), min(peak, 1.0)

    @staticmethod
    def _amplitude_to_db(amplitude: float) -> float:
        """Convert normalized amplitude to decibels."""
        if amplitude <= 0:
            return -float("inf")

        return 20.0 * math.log10(amplitude)

    def is_valid(self, audio_path: str | Path) -> bool:
        """Return True when the audio passes basic quality checks."""
        return self.analyze(audio_path).quality_ok

    def is_silent(self, audio_path: str | Path) -> bool:
        """Return True if the audio is effectively silent."""
        return self.analyze(audio_path).is_silent

    def get_duration(self, audio_path: str | Path) -> float:
        """Return audio duration in seconds."""
        return self.analyze(audio_path).duration


_default_analyzer = AudioAnalyzer()


def analyze_audio(audio_path: str | Path) -> dict[str, Any]:
    """
    Analyze an audio file and return a dictionary.

    This is the simplest API for other FalconAI-Kids modules.
    """
    return _default_analyzer.analyze(audio_path).to_dict()


def get_audio_duration(audio_path: str | Path) -> float:
    """Get audio duration in seconds."""
    return _default_analyzer.get_duration(audio_path)


def is_audio_valid(audio_path: str | Path) -> bool:
    """Check whether an audio file passes basic validation."""
    return _default_analyzer.is_valid(audio_path)


def is_audio_silent(audio_path: str | Path) -> bool:
    """Check whether an audio file is silent."""
    return _default_analyzer.is_silent(audio_path)
