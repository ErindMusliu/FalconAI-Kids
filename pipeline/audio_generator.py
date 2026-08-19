import asyncio
import math
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional, List

import numpy as np

from config.settings import AUDIO_CONFIG
from utils.logger import get_logger
from utils.exceptions import AudioGenerationError

logger = get_logger(__name__)


# ============================================================
# MUSIC THEORY / PROCEDURAL MUSIC
# ============================================================

_SCALES = {
    "adventure": [261.63, 293.66, 329.63, 392.00, 440.00, 523.25],
    "magical": [261.63, 311.13, 392.00, 466.16, 523.25, 622.25],
    "happy": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00],
    "mysterious": [261.63, 277.18, 311.13, 369.99, 415.30, 466.16],
    "heroic": [261.63, 329.63, 392.00, 523.25, 659.25, 783.99],
    "exciting": [329.63, 392.00, 493.88, 587.33, 659.25, 783.99],
}


_TEMPOS = {
    "adventure": 140,
    "magical": 80,
    "happy": 120,
    "mysterious": 60,
    "heroic": 130,
    "exciting": 150,
}


_CHORD_SETS = {
    "adventure": [
        (261.63, 329.63, 392.00),
        (349.23, 440.00, 523.25),
    ],
    "magical": [
        (261.63, 311.13, 392.00),
        (220.00, 261.63, 329.63),
    ],
    "happy": [
        (261.63, 329.63, 392.00),
        (293.66, 369.99, 440.00),
    ],
    "mysterious": [
        (130.81, 155.56, 196.00),
        (116.54, 138.59, 174.61),
    ],
    "heroic": [
        (261.63, 329.63, 392.00),
        (349.23, 440.00, 523.25),
    ],
    "exciting": [
        (329.63, 415.30, 493.88),
        (293.66, 369.99, 440.00),
    ],
}


_BASS_NOTES = {
    "adventure": [65.41, 73.42, 82.41, 98.00],
    "magical": [65.41, 69.30, 82.41, 87.31],
    "happy": [65.41, 87.31, 98.00, 73.42],
    "mysterious": [32.70, 36.71, 41.20, 43.65],
    "heroic": [65.41, 87.31, 130.81, 98.00],
    "exciting": [82.41, 98.00, 110.00, 123.47],
}


# ============================================================
# TTS VOICES
# ============================================================

_VOICE_MAP = {
    "albanian": (
        "sq-AL-AnilaNeural",
        "sq-AL-IlirNeural",
    ),
    "shqip": (
        "sq-AL-AnilaNeural",
        "sq-AL-IlirNeural",
    ),
    "english": (
        "en-US-EmmaNeural",
        "en-US-GuyNeural",
    ),
}


_DEFAULT_VOICE = (
    "en-US-EmmaNeural",
    "en-US-GuyNeural",
)


class AudioGenerator:
    """
    CPU-only audio generation pipeline.

    Responsibilities:
        1. Generate narration using Edge-TTS.
        2. Convert TTS output to real PCM WAV using FFmpeg.
        3. Generate procedural background music using NumPy.
        4. Concatenate scene narration.
        5. Duck background music underneath narration.
        6. Produce final_audio.wav.

    No GPU is required by this module.
    """

    def __init__(self, language: str = "Albanian"):
        self.language = language

        self.sample_rate = int(
            AUDIO_CONFIG.get("sample_rate", 22050)
        )

        self.tts_retries = int(
            AUDIO_CONFIG.get("tts_retries", 2)
        )

        self.tts_retry_delay_sec = float(
            AUDIO_CONFIG.get("tts_retry_delay_sec", 1.5)
        )

        self.ffmpeg_binary = AUDIO_CONFIG.get(
            "ffmpeg_binary",
            "ffmpeg"
        )

        self._edge_tts_available = False
        self._ffmpeg_available = False

        self._load_dependencies()

    # ========================================================
    # DEPENDENCY CHECKS
    # ========================================================

    def _load_dependencies(self) -> None:
        try:
            import edge_tts

            self._edge_tts_available = True
            logger.success(
                "Edge-TTS engine successfully initialized."
            )

        except ImportError:
            logger.warning(
                "edge-tts is not installed. "
                "TTS will fall back to generated silence."
            )

        self._ffmpeg_available = self._check_ffmpeg()

        if self._ffmpeg_available:
            logger.success(
                f"FFmpeg audio conversion available: "
                f"{self.ffmpeg_binary}"
            )
        else:
            logger.warning(
                "FFmpeg was not found. TTS conversion to WAV "
                "will be unavailable."
            )

    def _check_ffmpeg(self) -> bool:
        try:
            result = subprocess.run(
                [
                    self.ffmpeg_binary,
                    "-version",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

            return result.returncode == 0

        except (
            FileNotFoundError,
            OSError,
        ):
            return False

    # ========================================================
    # MAIN GENERATION
    # ========================================================

    def generate(
        self,
        story: dict,
        output_dir: Path,
        language: str = "English",
        gender: Optional[str] = None,
    ) -> Path:

        self.language = language

        scenes = story.get("scenes", [])

        if not scenes:
            raise AudioGenerationError(
                "The story contains no scenes; "
                "audio generation aborted."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        gap_sec = float(
            AUDIO_CONFIG.get(
                "scene_gap_sec",
                0.4,
            )
        )

        outro_tail_sec = float(
            AUDIO_CONFIG.get(
                "outro_tail_sec",
                2.0,
            )
        )

        logger.info(
            f"Generating CPU audio assets for "
            f"{len(scenes)} scenes."
        )

        narration_paths, scene_durations = (
            self._generate_all_narration(
                scenes=scenes,
                output_dir=output_dir,
                gender=gender,
            )
        )

        total_narration_duration = sum(
            scene_durations
        )

        # ----------------------------------------------------
        # FULL NARRATION
        # ----------------------------------------------------

        narration_full_path = (
            output_dir / "narration_full.wav"
        )

        self._concatenate_audio(
            audio_paths=narration_paths,
            output_path=narration_full_path,
            gap_sec=gap_sec,
        )

        # ----------------------------------------------------
        # BACKGROUND MUSIC
        # ----------------------------------------------------

        music_path = (
            output_dir / "background_music.wav"
        )

        self._generate_full_soundtrack(
            scenes=scenes,
            scene_durations=scene_durations,
            gap_sec=gap_sec,
            outro_tail_sec=outro_tail_sec,
            output_path=music_path,
        )

        # ----------------------------------------------------
        # FINAL MIX
        # ----------------------------------------------------

        final_path = (
            output_dir / "final_audio.wav"
        )

        self._mix_audio(
            narration_path=narration_full_path,
            music_path=music_path,
            output_path=final_path,
        )

        story["total_duration_sec"] = (
            total_narration_duration
            + gap_sec * max(0, len(scenes) - 1)
            + outro_tail_sec
        )

        logger.success(
            "Final audio generated successfully: "
            f"{final_path} "
            f"({story['total_duration_sec']:.1f}s)"
        )

        return final_path

    # ========================================================
    # NARRATION
    # ========================================================

    def _generate_all_narration(
        self,
        scenes: List[dict],
        output_dir: Path,
        gender: Optional[str],
    ) -> tuple[List[Path], List[float]]:

        narration_paths: List[Path] = []
        scene_durations: List[float] = []

        for i, scene in enumerate(scenes):

            narration = str(
                scene.get(
                    "narration",
                    "",
                )
            ).strip()

            narration_path = (
                output_dir
                / f"narration_scene_{i + 1:02d}.wav"
            )

            if not narration:

                logger.debug(
                    f"Scene {i + 1} has no narration. "
                    "Generating silence."
                )

                duration = 5.0

                self._generate_silence(
                    narration_path,
                    duration,
                )

            else:

                logger.step(
                    f"TTS [{i + 1}/{len(scenes)}]: "
                    f"{narration[:70]}..."
                )

                duration = (
                    self._synthesize_with_retries(
                        text=narration,
                        output_path=narration_path,
                        gender=gender,
                    )
                )

            scene["duration_sec"] = duration

            # Important for optional animation stages.
            scene["narration_audio_path"] = str(
                narration_path.resolve()
            )

            scene_durations.append(duration)
            narration_paths.append(narration_path)

        return narration_paths, scene_durations

    def _synthesize_with_retries(
        self,
        text: str,
        output_path: Path,
        gender: Optional[str],
    ) -> float:

        if not self._edge_tts_available:
            fallback_duration = self._estimate_speech_duration(
                text
            )

            self._generate_silence(
                output_path,
                fallback_duration,
            )

            return fallback_duration

        if not self._ffmpeg_available:
            logger.warning(
                "Edge-TTS is available but FFmpeg is not. "
                "Using silence fallback."
            )

            fallback_duration = self._estimate_speech_duration(
                text
            )

            self._generate_silence(
                output_path,
                fallback_duration,
            )

            return fallback_duration

        last_error = None

        for attempt in range(
            1,
            self.tts_retries + 2,
        ):

            try:

                with tempfile.TemporaryDirectory(
                    prefix="falconai_tts_"
                ) as temp_dir:

                    temp_dir = Path(temp_dir)

                    # Edge-TTS produces encoded audio.
                    raw_audio = (
                        temp_dir
                        / "tts_audio.mp3"
                    )

                    asyncio.run(
                        self._generate_edge_tts(
                            text=text,
                            output_path=raw_audio,
                            gender=gender,
                        )
                    )

                    self._convert_to_wav(
                        input_path=raw_audio,
                        output_path=output_path,
                    )

                duration = self._get_wav_duration(
                    output_path
                )

                if duration <= 0:
                    raise RuntimeError(
                        "Generated TTS audio has zero duration."
                    )

                return duration

            except Exception as e:

                last_error = e

                logger.warning(
                    f"TTS attempt {attempt} failed: {e}"
                )

                if attempt <= self.tts_retries:
                    time.sleep(
                        self.tts_retry_delay_sec
                    )

        logger.warning(
            "TTS failed after "
            f"{self.tts_retries + 1} attempts. "
            "Generating silence fallback."
        )

        fallback_duration = (
            self._estimate_speech_duration(text)
        )

        self._generate_silence(
            output_path,
            fallback_duration,
        )

        return fallback_duration

    async def _generate_edge_tts(
        self,
        text: str,
        output_path: Path,
        gender: Optional[str],
    ) -> None:

        import edge_tts

        voice = self._resolve_voice(
            gender
        )

        rate_str = self._speed_to_rate_string(
            AUDIO_CONFIG.get(
                "tts_speed",
                1.0,
            )
        )

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate_str,
        )

        await communicate.save(
            str(output_path)
        )

    # ========================================================
    # FFMPEG
    # ========================================================

    def _convert_to_wav(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            self.ffmpeg_binary,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-sample_fmt",
            "s16",
            str(output_path),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "FFmpeg failed to convert TTS audio: "
                + result.stderr.strip()
            )

        if not output_path.exists():
            raise RuntimeError(
                "FFmpeg reported success but WAV "
                "output was not created."
            )

    # ========================================================
    # VOICE
    # ========================================================

    def _resolve_voice(
        self,
        gender: Optional[str],
    ) -> str:

        override = AUDIO_CONFIG.get(
            "tts_voice_override"
        )

        if override:
            return override

        female_voice, male_voice = (
            _VOICE_MAP.get(
                self.language.lower(),
                _DEFAULT_VOICE,
            )
        )

        if gender and gender.lower() in (
            "male",
            "boy",
            "m",
        ):
            return male_voice

        return female_voice

    def _speed_to_rate_string(
        self,
        speed: float,
    ) -> str:

        try:
            pct = round(
                (float(speed) - 1.0)
                * 100
            )
        except (
            TypeError,
            ValueError,
        ):
            pct = 0

        pct = max(
            -80,
            min(100, pct),
        )

        sign = (
            "+"
            if pct >= 0
            else ""
        )

        return f"{sign}{pct}%"

    def _estimate_speech_duration(
        self,
        text: str,
    ) -> float:

        words = max(
            1,
            len(text.split()),
        )

        # Approx. 145 words/minute.
        duration = (
            words / 145.0
        ) * 60.0

        return max(
            1.5,
            min(30.0, duration),
        )

    # ========================================================
    # WAV UTILITIES
    # ========================================================

    def _get_wav_duration(
        self,
        path: Path,
    ) -> float:

        with wave.open(
            str(path),
            "rb",
        ) as wf:

            frames = wf.getnframes()
            rate = wf.getframerate()

            if rate <= 0:
                return 0.0

            return frames / float(rate)

    def _generate_silence(
        self,
        path: Path,
        duration_sec: float,
    ) -> None:

        duration_sec = max(
            0.1,
            float(duration_sec),
        )

        n = int(
            self.sample_rate
            * duration_sec
        )

        signal = np.zeros(
            n,
            dtype=np.float32,
        )

        self._save_wav(
            signal,
            path,
            self.sample_rate,
        )

    # ========================================================
    # PROCEDURAL MUSIC
    # ========================================================

    def _generate_full_soundtrack(
        self,
        scenes: List[dict],
        scene_durations: List[float],
        gap_sec: float,
        outro_tail_sec: float,
        output_path: Path,
    ) -> None:

        sr = self.sample_rate
        segments = []

        for i, (
            scene,
            duration,
        ) in enumerate(
            zip(
                scenes,
                scene_durations,
            )
        ):

            is_last = (
                i == len(scenes) - 1
            )

            segment_duration = (
                duration
                + (
                    outro_tail_sec
                    if is_last
                    else gap_sec
                )
            )

            mood = str(
                scene.get(
                    "mood",
                    "happy",
                )
            ).lower()

            if mood not in _SCALES:
                mood = "happy"

            segment = (
                self._generate_mood_segment(
                    mood,
                    segment_duration,
                )
            )

            segment = (
                self._apply_edge_fades(
                    segment,
                    sr,
                    fade_sec=0.25,
                )
            )

            segments.append(segment)

        if segments:
            full_track = np.concatenate(
                segments
            )
        else:
            full_track = np.zeros(
                int(sr * 2),
                dtype=np.float32,
            )

        full_track = self._apply_fade(
            full_track,
            sr,
            fade_in_sec=1.5,
            fade_out_sec=3.0,
        )

        full_track = self._normalize(
            full_track,
            target_peak=float(
                AUDIO_CONFIG.get(
                    "music_volume",
                    0.3,
                )
            ),
        )

        self._save_wav(
            full_track,
            output_path,
            sr,
        )

    def _generate_mood_segment(
        self,
        mood: str,
        duration_sec: float,
    ) -> np.ndarray:

        sr = self.sample_rate

        n = max(
            1,
            int(sr * duration_sec),
        )

        t = np.linspace(
            0,
            duration_sec,
            n,
            endpoint=False,
        )

        melody = self._compose_melody(
            t,
            mood,
            sr,
        )

        harmony = self._compose_harmony(
            t,
            mood,
        )

        bass = self._compose_bass(
            t,
            mood,
        )

        reverb = self._apply_simple_reverb(
            melody,
            sr,
            delay_ms=80,
        )

        return (
            melody * 0.45
            + harmony * 0.25
            + bass * 0.15
            + reverb * 0.15
        ).astype(np.float32)

    def _compose_melody(
        self,
        t: np.ndarray,
        mood: str,
        sr: int,
    ) -> np.ndarray:

        scale = _SCALES.get(
            mood,
            _SCALES["happy"],
        )

        tempo = _TEMPOS.get(
            mood,
            100,
        )

        beat = 60.0 / tempo

        total_duration = (
            t[-1] + 1.0 / sr
            if len(t)
            else 0.0
        )

        signal = np.zeros_like(t)

        rng = np.random.default_rng(
            seed=42
        )

        n_notes = (
            int(
                total_duration
                / beat
            )
            + 1
        )

        for i in range(n_notes):

            note_start = (
                i * beat
            )

            note_duration = (
                beat
                * rng.choice(
                    [
                        0.5,
                        1.0,
                        1.5,
                        2.0,
                    ],
                    p=[
                        0.3,
                        0.4,
                        0.2,
                        0.1,
                    ],
                )
            )

            frequency = (
                rng.choice(scale)
                * rng.choice(
                    [1.0, 2.0]
                )
            )

            mask = (
                (t >= note_start)
                & (
                    t
                    < note_start
                    + note_duration
                )
            )

            if not np.any(mask):
                continue

            local_t = (
                t[mask]
                - note_start
            )

            envelope = np.exp(
                -3.0
                * local_t
                / note_duration
            )

            signal[mask] += (
                np.sin(
                    2
                    * np.pi
                    * frequency
                    * t[mask]
                )
                * envelope
                * 0.6
            )

        return signal

    def _compose_harmony(
        self,
        t: np.ndarray,
        mood: str,
    ) -> np.ndarray:

        chords = _CHORD_SETS.get(
            mood,
            _CHORD_SETS["happy"],
        )

        signal = np.zeros_like(t)

        beat_duration = 2.0

        for i, chord in enumerate(
            chords * 100
        ):

            start = (
                i
                * beat_duration
            )

            if (
                len(t) == 0
                or start >= t[-1]
            ):
                break

            mask = (
                (t >= start)
                & (
                    t
                    < start
                    + beat_duration
                )
            )

            if not np.any(mask):
                continue

            for frequency in chord:

                signal[mask] += (
                    np.sin(
                        2
                        * np.pi
                        * frequency
                        * t[mask]
                    )
                    * 0.15
                )

                signal[mask] += (
                    np.sin(
                        2
                        * np.pi
                        * frequency
                        * 2
                        * t[mask]
                    )
                    * 0.05
                )

        return signal

    def _compose_bass(
        self,
        t: np.ndarray,
        mood: str,
    ) -> np.ndarray:

        notes = _BASS_NOTES.get(
            mood,
            _BASS_NOTES["happy"],
        )

        signal = np.zeros_like(t)

        duration = 1.5

        for i, note in enumerate(
            notes * 50
        ):

            start = (
                i * duration
            )

            if (
                len(t) == 0
                or start >= t[-1]
            ):
                break

            mask = (
                (t >= start)
                & (
                    t
                    < start
                    + duration
                )
            )

            if not np.any(mask):
                continue

            local_t = (
                t[mask]
                - start
            )

            envelope = np.exp(
                -1.5
                * local_t
                / duration
            )

            signal[mask] += (
                np.sin(
                    2
                    * np.pi
                    * note
                    * t[mask]
                )
                * envelope
                * 0.5
            )

            signal[mask] += (
                np.sin(
                    2
                    * np.pi
                    * note
                    * 2
                    * t[mask]
                )
                * envelope
                * 0.15
            )

        return signal

    # ========================================================
    # AUDIO EFFECTS
    # ========================================================

    def _apply_simple_reverb(
        self,
        signal: np.ndarray,
        sr: int,
        delay_ms: float = 80,
        decay: float = 0.3,
    ) -> np.ndarray:

        delay_samples = int(
            sr
            * delay_ms
            / 1000
        )

        reverb = np.zeros_like(
            signal
        )

        if (
            delay_samples > 0
            and delay_samples < len(signal)
        ):
            reverb[
                delay_samples:
            ] = (
                signal[
                    :-delay_samples
                ]
                * decay
            )

        return reverb

    def _apply_edge_fades(
        self,
        signal: np.ndarray,
        sr: int,
        fade_sec: float = 0.25,
    ) -> np.ndarray:

        result = signal.copy()

        n = min(
            int(sr * fade_sec),
            len(result) // 4,
        )

        if n <= 0:
            return result

        fade_curve = np.linspace(
            0.0,
            1.0,
            n,
        )

        result[:n] *= fade_curve

        result[-n:] *= (
            fade_curve[::-1]
        )

        return result

    def _apply_fade(
        self,
        signal: np.ndarray,
        sr: int,
        fade_in_sec: float = 1.0,
        fade_out_sec: float = 2.0,
    ) -> np.ndarray:

        result = signal.copy()

        if len(result) < 100:
            return result

        fade_in_n = min(
            int(sr * fade_in_sec),
            len(result) // 3,
        )

        fade_out_n = min(
            int(sr * fade_out_sec),
            len(result) // 3,
        )

        if fade_in_n > 0:
            result[:fade_in_n] *= (
                np.linspace(
                    0.0,
                    1.0,
                    fade_in_n,
                )
            )

        if fade_out_n > 0:
            fade_out_curve = (
                0.5
                * (
                    1
                    + np.cos(
                        np.linspace(
                            0,
                            np.pi,
                            fade_out_n,
                        )
                    )
                )
            )

            result[-fade_out_n:] *= (
                fade_out_curve
            )

        return result

    def _normalize(
        self,
        signal: np.ndarray,
        target_peak: float = 0.8,
    ) -> np.ndarray:

        peak = (
            np.max(
                np.abs(signal)
            )
            if len(signal)
            else 0.0
        )

        if peak > 1e-6:

            signal = (
                signal
                * (
                    target_peak
                    / peak
                )
            )

        return np.clip(
            signal,
            -1.0,
            1.0,
        ).astype(np.float32)

    # ========================================================
    # CONCATENATION
    # ========================================================

    def _concatenate_audio(
        self,
        audio_paths: List[Path],
        output_path: Path,
        gap_sec: float = 0.4,
    ) -> None:

        segments = []

        sr = self.sample_rate

        gap = np.zeros(
            int(sr * gap_sec),
            dtype=np.float32,
        )

        for path in audio_paths:

            if not path.exists():
                continue

            try:

                audio, file_sr = (
                    self._load_wav(path)
                )

                if file_sr != sr:
                    audio = (
                        self._resample(
                            audio,
                            file_sr,
                            sr,
                        )
                    )

                segments.append(audio)

            except Exception as e:

                logger.warning(
                    f"Could not read WAV "
                    f"{path.name}: {e}"
                )

        if not segments:

            self._generate_silence(
                output_path,
                5.0,
            )

            return

        parts = []

        for i, segment in enumerate(
            segments
        ):

            parts.append(segment)

            if i < len(segments) - 1:
                parts.append(gap)

        combined = np.concatenate(
            parts
        )

        self._save_wav(
            combined,
            output_path,
            sr,
        )

    # ========================================================
    # MIXING
    # ========================================================

    def _mix_audio(
        self,
        narration_path: Path,
        music_path: Path,
        output_path: Path,
    ) -> None:

        sr = self.sample_rate

        if narration_path.exists():

            narration, n_sr = (
                self._load_wav(
                    narration_path
                )
            )

        else:

            narration = np.zeros(
                sr,
                dtype=np.float32,
            )

            n_sr = sr

        if music_path.exists():

            music, m_sr = (
                self._load_wav(
                    music_path
                )
            )

        else:

            music = np.zeros(
                sr,
                dtype=np.float32,
            )

            m_sr = sr

        if n_sr != sr:

            narration = (
                self._resample(
                    narration,
                    n_sr,
                    sr,
                )
            )

        if m_sr != sr:

            music = (
                self._resample(
                    music,
                    m_sr,
                    sr,
                )
            )

        target_len = max(
            len(narration),
            len(music),
        )

        narration = (
            self._pad_or_trim(
                narration,
                target_len,
            )
        )

        music = (
            self._pad_or_trim(
                music,
                target_len,
            )
        )

        voice_volume = float(
            AUDIO_CONFIG.get(
                "voice_volume",
                1.0,
            )
        )

        music_volume = float(
            AUDIO_CONFIG.get(
                "music_volume",
                0.3,
            )
        )

        music_ducked = (
            self._duck_music(
                narration,
                music,
                sr,
            )
        )

        mixed = (
            narration * voice_volume
            + music_ducked
            * music_volume
        )

        mixed = self._normalize(
            mixed,
            target_peak=0.85,
        )

        self._save_wav(
            mixed,
            output_path,
            sr,
        )

    def _duck_music(
        self,
        narration: np.ndarray,
        music: np.ndarray,
        sr: int,
        duck_floor: float = 0.35,
    ) -> np.ndarray:

        if len(narration) == 0:
            return music

        window = max(
            1,
            int(sr * 0.03),
        )

        abs_narration = np.abs(
            narration
        )

        cumsum = np.cumsum(
            np.insert(
                abs_narration,
                0,
                0.0,
            )
        )

        envelope = (
            cumsum[window:]
            - cumsum[:-window]
        ) / window

        if len(envelope) == 0:
            return music

        envelope = np.pad(
            envelope,
            (
                0,
                max(
                    0,
                    len(narration)
                    - len(envelope),
                ),
            ),
            mode="edge",
        )

        envelope = envelope[
            : len(narration)
        ]

        peak = (
            np.max(envelope)
            if len(envelope)
            else 0.0
        )

        if peak <= 1e-6:
            return music

        envelope_norm = np.clip(
            envelope / peak,
            0.0,
            1.0,
        )

        smoothed = (
            np.copy(
                envelope_norm
            )
        )

        alpha = 0.002

        for i in range(
            1,
            len(smoothed),
        ):

            smoothed[i] = (
                smoothed[i - 1]
                + alpha
                * (
                    envelope_norm[i]
                    - smoothed[i - 1]
                )
            )

        gain = (
            1.0
            - (
                1.0
                - duck_floor
            )
            * smoothed
        )

        return music * gain

    # ========================================================
    # WAV I/O
    # ========================================================

    def _save_wav(
        self,
        signal: np.ndarray,
        path: Path,
        sample_rate: Optional[int] = None,
    ) -> None:

        sr = (
            sample_rate
            or self.sample_rate
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = (
            np.clip(
                signal,
                -1.0,
                1.0,
            )
            * 32767
        ).astype(
            np.int16
        )

        with wave.open(
            str(path),
            "wb",
        ) as wf:

            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(
                data.tobytes()
            )

    def _load_wav(
        self,
        path: Path,
    ) -> tuple[np.ndarray, int]:

        with wave.open(
            str(path),
            "rb",
        ) as wf:

            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(
                n_frames
            )

            n_channels = (
                wf.getnchannels()
            )

            sample_width = (
                wf.getsampwidth()
            )

        if sample_width == 2:

            data = (
                np.frombuffer(
                    raw,
                    dtype=np.int16,
                )
                .astype(
                    np.float32
                )
                / 32767.0
            )

        elif sample_width == 4:

            data = (
                np.frombuffer(
                    raw,
                    dtype=np.int32,
                )
                .astype(
                    np.float32
                )
                / 2147483647.0
            )

        else:

            data = (
                np.frombuffer(
                    raw,
                    dtype=np.uint8,
                )
                .astype(
                    np.float32
                )
                / 127.5
                - 1.0
            )

        if n_channels > 1:

            data = data.reshape(
                -1,
                n_channels,
            ).mean(axis=1)

        return (
            data.astype(np.float32),
            sr,
        )

    # ========================================================
    # RESAMPLING / LENGTH
    # ========================================================

    def _resample(
        self,
        signal: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:

        if (
            orig_sr == target_sr
            or len(signal) == 0
        ):
            return signal.astype(
                np.float32
            )

        ratio = (
            target_sr
            / orig_sr
        )

        new_len = max(
            1,
            int(
                len(signal)
                * ratio
            ),
        )

        old_idx = np.linspace(
            0,
            len(signal) - 1,
            new_len,
        )

        new_signal = np.interp(
            old_idx,
            np.arange(
                len(signal)
            ),
            signal,
        )

        return new_signal.astype(
            np.float32
        )

    def _pad_or_trim(
        self,
        signal: np.ndarray,
        target_len: int,
    ) -> np.ndarray:

        if len(signal) < target_len:

            return np.concatenate(
                [
                    signal,
                    np.zeros(
                        target_len
                        - len(signal),
                        dtype=np.float32,
                    ),
                ]
            )

        return signal[
            :target_len
        ]
