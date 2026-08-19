import asyncio
import gc
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
# MUSIC CONFIGURATION
# ============================================================

_SCALES = {
    "adventure": [
        261.63, 293.66, 329.63,
        392.00, 440.00, 523.25
    ],
    "magical": [
        261.63, 311.13, 392.00,
        466.16, 523.25, 622.25
    ],
    "happy": [
        261.63, 293.66, 329.63,
        349.23, 392.00, 440.00
    ],
    "mysterious": [
        261.63, 277.18, 311.13,
        369.99, 415.30, 466.16
    ],
    "heroic": [
        261.63, 329.63, 392.00,
        523.25, 659.25, 783.99
    ],
    "exciting": [
        329.63, 392.00, 493.88,
        587.33, 659.25, 783.99
    ],
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
    "adventure": [
        65.41, 73.42, 82.41, 98.00
    ],
    "magical": [
        65.41, 69.30, 82.41, 87.31
    ],
    "happy": [
        65.41, 87.31, 98.00, 73.42
    ],
    "mysterious": [
        32.70, 36.71, 41.20, 43.65
    ],
    "heroic": [
        65.41, 87.31, 130.81, 98.00
    ],
    "exciting": [
        82.41, 98.00, 110.00, 123.47
    ],
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
    FalconAI Kids CPU-friendly audio generation engine.

    Responsibilities:
        1. Generate narration using Edge-TTS when available.
        2. Generate fallback silence when TTS fails.
        3. Generate procedural background music using NumPy.
        4. Concatenate narration.
        5. Duck background music under narration.
        6. Mix narration + music.
        7. Expose per-scene narration audio paths.

    No GPU is required by this module.
    """

    def __init__(
        self,
        language: str = "Albanian",
    ):
        self.language = language

        self.sample_rate = int(
            AUDIO_CONFIG.get(
                "sample_rate",
                22050,
            )
        )

        self.tts_retries = max(
            0,
            int(
                AUDIO_CONFIG.get(
                    "tts_retries",
                    2,
                )
            ),
        )

        self.tts_retry_delay_sec = max(
            0.0,
            float(
                AUDIO_CONFIG.get(
                    "tts_retry_delay_sec",
                    1.5,
                )
            ),
        )

        self._edge_tts_available = False

        self._load_model()

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def _load_model(self) -> None:
        """
        Detect whether edge-tts is installed.

        edge-tts is network-based and does not require GPU.
        """

        try:
            import edge_tts  # noqa: F401

            self._edge_tts_available = True

            logger.success(
                "Edge-TTS engine successfully initialized."
            )

        except ImportError:
            self._edge_tts_available = False

            logger.warning(
                "edge-tts is not installed. "
                "Narration will use silence fallback."
            )

    # ============================================================
    # MAIN PIPELINE
    # ============================================================

    def generate(
        self,
        story: dict,
        output_dir: Path,
        language: str = "English",
        gender: Optional[str] = None,
    ) -> Path:

        if not isinstance(story, dict):
            raise AudioGenerationError(
                "Story must be a dictionary."
            )

        self.language = language

        scenes = story.get(
            "scenes",
            [],
        )

        if not scenes:
            raise AudioGenerationError(
                "The story contains no scenes; "
                "audio generation aborted."
            )

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        gap_sec = max(
            0.0,
            float(
                AUDIO_CONFIG.get(
                    "scene_gap_sec",
                    0.4,
                )
            ),
        )

        outro_tail_sec = max(
            0.0,
            float(
                AUDIO_CONFIG.get(
                    "outro_tail_sec",
                    2.0,
                )
            ),
        )

        logger.info(
            f"Generating audio for {len(scenes)} scenes."
        )

        # --------------------------------------------------------
        # NARRATION
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # FULL NARRATION
        # --------------------------------------------------------

        narration_full_path = (
            output_dir /
            "narration_full.wav"
        )

        self._concatenate_audio(
            audio_paths=narration_paths,
            output_path=narration_full_path,
            gap_sec=gap_sec,
        )

        # --------------------------------------------------------
        # BACKGROUND MUSIC
        # --------------------------------------------------------

        music_path = (
            output_dir /
            "background_music.wav"
        )

        self._generate_full_soundtrack(
            scenes=scenes,
            scene_durations=scene_durations,
            gap_sec=gap_sec,
            outro_tail_sec=outro_tail_sec,
            output_path=music_path,
        )

        # --------------------------------------------------------
        # FINAL MIX
        # --------------------------------------------------------

        final_path = (
            output_dir /
            "final_audio.wav"
        )

        self._mix_audio(
            narration_path=narration_full_path,
            music_path=music_path,
            output_path=final_path,
        )

        story["total_duration_sec"] = (
            total_narration_duration
        )

        logger.success(
            f"Final audio generated: "
            f"{final_path} "
            f"({total_narration_duration:.1f}s)"
        )

        self._free_memory()

        return final_path

    # ============================================================
    # NARRATION
    # ============================================================

    def _generate_all_narration(
        self,
        scenes: List[dict],
        output_dir: Path,
        gender: Optional[str],
    ) -> tuple[List[Path], List[float]]:

        narration_paths = []
        scene_durations = []

        total = len(scenes)

        for i, scene in enumerate(
            scenes
        ):

            narration = str(
                scene.get(
                    "narration",
                    "",
                )
            ).strip()

            narration_path = (
                output_dir /
                f"narration_scene_{i + 1:02d}.wav"
            )

            if not narration:

                logger.debug(
                    f"Scene {i + 1}: "
                    "no narration text. "
                    "Creating silence."
                )

                duration = 5.0

                self._generate_silence(
                    narration_path,
                    duration,
                )

            else:

                logger.info(
                    f"TTS [{i + 1}/{total}]: "
                    f"{narration[:60]}..."
                )

                duration = (
                    self._synthesize_with_retries(
                        text=narration,
                        output_path=narration_path,
                        gender=gender,
                    )
                )

            # ----------------------------------------------------
            # Store duration on scene.
            # ----------------------------------------------------

            scene["duration_sec"] = (
                duration
            )

            # ----------------------------------------------------
            # Store individual narration path.
            #
            # Required by character_animator.
            # ----------------------------------------------------

            scene[
                "narration_audio_path"
            ] = str(
                narration_path.resolve()
            )

            scene_durations.append(
                duration
            )

            narration_paths.append(
                narration_path
            )

            self._free_memory()

        return (
            narration_paths,
            scene_durations,
        )

    # ============================================================
    # TTS
    # ============================================================

    def _synthesize_with_retries(
        self,
        text: str,
        output_path: Path,
        gender: Optional[str],
    ) -> float:

        if not text.strip():
            self._generate_silence(
                output_path,
                4.0,
            )
            return 4.0

        if not self._edge_tts_available:

            logger.warning(
                "Edge-TTS unavailable. "
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

        attempts = (
            self.tts_retries + 1
        )

        for attempt in range(
            1,
            attempts + 1,
        ):

            try:

                asyncio.run(
                    self._generate_edge_tts(
                        text=text,
                        output_path=output_path,
                        gender=gender,
                    )
                )

                if not output_path.exists():
                    raise RuntimeError(
                        "TTS completed but output file "
                        "was not created."
                    )

                duration = (
                    self._get_audio_duration(
                        output_path
                    )
                )

                if duration <= 0:
                    raise RuntimeError(
                        "Generated TTS file has zero duration."
                    )

                return duration

            except Exception as exc:

                last_error = exc

                logger.warning(
                    f"TTS attempt "
                    f"{attempt}/{attempts} failed: "
                    f"{exc}"
                )

                if (
                    attempt < attempts
                    and self.tts_retry_delay_sec > 0
                ):
                    time.sleep(
                        self.tts_retry_delay_sec
                    )

        # --------------------------------------------------------
        # Final fallback
        # --------------------------------------------------------

        logger.warning(
            f"TTS failed after {attempts} attempts: "
            f"{last_error}"
        )

        fallback_duration = (
            self._estimate_speech_duration(
                text
            )
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

        speed = AUDIO_CONFIG.get(
            "tts_speed",
            1.0,
        )

        rate_str = (
            self._speed_to_rate_string(
                speed
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

    def _resolve_voice(
        self,
        gender: Optional[str],
    ) -> str:

        override = AUDIO_CONFIG.get(
            "tts_voice_override"
        )

        if override:
            return str(
                override
            )

        language_key = (
            str(
                self.language
            ).strip().lower()
        )

        female_voice, male_voice = (
            _VOICE_MAP.get(
                language_key,
                _DEFAULT_VOICE,
            )
        )

        if gender:
            gender_key = (
                str(
                    gender
                ).strip().lower()
            )

            if gender_key in (
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
            speed = float(speed)
        except (
            TypeError,
            ValueError,
        ):
            speed = 1.0

        speed = max(
            0.5,
            min(2.0, speed),
        )

        percentage = round(
            (speed - 1.0) * 100
        )

        sign = (
            "+"
            if percentage >= 0
            else ""
        )

        return (
            f"{sign}{percentage}%"
        )

    def _estimate_speech_duration(
        self,
        text: str,
    ) -> float:

        """
        Approximate speech duration used only
        when TTS is unavailable.

        Average:
            ~2.5 words/sec
        """

        words = len(
            text.split()
        )

        estimated = (
            words / 2.5
        )

        return max(
            1.0,
            min(
                60.0,
                estimated,
            ),
        )

    # ============================================================
    # AUDIO DURATION
    # ============================================================

    def _get_audio_duration(
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

            return (
                frames /
                float(rate)
            )

    # ============================================================
    # PROCEDURAL MUSIC
    # ============================================================

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
            ).strip().lower()

            if mood not in _SCALES:
                mood = "happy"

            segment = (
                self._generate_mood_segment(
                    mood=mood,
                    duration_sec=segment_duration,
                )
            )

            segment = (
                self._apply_edge_fades(
                    signal=segment,
                    sr=sr,
                    fade_sec=0.25,
                )
            )

            segments.append(
                segment
            )

        if segments:

            full_track = np.concatenate(
                segments
            )

        else:

            full_track = np.zeros(
                sr * 2,
                dtype=np.float32,
            )

        full_track = (
            self._apply_fade(
                full_track,
                sr,
                fade_in_sec=1.5,
                fade_out_sec=3.0,
            )
        )

        music_volume = float(
            AUDIO_CONFIG.get(
                "music_volume",
                0.3,
            )
        )

        full_track = (
            self._normalize(
                full_track,
                target_peak=min(
                    1.0,
                    max(
                        0.01,
                        music_volume,
                    ),
                ),
            )
        )

        self._save_wav(
            full_track,
            output_path,
            sr,
        )

        del full_track
        self._free_memory()

    def _generate_mood_segment(
        self,
        mood: str,
        duration_sec: float,
    ) -> np.ndarray:

        sr = self.sample_rate

        duration_sec = max(
            0.1,
            float(duration_sec),
        )

        n = max(
            1,
            int(
                sr * duration_sec
            ),
        )

        t = np.arange(
            n,
            dtype=np.float32,
        ) / sr

        melody = (
            self._compose_melody(
                t,
                mood,
                sr,
            )
        )

        harmony = (
            self._compose_harmony(
                t,
                mood,
            )
        )

        bass = (
            self._compose_bass(
                t,
                mood,
            )
        )

        reverb = (
            self._apply_simple_reverb(
                melody,
                sr,
                delay_ms=80,
            )
        )

        result = (
            melody * 0.45
            + harmony * 0.25
            + bass * 0.15
            + reverb * 0.15
        )

        return result.astype(
            np.float32
        )

    # ============================================================
    # MELODY
    # ============================================================

    def _compose_melody(
        self,
        t: np.ndarray,
        mood: str,
        sr: int,
    ) -> np.ndarray:

        if len(t) == 0:
            return np.zeros(
                0,
                dtype=np.float32,
            )

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
            len(t) / sr
        )

        signal = np.zeros_like(
            t,
            dtype=np.float32,
        )

        rng = np.random.default_rng(
            seed=42
        )

        note_count = (
            int(
                total_duration /
                beat
            ) + 1
        )

        for i in range(
            note_count
        ):

            start = (
                i * beat
            )

            duration = (
                beat
                * rng.choice(
                    [0.5, 1.0, 1.5, 2.0],
                    p=[
                        0.3,
                        0.4,
                        0.2,
                        0.1,
                    ],
                )
            )

            frequency = (
                float(
                    rng.choice(
                        scale
                    )
                )
                * float(
                    rng.choice(
                        [1.0, 2.0]
                    )
                )
            )

            mask = (
                (t >= start)
                & (
                    t <
                    start + duration
                )
            )

            if not np.any(mask):
                continue

            local_t = (
                t[mask] - start
            )

            envelope = np.exp(
                -3.0
                * local_t
                / max(
                    duration,
                    0.001,
                )
            )

            signal[mask] += (
                np.sin(
                    2.0
                    * np.pi
                    * frequency
                    * t[mask]
                )
                * envelope
                * 0.6
            )

        return np.clip(
            signal,
            -1.0,
            1.0,
        )

    # ============================================================
    # HARMONY
    # ============================================================

    def _compose_harmony(
        self,
        t: np.ndarray,
        mood: str,
    ) -> np.ndarray:

        if len(t) == 0:
            return np.zeros(
                0,
                dtype=np.float32,
            )

        chords = _CHORD_SETS.get(
            mood,
            _CHORD_SETS["happy"],
        )

        signal = np.zeros_like(
            t,
            dtype=np.float32,
        )

        chord_duration = 2.0

        chord_count = (
            int(
                (
                    len(t)
                    / self.sample_rate
                )
                / chord_duration
            ) + 1
        )

        for i in range(
            chord_count
        ):

            start = (
                i *
                chord_duration
            )

            mask = (
                (t >= start)
                & (
                    t <
                    start + chord_duration
                )
            )

            if not np.any(mask):
                continue

            chord = chords[
                i % len(chords)
            ]

            for frequency in chord:

                signal[mask] += (
                    np.sin(
                        2.0
                        * np.pi
                        * frequency
                        * t[mask]
                    )
                    * 0.15
                )

                signal[mask] += (
                    np.sin(
                        2.0
                        * np.pi
                        * frequency
                        * 2.0
                        * t[mask]
                    )
                    * 0.05
                )

        return signal

    # ============================================================
    # BASS
    # ============================================================

    def _compose_bass(
        self,
        t: np.ndarray,
        mood: str,
    ) -> np.ndarray:

        if len(t) == 0:
            return np.zeros(
                0,
                dtype=np.float32,
            )

        notes = _BASS_NOTES.get(
            mood,
            _BASS_NOTES["happy"],
        )

        signal = np.zeros_like(
            t,
            dtype=np.float32,
        )

        note_duration = 1.5

        note_count = (
            int(
                (
                    len(t)
                    / self.sample_rate
                )
                / note_duration
            ) + 1
        )

        for i in range(
            note_count
        ):

            start = (
                i *
                note_duration
            )

            mask = (
                (t >= start)
                & (
                    t <
                    start + note_duration
                )
            )

            if not np.any(mask):
                continue

            frequency = notes[
                i % len(notes)
            ]

            local_t = (
                t[mask] - start
            )

            envelope = np.exp(
                -1.5
                * local_t
                / max(
                    note_duration,
                    0.001,
                )
            )

            signal[mask] += (
                np.sin(
                    2.0
                    * np.pi
                    * frequency
                    * t[mask]
                )
                * envelope
                * 0.5
            )

            signal[mask] += (
                np.sin(
                    2.0
                    * np.pi
                    * frequency
                    * 2.0
                    * t[mask]
                )
                * envelope
                * 0.15
            )

        return signal

    # ============================================================
    # AUDIO EFFECTS
    # ============================================================

    def _apply_simple_reverb(
        self,
        signal: np.ndarray,
        sr: int,
        delay_ms: float = 80,
        decay: float = 0.3,
    ) -> np.ndarray:

        if len(signal) == 0:
            return signal.copy()

        delay_samples = max(
            1,
            int(
                sr
                * delay_ms
                / 1000.0
            ),
        )

        reverb = np.zeros_like(
            signal
        )

        if delay_samples < len(
            signal
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

        if len(signal) == 0:
            return signal.copy()

        result = signal.copy()

        fade_samples = min(
            int(
                sr * fade_sec
            ),
            len(result) // 4,
        )

        if fade_samples <= 0:
            return result

        fade_in = np.linspace(
            0.0,
            1.0,
            fade_samples,
            dtype=np.float32,
        )

        fade_out = fade_in[::-1]

        result[
            :fade_samples
        ] *= fade_in

        result[
            -fade_samples:
        ] *= fade_out

        return result

    def _apply_fade(
        self,
        signal: np.ndarray,
        sr: int,
        fade_in_sec: float = 1.0,
        fade_out_sec: float = 2.0,
    ) -> np.ndarray:

        if len(signal) < 100:
            return signal.copy()

        result = signal.copy()

        fade_in_n = min(
            int(
                sr * fade_in_sec
            ),
            len(result) // 3,
        )

        fade_out_n = min(
            int(
                sr * fade_out_sec
            ),
            len(result) // 3,
        )

        if fade_in_n > 0:

            result[
                :fade_in_n
            ] *= np.linspace(
                0.0,
                1.0,
                fade_in_n,
                dtype=np.float32,
            )

        if fade_out_n > 0:

            curve = 0.5 * (
                1.0
                + np.cos(
                    np.linspace(
                        0.0,
                        np.pi,
                        fade_out_n,
                        dtype=np.float32,
                    )
                )
            )

            result[
                -fade_out_n:
            ] *= curve

        return result

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def _normalize(
        self,
        signal: np.ndarray,
        target_peak: float = 0.8,
    ) -> np.ndarray:

        if len(signal) == 0:
            return signal.astype(
                np.float32
            )

        peak = float(
            np.max(
                np.abs(signal)
            )
        )

        if peak > 1e-6:

            target_peak = min(
                1.0,
                max(
                    0.01,
                    float(
                        target_peak
                    ),
                ),
            )

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
        ).astype(
            np.float32
        )

    # ============================================================
    # CONCATENATION
    # ============================================================

    def _concatenate_audio(
        self,
        audio_paths: List[Path],
        output_path: Path,
        gap_sec: float = 0.4,
    ) -> None:

        sr = self.sample_rate

        gap = np.zeros(
            int(
                sr
                * max(
                    0.0,
                    gap_sec,
                )
            ),
            dtype=np.float32,
        )

        segments = []

        for path in audio_paths:

            if not path.exists():
                logger.warning(
                    f"Audio file missing: "
                    f"{path}"
                )
                continue

            try:

                audio, file_sr = (
                    self._load_wav(
                        path
                    )
                )

                if file_sr != sr:

                    audio = (
                        self._resample(
                            audio,
                            file_sr,
                            sr,
                        )
                    )

                segments.append(
                    audio
                )

            except Exception as exc:

                logger.warning(
                    f"Failed to read "
                    f"{path.name}: "
                    f"{exc}"
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

            parts.append(
                segment
            )

            if i < len(
                segments
            ) - 1:

                if len(gap) > 0:
                    parts.append(
                        gap
                    )

        combined = np.concatenate(
            parts
        )

        self._save_wav(
            combined,
            output_path,
            sr,
        )

        del combined
        self._free_memory()

    # ============================================================
    # MIXING
    # ============================================================

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

        ducked_music = (
            self._duck_music(
                narration,
                music,
                sr,
            )
        )

        mixed = (
            narration * voice_volume
            + ducked_music * music_volume
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

        del narration
        del music
        del ducked_music
        del mixed

        self._free_memory()

    # ============================================================
    # MUSIC DUCKING
    # ============================================================

    def _duck_music(
        self,
        narration: np.ndarray,
        music: np.ndarray,
        sr: int,
        duck_floor: float = 0.35,
    ) -> np.ndarray:

        if len(narration) == 0:
            return music.copy()

        if len(music) == 0:
            return music.copy()

        if len(narration) != len(music):

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

        window = max(
            1,
            int(
                sr * 0.03
            ),
        )

        if len(narration) < window:
            return music.copy()

        absolute = np.abs(
            narration
        )

        cumulative = np.cumsum(
            np.insert(
                absolute,
                0,
                0.0,
            )
        )

        envelope = (
            cumulative[window:]
            - cumulative[:-window]
        ) / window

        envelope = np.pad(
            envelope,
            (
                0,
                len(narration)
                - len(envelope),
            ),
            mode="edge",
        )

        peak = float(
            np.max(envelope)
        )

        if peak <= 1e-6:
            return music.copy()

        normalized = np.clip(
            envelope / peak,
            0.0,
            1.0,
        )

        # Faster smoothing than the original
        # while still avoiding abrupt volume changes.
        alpha = 0.01

        smoothed = np.empty_like(
            normalized
        )

        smoothed[0] = normalized[0]

        for i in range(
            1,
            len(smoothed),
        ):
            smoothed[i] = (
                smoothed[i - 1]
                + alpha
                * (
                    normalized[i]
                    - smoothed[i - 1]
                )
            )

        duck_floor = np.clip(
            duck_floor,
            0.0,
            1.0,
        )

        gain = (
            1.0
            - (
                1.0
                - duck_floor
            )
            * smoothed
        )

        return (
            music * gain
        ).astype(
            np.float32
        )

    # ============================================================
    # WAV I/O
    # ============================================================

    def _save_wav(
        self,
        signal: np.ndarray,
        path: Path,
        sample_rate: Optional[int] = None,
    ) -> None:

        sr = int(
            sample_rate
            or self.sample_rate
        )

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        signal = np.asarray(
            signal,
            dtype=np.float32,
        )

        signal = np.nan_to_num(
            signal,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        signal = np.clip(
            signal,
            -1.0,
            1.0,
        )

        data = (
            signal
            * 32767.0
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

            sample_rate = (
                wf.getframerate()
            )

            frame_count = (
                wf.getnframes()
            )

            raw = wf.readframes(
                frame_count
            )

            channels = (
                wf.getnchannels()
            )

            sample_width = (
                wf.getsampwidth()
            )

        if sample_width == 1:

            data = (
                np.frombuffer(
                    raw,
                    dtype=np.uint8,
                ).astype(
                    np.float32
                )

            )

            data = (
                data
                / 127.5
                - 1.0
            )

        elif sample_width == 2:

            data = (
                np.frombuffer(
                    raw,
                    dtype=np.int16,
                ).astype(
                    np.float32
                )
                / 32767.0
            )

        elif sample_width == 4:

            data = (
                np.frombuffer(
                    raw,
                    dtype=np.int32,
                ).astype(
                    np.float32
                )
                / 2147483647.0
            )

        else:

            raise AudioGenerationError(
                f"Unsupported WAV sample width: "
                f"{sample_width} bytes."
            )

        if channels > 1:

            expected = (
                len(data) // channels
            )

            data = data[
                : expected * channels
            ]

            data = data.reshape(
                -1,
                channels,
            )

            data = data.mean(
                axis=1
            )

        return (
            data.astype(
                np.float32
            ),
            sample_rate,
        )

    # ============================================================
    # SILENCE
    # ============================================================

    def _generate_silence(
        self,
        path: Path,
        duration_sec: float,
    ) -> None:

        duration_sec = max(
            0.1,
            float(duration_sec),
        )

        sample_count = int(
            self.sample_rate
            * duration_sec
        )

        signal = np.zeros(
            sample_count,
            dtype=np.float32,
        )

        self._save_wav(
            signal,
            path,
            self.sample_rate,
        )

        del signal

    # ============================================================
    # RESAMPLING
    # ============================================================

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
                np.float32,
                copy=True,
            )

        if orig_sr <= 0:
            raise AudioGenerationError(
                f"Invalid source sample rate: "
                f"{orig_sr}"
            )

        if target_sr <= 0:
            raise AudioGenerationError(
                f"Invalid target sample rate: "
                f"{target_sr}"
            )

        ratio = (
            target_sr
            / orig_sr
        )

        new_length = max(
            1,
            int(
                round(
                    len(signal)
                    * ratio
                )
            ),
        )

        old_positions = np.arange(
            len(signal),
            dtype=np.float32,
        )

        new_positions = np.linspace(
            0,
            len(signal) - 1,
            new_length,
            dtype=np.float32,
        )

        resampled = np.interp(
            new_positions,
            old_positions,
            signal,
        )

        return resampled.astype(
            np.float32
        )

    # ============================================================
    # PADDING / TRIMMING
    # ============================================================

    def _pad_or_trim(
        self,
        signal: np.ndarray,
        target_len: int,
    ) -> np.ndarray:

        target_len = max(
            0,
            int(target_len),
        )

        if len(signal) == target_len:
            return signal.astype(
                np.float32,
                copy=True,
            )

        if len(signal) < target_len:

            padding = np.zeros(
                target_len
                - len(signal),
                dtype=np.float32,
            )

            return np.concatenate(
                [
                    signal,
                    padding,
                ]
            )

        return signal[
            :target_len
        ].astype(
            np.float32,
            copy=True,
        )

    # ============================================================
    # MEMORY
    # ============================================================

    def _free_memory(self) -> None:

        gc.collect()

    # ============================================================
    # CLEANUP
    # ============================================================

    def unload(self) -> None:

        logger.debug(
            "Releasing AudioGenerator resources."
        )

        self._free_memory()

    def __del__(self):

        try:
            self.unload()
        except Exception:
            pass
