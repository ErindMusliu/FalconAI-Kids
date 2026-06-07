import math
import struct
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import AUDIO_CONFIG, LLM_CONFIG
from utils.logger import get_logger
from utils.exceptions import AudioGenerationError, ModelLoadError

logger = get_logger(__name__)


class AudioGenerator:
    def __init__(self, language: str = "Albanian"):
        self.language   = language
        self.tts_model  = None
        self.sample_rate = 22050
        self._load_model()

    def _load_model(self) -> None:
        try:
            from TTS.api import TTS

            tts_model = AUDIO_CONFIG["tts_model"]
            logger.debug(f"Duke ngarkuar TTS: {tts_model}")

            try:
                self.tts_model = TTS(
                    model_name=tts_model,
                    progress_bar=False,
                    gpu=(True if __import__("torch").cuda.is_available() else False),
                )
                logger.success(f"TTS u ngarkua: {tts_model}")
            except Exception:
                fallback = "tts_models/en/ljspeech/tacotron2-DDC"
                logger.warning(
                    f"TTS '{tts_model}' nuk u gjet, "
                    f"duke u kthyer te: {fallback}"
                )
                self.tts_model = TTS(
                    model_name=fallback,
                    progress_bar=False,
                )

        except ImportError:
            logger.warning(
                "TTS (Coqui) nuk është instaluar. "
                "Audio do të gjenerohet me sinusoidal placeholder. "
                "Instalo me: pip install TTS"
            )
            self.tts_model = None
        except Exception as e:
            logger.warning(f"TTS nuk u ngarkua: {e}. Duke vazhduar pa TTS.")
            self.tts_model = None

    def generate(
        self,
        story: dict,
        output_dir: Path,
        language: str = "Albanian",
    ) -> Path:
        scenes = story.get("scenes", [])
        if not scenes:
            raise AudioGenerationError("Historia nuk ka skena, audio nuk gjenerohet")

        logger.debug(f"Duke gjeneruar audio për {len(scenes)} skena")

        narration_paths = []

        for i, scene in enumerate(scenes):
            narration = scene.get("narration", "").strip()
            if not narration:
                logger.debug(f"Skena {i+1} nuk ka narration, duke gjeneruar silence")
                duration = scene.get("duration_sec", LLM_CONFIG["scene_duration_sec"])
                silence_path = output_dir / f"narration_scene_{i+1:02d}.wav"
                self._generate_silence(silence_path, duration)
                narration_paths.append(silence_path)
                continue

            narration_path = output_dir / f"narration_scene_{i+1:02d}.wav"
            logger.step(f"TTS skena {i+1}/{len(scenes)}: {narration[:50]}...")

            try:
                self._generate_narration(
                    text=narration,
                    output_path=narration_path,
                    duration_sec=scene.get("duration_sec", LLM_CONFIG["scene_duration_sec"]),
                )
                narration_paths.append(narration_path)
            except Exception as e:
                logger.warning(f"TTS deshtoi për skenën {i+1}: {e}, duke gjeneruar silence")
                duration = scene.get("duration_sec", LLM_CONFIG["scene_duration_sec"])
                self._generate_silence(narration_path, duration)
                narration_paths.append(narration_path)

        narration_full_path = output_dir / "narration_full.wav"
        self._concatenate_audio(narration_paths, narration_full_path)
        logger.debug("Narration u bashkua")

        total_duration = story.get(
            "total_duration_sec",
            len(scenes) * LLM_CONFIG["scene_duration_sec"]
        )
        music_path = output_dir / "background_music.wav"
        theme = story.get("theme", "adventure")
        self._generate_background_music(
            output_path=music_path,
            duration_sec=total_duration + 4,
            theme=theme,
        )
        logger.debug("Muzika e sfondit u gjenerua")

        final_path = output_dir / "final_audio.wav"
        self._mix_audio(
            narration_path=narration_full_path,
            music_path=music_path,
            output_path=final_path,
        )

        logger.success(f"Audio finale u gjenerua: {final_path}")
        return final_path

    def _generate_narration(
        self,
        text: str,
        output_path: Path,
        duration_sec: int,
    ) -> None:
        if self.tts_model is not None:
            self._tts_coqui(text, output_path)
        else:
            self._generate_tone_placeholder(output_path, duration_sec)

        self._adjust_duration(output_path, duration_sec)

    def _tts_coqui(self, text: str, output_path: Path) -> None:
        try:
            self.tts_model.tts_to_file(
                text=text,
                file_path=str(output_path),
                speed=AUDIO_CONFIG.get("tts_speed", 1.0),
            )
        except Exception as e:
            raise AudioGenerationError(f"Coqui TTS deshtoi: {e}")

    def _generate_background_music(
        self,
        output_path: Path,
        duration_sec: float,
        theme: str = "adventure",
    ) -> None:
        logger.debug(f"Duke gjeneruar muzikë sfond | temë: {theme} | kohë: {duration_sec}s")

        sr      = self.sample_rate
        n       = int(sr * duration_sec)
        t       = np.linspace(0, duration_sec, n, endpoint=False)

        melody  = self._compose_melody(t, theme, sr)

        harmony = self._compose_harmony(t, theme)

        bass    = self._compose_bass(t, theme)

        reverb  = self._apply_simple_reverb(melody, sr, delay_ms=80)

        music   = (
            melody  * 0.45 +
            harmony * 0.25 +
            bass    * 0.15 +
            reverb  * 0.15
        )

        music = self._apply_fade(music, sr, fade_in_sec=1.5, fade_out_sec=3.0)

        music = self._normalize(music, target_peak=0.7)

        self._save_wav(music, output_path, sr)

    def _compose_melody(self, t: np.ndarray, theme: str, sr: int) -> np.ndarray:
        scales = {
            "adventure"  : [261.63, 293.66, 329.63, 392.00, 440.00, 523.25],
            "magical"    : [261.63, 311.13, 392.00, 466.16, 523.25, 622.25],
            "happy"      : [261.63, 293.66, 329.63, 349.23, 392.00, 440.00],
            "mysterious" : [261.63, 277.18, 311.13, 369.99, 415.30, 466.16],
            "heroic"     : [261.63, 329.63, 392.00, 523.25, 659.25, 783.99],
            "exciting"   : [329.63, 392.00, 493.88, 587.33, 659.25, 783.99],
        }

        tempos = {
            "adventure" : 140, "magical": 80, "happy": 120,
            "mysterious": 60,  "heroic": 130, "exciting": 150,
        }

        scale = scales.get(theme, scales["happy"])
        tempo = tempos.get(theme, 100)
        beat  = 60.0 / tempo
        total_dur = t[-1] + 1.0 / sr
        signal = np.zeros_like(t)

        rng = np.random.default_rng(seed=42)
        n_notes = int(total_dur / beat) + 1

        for i in range(n_notes):
            note_start = i * beat
            note_dur   = beat * rng.choice([0.5, 1.0, 1.5, 2.0], p=[0.3,0.4,0.2,0.1])
            freq       = rng.choice(scale) * rng.choice([1.0, 2.0])

            mask = (t >= note_start) & (t < note_start + note_dur)
            if not np.any(mask):
                continue

            t_note    = t[mask] - note_start
            envelope  = np.exp(-3.0 * t_note / note_dur)
            signal[mask] += np.sin(2 * np.pi * freq * t[mask]) * envelope * 0.6

        return signal

    def _compose_harmony(self, t: np.ndarray, theme: str) -> np.ndarray:
        chord_sets = {
            "adventure"  : [(261.63, 329.63, 392.00), (349.23, 440.00, 523.25)],
            "magical"    : [(261.63, 311.13, 392.00), (220.00, 261.63, 329.63)],
            "happy"      : [(261.63, 329.63, 392.00), (293.66, 369.99, 440.00)],
            "mysterious" : [(130.81, 155.56, 196.00), (116.54, 138.59, 174.61)],
            "heroic"     : [(261.63, 329.63, 392.00), (349.23, 440.00, 523.25)],
            "exciting"   : [(329.63, 415.30, 493.88), (293.66, 369.99, 440.00)],
        }

        chords   = chord_sets.get(theme, chord_sets["happy"])
        signal   = np.zeros_like(t)
        beat_dur = 2.0

        for i, chord in enumerate(chords * 100):
            start = i * beat_dur
            if start >= t[-1]:
                break
            mask = (t >= start) & (t < start + beat_dur)
            if not np.any(mask):
                continue
            for freq in chord:
                signal[mask] += (
                    np.sin(2 * np.pi * freq * t[mask]) * 0.15
                    + np.sin(2 * np.pi * freq * 2 * t[mask]) * 0.05
                )

        return signal

    def _compose_bass(self, t: np.ndarray, theme: str) -> np.ndarray:
        bass_notes = {
            "adventure"  : [65.41, 73.42, 82.41, 98.00],
            "magical"    : [65.41, 69.30, 82.41, 87.31],
            "happy"      : [65.41, 87.31, 98.00, 73.42],
            "mysterious" : [32.70, 36.71, 41.20, 43.65],
            "heroic"     : [65.41, 87.31, 130.81, 98.00],
            "exciting"   : [82.41, 98.00, 110.00, 123.47],
        }

        notes  = bass_notes.get(theme, bass_notes["happy"])
        signal = np.zeros_like(t)
        dur    = 1.5

        for i, note in enumerate(notes * 50):
            start = i * dur
            if start >= t[-1]:
                break
            mask = (t >= start) & (t < start + dur)
            if not np.any(mask):
                continue
            t_local  = t[mask] - start
            envelope = np.exp(-1.5 * t_local / dur)
            signal[mask] += (
                np.sin(2 * np.pi * note * t[mask]) * envelope * 0.5
                + np.sin(2 * np.pi * note * 2 * t[mask]) * envelope * 0.15
            )

        return signal

    def _apply_simple_reverb(
        self,
        signal: np.ndarray,
        sr: int,
        delay_ms: float = 80,
        decay: float = 0.3,
    ) -> np.ndarray:
        delay_samples = int(sr * delay_ms / 1000)
        reverb = np.zeros_like(signal)
        if delay_samples < len(signal):
            reverb[delay_samples:] = signal[:-delay_samples] * decay
        return reverb

    def _apply_fade(
        self,
        signal: np.ndarray,
        sr: int,
        fade_in_sec: float = 1.0,
        fade_out_sec: float = 2.0,
    ) -> np.ndarray:
        result        = signal.copy()
        fade_in_n     = min(int(sr * fade_in_sec), len(result) // 3)
        fade_out_n    = min(int(sr * fade_out_sec), len(result) // 3)

        result[:fade_in_n]  *= np.linspace(0.0, 1.0, fade_in_n)

        fade_out_curve = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_out_n)))
        result[-fade_out_n:] *= fade_out_curve

        return result

    def _normalize(self, signal: np.ndarray, target_peak: float = 0.8) -> np.ndarray:
        peak = np.max(np.abs(signal))
        if peak > 1e-6:
            signal = signal * (target_peak / peak)
        return np.clip(signal, -1.0, 1.0)

    def _adjust_duration(self, audio_path: Path, target_sec: float) -> None:
        try:
            audio, sr = self._load_wav(audio_path)
            target_n  = int(sr * target_sec)
            current_n = len(audio)

            if current_n < target_n:
                silence = np.zeros(target_n - current_n)
                audio   = np.concatenate([audio, silence])
            elif current_n > target_n + sr:
                audio = audio[:target_n]
                audio = self._apply_fade(audio, sr, fade_in_sec=0, fade_out_sec=0.3)

            self._save_wav(audio, audio_path, sr)
        except Exception as e:
            logger.debug(f"Adjust duration deshtoi (jo kritike): {e}")

    def _concatenate_audio(
        self,
        audio_paths: list[Path],
        output_path: Path,
        gap_sec: float = 0.3,
    ) -> None:
        segments = []
        sr       = self.sample_rate

        for path in audio_paths:
            if not path.exists():
                logger.debug(f"Audio file mungon: {path}, duke shtuar silence")
                segments.append(np.zeros(int(sr * LLM_CONFIG["scene_duration_sec"])))
                continue
            try:
                audio, file_sr = self._load_wav(path)
                if file_sr != sr:
                    audio = self._resample(audio, file_sr, sr)
                segments.append(audio)
            except Exception as e:
                logger.warning(f"Nuk u lexua audio {path}: {e}")
                segments.append(np.zeros(int(sr * 4)))

        if not segments:
            self._generate_silence(output_path, 20)
            return

        gap   = np.zeros(int(sr * gap_sec))
        parts = []
        for i, seg in enumerate(segments):
            parts.append(seg)
            if i < len(segments) - 1:
                parts.append(gap)

        combined = np.concatenate(parts)
        self._save_wav(combined, output_path, sr)

    def _mix_audio(
        self,
        narration_path: Path,
        music_path: Path,
        output_path: Path,
    ) -> None:
        sr = self.sample_rate

        narration = np.zeros(sr)
        music     = np.zeros(sr)

        if narration_path.exists():
            narration, n_sr = self._load_wav(narration_path)
            if n_sr != sr:
                narration = self._resample(narration, n_sr, sr)

        if music_path.exists():
            music, m_sr = self._load_wav(music_path)
            if m_sr != sr:
                music = self._resample(music, m_sr, sr)

        target_len = max(len(narration), len(music))
        narration  = self._pad_or_trim(narration, target_len)
        music      = self._pad_or_trim(music, target_len)

        voice_vol = AUDIO_CONFIG.get("voice_volume", 1.0)
        music_vol = AUDIO_CONFIG.get("music_volume", 0.3)

        mixed = narration * voice_vol + music * music_vol

        mixed = self._normalize(mixed, target_peak=0.85)

        self._save_wav(mixed, output_path, sr)
        logger.debug(
            f"Audio u miksua | "
            f"zë: {voice_vol:.0%} | muzikë: {music_vol:.0%} | "
            f"kohë: {len(mixed)/sr:.1f}s"
        )

    def _save_wav(
        self,
        signal: np.ndarray,
        path: Path,
        sample_rate: Optional[int] = None,
    ) -> None:
        """Ruaj numpy array si WAV file 16-bit."""
        sr   = sample_rate or self.sample_rate
        data = (signal * 32767).astype(np.int16)

        with wave.open(str(path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(data.tobytes())

    def _load_wav(self, path: Path) -> tuple[np.ndarray, int]:
        with wave.open(str(path), 'r') as wf:
            sr       = wf.getframerate()
            n_frames = wf.getnframes()
            raw      = wf.readframes(n_frames)
            n_ch     = wf.getnchannels()
            sampw    = wf.getsampwidth()

        if sampw == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        elif sampw == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483647.0
        else:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0

        if n_ch == 2:
            data = data.reshape(-1, 2).mean(axis=1)

        return data, sr

    def _generate_silence(self, path: Path, duration_sec: float) -> None:
        n      = int(self.sample_rate * duration_sec)
        signal = np.zeros(n, dtype=np.float32)
        self._save_wav(signal, path, self.sample_rate)

    def _generate_tone_placeholder(
        self,
        path: Path,
        duration_sec: float,
        freq: float = 440.0,
    ) -> None:
        sr  = self.sample_rate
        n   = int(sr * duration_sec)
        t   = np.linspace(0, duration_sec, n, endpoint=False)

        signal  = np.sin(2 * np.pi * freq * t) * 0.05
        signal  = self._apply_fade(signal, sr, fade_in_sec=0.1, fade_out_sec=0.3)

        self._save_wav(signal, path, sr)
        logger.debug(f"Tone placeholder u gjenerua: {path.name} ({duration_sec:.1f}s)")

    def _resample(
        self,
        signal: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resamplo audio nga orig_sr në target_sr (linear interpolation)."""
        if orig_sr == target_sr:
            return signal
        ratio    = target_sr / orig_sr
        new_len  = int(len(signal) * ratio)
        old_idx  = np.linspace(0, len(signal) - 1, new_len)
        new_sig  = np.interp(old_idx, np.arange(len(signal)), signal)
        return new_sig.astype(np.float32)

    def _pad_or_trim(self, signal: np.ndarray, target_len: int) -> np.ndarray:
        """Shto silence ose pritë sinjali për ta bërë target_len."""
        if len(signal) < target_len:
            return np.concatenate([
                signal,
                np.zeros(target_len - len(signal), dtype=np.float32)
            ])
        return signal[:target_len]