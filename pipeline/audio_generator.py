import math
import struct
import wave
import asyncio
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import AUDIO_CONFIG
from utils.logger import get_logger
from utils.exceptions import AudioGenerationError

logger = get_logger(__name__)

class AudioGenerator:
    def __init__(self, language: str = "English"):
        self.language = language
        self.sample_rate = 22050
        self._load_model()

    def _load_model(self) -> None:
        """For V1, edge-tts is an online/on-demand engine, so no local model weight loading is required into RAM/VRAM."""
        try:
            import edge_tts
            logger.success("Edge-TTS engine successfully initialized.")
        except ImportError:
            logger.warning(
                "edge-tts is not installed. System will fall back to silent/tone placeholders.\n"
                "Install it using: pip install edge-tts"
            )

    def generate(
        self,
        story: dict,
        output_dir: Path,
        language: str = "English",
    ) -> Path:
        self.language = language
        scenes = story.get("scenes", [])
        if not scenes:
            raise AudioGenerationError("The story contains no scenes; audio generation aborted.")

        logger.debug(f"Generating audio assets for {len(scenes)} scenes")
        narration_paths = []
        total_story_duration = 0.0

        for i, scene in enumerate(scenes):
            narration = scene.get("narration", "").strip()
            narration_path = output_dir / f"narration_scene_{i+1:02d}.wav"
            
            if not narration:
                logger.debug(f"Scene {i+1} has no narration text. Creating a 5.0 second silence card.")
                self._generate_silence(narration_path, 5.0)
                scene["duration_sec"] = 5.0
                narration_paths.append(narration_path)
                total_story_duration += 5.0
                continue

            logger.step(f"TTS Process [Scene {i+1}/{len(scenes)}]: {narration[:50]}...")

            try:
                # Execute the asynchronous edge-tts call synchronously using asyncio loop runner
                asyncio.run(self._generate_edge_tts(narration, narration_path))
                
                # Dynamically retrieve exact audio wave metadata file duration
                duration = self._get_wav_duration(narration_path)
                scene["duration_sec"] = duration  # Stored back into the scene dictionary for future steps
                total_story_duration += duration
                
                narration_paths.append(narration_path)
            except Exception as e:
                logger.warning(f"TTS engine failed on scene {i+1}: {e}. Generating fallback silence.")
                self._generate_silence(narration_path, 4.0)
                scene["duration_sec"] = 4.0
                total_story_duration += 4.0
                narration_paths.append(narration_path)

        # 1. Stitch and consolidate scene-by-scene voice overs
        narration_full_path = output_dir / "narration_full.wav"
        self._concatenate_audio(narration_paths, narration_full_path)

        # 2. Procedurally generate copyright-free background music matching the exact duration
        music_path = output_dir / "background_music.wav"
        theme = story.get("theme", "adventure")
        self._generate_background_music(
            output_path=music_path,
            duration_sec=total_story_duration + 2.0,
            theme=theme,
        )

        # 3. Mix the composite voice-over track and the procedural synth ambient pad together
        final_path = output_dir / "final_audio.wav"
        self._mix_audio(
            narration_path=narration_full_path,
            music_path=music_path,
            output_path=final_path,
        )

        # Write overall length attributes for application flow pipelines
        story["total_duration_sec"] = total_story_duration

        logger.success(f"Final combined audio production bundle outputted to: {final_path} ({total_story_duration:.1f}s)")
        return final_path

    async def _generate_edge_tts(self, text: str, output_path: Path) -> None:
        """Utilizes Microsoft Edge's downstream Azure Neural voices."""
        import edge_tts
        
        # Determine specific voice model string configurations depending on runtime settings
        if self.language.lower() in ["albanian", "shqip"]:
            voice = "sq-AL-AnilaNeural"  # Professional, natural-sounding native Albanian female voice profile
        else:
            voice = "en-US-EmmaNeural"   # Warm, slow-paced storybook narrator profile for English

        communicate = edge_tts.Communicate(text, voice, rate="+0%")
        await communicate.save(str(output_path))

    def _get_wav_duration(self, path: Path) -> float:
        with wave.open(str(path), 'r') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)

    def _generate_background_music(
        self,
        output_path: Path,
        duration_sec: float,
        theme: str = "adventure",
    ) -> None:
        sr = self.sample_rate
        n = int(sr * duration_sec)
        t = np.linspace(0, duration_sec, n, endpoint=False)

        melody = self._compose_melody(t, theme, sr)
        harmony = self._compose_harmony(t, theme)
        bass = self._compose_bass(t, theme)
        reverb = self._apply_simple_reverb(melody, sr, delay_ms=80)

        music = (
            melody * 0.45 +
            harmony * 0.25 +
            bass * 0.15 +
            reverb * 0.15
        )

        music = self._apply_fade(music, sr, fade_in_sec=1.5, fade_out_sec=3.0)
        music = self._normalize(music, target_peak=AUDIO_CONFIG.get("music_volume", 0.25))
        self._save_wav(music, output_path, sr)

    def _compose_melody(self, t: np.ndarray, theme: str, sr: int) -> np.ndarray:
        scales = {
            "adventure": [261.63, 293.66, 329.63, 392.00, 440.00, 523.25],
            "magical": [261.63, 311.13, 392.00, 466.16, 523.25, 622.25],
            "happy": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00],
            "mysterious": [261.63, 277.18, 311.13, 369.99, 415.30, 466.16],
            "heroic": [261.63, 329.63, 392.00, 523.25, 659.25, 783.99],
            "exciting": [329.63, 392.00, 493.88, 587.33, 659.25, 783.99],
        }
        tempos = {
            "adventure": 140, "magical": 80, "happy": 120,
            "mysterious": 60, "heroic": 130, "exciting": 150,
        }

        scale = scales.get(theme, scales["happy"])
        tempo = tempos.get(theme, 100)
        beat = 60.0 / tempo
        total_dur = t[-1] + 1.0 / sr
        signal = np.zeros_like(t)

        rng = np.random.default_rng(seed=42)
        n_notes = int(total_dur / beat) + 1

        for i in range(n_notes):
            note_start = i * beat
            note_dur = beat * rng.choice([0.5, 1.0, 1.5, 2.0], p=[0.3, 0.4, 0.2, 0.1])
            freq = rng.choice(scale) * rng.choice([1.0, 2.0])

            mask = (t >= note_start) & (t < note_start + note_dur)
            if not np.any(mask):
                continue

            t_note = t[mask] - note_start
            envelope = np.exp(-3.0 * t_note / note_dur)
            signal[mask] += np.sin(2 * np.pi * freq * t[mask]) * envelope * 0.6

        return signal

    def _compose_harmony(self, t: np.ndarray, theme: str) -> np.ndarray:
        chord_sets = {
            "adventure": [(261.63, 329.63, 392.00), (349.23, 440.00, 523.25)],
            "magical": [(261.63, 311.13, 392.00), (220.00, 261.63, 329.63)],
            "happy": [(261.63, 329.63, 392.00), (293.66, 369.99, 440.00)],
            "mysterious": [(130.81, 155.56, 196.00), (116.54, 138.59, 174.61)],
            "heroic": [(261.63, 329.63, 392.00), (349.23, 440.00, 523.25)],
            "exciting": [(329.63, 415.30, 493.88), (293.66, 369.99, 440.00)],
        }
        chords = chord_sets.get(theme, chord_sets["happy"])
        signal = np.zeros_like(t)
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
            "adventure": [65.41, 73.42, 82.41, 98.00],
            "magical": [65.41, 69.30, 82.41, 87.31],
            "happy": [65.41, 87.31, 98.00, 73.42],
            "mysterious": [32.70, 36.71, 41.20, 43.65],
            "heroic": [65.41, 87.31, 130.81, 98.00],
            "exciting": [82.41, 98.00, 110.00, 123.47],
        }
        notes = bass_notes.get(theme, bass_notes["happy"])
        signal = np.zeros_like(t)
        dur = 1.5

        for i, note in enumerate(notes * 50):
            start = i * dur
            if start >= t[-1]:
                break
            mask = (t >= start) & (t < start + dur)
            if not np.any(mask):
                continue
            t_local = t[mask] - start
            envelope = np.exp(-1.5 * t_local / dur)
            signal[mask] += (
                np.sin(2 * np.pi * note * t[mask]) * envelope * 0.5
                + np.sin(2 * np.pi * note * 2 * t[mask]) * envelope * 0.15
            )
        return signal

    def _apply_simple_reverb(self, signal: np.ndarray, sr: int, delay_ms: float = 80, decay: float = 0.3) -> np.ndarray:
        delay_samples = int(sr * delay_ms / 1000)
        reverb = np.zeros_like(signal)
        if delay_samples < len(signal):
            reverb[delay_samples:] = signal[:-delay_samples] * decay
        return reverb

    def _apply_fade(self, signal: np.ndarray, sr: int, fade_in_sec: float = 1.0, fade_out_sec: float = 2.0) -> np.ndarray:
        result = signal.copy()
        if len(result) < 100:
            return result
        fade_in_n = min(int(sr * fade_in_sec), len(result) // 3)
        fade_out_n = min(int(sr * fade_out_sec), len(result) // 3)

        if fade_in_n > 0:
            result[:fade_in_n] *= np.linspace(0.0, 1.0, fade_in_n)
        if fade_out_n > 0:
            fade_out_curve = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_out_n)))
            result[-fade_out_n:] *= fade_out_curve
        return result

    def _normalize(self, signal: np.ndarray, target_peak: float = 0.8) -> np.ndarray:
        peak = np.max(np.abs(signal))
        if peak > 1e-6:
            signal = signal * (target_peak / peak)
        return np.clip(signal, -1.0, 1.0)

    def _concatenate_audio(self, audio_paths: list[Path], output_path: Path, gap_sec: float = 0.4) -> None:
        segments = []
        sr = self.sample_rate
        gap = np.zeros(int(sr * gap_sec))

        for path in audio_paths:
            if not path.exists():
                continue
            try:
                audio, file_sr = self._load_wav(path)
                if file_sr != sr:
                    audio = self._resample(audio, file_sr, sr)
                segments.append(audio)
            except Exception as e:
                logger.warning(f"Error encountered reading raw WAV block {path.name}: {e}")

        if not segments:
            self._generate_silence(output_path, 5.0)
            return

        parts = []
        for i, seg in enumerate(segments):
            parts.append(seg)
            if i < len(segments) - 1:
                parts.append(gap)

        combined = np.concatenate(parts)
        self._save_wav(combined, output_path, sr)

    def _mix_audio(self, narration_path: Path, music_path: Path, output_path: Path) -> None:
        sr = self.sample_rate
        narration, n_sr = self._load_wav(narration_path) if narration_path.exists() else (np.zeros(sr), sr)
        music, m_sr = self._load_wav(music_path) if music_path.exists() else (np.zeros(sr), sr)

        if n_sr != sr: narration = self._resample(narration, n_sr, sr)
        if m_sr != sr: music = self._resample(music, m_sr, sr)

        target_len = max(len(narration), len(music))
        narration = self._pad_or_trim(narration, target_len)
        music = self._pad_or_trim(music, target_len)

        voice_vol = AUDIO_CONFIG.get("voice_volume", 1.0)
        music_vol = AUDIO_CONFIG.get("music_volume", 0.25)

        mixed = narration * voice_vol + music * music_vol
        mixed = self._normalize(mixed, target_peak=0.85)

        self._save_wav(mixed, output_path, sr)

    def _save_wav(self, signal: np.ndarray, path: Path, sample_rate: Optional[int] = None) -> None:
        sr = sample_rate or self.sample_rate
        data = (signal * 32767).astype(np.int16)
        with wave.open(str(path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(data.tobytes())

    def _load_wav(self, path: Path) -> tuple[np.ndarray, int]:
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

    def _generate_silence(self, path: Path, duration_sec: float) -> None:
        n = int(self.sample_rate * duration_sec)
        signal = np.zeros(n, dtype=np.float32)
        self._save_wav(signal, path, self.sample_rate)

    def _resample(self, signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return signal
        ratio = target_sr / orig_sr
        new_len = int(len(signal) * ratio)
        old_idx = np.linspace(0, len(signal) - 1, new_len)
        new_sig = np.interp(old_idx, np.arange(len(signal)), signal)
        return new_sig.astype(np.float32)

    def _pad_or_trim(self, signal: np.ndarray, target_len: int) -> np.ndarray:
        if len(signal) < target_len:
            return np.concatenate([signal, np.zeros(target_len - len(signal), dtype=np.float32)])
        return signal[:target_len]
