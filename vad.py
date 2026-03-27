"""Voice Activity Detection wrapper for MacWhisper.

Uses Silero VAD (loaded via torch.hub) for accurate speech/silence
boundary detection.  Falls back to simple RMS-based detection if
Silero is unavailable.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000


class VoiceActivityDetector:
    """Silero VAD wrapper with RMS fallback."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_ms: int = 500,
        min_speech_ms: int = 250,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.sample_rate = sample_rate
        self._model = None
        self._use_silero = False
        self._silence_samples = 0
        self._speech_samples = 0
        self._load_model()

    def _load_model(self):
        """Try to load Silero VAD; fall back to RMS if unavailable."""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            self._get_speech_timestamps = utils[0]
            self._use_silero = True
        except Exception as e:
            print(f"[WARN] Silero VAD unavailable, using RMS fallback: {e}")
            self._use_silero = False

    def is_speech(self, audio_chunk: np.ndarray, rms_threshold: float = 100.0) -> bool:
        """Check if an audio chunk contains speech.

        Args:
            audio_chunk: int16 or float32 numpy array.
            rms_threshold: RMS threshold for fallback mode.

        Returns:
            True if speech is detected.
        """
        if self._use_silero:
            return self._silero_is_speech(audio_chunk)
        return self._rms_is_speech(audio_chunk, rms_threshold)

    def _silero_is_speech(self, audio_chunk: np.ndarray) -> bool:
        import torch

        if audio_chunk.dtype == np.int16:
            audio_f32 = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_f32 = audio_chunk.astype(np.float32)

        audio_f32 = audio_f32.squeeze()

        # Silero VAD requires exactly 512 samples at 16kHz.
        # Split the chunk into 512-sample windows; return True if any has speech.
        window = 512
        if len(audio_f32) < window:
            # Pad short chunks with zeros
            padded = np.zeros(window, dtype=np.float32)
            padded[:len(audio_f32)] = audio_f32
            audio_f32 = padded

        for start in range(0, len(audio_f32) - window + 1, window):
            tensor = torch.from_numpy(audio_f32[start:start + window])
            confidence = self._model(tensor, self.sample_rate).item()
            if confidence >= self.threshold:
                return True
        return False

    def _rms_is_speech(self, audio_chunk: np.ndarray,
                       threshold: float) -> bool:
        audio = audio_chunk.squeeze().astype(np.float64)
        rms = np.sqrt(np.mean(audio ** 2))
        return rms >= threshold

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """Process a chunk and track speech/silence state.

        Returns:
            {"is_speech": bool, "silence_ms": int, "speech_ms": int}
        """
        chunk_samples = len(audio_chunk.squeeze())
        chunk_ms = int(chunk_samples * 1000 / self.sample_rate)

        if self.is_speech(audio_chunk):
            self._speech_samples += chunk_samples
            self._silence_samples = 0
        else:
            self._silence_samples += chunk_samples
            self._speech_samples = 0

        silence_ms = int(self._silence_samples * 1000 / self.sample_rate)
        speech_ms = int(self._speech_samples * 1000 / self.sample_rate)

        return {
            "is_speech": speech_ms > 0,
            "silence_ms": silence_ms,
            "speech_ms": speech_ms,
        }

    def is_speech_end(self) -> bool:
        """Return True if enough silence has accumulated to mark speech end."""
        silence_ms = int(self._silence_samples * 1000 / self.sample_rate)
        return silence_ms >= self.min_silence_ms

    def is_extended_silence(self, threshold_ms: int = 2000) -> bool:
        """Return True if silence exceeds a longer threshold.

        Used by Meeting Mode for paragraph breaks (longer than
        is_speech_end which is for segment commits at ~800ms).
        """
        silence_ms = int(self._silence_samples * 1000 / self.sample_rate)
        return silence_ms >= threshold_ms

    def is_active_speech(self) -> bool:
        """Return True if continuous speech meets the minimum speech threshold."""
        speech_ms = int(self._speech_samples * 1000 / self.sample_rate)
        return speech_ms >= self.min_speech_ms

    def reset(self):
        """Reset internal state for a new recording session."""
        self._silence_samples = 0
        self._speech_samples = 0
        if self._use_silero:
            self._model.reset_states()
