"""Pluggable ASR backend abstraction for MacWhisper.

Provides the ASRBackend base class and shared result types. The Qwen3-ASR
client (qwen_backend.QwenBackend) is the implementation the app uses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# ── Result types ─────────────────────────────────────────

@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[WordTimestamp] = field(default_factory=list)
    no_speech_prob: float = 0.0


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str = ""

    def all_words(self) -> list[WordTimestamp]:
        """Flatten all word timestamps from all segments."""
        words = []
        for seg in self.segments:
            words.extend(seg.words)
        return words


# ── Abstract base class ──────────────────────────────────

class ASRBackend(ABC):
    """Abstract interface for speech-to-text backends."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        """Transcribe audio with word-level timestamps.

        Args:
            audio: float32 numpy array, mono, at the backend's expected
                   sample rate (typically 16 kHz).
            language: ISO language code (e.g. "zh", "en") or None for auto.
            initial_prompt: Context prompt for the decoder.
            task: "transcribe" or "translate".

        Returns:
            TranscriptionResult with text, segments, and word timestamps.
        """
        ...
