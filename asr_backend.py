"""Pluggable ASR backend abstraction for MacWhisper.

Provides a base class and an mlx-whisper implementation.  Other backends
(whisper.cpp, faster-whisper, WhisperKit) can be added by subclassing
ASRBackend.
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


# ── mlx-whisper implementation ───────────────────────────

class MLXWhisperBackend(ASRBackend):
    """ASR backend using mlx-whisper (Apple Silicon optimized)."""

    def __init__(self, model_repo: str = "mlx-community/whisper-small-mlx"):
        self.model_repo = model_repo

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        import mlx_whisper

        kwargs: dict = {
            "path_or_hf_repo": self.model_repo,
            "task": task,
            "word_timestamps": True,
            "condition_on_previous_text": False,
            # Anti-hallucination: limit decoder output length.
            # Normal 5s audio produces 20-40 tokens; 100 gives 2-3x headroom
            # while preventing 200+ token hallucination loops (default is 224).
            "sample_len": 100,
        }
        if language is not None:
            kwargs["language"] = language
        if initial_prompt is not None:
            kwargs["initial_prompt"] = initial_prompt

        raw = mlx_whisper.transcribe(audio, **kwargs)

        segments = []
        for seg in raw.get("segments", []):
            words = []
            for w in seg.get("words", []):
                words.append(WordTimestamp(
                    word=w.get("word", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    probability=w.get("probability", 1.0),
                ))
            segments.append(Segment(
                text=seg.get("text", ""),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                words=words,
                no_speech_prob=seg.get("no_speech_prob", 0.0),
            ))

        return TranscriptionResult(
            text=raw.get("text", "").strip(),
            segments=segments,
            language=raw.get("language", ""),
        )
