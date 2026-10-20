"""Qwen3 live subtitles for Meeting Mode — one segment at a time.

The meeting loop (meeting.py) owns audio capture, VAD and segment commits.
This processor gives it the same small interface as OnlineASRProcessor:

  insert_audio_chunk(audio)   add float32 audio to the current segment
  process_iter()              about once per `step` s, re-decode the WHOLE
                              current segment and return (white, gray), split
                              by StableText (the approach PTT uses);
                              None when it did not decode this time
  segment_close()             decode the segment once more, return its final
                              text, and start a new, empty segment
  segment_full()              the segment reached max_segment_s
  reset()

Qwen3 returns no word timestamps, so `committed` stays empty and the loop's
timestamp-based buffer trim is skipped; segment_close() clears the audio
itself. Re-decoding the whole segment keeps text consistent (Qwen3 has no
streaming mode); max_segment_s bounds the decode cost. A backend failure
raises QwenUnavailable so the meeting can fall back to Whisper.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import numpy as np

import qwen_text
from stream_text import StableText
from text_utils import convert_t2s

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class _NoWordBuffer:
    """What the meeting loop reads: is there gray text right now?"""

    def __init__(self):
        self.tail: list = []

    def peek_unconfirmed(self) -> list:
        return self.tail

    def reset(self) -> None:
        self.tail = []


class QwenSegmentProcessor:
    def __init__(self, backend, *, step: float = 1.0, min_audio_s: float = 0.8,
                 max_segment_s: float = 15.0, strip_fillers: bool = True,
                 tight: bool = True, now=time.monotonic):
        self.backend = backend
        self.step = step
        self.min_audio_s = min_audio_s
        self.max_segment_s = max_segment_s
        self.strip_fillers = strip_fillers
        self.tight = tight
        self._now = now
        # Interface read by meeting.py
        self.committed: list = []          # always empty: no word timestamps
        self.committed_history: list = []
        self.buffer_time_offset = 0.0
        self._last_committed_raw = ""
        self.transcript_buffer = _NoWordBuffer()
        self.last_debug: dict = {}
        self.reset()

    # ── segment state ──
    def reset(self) -> None:
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self._stable = StableText()
        self._next_decode = 0.0
        self._decoded_once = False
        self._had_speech = False
        self.transcript_buffer.reset()

    def insert_audio_chunk(self, audio: np.ndarray) -> None:
        self.audio_buffer = np.concatenate([self.audio_buffer,
                                            np.asarray(audio, dtype=np.float32)])

    def segment_seconds(self) -> float:
        return len(self.audio_buffer) / SAMPLE_RATE

    def mark_speech(self) -> None:
        """The meeting loop saw speech (VAD) in audio fed to this segment."""
        self._had_speech = True

    def has_pending_speech(self) -> bool:
        """Untranscribed SPEECH in this segment (not just silence) — at stop
        it gets one more decode; silence alone must not (it re-decoded and
        duplicated the previous sentence)."""
        return self._had_speech and self.segment_seconds() >= 0.3

    def segment_full(self) -> bool:
        return self.segment_seconds() >= self.max_segment_s

    # A pause inside a sentence must not cut off a 1-2 character fragment
    # (eval 2026-10-20). Short segments wait for the Whisper-length silence.
    MIN_CHARS_FOR_SHORT_SILENCE = 4
    LONG_SILENCE_MS = 2000

    def ready_to_close(self, silence_ms: int) -> bool:
        text = "".join(t.text for t in self.transcript_buffer.tail)
        chars = len("".join(text.split()))
        if chars >= self.MIN_CHARS_FOR_SHORT_SILENCE:
            return True
        return silence_ms >= self.LONG_SILENCE_MS

    # ── decoding ──
    def _decode(self) -> str:
        t0 = self._now()
        text = convert_t2s(self.backend.transcribe(self.audio_buffer).text or "")
        took = self._now() - t0
        self.last_debug = {"engine": "qwen3", "audio_s": round(self.segment_seconds(), 2),
                           "decode_s": round(took, 2)}
        logger.debug("Qwen meeting: %.1fs segment decoded in %.2fs",
                     self.segment_seconds(), took)
        return text

    def process_iter(self):
        if self.segment_seconds() < self.min_audio_s:
            return None
        t = self._now()
        if self._decoded_once and t < self._next_decode:
            return None
        text = self._decode()                  # raises QwenUnavailable on failure
        self._decoded_once = True
        self._next_decode = self._now() + self.step
        if qwen_text.is_qwen_hallucination(text):
            self._stable = StableText()
            self.transcript_buffer.tail = []
            return ("", "")
        self._stable.update(text)
        white = qwen_text.tighten_spacing(self._stable.confirmed_text) if self.tight \
            else self._stable.confirmed_text
        gray = qwen_text.tighten_spacing(self._stable.tail_text) if self.tight \
            else self._stable.tail_text
        # The loop treats any confirmed/gray text as "segment has text"
        self.transcript_buffer.tail = [SimpleNamespace(text=white + gray)] \
            if (white or gray) else []
        return (white, gray)

    def segment_close(self) -> str:
        """Final text of the current segment (one more full decode), then reset."""
        if self.segment_seconds() < 0.3:
            self.reset()
            return ""
        raw = self._decode()      # on failure the audio stays for the fallback
        seconds = self.segment_seconds()
        self.reset()
        text = qwen_text.postprocess(raw, strip_filler_words=self.strip_fillers,
                                     tight=self.tight)
        logger.info("Qwen meeting segment closed: %.1fs -> %d chars", seconds, len(text))
        self._last_committed_raw = text
        return text
