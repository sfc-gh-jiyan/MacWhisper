"""Online ASR Processor with LocalAgreement for MacWhisper.

Core architecture inspired by ufal/whisper_streaming:
- Audio buffer continuously grows as new chunks arrive
- Each process_iter() call transcribes the full buffer with word timestamps
- HypothesisBuffer compares consecutive iterations word-by-word
- Only words confirmed by 2 consecutive iterations are output
- Buffer is trimmed at confirmed sentence boundaries to stay bounded

This gives us:
- Stable, non-flickering subtitles (only confirmed words displayed)
- Low latency (~2 × min_chunk_size for confirmation)
- High quality (full buffer context for each inference)
"""

from __future__ import annotations

import time
import numpy as np

from asr_backend import ASRBackend, TranscriptionResult
from text_utils import (
    BILINGUAL_PROMPT, convert_t2s, strip_trailing_repetition,
    hallucination_reason,
)

SAMPLE_RATE = 16000


# ── HypothesisBuffer ─────────────────────────────────────

class HypothesisBuffer:
    """Word-level consistency buffer: confirms words stable across 2 iterations.

    Inspired by ufal/whisper_streaming's HypothesisBuffer.
    Words are represented as (start, end, text) tuples.
    """

    def __init__(self):
        self.committed_in_buffer: list[tuple[float, float, str]] = []
        self.buffer: list[tuple[float, float, str]] = []
        self.new: list[tuple[float, float, str]] = []

    def insert(self, new_words: list[tuple[float, float, str]], offset: float = 0.0):
        """Insert a new transcription result and compare with previous.

        Words at matching positions that are identical across two iterations
        are moved to committed. Changed words stay in buffer.
        """
        # Apply time offset
        new_words = [(s + offset, e + offset, w) for s, e, w in new_words]

        if not self.buffer:
            # First iteration: just store, nothing to compare
            self.buffer = list(new_words)
            return

        # Compare new_words with self.buffer word by word
        # Find the longest matching prefix
        min_len = min(len(self.buffer), len(new_words))
        match_end = 0
        for i in range(min_len):
            if self.buffer[i][2].strip() == new_words[i][2].strip():
                match_end = i + 1
            else:
                break

        # Matched words → confirmed (newly confirmed only)
        already_committed = len(self.committed_in_buffer)
        for i in range(already_committed, match_end):
            self.new.append(new_words[i])
            self.committed_in_buffer.append(new_words[i])

        # Update buffer to new words for next comparison
        self.buffer = list(new_words)

    def flush(self) -> list[tuple[float, float, str]]:
        """Return newly confirmed words and clear the output queue."""
        flushed = list(self.new)
        self.new = []
        return flushed

    def peek_unconfirmed(self) -> list[tuple[float, float, str]]:
        """Return words in buffer that haven't been confirmed yet."""
        n_committed = len(self.committed_in_buffer)
        return self.buffer[n_committed:]

    def reset(self):
        """Reset buffer state for a new segment."""
        self.committed_in_buffer = []
        self.buffer = []
        self.new = []


# ── OnlineASRProcessor ───────────────────────────────────

class OnlineASRProcessor:
    """Online ASR with LocalAgreement, VAD integration, and buffer management.

    Usage:
        proc = OnlineASRProcessor(backend, vad, min_chunk_size=0.7)
        proc.insert_audio_chunk(audio_float32)
        result = proc.process_iter()  # returns (confirmed, unconfirmed) or None
        words = proc.get_confirmed_words()  # for SRT export
        text = proc.segment_close()  # force-close segment
    """

    def __init__(
        self,
        backend: ASRBackend,
        vad=None,
        min_chunk_size: float = 0.7,
        max_buffer_s: float = 20.0,
        buffer_trimming: str = "segment",
        language: str | None = None,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.backend = backend
        self.vad = vad
        self.min_chunk_size = min_chunk_size
        self.max_buffer_s = max_buffer_s
        self.buffer_trimming = buffer_trimming
        self.language = language
        self.sample_rate = sample_rate

        # Audio buffer (float32, mono)
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset: float = 0.0

        # Transcript state
        self.transcript_buffer = HypothesisBuffer()
        self.committed: list[tuple[float, float, str]] = []
        self.last_unconfirmed: list[tuple[float, float, str]] = []
        self._committed_end_time: float = 0.0  # end time of last committed word

        # Timing
        self._last_process_time: float = 0.0
        self._iter_count: int = 0
        self.throttle: bool = True  # can disable for testing

        # Debug info (for logging)
        self.last_debug: dict = {}

    def insert_audio_chunk(self, chunk: np.ndarray):
        """Append float32 audio to the processing buffer."""
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        chunk = chunk.squeeze()
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])

    def process_iter(self) -> tuple[str, str] | None:
        """Run one transcription iteration on the full audio buffer.

        Returns:
            (confirmed_text, unconfirmed_text) or None if audio too short.
            confirmed_text: words confirmed by LocalAgreement (display in white)
            unconfirmed_text: latest tail not yet confirmed (display in gray)
        """
        buffer_duration = len(self.audio_buffer) / self.sample_rate

        # Not enough audio yet
        if buffer_duration < self.min_chunk_size:
            return None

        # Throttle: ensure at least min_chunk_size between iterations
        # (can be disabled for testing via self.throttle = False)
        now = time.time()
        if self._last_process_time > 0 and self.throttle:
            elapsed = now - self._last_process_time
            if elapsed < self.min_chunk_size * 0.8:
                return None

        self._iter_count += 1
        t0 = time.time()

        # Build dynamic prompt from committed text
        prompt = self._build_prompt()

        # Run ASR
        result = self.backend.transcribe(
            self.audio_buffer,
            language=self.language,
            initial_prompt=prompt,
            task="transcribe",
        )

        inference_ms = int((time.time() - t0) * 1000)

        # Extract word timestamps
        raw_words = self._extract_words(result)

        # Post-process: t2s conversion, hallucination filter
        raw_text = result.text
        raw_text = convert_t2s(raw_text)
        if prompt and raw_text.startswith(BILINGUAL_PROMPT):
            raw_text = raw_text[len(BILINGUAL_PROMPT):].strip()
        raw_text = strip_trailing_repetition(raw_text)
        if not raw_text or hallucination_reason(raw_text):
            self._last_process_time = time.time()
            # Return current state (don't reset display to empty)
            confirmed_text = "".join(w[2] for w in self.committed)
            unconfirmed_text = "".join(w[2] for w in self.last_unconfirmed)
            return (confirmed_text, unconfirmed_text)

        # Insert into HypothesisBuffer for LocalAgreement
        self.transcript_buffer.insert(raw_words, offset=self.buffer_time_offset)

        # Flush confirmed words
        newly_confirmed = self.transcript_buffer.flush()
        self.committed.extend(newly_confirmed)
        if newly_confirmed:
            self._committed_end_time = newly_confirmed[-1][1]  # end time

        # Get unconfirmed tail
        self.last_unconfirmed = self.transcript_buffer.peek_unconfirmed()

        # Build text outputs
        confirmed_text = "".join(w[2] for w in self.committed)
        unconfirmed_text = "".join(w[2] for w in self.last_unconfirmed)

        # Buffer trimming: keep buffer bounded
        self._maybe_trim_buffer()

        self._last_process_time = time.time()

        # Debug info
        self.last_debug = {
            "iter": self._iter_count,
            "buffer_s": round(buffer_duration, 1),
            "inference_ms": inference_ms,
            "raw_text": raw_text[:50],
            "confirmed_words": len(self.committed),
            "unconfirmed_words": len(self.last_unconfirmed),
            "newly_confirmed": len(newly_confirmed),
        }

        return (confirmed_text, unconfirmed_text)

    def segment_close(self) -> str:
        """Force-confirm all remaining words and return full segment text.

        Call when VAD detects speech end or at safety timeout.
        """
        # Confirm everything still in buffer
        unconfirmed = self.transcript_buffer.peek_unconfirmed()
        self.committed.extend(unconfirmed)
        full_text = "".join(w[2] for w in self.committed)

        # Reset for next segment (keep committed words for history)
        self.transcript_buffer.reset()
        self.last_unconfirmed = []

        return full_text

    def get_confirmed_words(self) -> list[tuple[float, float, str]]:
        """Return all confirmed words with timestamps for SRT export."""
        return list(self.committed)

    def get_all_words(self) -> list[tuple[float, float, str]]:
        """Return confirmed + unconfirmed words."""
        return list(self.committed) + list(self.last_unconfirmed)

    def reset(self):
        """Full reset for a new recording session."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0.0
        self.transcript_buffer.reset()
        self.committed = []
        self.last_unconfirmed = []
        self._committed_end_time = 0.0
        self._last_process_time = 0.0
        self._iter_count = 0
        self.last_debug = {}

    # ── Internal ─────────────────────────────────────────

    def _build_prompt(self) -> str:
        """Build dynamic init_prompt from committed text + bilingual hint."""
        committed_text = "".join(w[2] for w in self.committed)
        if committed_text:
            # Use last 200 chars of committed text as context
            suffix = committed_text[-200:]
            return suffix + BILINGUAL_PROMPT
        return BILINGUAL_PROMPT

    def _extract_words(self, result: TranscriptionResult) -> list[tuple[float, float, str]]:
        """Extract (start, end, word) tuples from transcription result."""
        words = []
        for seg in result.segments:
            for w in seg.words:
                text = convert_t2s(w.word)
                if text.strip():
                    words.append((w.start, w.end, text))
        # Fallback: if no word timestamps, treat entire text as one "word"
        if not words and result.text.strip():
            text = convert_t2s(result.text.strip())
            start = result.segments[0].start if result.segments else 0.0
            end = result.segments[-1].end if result.segments else 0.0
            words.append((start, end, text))
        return words

    def _maybe_trim_buffer(self):
        """Trim audio buffer if it exceeds max_buffer_s.

        Strategy: find the timestamp of the last confirmed sentence boundary
        and trim everything before it.
        """
        buffer_duration = len(self.audio_buffer) / self.sample_rate
        if buffer_duration <= self.max_buffer_s:
            return

        # Find trim point: last confirmed word's end time, or half the buffer
        if self.committed:
            # Find a sentence boundary in committed text
            trim_time = self._find_trim_point()
        else:
            # No confirmed words: trim to keep last max_buffer_s/2
            trim_time = buffer_duration - self.max_buffer_s / 2

        if trim_time <= self.buffer_time_offset:
            return

        # Trim audio
        trim_samples = int((trim_time - self.buffer_time_offset) * self.sample_rate)
        trim_samples = min(trim_samples, len(self.audio_buffer))

        self.audio_buffer = self.audio_buffer[trim_samples:]
        self.buffer_time_offset = trim_time

        # Reset HypothesisBuffer after trim — word positions change completely
        # so old buffer comparison would never match. Committed words are safe
        # in self.committed already.
        self.transcript_buffer.reset()

        # Pre-populate committed_in_buffer with committed words that fall
        # within the retained audio region. This prevents the HypothesisBuffer
        # from re-confirming words that were already committed before the trim.
        retained_start = trim_time
        for w in self.committed:
            if w[0] >= retained_start - 0.2:  # word start >= retained region
                self.transcript_buffer.committed_in_buffer.append(w)
                self.transcript_buffer.buffer.append(w)

    def _find_trim_point(self) -> float:
        """Find a good trim point in committed text (sentence boundary).

        Strategy: find the LAST sentence boundary in committed words,
        then trim up to that point. This maximizes the amount of audio
        we discard (reducing inference time on retained buffer).
        """
        if not self.committed:
            return self.buffer_time_offset

        # Look for the last sentence-ending punctuation
        sentence_enders = set('。！？.!?\n')
        best_idx = -1
        for i, word in enumerate(self.committed):
            for ch in word[2]:
                if ch in sentence_enders:
                    best_idx = i

        if best_idx >= 0 and best_idx < len(self.committed) - 1:
            # Trim up to the word after the sentence boundary
            return self.committed[best_idx][1]  # end time of boundary word

        # No sentence boundary found: trim to keep latest 60% of buffer
        buffer_duration = len(self.audio_buffer) / self.sample_rate
        return self.buffer_time_offset + buffer_duration * 0.4
