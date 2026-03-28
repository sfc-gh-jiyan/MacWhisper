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

import logging
import time
import numpy as np

from asr_backend import ASRBackend, TranscriptionResult

logger = logging.getLogger(__name__)
from text_utils import (
    BILINGUAL_PROMPT, convert_t2s, normalize_punctuation,
    strip_trailing_repetition, hallucination_reason,
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
        min_first_buffer_s: float = 2.0,
        max_buffer_s: float = 20.0,
        buffer_trimming: str = "segment",
        language: str | None = None,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.backend = backend
        self.vad = vad
        self.min_chunk_size = min_chunk_size
        self.min_first_buffer_s = min_first_buffer_s
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
        self._last_committed_raw: str = ""  # for post-commit echo detection

        # Timing
        self._last_process_time: float = 0.0
        self._iter_count: int = 0
        self.throttle: bool = True  # can disable for testing
        self._last_inference_ms: int = 0  # track last inference time for adaptive trimming
        self._last_trim_info: dict | None = None  # set by _maybe_trim_buffer

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

        # First iteration needs more audio for reliable language detection
        if self._iter_count == 0 and buffer_duration < self.min_first_buffer_s:
            return None

        # Throttle: ensure at least min_chunk_size between iterations
        # (can be disabled for testing via self.throttle = False)
        now = time.time()
        if self._last_process_time > 0 and self.throttle:
            elapsed = now - self._last_process_time
            if elapsed < self.min_chunk_size * 0.5:
                return None

        self._iter_count += 1
        t0 = time.time()

        # Build dynamic prompt from committed text
        prompt = self._build_prompt()

        # Run ASR — emergency cap if buffer grew far beyond max during
        # a previous blocking inference. Normal trimming happens in
        # _maybe_trim_buffer() after inference. This only triggers when
        # the buffer is >2x the max (i.e., lots of audio accumulated
        # while we were blocked).
        max_samples = int(self.max_buffer_s * 2 * self.sample_rate)
        if len(self.audio_buffer) > max_samples:
            target_samples = int(self.max_buffer_s * self.sample_rate)
            excess = len(self.audio_buffer) - target_samples
            self.audio_buffer = self.audio_buffer[excess:]
            self.buffer_time_offset += excess / self.sample_rate
            # Reset HypothesisBuffer — word positions shifted
            self.transcript_buffer.reset()
            # Re-populate with committed words in retained region
            for w in self.committed:
                if w[0] >= self.buffer_time_offset - 0.2:
                    self.transcript_buffer.committed_in_buffer.append(w)
                    self.transcript_buffer.buffer.append(w)

        result = self.backend.transcribe(
            self.audio_buffer,
            language=self.language,
            initial_prompt=prompt,
            task="transcribe",
        )

        inference_ms = int((time.time() - t0) * 1000)
        self._last_inference_ms = inference_ms

        # Extract word timestamps
        raw_words = self._extract_words(result)

        # Log inference results for debugging language/translation issues
        logger.debug(
            "iter=%d buf=%.1fs inf=%dms lang=%s words=%d raw=%.80s",
            self._iter_count, buffer_duration, inference_ms,
            result.language, len(raw_words),
            result.text.replace("\n", "\\n")[:80],
        )

        # Anti-hallucination layer 3: discard if word count is unreasonable
        # Normal speech is ~3-5 words/sec; >12 words/sec is hallucination
        max_reasonable_words = max(10, int(buffer_duration * 12))
        if len(raw_words) > max_reasonable_words:
            logger.warning(
                "Excessive words: %d for %.1fs audio — discarding",
                len(raw_words), buffer_duration,
            )
            self._last_process_time = time.time()
            confirmed_text = "".join(w[2] for w in self.committed)
            unconfirmed_text = "".join(w[2] for w in self.last_unconfirmed)
            self.last_debug = {
                "iter": self._iter_count, "buffer_s": round(buffer_duration, 1),
                "inference_ms": inference_ms, "raw_text": "(word count discard)",
                "confirmed_words": len(self.committed),
                "unconfirmed_words": len(self.last_unconfirmed),
                "newly_confirmed": 0, "trim": self._last_trim_info,
            }
            return (confirmed_text, unconfirmed_text)

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

        # Post-commit echo detection: if newly confirmed words just repeat
        # the tail of the previous segment's committed text, skip them.
        echo_dropped = 0
        if newly_confirmed and self._last_committed_raw:
            new_text = "".join(w[2] for w in newly_confirmed)
            # Check if the new text is a substring of the committed tail
            tail = self._last_committed_raw[-len(new_text) - 20:]
            if new_text in tail:
                echo_dropped = len(newly_confirmed)
                newly_confirmed = []
            else:
                # Partial echo: drop leading words that match committed tail
                kept = []
                running = ""
                for w in newly_confirmed:
                    running += w[2]
                    if running in tail:
                        echo_dropped += 1
                    else:
                        kept.append(w)
                newly_confirmed = kept
            if echo_dropped:
                # Clear echo state once non-echo content arrives
                # (only clear after full echo is consumed)
                if newly_confirmed:
                    self._last_committed_raw = ""

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
            "echo_dropped": echo_dropped,
            "trim": self._last_trim_info,
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

        logger.debug(
            "segment_close: force_confirmed=%d total_committed=%d text_len=%d text=%.80s",
            len(unconfirmed), len(self.committed), len(full_text),
            full_text.replace("\n", "\\n")[:80],
        )

        # Save committed text for post-commit echo detection
        self._last_committed_raw = full_text

        # Reset for next segment (keep committed words for history)
        self.transcript_buffer.reset()
        self.last_unconfirmed = []

        # Pre-populate HypothesisBuffer with committed words still in
        # retained audio region — prevents re-confirmation of old words
        # (same pattern as _maybe_trim_buffer)
        for w in self.committed:
            if w[0] >= self.buffer_time_offset - 0.2:
                self.transcript_buffer.committed_in_buffer.append(w)
                self.transcript_buffer.buffer.append(w)

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
        self._last_committed_raw = ""
        self._last_process_time = 0.0
        self._iter_count = 0
        self._last_inference_ms = 0
        self._last_trim_info = None
        self.last_debug = {}

    # ── Internal ─────────────────────────────────────────

    def _build_prompt(self) -> str:
        """Build dynamic init_prompt from committed text + bilingual hint."""
        committed_text = "".join(w[2] for w in self.committed)
        if committed_text:
            # Use last 200 chars of committed text as context
            suffix = committed_text[-200:]
            prompt = suffix + BILINGUAL_PROMPT
        else:
            prompt = BILINGUAL_PROMPT
        logger.debug(
            "prompt len=%d committed_chars=%d prompt=%.120s",
            len(prompt), len(committed_text),
            prompt.replace("\n", "\\n")[:120],
        )
        return prompt

    def _extract_words(self, result: TranscriptionResult) -> list[tuple[float, float, str]]:
        """Extract (start, end, word) tuples from transcription result."""
        words = []
        for seg in result.segments:
            for w in seg.words:
                text = normalize_punctuation(convert_t2s(w.word))
                if text.strip():
                    words.append((w.start, w.end, text))
                elif w.word.strip():
                    # Word became empty after normalization — log for diagnosis
                    logger.debug("Dropped word after normalize: %r", w.word)
            # Log segment-level newlines or suspicious content
            if "\n" in seg.text:
                logger.debug("Segment contains newline: %r", seg.text[:60])
        # Fallback: if no word timestamps, treat entire text as one "word"
        if not words and result.text.strip():
            text = normalize_punctuation(convert_t2s(result.text.strip()))
            start = result.segments[0].start if result.segments else 0.0
            end = result.segments[-1].end if result.segments else 0.0
            words.append((start, end, text))
        return words

    def _maybe_trim_buffer(self):
        """Trim audio buffer if it exceeds max_buffer_s.

        Uses adaptive threshold: if last inference was slow (>800ms),
        trim more aggressively to keep inference fast. If inference was
        very slow (>2s), force trim to 3s regardless.
        """
        buffer_duration = len(self.audio_buffer) / self.sample_rate

        # Adaptive: if inference was slow, use a tighter limit
        effective_max = self.max_buffer_s
        if self._last_inference_ms > 2000:
            # Very slow — emergency trim to 3s
            effective_max = 3.0
        elif self._last_inference_ms > 800:
            # Moderately slow — trim to 60% of current buffer
            effective_max = min(effective_max, buffer_duration * 0.6)
            effective_max = max(effective_max, 3.0)  # never trim below 3s

        if buffer_duration <= effective_max:
            self._last_trim_info = None
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
        old_offset = self.buffer_time_offset
        self.buffer_time_offset = trim_time

        self._last_trim_info = {
            "trimmed": True,
            "from_s": round(old_offset, 1),
            "to_s": round(trim_time, 1),
            "effective_max": round(effective_max, 1),
            "retained_s": round(len(self.audio_buffer) / self.sample_rate, 1),
        }

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
