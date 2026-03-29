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

        Uses sliding-window alignment: instead of requiring position-0 match,
        searches for the best overlap between old unconfirmed words and the
        new transcription. This handles pre-inference trim shifting the audio
        start — Whisper sees different prefixes each iteration, but the
        overlapping portion in the middle stays stable and can be confirmed.
        """
        # Apply time offset
        new_words = [(s + offset, e + offset, w) for s, e, w in new_words]

        if not self.buffer:
            # First iteration: just store, nothing to compare
            self.buffer = list(new_words)
            return

        # Extract unconfirmed words from previous buffer
        already_committed = len(self.committed_in_buffer)
        old_unconfirmed = self.buffer[already_committed:]
        old_texts = [w[2].strip() for w in old_unconfirmed]
        new_texts = [w[2].strip() for w in new_words]

        # Find best alignment: search for overlapping words between
        # old unconfirmed and new_words. Try all (old_start, new_start) pairs,
        # find the longest consecutive match of >= 2 words.
        best_old_start = -1
        best_new_start = -1
        best_match_len = 0

        if old_texts:
            for i in range(len(old_texts)):
                for j in range(len(new_texts)):
                    if old_texts[i] != new_texts[j]:
                        continue
                    # Found a potential start — count consecutive matches
                    match_len = 0
                    while (i + match_len < len(old_texts)
                           and j + match_len < len(new_texts)
                           and old_texts[i + match_len] == new_texts[j + match_len]):
                        match_len += 1
                    if match_len > best_match_len:
                        best_old_start = i
                        best_new_start = j
                        best_match_len = match_len
                # Stop after first old_start that gives a match >= 2
                if best_match_len >= 2:
                    break

        # Require at least 2 consecutive matching words to confirm
        # (prevents false positives from coincidental single-word matches).
        # Exception: if old_unconfirmed has only 1 word, 1 match is enough.
        min_required = min(2, len(old_texts))
        if best_match_len >= min_required:
            # Alignment found: confirm matched words from new_words
            for k in range(best_match_len):
                w = new_words[best_new_start + k]
                self.new.append(w)

        # Update buffer to new words for next comparison.
        # committed_in_buffer tracks how many leading words in buffer are
        # "done" (either confirmed or before the confirmed region). This
        # ensures peek_unconfirmed() returns only the true tail, and the
        # next insert() only searches the genuinely new portion.
        if best_match_len >= min_required:
            confirmed_end = best_new_start + best_match_len
            self.committed_in_buffer = list(new_words[:confirmed_end])
        else:
            # No alignment — keep old committed count if buffer prefix
            # still matches (handles stable prefix case), else clear.
            self.committed_in_buffer = []
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
        # committed_history: append-only full history of all confirmed words.
        # Unlike self.committed (which gets trimmed by _maybe_trim_buffer),
        # this list never loses words. Used for output text, SRT export,
        # prompt building, and dedup checks.
        self.committed_history: list[tuple[float, float, str]] = []
        self.last_unconfirmed: list[tuple[float, float, str]] = []
        self._committed_end_time: float = 0.0  # end time of last committed word
        self._last_committed_raw: str = ""  # for post-commit echo detection

        # Timing
        self._last_process_time: float = 0.0
        self._iter_count: int = 0
        self.throttle: bool = True  # can disable for testing
        self._last_inference_ms: int = 0  # track last inference time for adaptive trimming
        self._last_trim_info: dict | None = None  # set by _maybe_trim_buffer
        self._trim_cooldown: int = 0  # iterations to skip _maybe_trim_buffer after trim

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

        # Skip inference if buffer tail is silent (prevents hallucination on silence)
        tail_samples = int(0.5 * self.sample_rate)
        if len(self.audio_buffer) > tail_samples:
            tail_rms = float(np.sqrt(np.mean(self.audio_buffer[-tail_samples:] ** 2)))
            if tail_rms < 0.003:  # ~100 in int16 scale
                logger.debug("skip: buffer tail silent (rms=%.5f)", tail_rms)
                self._last_process_time = time.time()
                confirmed_text = "".join(w[2] for w in self.committed_history)
                unconfirmed_text = "".join(w[2] for w in self.last_unconfirmed)
                return (confirmed_text, unconfirmed_text)

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

        # Run ASR — pre-inference trim: keep buffer within max_buffer_s
        # to avoid sending oversized buffers (which accumulate during slow
        # inference). The 2x emergency cap below handles extreme cases.
        buffer_duration = len(self.audio_buffer) / self.sample_rate
        if buffer_duration > self.max_buffer_s:
            excess = len(self.audio_buffer) - int(self.max_buffer_s * self.sample_rate)
            self.audio_buffer = self.audio_buffer[excess:]
            self.buffer_time_offset += excess / self.sample_rate
            # Do NOT touch HypothesisBuffer here. Pre-inference trim fires
            # every iteration in dual-thread mode. insert() handles shifted
            # content via mismatch-aware committed tracking reset.

        # Emergency cap if buffer grew far beyond max during
        # a previous blocking inference (>2x the max).
        max_samples = int(self.max_buffer_s * 2 * self.sample_rate)
        if len(self.audio_buffer) > max_samples:
            target_samples = int(self.max_buffer_s * self.sample_rate)
            excess = len(self.audio_buffer) - target_samples
            self.audio_buffer = self.audio_buffer[excess:]
            self.buffer_time_offset += excess / self.sample_rate

        # Strip trailing silence before inference to prevent hallucination
        # on silent tails. Only affects what we send to Whisper, not the
        # actual buffer (so timestamps stay aligned).
        inference_audio = self.audio_buffer
        _window = int(0.1 * self.sample_rate)  # 100ms windows
        while len(inference_audio) > _window * 3:
            tail_rms = float(np.sqrt(np.mean(inference_audio[-_window:] ** 2)))
            if tail_rms >= 0.003:
                break
            inference_audio = inference_audio[:-_window]

        result = self.backend.transcribe(
            inference_audio,
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
            confirmed_text = "".join(w[2] for w in self.committed_history)
            unconfirmed_text = "".join(w[2] for w in self.last_unconfirmed)
            self.last_debug = {
                "iter": self._iter_count, "buffer_s": round(buffer_duration, 1),
                "inference_ms": inference_ms, "raw_text": "(word count discard)",
                "confirmed_words": len(self.committed_history),
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
            confirmed_text = "".join(w[2] for w in self.committed_history)
            unconfirmed_text = "".join(w[2] for w in self.last_unconfirmed)
            return (confirmed_text, unconfirmed_text)

        # Strip word-level repetitions before LocalAgreement.
        # Whisper hallucination loops produce "四四四四..." or "雪。雪。雪。..."
        # where the same word repeats 40-80 times. strip_trailing_repetition()
        # operates on raw_text but raw_words is extracted from the original
        # result before stripping, so repetitive words survive into here.
        raw_words = self._strip_word_repetitions(raw_words)

        # Insert into HypothesisBuffer for LocalAgreement
        self.transcript_buffer.insert(raw_words, offset=self.buffer_time_offset)

        # Flush confirmed words
        newly_confirmed = self.transcript_buffer.flush()

        # Dedup: sliding window alignment may re-confirm words that are
        # already in self.committed (e.g., after trim+reset, or when the same
        # overlapping region matches in consecutive iterations). Drop if the
        # newly confirmed text appears as a substring in committed text.
        # NOTE: use self.committed (active words), NOT committed_history.
        # committed_history retains all words including pre-trim ones, so
        # post-trim re-confirmations would be falsely rejected.
        if newly_confirmed and self.committed:
            new_text = "".join(w[2] for w in newly_confirmed)
            committed_text = "".join(w[2] for w in self.committed)
            if new_text in committed_text:
                newly_confirmed = []

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
        self.committed_history.extend(newly_confirmed)
        if newly_confirmed:
            self._committed_end_time = newly_confirmed[-1][1]  # end time

        # Get unconfirmed tail
        self.last_unconfirmed = self.transcript_buffer.peek_unconfirmed()

        # Build text outputs (use committed_history for complete output)
        confirmed_text = "".join(w[2] for w in self.committed_history)
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
            "confirmed_words": len(self.committed_history),
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
        self.committed_history.extend(unconfirmed)
        full_text = "".join(w[2] for w in self.committed)

        logger.debug(
            "segment_close: force_confirmed=%d total_committed=%d text_len=%d text=%.80s",
            len(unconfirmed), len(self.committed), len(full_text),
            full_text.replace("\n", "\\n")[:80],
        )

        # Save committed text for post-commit echo detection
        self._last_committed_raw = full_text

        # Reset for next segment (keep committed words for history).
        # Do NOT pre-populate — same rationale as _maybe_trim_buffer:
        # fresh start prevents stale words from blocking LocalAgreement.
        self.transcript_buffer.reset()
        self.last_unconfirmed = []

        return full_text

    def get_confirmed_words(self) -> list[tuple[float, float, str]]:
        """Return all confirmed words with timestamps for SRT export."""
        return list(self.committed_history)

    def get_all_words(self) -> list[tuple[float, float, str]]:
        """Return confirmed + unconfirmed words."""
        return list(self.committed_history) + list(self.last_unconfirmed)

    def reset(self):
        """Full reset for a new recording session."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0.0
        self.transcript_buffer.reset()
        self.committed = []
        self.committed_history = []
        self.last_unconfirmed = []
        self._committed_end_time = 0.0
        self._last_committed_raw = ""
        self._last_process_time = 0.0
        self._iter_count = 0
        self._last_inference_ms = 0
        self._last_trim_info = None
        self._trim_cooldown = 0
        self.last_debug = {}

    # ── Internal ─────────────────────────────────────────

    def _build_prompt(self) -> str:
        """Build dynamic init_prompt from committed text + bilingual hint."""
        committed_text = "".join(w[2] for w in self.committed_history)
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

    @staticmethod
    def _strip_word_repetitions(
        words: list[tuple[float, float, str]],
        max_repeat: int = 3,
    ) -> list[tuple[float, float, str]]:
        """Remove consecutive repeated words that indicate hallucination loops.

        Whisper decoder loops produce sequences like "四四四四..." (77×) or
        "雪。雪。雪。..." (41×). This strips runs of ≥ max_repeat identical
        words down to a single occurrence, preserving the first word's
        timestamps.

        Comparison is case-insensitive and strip-punctuation to catch
        variations like "雪。" vs "雪," which are the same hallucination.
        """
        if len(words) < max_repeat:
            return words

        strip_chars = set(' \t\n，。！？、,.!?-\u3000')

        def _norm(text: str) -> str:
            return ''.join(c.lower() for c in text if c not in strip_chars)

        result = []
        i = 0
        while i < len(words):
            norm_i = _norm(words[i][2])
            # Count consecutive identical words
            run_end = i + 1
            while run_end < len(words) and _norm(words[run_end][2]) == norm_i:
                run_end += 1
            run_len = run_end - i
            if run_len >= max_repeat:
                # Hallucination loop: keep only the first occurrence
                result.append(words[i])
                logger.debug(
                    "Stripped word repetition: %r × %d",
                    words[i][2], run_len,
                )
                i = run_end
            else:
                # Normal: keep all words in this short run
                result.extend(words[i:run_end])
                i = run_end
        return result

    def _maybe_trim_buffer(self):
        """Trim audio buffer if it exceeds max_buffer_s.

        Uses adaptive threshold: if last inference was slow (>1500ms),
        trim more aggressively to keep inference fast. If inference was
        very slow (>3s), force trim to 3s regardless.

        Cooldown: after trimming, skip 2 iterations to let LocalAgreement
        confirm new words and advance _find_trim_point() past the old
        sentence boundary. Without this, trim loops to the same position.
        """
        # Cooldown: after a trim, give LocalAgreement time to advance
        if self._trim_cooldown > 0:
            self._trim_cooldown -= 1
            self._last_trim_info = None
            return

        buffer_duration = len(self.audio_buffer) / self.sample_rate

        # Only trim when buffer genuinely exceeds max_buffer_s.
        # Pre-inference trim (line ~246) already keeps buffer within
        # max_buffer_s before each inference. This post-inference trim
        # handles edge cases where audio accumulated during a slow call.
        #
        # Previous adaptive thresholds (trim to 3s when inference > 3000ms)
        # were too aggressive: a single GPU spike would slash the threshold,
        # triggering trim → reset LocalAgreement → massive confirmation loss.
        # Run 4 (best: 277/245 chars, 0 trims) vs Run 9 (129/173, 2 trims)
        # showed that fewer trims = more confirmed text.
        effective_max = self.max_buffer_s

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

        # Remove committed words whose audio has been trimmed away.
        # This prevents _find_trim_point() from re-finding the same stale
        # sentence boundary → no more trim loops to the same position.
        # Safe because committed_history retains the full word history.
        self.committed = [w for w in self.committed if w[1] > trim_time]

        self._last_trim_info = {
            "trimmed": True,
            "from_s": round(old_offset, 1),
            "to_s": round(trim_time, 1),
            "effective_max": round(effective_max, 1),
            "retained_s": round(len(self.audio_buffer) / self.sample_rate, 1),
        }

        # Cooldown: skip 2 iterations of _maybe_trim_buffer so
        # LocalAgreement can confirm new words and advance past this
        # sentence boundary before we consider trimming again.
        self._trim_cooldown = 2

        # Reset HypothesisBuffer after trim — word positions change completely
        # so old buffer comparison would never match. Committed words are safe
        # in self.committed already.
        # NOTE: do NOT pre-populate buffer after reset. The trimmed audio will
        # produce entirely new word sequences from Whisper, so pre-populated
        # old words would never match and LocalAgreement would stall forever.
        # Fresh start costs 1 iter of confirm latency but prevents deadlock.
        self.transcript_buffer.reset()

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
