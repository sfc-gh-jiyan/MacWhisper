"""TDD tests for OnlineASRProcessor and HypothesisBuffer.

RED phase: these tests define expected behavior BEFORE implementation.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asr_backend import ASRBackend, TranscriptionResult, Segment, WordTimestamp


# ── Mock backend for testing ─────────────────────────────

class MockASRBackend(ASRBackend):
    """Returns pre-configured results for testing."""

    def __init__(self):
        self.results: list[TranscriptionResult] = []
        self._call_count = 0

    def add_result(self, text: str, words: list[tuple[str, float, float]]):
        """Add a result to the queue. words = [(word, start, end), ...]"""
        word_ts = [WordTimestamp(word=w, start=s, end=e) for w, s, e in words]
        seg = Segment(text=text, start=words[0][1] if words else 0.0,
                      end=words[-1][2] if words else 0.0, words=word_ts)
        self.results.append(TranscriptionResult(text=text, segments=[seg]))

    def transcribe(self, audio, **kwargs) -> TranscriptionResult:
        if self._call_count < len(self.results):
            result = self.results[self._call_count]
        else:
            result = self.results[-1] if self.results else TranscriptionResult(text="")
        self._call_count += 1
        return result


# ── Helper: generate audio ───────────────────────────────

def make_audio(duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate silent float32 audio of given duration."""
    n_samples = int(duration_s * sample_rate)
    return np.zeros(n_samples, dtype=np.float32)


def make_speech_audio(duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate non-silent float32 audio (sine wave)."""
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ══════════════════════════════════════════════════════════
# HypothesisBuffer tests
# ══════════════════════════════════════════════════════════

class TestHypothesisBuffer:
    """Test the core word-level consistency mechanism."""

    def test_two_identical_results_confirms_all(self):
        """When two consecutive iterations produce the same words, all are confirmed."""
        from online_processor import HypothesisBuffer

        buf = HypothesisBuffer()
        words1 = [(0.0, 0.5, "今天"), (0.5, 1.0, "天气"), (1.0, 1.5, "很好")]
        words2 = [(0.0, 0.5, "今天"), (0.5, 1.0, "天气"), (1.0, 1.5, "很好")]

        buf.insert(words1, offset=0.0)
        confirmed1 = buf.flush()
        assert confirmed1 == [], "First iteration should not confirm anything"

        buf.insert(words2, offset=0.0)
        confirmed2 = buf.flush()
        assert len(confirmed2) == 3, "Second identical iteration should confirm all 3 words"
        assert [w[2] for w in confirmed2] == ["今天", "天气", "很好"]

    def test_partial_match_confirms_prefix(self):
        """Only matching prefix words are confirmed when tail changes."""
        from online_processor import HypothesisBuffer

        buf = HypothesisBuffer()
        words1 = [(0.0, 0.5, "今天"), (0.5, 1.0, "天气"), (1.0, 1.5, "很好")]
        words2 = [(0.0, 0.5, "今天"), (0.5, 1.0, "天气"), (1.0, 1.8, "很好我们")]

        buf.insert(words1, offset=0.0)
        buf.flush()

        buf.insert(words2, offset=0.0)
        confirmed = buf.flush()
        # "今天" and "天气" match, "很好" vs "很好我们" differ
        assert len(confirmed) == 2
        assert [w[2] for w in confirmed] == ["今天", "天气"]

    def test_empty_input_no_crash(self):
        """Empty word list should not cause errors."""
        from online_processor import HypothesisBuffer

        buf = HypothesisBuffer()
        buf.insert([], offset=0.0)
        confirmed = buf.flush()
        assert confirmed == []

    def test_single_iteration_no_confirmation(self):
        """A single iteration never confirms anything (need 2 matches)."""
        from online_processor import HypothesisBuffer

        buf = HypothesisBuffer()
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]
        buf.insert(words, offset=0.0)
        confirmed = buf.flush()
        assert confirmed == []

    def test_three_iterations_incremental(self):
        """Third iteration with new tail confirms more words."""
        from online_processor import HypothesisBuffer

        buf = HypothesisBuffer()
        # Iter 1: "A B"
        buf.insert([(0, 0.5, "A"), (0.5, 1, "B")], offset=0)
        buf.flush()
        # Iter 2: "A B C" — A,B match → confirmed
        buf.insert([(0, 0.5, "A"), (0.5, 1, "B"), (1, 1.5, "C")], offset=0)
        c2 = buf.flush()
        assert [w[2] for w in c2] == ["A", "B"]
        # Iter 3: "A B C D" — C matches → confirmed
        buf.insert([(0, 0.5, "A"), (0.5, 1, "B"), (1, 1.5, "C"), (1.5, 2, "D")], offset=0)
        c3 = buf.flush()
        assert [w[2] for w in c3] == ["C"]

    def test_complete_rewrite_resets(self):
        """When output completely changes, nothing is confirmed."""
        from online_processor import HypothesisBuffer

        buf = HypothesisBuffer()
        buf.insert([(0, 0.5, "hello"), (0.5, 1, "world")], offset=0)
        buf.flush()
        buf.insert([(0, 0.5, "bonjour"), (0.5, 1, "monde")], offset=0)
        confirmed = buf.flush()
        assert confirmed == []

    def test_sliding_window_alignment(self):
        """When buffer prefix shifts (simulating pre-inference trim), words that
        appear at a different position in new_words should still be confirmed
        if they match old unconfirmed words.

        This is the core scenario in dual-thread Meeting Mode: pre-inference
        trim shifts the audio start each iteration, so Whisper returns different
        prefixes but the overlapping middle portion stays stable.
        """
        from online_processor import HypothesisBuffer

        buf = HypothesisBuffer()
        # Iter 1: "A B C D" — stored
        buf.insert([(0, 1, "A"), (1, 2, "B"), (2, 3, "C"), (3, 4, "D")])
        assert buf.flush() == []

        # Iter 2: trim shifted start → "C D E F" — "C D" overlaps old unconfirmed
        buf.insert([(0, 1, "C"), (1, 2, "D"), (2, 3, "E"), (3, 4, "F")])
        confirmed = buf.flush()
        texts = [w[2] for w in confirmed]
        assert "C" in texts and "D" in texts, \
            f"Sliding window should confirm overlapping words, got: {texts}"
        # "A" and "B" should NOT be confirmed (no match in new_words)
        assert "A" not in texts and "B" not in texts

    def test_sliding_window_no_false_match(self):
        """Single word overlap should not trigger confirmation (need >= 2)."""
        from online_processor import HypothesisBuffer

        buf = HypothesisBuffer()
        buf.insert([(0, 1, "X"), (1, 2, "Y"), (2, 3, "Z")])
        buf.flush()
        # Only "Z" overlaps — should NOT confirm (need 2+ consecutive)
        buf.insert([(0, 1, "A"), (1, 2, "Z"), (2, 3, "B")])
        confirmed = buf.flush()
        assert confirmed == [], \
            f"Single word overlap should not confirm, got: {[w[2] for w in confirmed]}"

class TestOnlineASRProcessor:
    """Test the main processor that orchestrates backend + buffer + VAD."""

    def test_returns_none_when_audio_too_short(self):
        """process_iter returns None when audio < min_chunk_size."""
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.7)
        proc.insert_audio_chunk(make_audio(0.3))
        result = proc.process_iter()
        assert result is None

    def test_confirms_stable_words_after_two_iterations(self):
        """Two iterations with matching words → confirmed text returned."""
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        backend.add_result("今天天气", [(  "今天", 0.0, 0.5), ("天气", 0.5, 1.0)])
        backend.add_result("今天天气很好", [("今天", 0.0, 0.5), ("天气", 0.5, 1.0), ("很好", 1.0, 1.5)])

        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5)
        proc.throttle = False

        # First iteration: insert enough audio and process
        proc.insert_audio_chunk(make_speech_audio(1.0))
        result1 = proc.process_iter()
        assert result1 is not None
        confirmed1, unconfirmed1 = result1
        assert confirmed1 == ""  # First iter: nothing confirmed yet

        # Second iteration: add more audio
        proc.insert_audio_chunk(make_speech_audio(1.0))
        result2 = proc.process_iter()
        assert result2 is not None
        confirmed2, unconfirmed2 = result2
        assert "今天" in confirmed2
        assert "天气" in confirmed2

    def test_buffer_trimming_at_max(self):
        """Buffer is trimmed when it exceeds max_buffer_s."""
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        backend.add_result("测试内容", [("测试", 0.0, 0.5), ("内容", 0.5, 1.0)])

        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5, max_buffer_s=5.0)
        proc.throttle = False

        # Insert 10 seconds of audio
        proc.insert_audio_chunk(make_speech_audio(10.0))
        proc.process_iter()

        # Buffer should have been trimmed
        buffer_duration = len(proc.audio_buffer) / 16000
        assert buffer_duration <= 6.0, f"Buffer should be trimmed, got {buffer_duration:.1f}s"

    def test_dynamic_prompt_uses_committed_text(self):
        """init_prompt should include recently committed text."""
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        # Two identical results → words confirmed
        backend.add_result("你好世界", [("你好", 0, 0.5), ("世界", 0.5, 1.0)])
        backend.add_result("你好世界", [("你好", 0, 0.5), ("世界", 0.5, 1.0)])
        backend.add_result("你好世界再见", [("你好", 0, 0.5), ("世界", 0.5, 1.0), ("再见", 1.0, 1.5)])

        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5)
        proc.throttle = False

        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()

        # After confirming words, the prompt should contain committed text
        prompt = proc._build_prompt()
        assert "你好" in prompt or "世界" in prompt

    def test_segment_close_confirms_remaining(self):
        """segment_close() should confirm all remaining unconfirmed words."""
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        backend.add_result("你好世界", [("你好", 0, 0.5), ("世界", 0.5, 1.0)])

        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5)
        proc.throttle = False

        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()  # First iter: nothing confirmed

        # Force close segment
        text = proc.segment_close()
        assert "你好" in text
        assert "世界" in text

    def test_get_confirmed_words_returns_timestamps(self):
        """get_confirmed_words() returns list with timing info."""
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        backend.add_result("你好", [("你", 0, 0.5), ("好", 0.5, 1.0)])
        backend.add_result("你好", [("你", 0, 0.5), ("好", 0.5, 1.0)])

        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5)
        proc.throttle = False

        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()

        words = proc.get_confirmed_words()
        assert len(words) >= 2
        assert all(len(w) == 3 for w in words)  # (start, end, text)

    def test_unconfirmed_text_contains_tail(self):
        """Unconfirmed text should contain the not-yet-confirmed tail."""
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        backend.add_result("今天很好", [("今天", 0, 0.3), ("很", 0.3, 0.6), ("好", 0.6, 1.0)])
        backend.add_result("今天很好啊", [("今天", 0, 0.3), ("很", 0.3, 0.6), ("好啊", 0.6, 1.2)])

        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5)
        proc.throttle = False

        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(0.5))
        result = proc.process_iter()

        assert result is not None
        confirmed, unconfirmed = result
        # "今天" and "很" confirmed; "好" vs "好啊" differ
        assert "今天" in confirmed
        assert len(unconfirmed) > 0

    # ── Bug regression tests (found via replay testing) ────

    def test_buffer_trim_does_not_stall_confirmation(self):
        """After buffer trim, new words should still get confirmed.

        Bug: buffer trim reset HypothesisBuffer but didn't pre-populate
        committed_in_buffer, so post-trim transcriptions saw completely
        different word positions and nothing could match → confirmation stalled.
        """
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        # Phase 1: two identical results → confirm "你好世界"
        backend.add_result("你好世界", [("你好", 0, 0.5), ("世界", 0.5, 1.0)])
        backend.add_result("你好世界", [("你好", 0, 0.5), ("世界", 0.5, 1.0)])
        # Phase 2: after trim, new content arrives (timestamps relative to trimmed buffer)
        backend.add_result("新的内容", [("新的", 0, 0.5), ("内容", 0.5, 1.0)])
        backend.add_result("新的内容来了", [("新的", 0, 0.5), ("内容", 0.5, 1.0), ("来了", 1.0, 1.5)])

        # max_buffer_s=5.0 so Phase 1 (4s total) doesn't trigger trim
        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5, max_buffer_s=5.0)
        proc.throttle = False

        # Phase 1: confirm initial words (4s total, under max_buffer_s=5)
        proc.insert_audio_chunk(make_speech_audio(2.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(2.0))
        r2 = proc.process_iter()
        assert r2 is not None
        confirmed_before_trim = r2[0]
        assert "你好" in confirmed_before_trim

        # Phase 2: add enough audio to trigger trim (>5s already + more)
        proc.insert_audio_chunk(make_speech_audio(4.0))
        r3 = proc.process_iter()
        # After trim, still getting results (not stalled)
        assert r3 is not None

        proc.insert_audio_chunk(make_speech_audio(2.0))
        r4 = proc.process_iter()
        assert r4 is not None
        # New words should eventually appear in confirmed or unconfirmed
        confirmed4, unconfirmed4 = r4
        total_text = confirmed4 + unconfirmed4
        # The key assertion: we're not stuck — total text keeps growing
        assert len(total_text) > len(confirmed_before_trim)

    def test_buffer_trim_no_duplicate_committed_words(self):
        """After buffer trim, committed words should not be re-confirmed.

        Bug: after trim + HypothesisBuffer reset, retained audio was
        retranscribed and words that were already committed got confirmed
        again → duplicated sentences in output.
        """
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        # Two identical → confirm "你好世界"
        backend.add_result("你好世界", [("你好", 0, 0.5), ("世界", 0.5, 1.0)])
        backend.add_result("你好世界", [("你好", 0, 0.5), ("世界", 0.5, 1.0)])
        # After trim: retained audio produces same words + new ones
        # Word timestamps overlap with committed region
        backend.add_result("你好世界新词", [("你好", 0, 0.5), ("世界", 0.5, 1.0), ("新词", 1.0, 1.5)])
        backend.add_result("你好世界新词", [("你好", 0, 0.5), ("世界", 0.5, 1.0), ("新词", 1.0, 1.5)])

        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5, max_buffer_s=3.0)
        proc.throttle = False

        # Phase 1: confirm
        proc.insert_audio_chunk(make_speech_audio(2.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(2.0))
        proc.process_iter()

        # Phase 2: trigger trim + continue
        proc.insert_audio_chunk(make_speech_audio(3.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(2.0))
        proc.process_iter()

        # Check: no duplicates in committed words
        committed_texts = [w[2] for w in proc.committed_history]
        # Count occurrences — "你好" should appear at most once
        from collections import Counter
        counts = Counter(committed_texts)
        for word, count in counts.items():
            assert count == 1, f"Word '{word}' committed {count} times (expected 1)"

    def test_hallucination_filter_preserves_confirmed_state(self):
        """When hallucination is detected, process_iter should return
        current confirmed text, not empty strings.

        Bug: hallucination filter returned ("", "") which made the overlay
        flash to empty, causing stability_jumps in replay tests.
        """
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        # Two identical → confirm "你好世界"
        backend.add_result("你好世界", [("你好", 0, 0.5), ("世界", 0.5, 1.0)])
        backend.add_result("你好世界", [("你好", 0, 0.5), ("世界", 0.5, 1.0)])
        # Third result triggers hallucination filter (known hallucination phrase)
        backend.add_result("Thank you for watching.", [("Thank", 0, 0.3), ("you", 0.3, 0.5),
                           ("for", 0.5, 0.7), ("watching.", 0.7, 1.0)])

        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5)
        proc.throttle = False

        # Confirm "你好世界"
        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(1.0))
        r2 = proc.process_iter()
        assert r2 is not None
        confirmed_before = r2[0]
        assert "你好" in confirmed_before

        # Hallucination result — should NOT clear the display
        proc.insert_audio_chunk(make_speech_audio(1.0))
        r3 = proc.process_iter()
        assert r3 is not None
        confirmed_after, unconfirmed_after = r3
        # Key: confirmed text is preserved (not empty)
        assert confirmed_after == confirmed_before, \
            f"Hallucination should preserve confirmed text, got '{confirmed_after}'"

    def test_punctuation_normalization_in_word_matching(self):
        """Words with English vs Chinese punctuation should still match."""
        from online_processor import OnlineASRProcessor
        backend = MockASRBackend()
        # First iteration: English comma
        backend.add_result("你好, 世界.", [("你好,", 0, 0.5), (" 世界.", 0.5, 1.0)])
        # Second iteration: Chinese comma (same words, different punctuation style)
        # After normalize_punctuation, both become "你好，" and " 世界。"
        backend.add_result("你好, 世界.", [("你好,", 0, 0.5), (" 世界.", 0.5, 1.0)])

        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5)
        proc.throttle = False

        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(1.0))
        r2 = proc.process_iter()
        assert r2 is not None
        confirmed = r2[0]
        # Punctuation should be normalized to Chinese style
        assert "，" in confirmed or "。" in confirmed, \
            f"Expected Chinese punctuation in '{confirmed}'"
        # English punctuation should NOT appear
        assert "," not in confirmed, \
            f"English comma should be normalized, got '{confirmed}'"

    def test_segment_close_no_echo_duplication(self):
        """After segment_close(), re-transcribed words should not duplicate.

        Bug: segment_close() reset HypothesisBuffer without pre-populating
        committed_in_buffer, so post-close transcription re-confirmed words
        that were already in self.committed → duplicated text.
        """
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        # Phase 1: two identical → confirm "你好世界"
        backend.add_result("你好世界", [("你好", 0, 0.5), ("世界", 0.5, 1.0)])
        backend.add_result("你好世界", [("你好", 0, 0.5), ("世界", 0.5, 1.0)])
        # Phase 2: after segment_close, retained audio re-transcribes same words + new
        backend.add_result("你好世界新内容", [("你好", 0, 0.5), ("世界", 0.5, 1.0), ("新内容", 1.0, 1.5)])
        backend.add_result("你好世界新内容", [("你好", 0, 0.5), ("世界", 0.5, 1.0), ("新内容", 1.0, 1.5)])

        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5)
        proc.throttle = False

        # Phase 1: confirm initial words
        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(1.0))
        r2 = proc.process_iter()
        assert r2 is not None
        assert "你好" in r2[0]

        # Force close segment (simulates VAD speech end)
        proc.segment_close()

        # Phase 2: continue processing (retained audio)
        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()

        # Check: no duplicates in committed words
        committed_texts = [w[2] for w in proc.committed_history]
        from collections import Counter
        counts = Counter(committed_texts)
        for word, count in counts.items():
            assert count == 1, f"Word '{word}' committed {count} times after segment_close (expected 1)"

    def test_echo_detection_drops_echoed_words(self):
        """Post-commit echo detection should drop words that repeat committed tail.

        Even if HypothesisBuffer pre-population fails (e.g., shifted timestamps),
        the echo detection safety net should catch duplicates.
        """
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        # Phase 1: confirm "今天天气很好"
        backend.add_result("今天天气很好", [("今天", 0, 0.3), ("天气", 0.3, 0.6), ("很好", 0.6, 1.0)])
        backend.add_result("今天天气很好", [("今天", 0, 0.3), ("天气", 0.3, 0.6), ("很好", 0.6, 1.0)])
        # Phase 2: after segment_close, echo with SHIFTED timestamps (won't match pre-population)
        backend.add_result("今天天气很好明天", [("今天", 0.1, 0.4), ("天气", 0.4, 0.7), ("很好", 0.7, 1.1), ("明天", 1.1, 1.5)])
        backend.add_result("今天天气很好明天", [("今天", 0.1, 0.4), ("天气", 0.4, 0.7), ("很好", 0.7, 1.1), ("明天", 1.1, 1.5)])

        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5)
        proc.throttle = False

        # Phase 1
        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()

        # Close segment
        proc.segment_close()

        # Phase 2 — new content after echo
        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(1.0))
        r4 = proc.process_iter()
        assert r4 is not None

        # "今天", "天气", "很好" should appear exactly once
        committed_texts = [w[2] for w in proc.committed_history]
        from collections import Counter
        counts = Counter(committed_texts)
        for word in ["今天", "天气", "很好"]:
            assert counts.get(word, 0) == 1, \
                f"Echo word '{word}' appeared {counts.get(word, 0)} times (expected 1)"
        # "明天" is new — should be committed
        assert "明天" in committed_texts, "New word '明天' should be committed"

    def test_trim_stall_with_unstable_transcription(self):
        """After trim, unstable transcription (different text each iter) must not
        permanently stall LocalAgreement.

        Regression: when HypothesisBuffer was pre-populated with committed words
        after trim, but Whisper produced entirely different text for the trimmed
        audio, position-0 mismatch caused `break` in insert() → match_end=0
        forever. Fix: don't pre-populate after reset.
        """
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        # Phase 1: two identical → confirm "你好世界"
        backend.add_result("你好世界", [("你好", 0, 0.5), ("世界", 0.5, 1.0)])
        backend.add_result("你好世界", [("你好", 0, 0.5), ("世界", 0.5, 1.0)])
        # Phase 2: after trim, unstable results (simulates Chinese-English switching)
        backend.add_result("现代人的数字生活", [("现代人", 0, 0.5), ("的", 0.5, 0.7), ("数字", 0.7, 1.0), ("生活", 1.0, 1.3)])
        backend.add_result("That is the past", [("That", 0, 0.3), ("is", 0.3, 0.5), ("the", 0.5, 0.7), ("past", 0.7, 1.0)])
        backend.add_result("保持生活的balance", [("保持", 0, 0.4), ("生活", 0.4, 0.7), ("的", 0.7, 0.8), ("balance", 0.8, 1.2)])
        # Phase 3: eventually stabilizes
        backend.add_result("越来越难", [("越来越", 0, 0.5), ("难", 0.5, 0.8)])
        backend.add_result("越来越难", [("越来越", 0, 0.5), ("难", 0.5, 0.8)])

        # max_buffer_s=5.0 so Phase 1 (4s total) doesn't trigger trim;
        # Phase 2 (4s + 6s = 10s) forces trim to kick in.
        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5, max_buffer_s=5.0)
        proc.throttle = False

        # Phase 1: confirm initial words (4s total, under max_buffer_s=5)
        proc.insert_audio_chunk(make_speech_audio(2.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(2.0))
        r2 = proc.process_iter()
        assert r2 is not None
        confirmed_phase1 = r2[0]
        assert "你好" in confirmed_phase1

        # Phase 2: add enough audio to trigger trim + unstable iterations
        for _ in range(3):
            proc.insert_audio_chunk(make_speech_audio(3.0))
            proc.process_iter()

        # Phase 3: stable results → should confirm new words
        # After trim + reset, HypothesisBuffer fresh-starts: first identical
        # result is stored, second confirms. We give 3 iterations to be safe.
        for _ in range(3):
            proc.insert_audio_chunk(make_speech_audio(2.0))
            proc.process_iter()

        confirmed_final = "".join(w[2] for w in proc.committed_history)
        unconfirmed_final = "".join(w[2] for w in proc.last_unconfirmed)
        total = confirmed_final + unconfirmed_final
        # Key: "越来越" or "难" should appear (LocalAgreement recovered)
        assert "越来越" in total or "难" in total, \
            f"LocalAgreement should recover after unstable period, got: '{total}'"

    def test_pre_inference_trim_does_not_block_confirmation(self):
        """Pre-inference trim firing every iteration must not prevent confirmation.

        Regression: in dual-thread Meeting Mode, buffer always exceeds max_buffer_s
        because audio accumulates during inference. Pre-inference trim fires every
        iteration. If it does a full reset(), HypothesisBuffer can never compare
        2 consecutive results → zero confirmation after initial burst.

        Fix: pre-inference trim uses clear_committed() (keeps buffer for comparison)
        + dedup in process_iter prevents re-confirming already-committed words.
        """
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        # Sentence 1: confirmed normally (buffer still small)
        backend.add_result("今天天气很好", [("今天", 0, 0.5), ("天气", 0.5, 1.0), ("很好", 1.0, 1.5)])
        backend.add_result("今天天气很好", [("今天", 0, 0.5), ("天气", 0.5, 1.0), ("很好", 1.0, 1.5)])
        # Sentence 2: new content after buffer exceeds max → pre-inference trim fires
        # These iterations all have buffer > max_buffer_s, so trim fires each time
        backend.add_result("我们去公园", [("我们", 0, 0.4), ("去", 0.4, 0.6), ("公园", 0.6, 1.0)])
        backend.add_result("我们去公园", [("我们", 0, 0.4), ("去", 0.4, 0.6), ("公园", 0.6, 1.0)])
        # Sentence 3: even more content, trim continues firing
        backend.add_result("散步聊天", [("散步", 0, 0.5), ("聊天", 0.5, 1.0)])
        backend.add_result("散步聊天", [("散步", 0, 0.5), ("聊天", 0.5, 1.0)])

        # Small max_buffer_s=2.0 to ensure pre-inference trim fires frequently
        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5, max_buffer_s=2.0)
        proc.throttle = False

        # Phase 1: buffer under max, confirm sentence 1
        proc.insert_audio_chunk(make_speech_audio(1.5))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(1.0))
        r = proc.process_iter()
        assert r is not None
        assert "今天" in r[0], f"Sentence 1 should be confirmed, got: {r[0]}"

        # Phase 2: buffer now exceeds max (2.5s total > 2.0), trim fires
        proc.insert_audio_chunk(make_speech_audio(2.0))
        proc.process_iter()  # trim fires, then Whisper → "我们去公园" stored
        proc.insert_audio_chunk(make_speech_audio(2.0))
        proc.process_iter()  # trim fires, "我们去公园" matches → confirmed

        # Phase 3: more speech, trim keeps firing
        proc.insert_audio_chunk(make_speech_audio(2.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(2.0))
        proc.process_iter()

        total_text = "".join(w[2] for w in proc.committed_history)

        # Key assertions:
        # 1. Words from sentences 2 and 3 should be confirmed (not blocked by trim)
        assert "我们" in total_text or "公园" in total_text, \
            f"Sentence 2 should be confirmed despite pre-inference trim, got: '{total_text}'"
        # 2. No duplicates from sentence 1
        count_today = total_text.count("今天")
        assert count_today <= 1, \
            f"'今天' appears {count_today} times — dedup should prevent re-confirmation"

    def test_trim_cleans_committed_prevents_loop(self):
        """After _maybe_trim_buffer, committed (active) list must not contain
        stale words from before trim_time. This ensures _find_trim_point()
        finds fresh sentence boundaries instead of looping on the same one.

        Meanwhile, committed_history must retain ALL words (never lose text).

        Regression: Run 5 showed 9 trims in 12s, all to the same 3.3-5.3s
        position, because committed was never cleaned after trim.

        Fix: _maybe_trim_buffer filters committed after trim. Output uses
        committed_history (append-only) so no text is lost.
        """
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        # Sentence 1 with period → sentence boundary for trim
        backend.add_result("你好世界。", [("你好", 0, 0.5), ("世界。", 0.5, 1.0)])
        backend.add_result("你好世界。", [("你好", 0, 0.5), ("世界。", 0.5, 1.0)])
        # More results for continued processing
        backend.add_result("今天好。", [("今天", 0, 0.4), ("好。", 0.4, 0.8)])
        backend.add_result("今天好。", [("今天", 0, 0.4), ("好。", 0.4, 0.8)])
        backend.add_result("明天见", [("明天", 0, 0.5), ("见", 0.5, 0.8)])
        backend.add_result("明天见", [("明天", 0, 0.5), ("见", 0.5, 0.8)])

        # Use larger max_buffer_s so pre-inference trim doesn't fire,
        # but _maybe_trim_buffer fires when we simulate slow inference
        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5, max_buffer_s=5.0)
        proc.throttle = False

        # Confirm sentence 1
        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()

        assert any("你好" in w[2] for w in proc.committed_history), \
            "Sentence 1 should be confirmed"

        # Simulate slow inference → triggers adaptive trim (effective_max=3.0)
        proc._last_inference_ms = 4000

        # Add enough audio to exceed effective_max of 3.0
        proc.insert_audio_chunk(make_speech_audio(3.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(2.0))
        proc.process_iter()

        # Key 1: committed_history must retain ALL words (no text loss)
        history_text = "".join(w[2] for w in proc.committed_history)
        assert "你好" in history_text, \
            f"committed_history should retain all words, got: '{history_text}'"

        # Key 2: if _maybe_trim_buffer fired, committed should be cleaned
        if proc._last_trim_info and proc._last_trim_info.get("trimmed"):
            stale = [w for w in proc.committed if w[1] <= proc.buffer_time_offset]
            assert not stale, \
                f"committed should not contain stale words after trim, got: {stale}"

    def test_echo_detection_debug_field(self):
        """last_debug should include echo_dropped count."""
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        backend.add_result("测试", [("测试", 0, 0.5)])
        backend.add_result("测试", [("测试", 0, 0.5)])

        proc = OnlineASRProcessor(backend=backend, vad=None, min_first_buffer_s=0,
                                  min_chunk_size=0.5)
        proc.throttle = False

        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(1.0))
        proc.process_iter()

        assert "echo_dropped" in proc.last_debug
