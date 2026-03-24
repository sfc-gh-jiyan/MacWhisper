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


# ══════════════════════════════════════════════════════════
# OnlineASRProcessor tests
# ══════════════════════════════════════════════════════════

class TestOnlineASRProcessor:
    """Test the main processor that orchestrates backend + buffer + VAD."""

    def test_returns_none_when_audio_too_short(self):
        """process_iter returns None when audio < min_chunk_size."""
        from online_processor import OnlineASRProcessor

        backend = MockASRBackend()
        proc = OnlineASRProcessor(backend=backend, vad=None,
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

        proc = OnlineASRProcessor(backend=backend, vad=None,
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

        proc = OnlineASRProcessor(backend=backend, vad=None,
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

        proc = OnlineASRProcessor(backend=backend, vad=None,
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

        proc = OnlineASRProcessor(backend=backend, vad=None,
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

        proc = OnlineASRProcessor(backend=backend, vad=None,
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

        proc = OnlineASRProcessor(backend=backend, vad=None,
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

        proc = OnlineASRProcessor(backend=backend, vad=None,
                                  min_chunk_size=0.5, max_buffer_s=3.0)
        proc.throttle = False

        # Phase 1: confirm initial words
        proc.insert_audio_chunk(make_speech_audio(2.0))
        proc.process_iter()
        proc.insert_audio_chunk(make_speech_audio(2.0))
        r2 = proc.process_iter()
        assert r2 is not None
        confirmed_before_trim = r2[0]
        assert "你好" in confirmed_before_trim

        # Phase 2: add enough audio to trigger trim (>3s already + more)
        proc.insert_audio_chunk(make_speech_audio(3.0))
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

        proc = OnlineASRProcessor(backend=backend, vad=None,
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
        committed_texts = [w[2] for w in proc.committed]
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

        proc = OnlineASRProcessor(backend=backend, vad=None,
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
