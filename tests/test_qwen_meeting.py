"""Tests for qwen_meeting.QwenSegmentProcessor (fake backend, no model)."""

from types import SimpleNamespace

import numpy as np
import pytest

import qwen_meeting as qm
from qwen_backend import QwenUnavailable

SR = 16000


class FakeQwen:
    """Returns scripted texts; records the audio length of every call."""

    def __init__(self, texts, fail_on=None):
        self.texts = list(texts)
        self.calls = []
        self.fail_on = fail_on

    def transcribe(self, audio):
        self.calls.append(len(audio) / SR)
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise QwenUnavailable("sidecar died")
        text = self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]
        return SimpleNamespace(text=text)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _proc(texts, clock=None, **kw):
    clock = clock or Clock()
    return qm.QwenSegmentProcessor(FakeQwen(texts, kw.pop("fail_on", None)),
                                   now=clock, **kw), clock


def _feed(p, seconds):
    p.insert_audio_chunk(np.full(int(seconds * SR), 0.1, dtype=np.float32))


def test_needs_minimum_audio_before_first_decode():
    p, clock = _proc(["你好"])
    _feed(p, 0.3)
    assert p.process_iter() is None
    assert p.backend.calls == []


def test_decodes_whole_segment_about_once_per_second():
    p, clock = _proc(["hello", "hello world", "hello world again"])
    _feed(p, 1.0)
    assert p.process_iter() is not None
    _feed(p, 0.5)
    clock.t += 0.5
    assert p.process_iter() is None                    # throttled (< step)
    clock.t += 0.6
    _feed(p, 0.5)
    assert p.process_iter() is not None
    assert p.backend.calls == [1.0, 2.0]               # whole segment each time


def test_white_and_gray_from_stable_text():
    p, clock = _proc(["hello world", "hello world how", "hello world how are you"])
    out = []
    for _ in range(3):
        _feed(p, 1.0)
        clock.t += 1.1
        out.append(p.process_iter())
    confirmed, gray = out[-1]
    assert confirmed.startswith("hello world")         # agreed by two decodes
    assert "you" in gray or "you" in confirmed


def test_segment_close_decodes_once_more_and_resets():
    p, clock = _proc(["draft", "final text."])
    _feed(p, 1.0)
    clock.t += 1.1
    p.process_iter()
    text = p.segment_close()
    assert text == "final text."
    assert p.backend.calls == [1.0, 1.0]               # the closing full decode
    assert len(p.audio_buffer) == 0
    assert p.committed == [] and p.transcript_buffer.peek_unconfirmed() == []
    assert p.process_iter() is None                    # fresh segment


def test_segment_close_on_empty_segment_does_not_call_backend():
    p, _ = _proc(["x"])
    assert p.segment_close() == ""
    assert p.backend.calls == []


def test_gray_text_is_visible_to_the_meeting_loop():
    p, clock = _proc(["alpha", "beta"])
    _feed(p, 1.0)
    clock.t += 1.1
    p.process_iter()
    assert p.transcript_buffer.peek_unconfirmed()      # loop may close on gray-only


def test_segment_over_max_length_reports_it():
    p, clock = _proc(["a"], max_segment_s=15.0)
    _feed(p, 14.0)
    assert not p.segment_full()
    _feed(p, 1.5)
    assert p.segment_full()


def test_backend_failure_raises_for_fallback():
    p, clock = _proc(["ok", "ok"], fail_on=2)
    _feed(p, 1.0)
    clock.t += 1.1
    p.process_iter()
    _feed(p, 1.0)
    clock.t += 1.1
    with pytest.raises(QwenUnavailable):
        p.process_iter()


def test_hallucination_on_silence_is_dropped():
    p, clock = _proc(["谢谢观看"])
    _feed(p, 1.0)
    clock.t += 1.1
    assert p.process_iter() in (None, ("", ""))
    assert p.segment_close() == ""


def test_traditional_chinese_is_converted():
    p, clock = _proc(["這個問題"])
    _feed(p, 1.0)
    clock.t += 1.1
    p.process_iter()
    assert p.segment_close() == "这个问题"


def test_reset_clears_everything():
    p, clock = _proc(["a"])
    _feed(p, 2.0)
    clock.t += 1.1
    p.process_iter()
    p.reset()
    assert len(p.audio_buffer) == 0 and p.transcript_buffer.peek_unconfirmed() == []


def test_failed_segment_close_keeps_audio_for_the_whisper_fallback():
    """The meeting hands the unfinished segment's audio to Whisper on failure;
    segment_close must not clear it before raising."""
    p, clock = _proc(["ok"], fail_on=1)
    _feed(p, 2.0)
    with pytest.raises(QwenUnavailable):
        p.segment_close()
    assert p.segment_seconds() == pytest.approx(2.0)


def test_tiny_fragment_waits_for_a_longer_silence():
    """Eval 2026-10-20: 0.8 s silence cut a 1-character segment off a
    sentence. A segment with < 4 characters waits for 2 s of silence."""
    p, clock = _proc(["嗯", "嗯"])
    _feed(p, 1.0)
    clock.t += 1.1
    p.process_iter()
    assert not p.ready_to_close(silence_ms=900)
    assert p.ready_to_close(silence_ms=2100)


def test_normal_sentence_closes_after_short_silence():
    p, clock = _proc(["大家好今天我们开会"])
    _feed(p, 2.0)
    clock.t += 1.1
    p.process_iter()
    assert p.ready_to_close(silence_ms=900)


def test_pending_only_after_speech_was_heard():
    """Audio without speech (silence after a closed segment) must not get a
    decode at stop: it duplicated the previous sentence in a replay."""
    p, clock = _proc(["x"])
    _feed(p, 1.0)
    assert not p.has_pending_speech()
    p.mark_speech()
    assert p.has_pending_speech()
    p.segment_close()
    _feed(p, 1.0)                               # silence after the close
    assert not p.has_pending_speech()


def test_silent_segment_closes_without_decoding():
    """15 s of meeting silence (2026-10-23): decoding it made Qwen read the
    vocabulary back ('会话。') and run until the 21 s timeout."""
    p, clock = _proc(["会话。"])
    p.uses_vad = True
    _feed(p, 15.0)
    assert p.segment_close() == ""
    assert p.backend.calls == []
    assert p.segment_seconds() == 0.0


def test_segment_with_speech_still_decodes_on_close():
    p, clock = _proc(["hello there"])
    p.uses_vad = True
    _feed(p, 3.0)
    p.mark_speech()
    assert p.segment_close() != ""
    assert len(p.backend.calls) == 1


def test_process_iter_skips_silent_segment():
    p, clock = _proc(["会话。"])
    p.uses_vad = True
    _feed(p, 3.0)
    clock.t += 1.1
    assert p.process_iter() is None
    assert p.backend.calls == []


def test_without_vad_segments_still_decode():
    p, clock = _proc(["hello"])           # uses_vad defaults to False
    _feed(p, 3.0)
    assert p.segment_close() != ""
