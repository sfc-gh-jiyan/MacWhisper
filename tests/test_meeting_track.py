"""Tests for ChannelTrack and the single-track MeetingSession refactor.

No ML model: a fake processor, a scripted VAD and a fake clock.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import meeting
from meeting import MeetingSession, MeetingSegment
from meeting_track import ChannelTrack


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def fake_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(meeting.time, "time", clock)
    return clock


class FakeProc:
    """Processor stub: words carry buffer-relative times, like the real one."""

    def __init__(self, word_start=0.0):
        self.committed = []
        self.committed_history = []
        self.buffer_time_offset = 0.0
        self._last_committed_raw = ""
        self.last_debug = {}
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.transcript_buffer = SimpleNamespace(peek_unconfirmed=lambda: [])
        self._word_start = word_start
        self._n = 0

    def insert_audio_chunk(self, a):
        self.audio_buffer = np.concatenate([self.audio_buffer, a])

    def process_iter(self):
        if not self.committed:
            self.committed = [(self._word_start, self._word_start + 0.5, "hi")]
        return ("hi", "")

    def segment_close(self):
        self._n += 1
        return f"text {self._n}" if self.committed else ""


class ScriptVAD:
    def __init__(self):
        self.speech = False
        self.silence = False

    def process_chunk(self, chunk):
        return {"is_speech": self.speech}

    def is_extended_silence(self, ms):
        return self.silence

    def reset(self):
        pass


def _session(clock, vad, proc):
    s = MeetingSession(backend=None, vad=vad, audio_source=SimpleNamespace(),
                       processor_factory=lambda: FakeProc())
    s._meeting_start = clock.t
    s._state = "recording"
    s._segments = []
    s._processor = proc
    return s


CHUNK = [np.full((16000, 1), 1000, dtype=np.int16)]   # 1 s of audio


def _feed(s, clock, t, speech, silence=False):
    clock.t = t
    s.vad.speech, s.vad.silence = speech, silence
    s._process_new_audio(CHUNK, [])


def test_segment_start_is_first_speech_not_previous_close(fake_clock):
    start = fake_clock.t
    s = _session(fake_clock, ScriptVAD(), FakeProc())
    for t in range(1, 10):
        _feed(s, fake_clock, start + t, speech=True)
    _feed(s, fake_clock, start + 10, speech=False, silence=True)    # close #1
    assert len(s.segments) == 1
    for t in range(11, 70):
        _feed(s, fake_clock, start + t, speech=False)
    for t in range(70, 75):
        _feed(s, fake_clock, start + t, speech=True)
    _feed(s, fake_clock, start + 80, speech=False, silence=True)    # close #2
    assert len(s.segments) == 2
    assert s.segments[0].start_time == pytest.approx(1.0, abs=0.1)
    assert s.segments[1].start_time == pytest.approx(70.0, abs=0.1)
    assert s.segments[1].end_time == pytest.approx(80.0, abs=0.1)


def test_segment_times_never_negative(fake_clock):
    """Buffer trimmed mid-meeting: word times are relative to the trimmed
    buffer and used to give [-1:56]-style starts."""
    start = fake_clock.t
    s = _session(fake_clock, ScriptVAD(), FakeProc(word_start=-116.0))
    for t in range(1, 5):
        _feed(s, fake_clock, start + t, speech=True)
    _feed(s, fake_clock, start + 5, speech=False, silence=True)
    # a commit with no recorded speech falls back to "now"
    fake_clock.t = start + 9
    s._commit_segment(s._tracks[0], "late text")
    assert len(s.segments) == 2
    assert all(seg.start_time >= 0 for seg in s.segments)
    assert s.segments[1].start_time == pytest.approx(9.0)


GOLDEN_MD = """# Meeting Transcript

- **Date**: 20261107
- **Segments**: 2
- **Duration**: 1m 5s

---

**[00:01]** First line.

**[01:10]** Second line.
"""


@pytest.fixture
def golden_segments():
    s = MeetingSession(backend=None, vad=None, audio_source=SimpleNamespace())
    s._session_id = "20261107_120000"
    s._segments = [MeetingSegment(text="First line.", start_time=1.2, end_time=9.0),
                   MeetingSegment(text="Second line.", start_time=70.4, end_time=65.9)]
    return s


def test_single_channel_markdown_unchanged(golden_segments):
    session = golden_segments
    assert session._format_markdown() == GOLDEN_MD   # no labels, today's format


def test_mark_speech_start_first_call_wins():
    t = ChannelTrack(None, FakeProc(), None)
    t.mark_speech_start(5.0)
    t.mark_speech_start(9.0)
    assert t.segment_start == 5.0
    t.reset_segment()
    assert t.segment_start is None


def test_processor_setter_works_before_start():
    s = MeetingSession(backend=None, vad=None, audio_source=SimpleNamespace())
    p = FakeProc()
    s._processor = p
    assert s._tracks[0].processor is p and s._processor is p
