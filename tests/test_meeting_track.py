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

    def reset(self):
        pass

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
    t.reset_segment(7.0)
    assert t.segment_start is None and t.buffer_start == 7.0


def test_processor_setter_works_before_start():
    s = MeetingSession(backend=None, vad=None, audio_source=SimpleNamespace())
    p = FakeProc()
    s._processor = p
    assert s._tracks[0].processor is p and s._processor is p


# ── Review fixes: outage offset = buffer start; close failure path ──

class FailProc(FakeProc):
    """Fails like Qwen3 once the clock passes fail_at (process_iter or close)."""

    def __init__(self, clock, fail_at, on="iter"):
        super().__init__()
        self._clock, self._fail_at, self._on = clock, fail_at, on

    def _maybe_fail(self, where):
        from qwen_backend import QwenUnavailable
        if where == self._on and self._clock.t >= self._fail_at:
            raise QwenUnavailable("down")

    def process_iter(self):
        self._maybe_fail("iter")
        return super().process_iter()

    def segment_close(self):
        self._maybe_fail("close")
        return super().segment_close()


class CatchProc(FakeProc):
    def segment_close(self):
        return "caught" if len(self.audio_buffer) else ""


def _outage_session(clock, proc):
    s = MeetingSession(backend=None, vad=ScriptVAD(), audio_source=SimpleNamespace(),
                       processor_factory=lambda: CatchProc())
    s._meeting_start = clock.t
    s._state = "recording"
    s._segments = []
    s._processor = proc
    s._tracks[0].buffer_start = clock.t        # what start() does
    return s


def test_catch_up_starts_at_buffer_start_not_first_speech(fake_clock):
    start = fake_clock.t
    s = _outage_session(fake_clock, FailProc(fake_clock, fail_at=start + 25))
    for t in range(1, 10):
        _feed(s, fake_clock, start + t, speech=True)
    _feed(s, fake_clock, start + 10, speech=False, silence=True)    # close #1 at 10
    for t in range(11, 20):
        _feed(s, fake_clock, start + t, speech=False)
    for t in range(20, 30):                                         # fails at 25
        _feed(s, fake_clock, start + t, speech=True)
    assert s._engine_down
    s._catch_up_buffered()
    caught = [g for g in s.segments if g.text == "caught"]
    assert caught and caught[0].start_time == pytest.approx(10.0, abs=1.0)


def test_close_failure_then_catch_up_keeps_content_in_order(fake_clock):
    start = fake_clock.t
    s = _outage_session(fake_clock, FailProc(fake_clock, fail_at=start + 10, on="close"))
    for t in range(1, 10):
        _feed(s, fake_clock, start + t, speech=True)
    _feed(s, fake_clock, start + 10, speech=False, silence=True)    # close fails
    assert s._engine_down and not s.segments
    for t in range(11, 15):
        _feed(s, fake_clock, start + t, speech=True)
    fake_clock.t = start + 15
    s._catch_up_buffered()
    assert s.segments and all(g.text == "caught" for g in s.segments)
    assert s.segments[0].start_time == pytest.approx(0.0, abs=0.1)   # nothing lost
    total = sum(g.end_time - g.start_time for g in s.segments)
    assert total == pytest.approx(14.0, abs=1.0)                     # all buffered audio
    starts = [g.start_time for g in s.segments]
    assert starts == sorted(starts) and all(g.end_time >= g.start_time for g in s.segments)
    assert s._tracks[0].segment_start is None                        # fresh segment after catch-up


# ── Task 3: two labelled tracks, echo removal ──

@pytest.fixture(autouse=True)
def _tmp_meetings_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(meeting, "MEETINGS_DIR", str(tmp_path))   # stop() auto-saves

class TagProc(FakeProc):
    """Returns text tagged by which channel's processor it is."""

    def __init__(self, text):
        super().__init__()
        self._text = text

    def segment_close(self):
        return self._text if self.committed else ""


def _dual_session(clock, mic_text="hello from mic", remote_text="hi from remote",
                  vad_factory=ScriptVAD):
    procs = iter([TagProc(mic_text), TagProc(remote_text)])
    s = MeetingSession(backend=None, vad=ScriptVAD(), audio_source=SimpleNamespace(),
                       processor_factory=lambda: next(procs, None) or CatchProc(),
                       vad_factory=vad_factory)
    s._dual_channel = True
    s._session_id = "20261107_120000"
    s._meeting_start = clock.t
    s._state = "recording"
    s._segments = []
    s._build_tracks()
    return s


def _tick(s, clock, t, mic=None, remote=None):
    """mic/remote: None = no audio, "speech" / "quiet" / "close" (silence)."""
    clock.t = t
    for track, mode in zip(s._tracks, (mic, remote)):
        if mode is not None:
            track.vad.speech = mode == "speech"
            track.vad.silence = mode == "close"
    s._process_new_audio(CHUNK if mic else [], CHUNK if remote else [])


def test_dual_lines_are_labelled_and_time_ordered(fake_clock):
    t0 = fake_clock.t
    s = _dual_session(fake_clock)
    for t in range(2, 5):
        _tick(s, fake_clock, t0 + t, mic="speech")
    for t in range(5, 8):
        _tick(s, fake_clock, t0 + t, remote="speech")
    _tick(s, fake_clock, t0 + 8, remote="close")      # remote commits first
    _tick(s, fake_clock, t0 + 9, mic="close")
    md = s._format_markdown()
    assert "**[00:02] Mic:** hello from mic" in md
    assert md.index("Mic:** hello") < md.index("Remote:** hi from remote")


def test_crosstalk_keeps_both_channels(fake_clock):
    t0 = fake_clock.t
    s = _dual_session(fake_clock)
    for t in range(1, 6):
        _tick(s, fake_clock, t0 + t, mic="speech", remote="speech")
    _tick(s, fake_clock, t0 + 6, mic="close", remote="close")
    assert {g.source for g in s.segments} == {"mic", "remote"}


def test_remote_only_meeting_has_no_empty_mic_lines(fake_clock):
    t0 = fake_clock.t
    s = _dual_session(fake_clock)
    for t in range(1, 6):
        _tick(s, fake_clock, t0 + t, mic="quiet", remote="speech")
    _tick(s, fake_clock, t0 + 6, mic="close", remote="close")
    s._mic_source = s._sys_source = SimpleNamespace(stop=lambda: None)
    s.stop(wait_s=0)
    assert s.segments and all(g.source == "remote" and g.text for g in s.segments)


def test_echo_mic_segment_dropped_remote_kept(fake_clock, caplog):
    t0 = fake_clock.t
    s = _dual_session(fake_clock, mic_text="OK, which warehouse is this?",
                      remote_text="OK which warehouse is this")
    for t in range(1, 6):
        _tick(s, fake_clock, t0 + t, mic="speech", remote="speech")
    # WARNING: the app's default log level; a dropped line must reach the log file
    with caplog.at_level("WARNING", logger="meeting"):
        _tick(s, fake_clock, t0 + 6, mic="close", remote="close")
    assert [g.source for g in s.segments] == ["remote"]
    assert "Echo dropped (Mic)" in caplog.text


def test_channel_mode_logged_at_default_level(fake_clock, caplog):
    with caplog.at_level("WARNING", logger="meeting"):
        _dual_session(fake_clock)
    assert "Meeting started: dual channel (Mic + Remote)" in caplog.text


def test_echo_dropped_when_mic_commits_after_remote(fake_clock):
    t0 = fake_clock.t
    s = _dual_session(fake_clock, mic_text="same words here", remote_text="same words here")
    for t in range(1, 6):
        _tick(s, fake_clock, t0 + t, mic="speech", remote="speech")
    _tick(s, fake_clock, t0 + 6, remote="close")
    _tick(s, fake_clock, t0 + 7, mic="close")
    assert [g.source for g in s.segments] == ["remote"]


class _Caught(FakeProc):
    def __init__(self, text):
        super().__init__()
        self._text = text

    def segment_close(self):
        return self._text if len(self.audio_buffer) else ""


def test_outage_catch_up_keeps_labels(fake_clock):
    t0 = fake_clock.t
    s = _dual_session(fake_clock)
    s._switch_to_fallback(RuntimeError("down"))
    for t in range(1, 6):
        _tick(s, fake_clock, t0 + t, mic="speech", remote="speech")
    texts = iter(["mic words caught", "", "remote reply caught", ""])   # chunk, fresh proc, ... (distinct: no echo)
    s._new_processor = lambda: _Caught(next(texts, ""))
    s._catch_up_buffered()
    assert {g.source for g in s.segments} == {"mic", "remote"}
    assert all(g.start_time >= 0 for g in s.segments)
    assert not s._engine_down


def test_stop_commits_pending_segments_of_both_tracks(fake_clock):
    t0 = fake_clock.t
    s = _dual_session(fake_clock)
    for t in range(1, 4):
        _tick(s, fake_clock, t0 + t, mic="speech", remote="speech")
    s._mic_source = s._sys_source = SimpleNamespace(stop=lambda: None)
    s.stop(wait_s=0)
    assert sorted(g.source for g in s.segments) == ["mic", "remote"]


def test_txt_and_cues_have_labels(fake_clock):
    s = _dual_session(fake_clock)
    s._segments = [MeetingSegment("hello", 1.0, 2.0, source="mic"),
                   MeetingSegment("hi", 3.0, 4.0, source="remote")]
    assert s._build_transcript().splitlines()[0].startswith("Mic: ")
    assert s._subtitle_cues()[0][2].startswith("Mic: ")
    assert s._subtitle_cues()[1][2] == "Remote: hi"


def test_each_track_has_its_own_vad(fake_clock):
    s = _dual_session(fake_clock)
    assert [t.source for t in s._tracks] == ["mic", "remote"]
    assert s._tracks[0].vad is s.vad
    assert s._tracks[1].vad is not None and s._tracks[1].vad is not s._tracks[0].vad


def test_no_vad_factory_remote_track_has_no_vad(fake_clock):
    s = _dual_session(fake_clock, vad_factory=None)
    assert s._tracks[1].vad is None


def test_mic_chunks_only_reach_mic_track(fake_clock):
    s = _dual_session(fake_clock)
    _tick(s, fake_clock, fake_clock.t + 1, mic="speech")
    assert len(s._tracks[0].processor.audio_buffer) == 16000
    assert len(s._tracks[1].processor.audio_buffer) == 0
