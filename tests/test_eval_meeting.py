"""Tests for the meeting transcription evaluation harness (eval_meeting.py).

Pure logic only — no ML model, no user recordings.
"""

import json
import os
import wave

import numpy as np
import pytest

import eval_meeting


def _write_wav(path, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


@pytest.fixture
def dirs(tmp_path):
    meetings = tmp_path / "meetings"
    refs = tmp_path / "eval"
    meetings.mkdir()
    refs.mkdir()
    return meetings, refs


def test_discover_only_sessions_with_reference_and_audio(dirs):
    meetings, refs = dirs
    _write_wav(meetings / "meeting_A_mic.wav", [0] * 16000)
    (refs / "A.ref.txt").write_text("hello")
    _write_wav(meetings / "meeting_B_mic.wav", [0] * 16000)      # no reference
    (refs / "C.ref.txt").write_text("no audio")                  # no audio
    found = eval_meeting.discover_sessions(str(meetings), str(refs))
    assert [s.session_id for s in found] == ["A"]
    assert found[0].reference == "hello"


def test_reference_comments_are_ignored(dirs):
    meetings, refs = dirs
    _write_wav(meetings / "meeting_A_mic.wav", [0] * 16000)
    (refs / "A.ref.txt").write_text("# DRAFT — please confirm\nhello world\n")
    s = eval_meeting.discover_sessions(str(meetings), str(refs))[0]
    assert s.reference == "hello world"


def test_load_audio_mixes_mic_and_sys(dirs):
    meetings, refs = dirs
    _write_wav(meetings / "meeting_A_mic.wav", [1000] * 16000)
    _write_wav(meetings / "meeting_A_sys.wav", [500] * 8000)    # shorter channel
    (refs / "A.ref.txt").write_text("x")
    s = eval_meeting.discover_sessions(str(meetings), str(refs))[0]
    audio = s.load_audio()
    assert audio.dtype == np.float32
    assert len(audio) == 16000
    assert audio[0] == pytest.approx(1500 / 32768, rel=1e-3)
    assert audio[-1] == pytest.approx(1000 / 32768, rel=1e-3)


def test_load_audio_mix_does_not_overflow(dirs):
    meetings, refs = dirs
    _write_wav(meetings / "meeting_A_mic.wav", [30000] * 100)
    _write_wav(meetings / "meeting_A_sys.wav", [30000] * 100)
    (refs / "A.ref.txt").write_text("x")
    audio = eval_meeting.discover_sessions(str(meetings), str(refs))[0].load_audio()
    assert np.max(np.abs(audio)) <= 1.0


def test_realtime_strategy_reads_saved_transcript(dirs):
    meetings, refs = dirs
    _write_wav(meetings / "meeting_A_mic.wav", [0] * 16000)
    (meetings / "meeting_A.json").write_text(json.dumps({"transcript": "app output"}))
    (refs / "A.ref.txt").write_text("x")
    s = eval_meeting.discover_sessions(str(meetings), str(refs))[0]
    assert eval_meeting.STRATEGIES["realtime"](s, backend=None) == "app output"


def test_fullpass_strategy_uses_backend_once(dirs):
    meetings, refs = dirs
    _write_wav(meetings / "meeting_A_mic.wav", [0] * 32000)
    (refs / "A.ref.txt").write_text("x")
    s = eval_meeting.discover_sessions(str(meetings), str(refs))[0]

    class FakeResult:
        text = "  full text  "

    class FakeBackend:
        calls = []

        def transcribe(self, audio, **kw):
            self.calls.append((len(audio), kw))
            return FakeResult()

    b = FakeBackend()
    assert eval_meeting.STRATEGIES["fullpass"](s, backend=b) == "full text"
    assert len(b.calls) == 1 and b.calls[0][0] == 32000


def test_score_perfect_and_partial():
    perfect = eval_meeting.score("今天 hello world", "今天 hello world")
    assert perfect["mer"] == 0.0 and perfect["recall"] == 1.0
    partial = eval_meeting.score("今天 hello", "今天 hello world")
    assert 0 < partial["mer"] < 1
    assert partial["recall"] < 1.0


def test_format_table_contains_rows():
    rows = [{"session": "A", "strategy": "realtime", "model": "-", "mer": 0.5,
             "recall": 0.6, "seconds": 0.0, "audio_s": 30.0, "text": "abc"}]
    out = eval_meeting.format_table(rows)
    assert "realtime" in out and "0.500" in out and "A" in out
