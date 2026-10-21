"""eval_meeting_live: the harness itself, with a fake backend (no model)."""

import os
from types import SimpleNamespace

import numpy as np

import eval_meeting_live as em
import meeting


class FakeQwen:
    def __init__(self):
        self.n = 0

    def transcribe(self, audio, *a, **k):
        self.n += 1
        return SimpleNamespace(text="大家好今天开会")


class TalkThenSilenceVad:
    """Speech for the first 2 s of audio, then silence."""

    def __init__(self):
        self.samples = 0

    def reset(self):
        pass

    def process_chunk(self, chunk):
        self.samples += len(chunk)
        sp = self.samples < 2 * 16000
        return {"is_speech": sp, "silence_ms": 0 if sp else 1500, "speech_ms": 100}

    def is_extended_silence(self, *a, **k):
        return self.samples >= 2 * 16000


def test_replay_runs_a_real_meeting_session_and_measures(tmp_path):
    audio = (np.random.RandomState(0).randn(16000 * 4) * 0.1).astype(np.float32)
    before = set(os.listdir(meeting.MEETINGS_DIR)) if os.path.isdir(meeting.MEETINGS_DIR) else set()
    r = em.replay("synthetic", audio, None, engine="qwen3", backend=FakeQwen(),
                  vad=TalkThenSilenceVad(), reference="大家好今天开会", speed=4.0)
    s = r.summary()
    assert s["decodes"] >= 1 and s["segments"] >= 1
    assert s["mer"] is not None and s["mer"] < 0.5
    assert s["first_text_s"] is not None
    after = set(os.listdir(meeting.MEETINGS_DIR)) if os.path.isdir(meeting.MEETINGS_DIR) else set()
    assert after == before                      # wrote to a temp dir only


def test_no_recordings_means_nothing_to_measure(monkeypatch, capsys):
    monkeypatch.setattr(em, "discover_sessions", lambda: [])
    assert em.main(["--engine", "qwen3"]) == 1
    assert "nothing to measure" in capsys.readouterr().out
