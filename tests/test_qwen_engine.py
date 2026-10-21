"""Tests for qwen_engine: live loop, final pass, fallback (no model)."""

import threading

import numpy as np
import pytest

import qwen_engine as qe
from qwen_backend import QwenUnavailable
from asr_backend import TranscriptionResult


class FakeBackend:
    """Echoes a growing sentence: N seconds of audio -> first N words."""

    WORDS = "我 上 周 去 了 San Francisco 参 加 会 议".split()

    def __init__(self, fail=False, text=None):
        self.fail = fail
        self.text = text
        self.calls = []
        self.is_ready = True
        self.last_error = ""

    def transcribe(self, audio, **kw):
        self.calls.append(len(audio))
        if self.fail:
            raise QwenUnavailable("boom")
        if self.text is not None:
            return TranscriptionResult(text=self.text)
        n = int(len(audio) / qe.SAMPLE_RATE)
        return TranscriptionResult(text=" ".join(self.WORDS[:n]))

    def start(self, timeout=0):
        return True

    def close(self):
        pass


class Clock:
    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _frames(seconds):
    return [np.zeros((qe.SAMPLE_RATE, 1), dtype=np.int16) for _ in range(int(seconds))]


def test_live_loop_updates_overlay_with_stable_text():
    clock = Clock()
    frames = []
    updates = []
    state = {"recording": True}

    def is_recording():
        # one second of audio arrives per second of wall time; stop after 6 s
        while len(frames) < int(clock.t):
            frames.append(np.zeros((qe.SAMPLE_RATE, 1), dtype=np.int16))
        if clock.t >= 6:
            state["recording"] = False
        return state["recording"]

    qe.run_live_loop(FakeBackend(), lambda: list(frames), is_recording,
                     lambda c, u: updates.append((c, u)), step=1.0,
                     now=clock.time, sleep=clock.sleep)
    assert updates, "overlay never updated"
    confirmed, tail = updates[-1]
    assert confirmed.replace(" ", "").startswith("我上周去")


def test_live_loop_shows_hint_when_no_result():
    clock = Clock()
    updates = []
    t = {"stop": 6}

    qe.run_live_loop(FakeBackend(text=""), lambda: _frames(clock.t),
                     lambda: clock.t < t["stop"], lambda c, u: updates.append((c, u)),
                     step=1.0, hint_after_s=4.0, now=clock.time, sleep=clock.sleep)
    assert any(qe.SLOW_HINT in u for _, u in updates)


def test_live_loop_backend_failure_stops_quietly_and_reports():
    clock = Clock()
    errors = []
    qe.run_live_loop(FakeBackend(fail=True), lambda: _frames(max(clock.t, 2)),
                     lambda: clock.t < 5, lambda c, u: None, step=1.0,
                     on_error=errors.append, now=clock.time, sleep=clock.sleep)
    assert len(errors) == 1                      # reported once, not every second
    assert isinstance(errors[0], QwenUnavailable)


def test_final_pass_postprocesses():
    eng = qe.QwenEngine(backend_factory=lambda repo: FakeBackend(text="嗯，这个 commit 的情况"))
    eng.start("m")
    assert eng.transcribe_final(np.zeros(16000, dtype=np.float32)) == "这个commit的情况"


def test_final_pass_raises_for_fallback():
    eng = qe.QwenEngine(backend_factory=lambda repo: FakeBackend(fail=True))
    eng.start("m")
    with pytest.raises(QwenUnavailable):
        eng.transcribe_final(np.zeros(16000, dtype=np.float32))


def test_engine_not_started_is_unavailable():
    eng = qe.QwenEngine(backend_factory=lambda repo: FakeBackend())
    assert not eng.is_ready
    with pytest.raises(QwenUnavailable):
        eng.transcribe_final(np.zeros(16000, dtype=np.float32))


def test_switch_model_closes_previous():
    closed = []

    class B(FakeBackend):
        def __init__(self, repo):
            super().__init__()
            self.repo = repo

        def close(self):
            closed.append(self.repo)

    eng = qe.QwenEngine(backend_factory=B)
    eng.start("small")
    eng.start("big")
    assert closed == ["small"]
    assert eng.model_repo == "big"


def test_start_async_reports_result():
    done = threading.Event()
    result = {}

    class Slow(FakeBackend):
        def __init__(self):
            super().__init__()
            self.last_error = "env missing"

        def start(self, timeout=0):
            return False

    eng = qe.QwenEngine(backend_factory=lambda repo: Slow())
    eng.start_async("m", on_done=lambda ok, err: (result.update(ok=ok, err=err), done.set()))
    assert done.wait(5)
    assert result == {"ok": False, "err": "env missing"}
    assert not eng.is_ready


def test_hotwords_applied_to_current_and_new_backends():
    made = []

    def factory(repo):
        b = FakeBackend()
        made.append(b)
        return b

    eng = qe.QwenEngine(backend_factory=factory)
    eng.set_hotwords(["CSE"])
    eng.start("m1")
    assert made[-1].hotwords == ["CSE"]
    eng.set_hotwords(["CSE", "Snowflake"])
    assert made[-1].hotwords == ["CSE", "Snowflake"]
    eng.start("m2")
    assert made[-1].hotwords == ["CSE", "Snowflake"]
