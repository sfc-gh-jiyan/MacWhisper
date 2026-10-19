"""Tests for QwenSidecarBackend against the real sidecar in --fake mode."""

import os
import sys
import threading

import numpy as np
import pytest

import qwen_backend as qb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _backend(tmp_path, env=None, **kw):
    b = qb.QwenSidecarBackend(
        model_repo="fake",
        python_path=sys.executable,
        extra_args=["--fake"],
        env=env,
        log_path=str(tmp_path / "sidecar.log"),
        **kw,
    )
    return b


def _audio(seconds):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def test_start_and_transcribe(tmp_path):
    b = _backend(tmp_path)
    try:
        assert b.start(timeout=20) is True
        assert b.is_ready
        r = b.transcribe(_audio(2.0))
        assert r.text == "fake 2.0s Chinese"
        assert r.segments == []          # Qwen3 gives no word timestamps
    finally:
        b.close()
    assert not b.is_ready


def test_translate_is_rejected_so_caller_uses_whisper(tmp_path):
    b = _backend(tmp_path)
    with pytest.raises(qb.QwenUnavailable):
        b.transcribe(_audio(1.0), task="translate")
    b.close()


def test_missing_env_is_unavailable_not_crash(tmp_path):
    b = qb.QwenSidecarBackend(model_repo="x", python_path=str(tmp_path / "nope" / "python"),
                              log_path=str(tmp_path / "s.log"))
    assert b.start(timeout=2) is False
    assert "not found" in b.last_error
    with pytest.raises(qb.QwenUnavailable):
        b.transcribe(_audio(1.0))


def test_timeout_kills_and_raises(tmp_path):
    b = _backend(tmp_path, env={"QWEN_FAKE_DELAY_S": "3"}, base_timeout_s=0.5, per_audio_s=0.0)
    try:
        assert b.start(timeout=20)
        with pytest.raises(qb.QwenUnavailable, match="timed out"):
            b.transcribe(_audio(1.0))
        assert not b.is_ready
    finally:
        b.close()


def test_crash_then_auto_restart(tmp_path):
    b = _backend(tmp_path, env={"QWEN_FAKE_CRASH_AFTER": "1"})
    try:
        assert b.start(timeout=20)
        assert b.transcribe(_audio(1.0)).text.startswith("fake")
        with pytest.raises(qb.QwenUnavailable):
            b.transcribe(_audio(1.0))           # sidecar exits on 2nd call
        assert not b.is_ready
        r = b.transcribe(_audio(1.0))           # restarted transparently
        assert r.text.startswith("fake")
        assert b.restarts == 1
    finally:
        b.close()


def test_restart_budget_is_limited(tmp_path):
    b = _backend(tmp_path, env={"QWEN_FAKE_CRASH_AFTER": "0"}, max_restarts=1)
    try:
        b.start(timeout=20)
        for _ in range(3):
            with pytest.raises(qb.QwenUnavailable):
                b.transcribe(_audio(1.0))
        assert b.restarts == 1
        assert "gave up" in b.last_error
        assert b.last_error.count("gave up") == 1        # no nesting on repeats
    finally:
        b.close()


def test_concurrent_requests_are_serialized(tmp_path):
    b = _backend(tmp_path, env={"QWEN_FAKE_DELAY_S": "0.2"})
    try:
        assert b.start(timeout=20)
        out = []
        ts = [threading.Thread(target=lambda s=s: out.append(b.transcribe(_audio(s)).text))
              for s in (1.0, 2.0, 3.0)]
        [t.start() for t in ts]
        [t.join(10) for t in ts]
        assert sorted(out) == ["fake 1.0s Chinese", "fake 2.0s Chinese", "fake 3.0s Chinese"]
    finally:
        b.close()


def test_sidecar_stderr_goes_to_log_file(tmp_path):
    b = _backend(tmp_path)
    b.start(timeout=20)
    b.close()
    assert os.path.isfile(tmp_path / "sidecar.log")


def test_default_python_path_points_at_repo_venv():
    assert qb.default_python_path(REPO).endswith(os.path.join(".venv-qwen", "bin", "python"))


def test_sidecar_killed_between_requests_records_exit_code(tmp_path):
    b = _backend(tmp_path, max_restarts=0)
    try:
        assert b.start(timeout=20)
        b._proc.kill()
        b._proc.wait(5)
        with pytest.raises(qb.QwenUnavailable):
            b.transcribe(_audio(1.0))
        assert "exited (code" in b.last_error
    finally:
        b.close()
