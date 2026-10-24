"""LLM sidecar protocol + LLMSidecarBackend against the real sidecar in --fake mode."""

import sys

import pytest

import llm_backend as lb
import llm_sidecar as ls


def test_handle_complete_and_errors():
    fn = lambda system, prompt, n: f"{system}|{prompt}|{n}"
    assert ls.handle({"id": 1, "cmd": "complete", "system": "S", "prompt": "P", "max_tokens": 9999}, fn) == \
        {"id": 1, "ok": True, "text": f"S|P|{ls.MAX_TOKENS_CAP}", "infer_s": pytest.approx(0, abs=1)}
    assert ls.handle({"id": 2, "cmd": "complete"}, fn)["ok"] is False
    assert ls.handle({"id": 3, "cmd": "nope"}, fn)["ok"] is False
    assert ls.handle({"id": 4, "cmd": "ping"}, fn)["pong"] is True

    def boom(*a):
        raise RuntimeError("x")
    assert ls.handle({"id": 5, "cmd": "complete", "prompt": "p"}, boom)["ok"] is False


def _backend(tmp_path, env=None, **kw):
    return lb.LLMSidecarBackend(model_repo="fake", python_path=sys.executable, extra_args=["--fake"],
                                env=env, log_path=str(tmp_path / "llm.log"), **kw)


def test_complete_via_fake_sidecar(tmp_path):
    b = _backend(tmp_path, env={"MW_LLM_FAKE_RESPONSE": '[{"heard": "哈什", "correct": "hash"}]'})
    try:
        assert b.start(timeout=30)
        assert b.complete("找出听错的词").startswith('[{"heard"')
    finally:
        b.close()


def test_timeout_raises_llm_unavailable(tmp_path):
    b = _backend(tmp_path, env={"MW_LLM_FAKE_DELAY_S": "3"}, max_restarts=0)
    try:
        assert b.start(timeout=30)
        with pytest.raises(lb.LLMUnavailable):
            b.complete("x", timeout=0.5)
    finally:
        b.close()


def test_missing_env_is_unavailable(tmp_path):
    b = lb.LLMSidecarBackend(python_path=str(tmp_path / "nope"), log_path=str(tmp_path / "l.log"),
                             max_restarts=0)
    with pytest.raises(lb.LLMUnavailable):
        b.complete("x")


def test_busy_backend_gives_up_quickly_with_lock_timeout(tmp_path):
    """The beautifier must not wait behind a long vocabulary-learning request."""
    import threading
    import time as _t
    b = _backend(tmp_path, env={"MW_LLM_FAKE_DELAY_S": "2"}, max_restarts=0)
    try:
        assert b.start(timeout=30)
        t = threading.Thread(target=lambda: b.complete("long job", timeout=10))
        t.start()
        _t.sleep(0.3)
        t0 = _t.time()
        with pytest.raises(lb.LLMUnavailable, match="busy"):
            b.complete("dictation", timeout=5, lock_timeout=0.2)
        assert _t.time() - t0 < 1.0
        t.join(10)
    finally:
        b.close()
