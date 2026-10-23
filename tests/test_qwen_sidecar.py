"""Tests for qwen_sidecar's request handling (no model, no subprocess)."""

import base64
import json

import numpy as np

import qwen_sidecar as qs


def _pcm_b64(samples):
    return base64.b64encode(np.asarray(samples, dtype=np.int16).tobytes()).decode()


def test_decode_audio_roundtrip():
    audio = qs.decode_audio(_pcm_b64([0, 16384, -32768]))
    assert audio.dtype == np.float32
    assert list(audio) == [0.0, 0.5, -1.0]


def test_transcribe_request_calls_model_with_chinese():
    calls = []

    def model_fn(audio, language):
        calls.append((len(audio), language))
        return "你好 world"

    resp = qs.handle({"id": 7, "cmd": "transcribe", "pcm16_b64": _pcm_b64([0] * 1600)}, model_fn)
    assert resp["id"] == 7 and resp["ok"] is True
    assert resp["text"] == "你好 world"
    assert calls == [(1600, "Chinese")]
    assert resp["audio_s"] == 0.1


def test_language_override():
    seen = []
    qs.handle({"id": 1, "cmd": "transcribe", "pcm16_b64": _pcm_b64([0]), "language": "English"},
              lambda a, language: seen.append(language) or "")
    assert seen == ["English"]


def test_model_error_is_reported_not_raised():
    def boom(audio, language):
        raise RuntimeError("metal oom")

    resp = qs.handle({"id": 2, "cmd": "transcribe", "pcm16_b64": _pcm_b64([0])}, boom)
    assert resp["ok"] is False and "metal oom" in resp["error"]


def test_bad_requests():
    assert qs.handle({"id": 3, "cmd": "nope"}, None)["ok"] is False
    assert qs.handle({"id": 4, "cmd": "transcribe"}, None)["ok"] is False   # no audio
    assert qs.handle({"id": 5, "cmd": "ping"}, None) == {"id": 5, "ok": True, "pong": True}


def test_parse_line_rejects_garbage():
    assert qs.parse_line("not json") is None
    assert qs.parse_line(json.dumps([1, 2])) is None
    assert qs.parse_line(json.dumps({"cmd": "ping"})) == {"cmd": "ping"}


def test_hotwords_passed_to_model():
    seen = {}

    def model_fn(audio, language, hotwords=None):
        seen["hw"] = hotwords
        return "ok"

    qs.handle({"id": 8, "cmd": "transcribe", "pcm16_b64": _pcm_b64([0]), "hotwords": ["CSE", "Snowflake"]},
              model_fn)
    assert seen["hw"] == ["CSE", "Snowflake"]


def test_no_hotwords_keeps_two_arg_call():
    calls = []
    qs.handle({"id": 9, "cmd": "transcribe", "pcm16_b64": _pcm_b64([0])},
              lambda audio, language: calls.append(language) or "ok")
    assert calls


def test_bad_hotwords_ignored():
    calls = []
    qs.handle({"id": 10, "cmd": "transcribe", "pcm16_b64": _pcm_b64([0]), "hotwords": "CSE"},
              lambda audio, language: calls.append(1) or "ok")
    assert calls


# ── Runaway generation guard (2026-10-23: 3 meeting timeouts from repetition loops) ──

class _Model:
    def __init__(self):
        self.calls = []

    def generate(self, audio, **kw):
        self.calls.append(kw)
        return type("R", (), {"text": "ok"})()


def test_token_cap_scales_with_audio_and_has_floor():
    assert qs.token_cap(1.0) == qs.MIN_TOKENS
    assert qs.token_cap(15.0) == 15 * qs.TOKENS_PER_AUDIO_S
    # 1321 s meeting (10/23): the budget is a total over all chunks, so it
    # must cover ~5 tokens/s of dense speech for the whole meeting
    assert qs.token_cap(1321.0) >= 5 * 1321
    # real speech: 120 chars ≈ 30 tokens in ~8 s; cap must leave a wide margin
    assert qs.token_cap(8.0) >= 5 * 30


def test_run_passes_max_tokens_to_generate():
    m = _Model()
    run = qs.make_run(m)
    run(np.zeros(qs.SAMPLE_RATE * 15, dtype=np.float32), "Chinese")
    run(np.zeros(qs.SAMPLE_RATE * 15, dtype=np.float32), "Chinese", ["CSE"])
    assert [c["max_tokens"] for c in m.calls] == [qs.token_cap(15.0)] * 2
    assert m.calls[1]["hotwords"] == ["CSE"]
