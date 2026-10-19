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
