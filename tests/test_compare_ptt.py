"""Pure-logic tests for compare_ptt (no models, no user recordings)."""

import json
import wave

import numpy as np

import compare_ptt


def test_looks_translated():
    assert compare_ptt.looks_translated("我们 hello", "we hello")
    assert not compare_ptt.looks_translated("我们 hello", "我们 hello")
    assert not compare_ptt.looks_translated("hello", "hello")     # no Chinese to lose
    assert not compare_ptt.looks_translated("我们", "")             # empty is not "translated"


def test_code_switched():
    assert compare_ptt.is_code_switched("这个 repository 很好")
    assert not compare_ptt.is_code_switched("这个很好")
    assert not compare_ptt.is_code_switched("all english")


def test_bucket_edges():
    assert [compare_ptt.bucket(x) for x in (4.9, 5, 14.9, 15, 30.9, 31)] == [
        "<5s", "5-15s", "5-15s", "15-31s", "15-31s", ">=31s"]


def test_stratified_sample_spreads_and_prefers_mixed():
    clips = [{"duration_s": d, "pasted": t, "wav": f"{i}"}
             for i, (d, t) in enumerate([(3, "短"), (3, "短 mixed"), (10, "中"), (20, "长"), (40, "很长")])]
    s = compare_ptt.stratified_sample(clips, 4)
    assert {compare_ptt.bucket(c["duration_s"]) for c in s} == {"<5s", "5-15s", "15-31s", ">=31s"}
    assert any(c["pasted"] == "短 mixed" for c in s)


def test_load_clips_links_wav_and_skips_translate(tmp_path):
    (tmp_path / "audio").mkdir()
    for name in ("a.wav", "b.wav"):
        with wave.open(str(tmp_path / "audio" / name), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(np.zeros(160, dtype=np.int16).tobytes())
    rows = [
        {"audio_file": "/old/path/a.wav", "text": "你好", "duration_s": 1.0, "translate": False},
        {"audio_file": "b.wav", "text": "hi", "duration_s": 1.0, "translate": True},
        {"audio_file": None, "text": "no audio", "duration_s": 1.0},
        {"audio_file": "missing.wav", "text": "gone", "duration_s": 1.0},
    ]
    (tmp_path / "transcripts.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    clips = compare_ptt.load_clips(str(tmp_path))
    assert [c["pasted"] for c in clips] == ["你好"]
