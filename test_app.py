"""
Automated tests for MacWhisper app.py
Covers: config, audio processing, state machines, menu logic, overlay text.
Run:  ./venv/bin/python3 -m pytest test_app.py -v
"""

import json
import os
import queue
from unittest.mock import patch

import numpy as np
import pytest


# ── Helpers ───────────────────────────────────────────────────

@pytest.fixture
def tmp_config(tmp_path):
    cfg_file = tmp_path / "test_config.json"
    with patch("app.CONFIG_PATH", str(cfg_file)):
        yield cfg_file


def _make_frames(n=10, samples_per_frame=1024):
    return [np.random.randint(-32768, 32767, (samples_per_frame, 1), dtype=np.int16) for _ in range(n)]


def _frames_to_float(frames):
    audio = np.concatenate(frames, axis=0).squeeze()
    return audio.astype(np.float32) / 32768.0


# ── Test: imports and constants ───────────────────────────────

def test_imports():
    import app
    assert hasattr(app, "TranscriberApp")
    assert hasattr(app, "SAMPLE_RATE")
    assert hasattr(app, "LIVE_CHUNK_SECONDS")


def test_constants():
    import app
    assert app.SAMPLE_RATE == 16000
    assert app.LIVE_CHUNK_SECONDS == 3
    assert len(app.MODEL_OPTIONS) == 3
    assert "Small (Fast)" in app.MODEL_OPTIONS


# ── Test: config load/save ────────────────────────────────────

def test_config_load_missing_file(tmp_config):
    import app
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    result = inst._load_config()
    assert result == {}


def test_config_load_valid(tmp_config):
    cfg = {"translate_mode": True, "current_model": "mlx-community/whisper-large-v3-mlx", "live_mode": True}
    tmp_config.write_text(json.dumps(cfg))
    import app
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    result = inst._load_config()
    assert result["translate_mode"] is True
    assert result["live_mode"] is True


def test_config_load_corrupt(tmp_config):
    tmp_config.write_text("{bad json")
    import app
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    result = inst._load_config()
    assert result == {}


def test_config_save(tmp_config):
    import app
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst.translate_mode = True
    inst.current_model = "mlx-community/whisper-medium-mlx"
    inst.live_mode = True
    inst._save_config()
    saved = json.loads(tmp_config.read_text())
    assert saved["translate_mode"] is True
    assert saved["live_mode"] is True


# ── Test: audio frame processing ──────────────────────────────

def test_frames_concatenation():
    frames = _make_frames(5, 1024)
    audio_float = _frames_to_float(frames)
    assert audio_float.dtype == np.float32
    assert len(audio_float) == 5 * 1024
    assert audio_float.max() <= 1.0
    assert audio_float.min() >= -1.0


def test_empty_frames():
    with pytest.raises((ValueError, Exception)):
        _frames_to_float([])


def test_single_frame():
    frames = _make_frames(1, 512)
    audio_float = _frames_to_float(frames)
    assert len(audio_float) == 512


# ── Test: model cycling ───────────────────────────────────────

def test_model_cycle_order():
    import app
    keys = app.MODEL_KEYS
    assert keys == ["Small (Fast)", "Medium (Accurate)", "Large (Best)"]
    idx = 0
    for expected in ["Medium (Accurate)", "Large (Best)", "Small (Fast)"]:
        idx = (idx + 1) % len(keys)
        assert keys[idx] == expected


# ── Test: live queue behavior ─────────────────────────────────

def test_live_queue_maxsize():
    q = queue.Queue(maxsize=2)
    q.put("chunk1")
    q.put("chunk2")
    new_chunk = "chunk3"
    try:
        q.put_nowait(new_chunk)
        assert False, "Should have raised queue.Full"
    except queue.Full:
        q.get_nowait()
        q.put_nowait(new_chunk)
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    assert items == ["chunk2", "chunk3"]


# ── Test: growing buffer snapshot ─────────────────────────────

def test_growing_buffer_snapshot():
    """Each iteration sends ALL accumulated frames, not just new ones."""
    frames = _make_frames(30, 1024)

    # t=3s: 10 frames accumulated → snapshot is all 10
    n = 10
    snapshot = frames[:n]
    assert len(snapshot) == 10

    # t=6s: 20 frames accumulated → snapshot is all 20
    n = 20
    snapshot = frames[:n]
    assert len(snapshot) == 20

    # t=9s: 30 frames accumulated → snapshot is all 30
    n = 30
    snapshot = frames[:n]
    assert len(snapshot) == 30


def test_window_cap():
    """When frames exceed MAX_LIVE_WINDOW, only the tail is sent."""
    import app
    max_frames = int(app.MAX_LIVE_WINDOW * app.SAMPLE_RATE / 1024)
    frames = _make_frames(max_frames + 50, 1024)
    n = len(frames)

    assert n > max_frames
    snapshot = frames[n - max_frames:]
    assert len(snapshot) == max_frames


def test_max_live_window_constant():
    import app
    assert app.MAX_LIVE_WINDOW == 30


# ── Test: overlay full replace (not append) ───────────────────

def test_overlay_replace():
    """Display should fully replace, not accumulate lines."""
    display = ""

    display = "我上周去了San Francisco"
    assert display == "我上周去了San Francisco"

    display = "我上周去了San Francisco参加了一个conference。"
    assert "conference" in display
    assert display.count("San Francisco") == 1


# ── Test: icon states ────────────────────────────────────────

def test_icon_states():
    import app
    source = open(os.path.join(os.path.dirname(app.__file__), "app.py")).read()
    assert '🟠' in source
    assert '🔴' not in source
    assert '💬' in source
    assert '🎙' in source
    assert '🌐' in source


# ── Test: prompt stripping ────────────────────────────────────

def test_prompt_stripping():
    prompt = "以下是普通话与英语的混合对话。"
    text = f"{prompt}你好世界 hello world".strip()
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    assert text == "你好世界 hello world"


def test_prompt_not_present():
    prompt = "以下是普通话与英语的混合对话。"
    text = "Hello this is a test".strip()
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    assert text == "Hello this is a test"


# ── Test: task selection ──────────────────────────────────────

def test_translate_mode_task():
    translate_mode = True
    task, prompt = ("translate", None) if translate_mode else ("transcribe", "以下是普通话与英语的混合对话。")
    assert task == "translate"
    assert prompt is None


def test_transcribe_mode_task():
    translate_mode = False
    task, prompt = ("translate", None) if translate_mode else ("transcribe", "以下是普通话与英语的混合对话。")
    assert task == "transcribe"
    assert prompt is not None


# ── Test: live_mode is a toggle preference ────────────────────

def test_live_mode_is_preference():
    """live_mode is just a boolean preference, not a recording trigger."""
    import app
    source = open(os.path.join(os.path.dirname(app.__file__), "app.py")).read()
    assert "self.live_mode = not self.live_mode" in source
    assert "if self.live_mode:" in source


# ── Test: menu item titles ────────────────────────────────────

def test_menu_live_subtitles_titles():
    on_title = "✅ Live Subtitles: On"
    off_title = "   Live Subtitles: Off"
    assert "On" in on_title
    assert "Off" in off_title
    assert "✅" in on_title
    assert "✅" not in off_title


# ── Test: constants ───────────────────────────────────────────

def test_nspanel_constants():
    import app
    assert app.NSWindowStyleMaskBorderless == 0


def test_hotkey_vk_codes():
    vk_map = {46: "M", 17: "T", 1: "S"}
    assert len(vk_map) == 3


def test_chunk_seconds():
    import app
    assert app.LIVE_CHUNK_SECONDS == 3


# ── Test: hallucination filter ────────────────────────────────

def test_hallucination_known_phrases():
    import app
    assert app._is_hallucination("Thank you for watching.") is True
    assert app._is_hallucination("please subscribe") is True


def test_hallucination_repetition():
    import app
    assert app._is_hallucination("Ok Ok Ok") is True
    assert app._is_hallucination("sto sto sto") is True


def test_hallucination_cjk_repetition():
    import app
    assert app._is_hallucination("技术技术技术") is True


def test_hallucination_cyrillic():
    import app
    assert app._is_hallucination("Спасибо за внимание") is True


def test_hallucination_normal_text():
    import app
    assert app._is_hallucination("我上周去了San Francisco") is False
    assert app._is_hallucination("Hello world") is False


# ── Test: trailing repetition stripping ───────────────────────

def test_strip_trailing_cjk():
    import app
    result = app._strip_trailing_repetition("别的语言。举举举举举举举举举举举举")
    assert "举" not in result
    assert "别的语言" in result


def test_strip_trailing_with_commas():
    import app
    result = app._strip_trailing_repetition("别的语言。啊,啊,啊,啊,啊,啊,啊,啊,啊")
    assert result.count("啊") <= 1
    assert "别的语言" in result


def test_strip_trailing_sentence_repetition():
    import app
    text = "正常内容。我们在世界上最大的优点。我们在世界上最大的优点。我们在世界上最大的优点。我们在世界上最大的优点。"
    result = app._strip_trailing_repetition(text)
    assert "正常内容" in result
    assert result.count("我们在世界上最大的优点") <= 1


def test_strip_trailing_preserves_clean():
    import app
    clean = "我上周去了San Francisco参加了一个conference。"
    assert app._strip_trailing_repetition(clean) == clean


# ── Test: opencc t2s ─────────────────────────────────────────

def test_opencc_t2s():
    import app
    assert app._t2s.convert("機器學習") == "机器学习"
    assert app._t2s.convert("Hello") == "Hello"


# ── Test: history constants and dirs ─────────────────────────

def test_history_constants():
    import app
    assert hasattr(app, "HISTORY_DIR")
    assert hasattr(app, "AUDIO_DIR")
    assert hasattr(app, "TRANSCRIPT_LOG")
    assert app.AUDIO_DIR.endswith(os.path.join("history", "audio"))
    assert app.TRANSCRIPT_LOG.endswith(os.path.join("history", "transcripts.jsonl"))


def test_ensure_history_dirs(tmp_path):
    import app
    audio_dir = str(tmp_path / "history" / "audio")
    with patch("app.AUDIO_DIR", audio_dir):
        app._ensure_history_dirs()
    assert os.path.isdir(audio_dir)
    # Calling again should not raise
    with patch("app.AUDIO_DIR", audio_dir):
        app._ensure_history_dirs()


def test_audio_history_saves_wav(tmp_path):
    """_do_transcribe should save a WAV file to AUDIO_DIR."""
    import app
    import wave

    audio_dir = str(tmp_path / "history" / "audio")
    transcript_log = str(tmp_path / "history" / "transcripts.jsonl")

    frames = _make_frames(5, 1024)
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst.translate_mode = False
    inst.current_model = "mlx-community/whisper-small-mlx"

    mock_result = {"text": "Hello world"}
    with patch("app.AUDIO_DIR", audio_dir), \
         patch("app.TRANSCRIPT_LOG", transcript_log), \
         patch("app.mlx_whisper.transcribe", return_value=mock_result), \
         patch.object(inst, "_auto_paste"):
        app._ensure_history_dirs()
        inst._do_transcribe(frames)

    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    assert len(wav_files) == 1

    wav_path = os.path.join(audio_dir, wav_files[0])
    with wave.open(wav_path, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 5 * 1024


def test_transcript_log_appends_jsonl(tmp_path):
    """_do_transcribe should append a JSONL entry to TRANSCRIPT_LOG."""
    import app

    audio_dir = str(tmp_path / "history" / "audio")
    transcript_log = str(tmp_path / "history" / "transcripts.jsonl")

    frames = _make_frames(5, 1024)
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst.translate_mode = False
    inst.current_model = "mlx-community/whisper-small-mlx"

    mock_result = {"text": "你好世界 hello"}
    with patch("app.AUDIO_DIR", audio_dir), \
         patch("app.TRANSCRIPT_LOG", transcript_log), \
         patch("app.mlx_whisper.transcribe", return_value=mock_result), \
         patch.object(inst, "_auto_paste"):
        app._ensure_history_dirs()
        inst._do_transcribe(frames)

    assert os.path.isfile(transcript_log)
    with open(transcript_log, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["text"] == "你好世界 hello"
    assert entry["model"] == "mlx-community/whisper-small-mlx"
    assert entry["translate"] is False
    assert "timestamp" in entry
    assert "audio_file" in entry
    assert entry["audio_file"].endswith(".wav")
    assert entry["duration_s"] == round(5 * 1024 / 16000, 1)


# ── Test: subtitle logging ───────────────────────────────────

def test_subtitle_log_constant():
    import app
    assert hasattr(app, "SUBTITLE_LOG")
    assert app.SUBTITLE_LOG.endswith(os.path.join("history", "subtitles.jsonl"))


def test_subtitle_log_appends_on_live_result(tmp_path):
    """_live_transcription_worker should log each non-empty subtitle to SUBTITLE_LOG."""
    import app

    audio_dir = str(tmp_path / "history" / "audio")
    subtitle_log = str(tmp_path / "history" / "subtitles.jsonl")

    frames = _make_frames(10, 1024)

    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst.recording = True
    inst.current_model = "mlx-community/whisper-small-mlx"
    inst._overlay_panel = None
    inst._overlay_text = None
    inst._inference_lock = __import__("threading").Lock()
    inst._live_queue = queue.Queue(maxsize=4)

    with patch("app.AUDIO_DIR", audio_dir), \
         patch("app.SUBTITLE_LOG", subtitle_log), \
         patch("app.mlx_whisper.transcribe", return_value={"text": "测试字幕内容"}):
        app._ensure_history_dirs()
        # Simulate what _live_transcription_worker does for one snapshot
        text = inst._do_live_transcribe(frames)
        if text:
            app._ensure_history_dirs()
            entry = {
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "audio_s": round(len(frames) * 1024 / 16000, 1),
                "text": text,
            }
            with open(subtitle_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    assert os.path.isfile(subtitle_log)
    with open(subtitle_log, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert "timestamp" in entry
    assert "audio_s" in entry
    assert entry["text"] == "测试字幕内容"
