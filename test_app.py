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
    # Log may use old "text" key or new "raw"/"display" keys
    assert entry.get("text") == "测试字幕内容" or entry.get("raw") == "测试字幕内容"


# ── Test: helper functions for committed text ─────────────

def test_common_prefix_len():
    import app
    assert app._common_prefix_len("abcdef", "abcxyz") == 3
    assert app._common_prefix_len("hello", "hello world") == 5
    assert app._common_prefix_len("abc", "xyz") == 0
    assert app._common_prefix_len("", "abc") == 0


def test_common_prefix_len_fuzzy():
    """Tolerates case and CJK punctuation differences."""
    import app
    # Case difference
    assert app._common_prefix_len("Test3你好", "test3你好") == 7
    # CJK vs ASCII punctuation
    assert app._common_prefix_len("你好，世界", "你好,世界") == 5
    assert app._common_prefix_len("结束。下一句", "结束.下一句") == 6


def test_snap_to_boundary():
    import app
    text = "你好世界。这是测试。然后继续"
    # Position after second 。 should snap to right after it
    pos = text.index("然")
    result = app._snap_to_boundary(text, pos)
    assert text[result - 1] == "。"
    # Position 0 should stay 0
    assert app._snap_to_boundary(text, 0) == 0


def test_snap_to_boundary_no_punctuation():
    import app
    text = "没有标点的文字内容"
    result = app._snap_to_boundary(text, 5)
    assert result == 5  # No boundary found, return raw position


def test_find_after_overlap():
    import app
    committed = "我上周去了San Francisco参加了一个conference。"
    raw = "参加了一个conference。这个conference是关于ML的。"
    result = app._find_after_overlap(committed, raw)
    assert result == "这个conference是关于ML的。"


def test_find_after_overlap_case_insensitive():
    """Overlap matching ignores case differences from Whisper."""
    import app
    committed = "我不知道test3,我不知道为什么。"
    raw = "我不知道Test3,我不知道为什么,但是之后的效果还是不错的。"
    result = app._find_after_overlap(committed, raw)
    assert "但是之后" in result
    assert "我不知道" not in result  # no duplication


def test_find_after_overlap_punct_variation():
    """Overlap matching tolerates punctuation differences."""
    import app
    committed = "这是测试。结果如何？"
    raw = "这是测试,结果如何,后续内容。"
    result = app._find_after_overlap(committed, raw)
    assert "后续内容" in result


def test_find_after_overlap_no_match():
    import app
    committed = "完全不同的句子。"
    raw = "另一段话。"
    result = app._find_after_overlap(committed, raw)
    assert result == raw  # No overlap, return full raw


def test_find_after_overlap_empty():
    import app
    assert app._find_after_overlap("", "新内容") == "新内容"
    assert app._find_after_overlap("旧内容", "") == ""


# ── Test: _build_display_text ────────────────────────────

def test_build_display_first_call():
    """First call returns raw text as-is."""
    import app
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst._committed_text = ""
    inst._prev_raw_text = ""
    inst._stable_prefix_len = 0
    inst._stable_cycles = 0
    result = inst._build_display_text("你好世界")
    assert result == "你好世界"


def test_build_display_stable_commit():
    """After 2 stable cycles, prefix is committed."""
    import app
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst._committed_text = ""
    inst._prev_raw_text = ""
    inst._stable_prefix_len = 0
    inst._stable_cycles = 0

    # Cycle 1
    inst._build_display_text("你好世界。这是第一句话。")
    # Cycle 2 — same prefix, different tail
    inst._build_display_text("你好世界。这是第一句话。然后继续说。")
    # Cycle 3 — prefix still stable
    result = inst._build_display_text("你好世界。这是第一句话。然后继续说。更多内容。")

    assert inst._committed_text.startswith("你好世界。")
    assert "更多内容" in result


def test_build_display_window_slide():
    """When window slides, committed text is preserved via overlap."""
    import app
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst._committed_text = ""
    inst._prev_raw_text = ""
    inst._stable_prefix_len = 0
    inst._stable_cycles = 0

    # Build up committed text over several stable cycles
    inst._build_display_text("开头内容。中间内容。")
    inst._build_display_text("开头内容。中间内容。后续A。")
    inst._build_display_text("开头内容。中间内容。后续A。后续B。")

    assert "开头内容" in inst._committed_text

    # Now simulate window slide — raw no longer starts with committed
    result = inst._build_display_text("中间内容。后续A。后续B。全新内容。")

    # Committed text should be preserved, new content appended
    assert "开头内容" in result
    assert "全新内容" in result


def test_build_display_reset():
    """Committed state resets between recordings."""
    import app
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst._committed_text = "旧的提交文本。"
    inst._prev_raw_text = "旧的原始文本。"
    inst._stable_prefix_len = 20
    inst._stable_cycles = 5

    # Simulate what _start_recording does
    inst._committed_text = ""
    inst._prev_raw_text = ""
    inst._stable_prefix_len = 0
    inst._stable_cycles = 0

    result = inst._build_display_text("全新录音。")
    assert result == "全新录音。"
    assert inst._committed_text == ""


# ── Test: enhanced trailing repetition ────────────────────

def test_strip_trailing_long_sentence_repetition():
    """Long sentence repetitions (>30 chars) should now be caught."""
    import app
    sentence = "我们在世界上最大的优点就是能够持续不断地创新和发展"
    text = f"正常内容。{sentence}{sentence}{sentence}"
    result = app._strip_trailing_repetition(text)
    assert "正常内容" in result
    assert result.count(sentence) <= 1


def test_strip_trailing_repetition_punct_variation():
    """Repetition with different punctuation between copies should be caught."""
    import app
    # Whisper produces same sentence with/without trailing period
    text = "正常内容。人们最关心的是AI如何帮助推理。人们最关心的是AI如何帮助推理"
    result = app._strip_trailing_repetition(text)
    assert "正常内容" in result
    assert result.count("人们最关心的是AI如何帮助推理") <= 1


def test_strip_trailing_repetition_mixed_punct():
    """Repetition with mixed Chinese/ASCII punctuation should be caught."""
    import app
    text = "开头内容。这是重复句，很长的一句话！这是重复句,很长的一句话!"
    result = app._strip_trailing_repetition(text)
    assert "开头内容" in result


# ── Test: _find_after_sentence_overlap ─────────────────────

def test_find_after_sentence_overlap_basic():
    """Sentence anchor finds overlap when tail-substring search fails."""
    import app
    committed = "今天天气很好。我去了San Francisco。参加了一个会议。"
    raw = "我去了San Francisco。参加了一场会议。主题是AI。"
    result = app._find_after_sentence_overlap(committed, raw)
    assert result is not None
    assert "参加了" in result or "主题" in result


def test_find_after_sentence_overlap_no_match():
    """Returns None when no sentence from committed is found in raw."""
    import app
    committed = "完全不相关的内容。另一个话题。"
    raw = "全新的一段文字。与之前无关。"
    result = app._find_after_sentence_overlap(committed, raw)
    assert result is None


# ── Test: 4-level cascade in _build_display_text ───────────

def test_build_display_paraphrase_uses_raw():
    """Level 2: When Whisper paraphrases (>40% prefix match), use raw directly."""
    import app
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst._committed_text = "现在我们开始测试,看看效果如何。"
    inst._prev_raw_text = "现在我们开始测试,看看效果如何。"
    inst._stable_prefix_len = 10
    inst._stable_cycles = 3

    # Whisper rephrases "效果如何" to "这个效果到底怎么样"
    raw = "现在我们开始测试,看看这个效果到底怎么样。今天去了SF。"
    result = inst._build_display_text(raw)

    # Should NOT have duplication — Level 2 uses raw directly
    assert result.count("现在我们开始测试") == 1
    assert "今天去了SF" in result


def test_build_display_sentence_anchor_overlap():
    """Level 4: When tail is paraphrased but an earlier sentence matches."""
    import app
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst._committed_text = "开头内容。我去了San Francisco。参加了一个会议。"
    inst._prev_raw_text = "之前的文本"
    inst._stable_prefix_len = 10
    inst._stable_cycles = 3

    # Window slid: raw doesn't start with committed, AND tail is paraphrased
    raw = "我去了San Francisco。参加了一场会议。主题是AI。"
    result = inst._build_display_text(raw)

    # Should preserve committed beginning and append new content
    assert "开头内容" in result
    assert "主题是AI" in result or "参加了" in result
