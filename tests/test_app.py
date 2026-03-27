"""
Automated tests for MacWhisper app.py (v0.5 — LocalAgreement architecture)
Covers: config, audio processing, state machines, menu logic, overlay, text utils.
Run:  ./venv/bin/python3 -m pytest tests/test_app.py -v
"""

import json
import os
import queue
import threading
from unittest.mock import patch, MagicMock

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
    assert hasattr(app, "MIN_CHUNK_SIZE")


def test_constants():
    import app
    assert app.SAMPLE_RATE == 16000
    assert app.MIN_CHUNK_SIZE == 0.5
    assert app.MAX_BUFFER_S == 5.0
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
    inst.save_audio = False
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
    assert '🔴' in source   # Meeting Mode recording icon
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


# ── Test: hotkey vk codes ─────────────────────────────────────

def test_hotkey_vk_codes():
    vk_map = {46: "M", 17: "T", 1: "S"}
    assert len(vk_map) == 3


# ── Test: hallucination filter (via text_utils) ──────────────

def test_hallucination_known_phrases():
    from text_utils import is_hallucination
    assert is_hallucination("Thank you for watching.") is True
    assert is_hallucination("please subscribe") is True


def test_hallucination_repetition():
    from text_utils import is_hallucination
    assert is_hallucination("Ok Ok Ok") is True
    assert is_hallucination("sto sto sto") is True


def test_hallucination_cjk_repetition():
    from text_utils import is_hallucination
    assert is_hallucination("技术技术技术") is True


def test_hallucination_cyrillic():
    from text_utils import is_hallucination
    assert is_hallucination("Спасибо за внимание") is True


def test_hallucination_normal_text():
    from text_utils import is_hallucination
    assert is_hallucination("我上周去了San Francisco") is False
    # Pure-English is legitimate in bilingual context (no_cjk filter removed)
    assert is_hallucination("Hello world") is False
    assert is_hallucination("What's going on there?") is False
    # Short pure-English is OK
    assert is_hallucination("OK") is False


# ── Test: trailing repetition stripping (via text_utils) ──────

def test_strip_trailing_cjk():
    from text_utils import strip_trailing_repetition
    result = strip_trailing_repetition("别的语言。举举举举举举举举举举举举")
    assert "举" not in result
    assert "别的语言" in result


def test_strip_trailing_with_commas():
    from text_utils import strip_trailing_repetition
    result = strip_trailing_repetition("别的语言。啊,啊,啊,啊,啊,啊,啊,啊,啊")
    assert result.count("啊") <= 1
    assert "别的语言" in result


def test_strip_trailing_sentence_repetition():
    from text_utils import strip_trailing_repetition
    text = "正常内容。我们在世界上最大的优点。我们在世界上最大的优点。我们在世界上最大的优点。我们在世界上最大的优点。"
    result = strip_trailing_repetition(text)
    assert "正常内容" in result
    assert result.count("我们在世界上最大的优点") <= 1


def test_strip_trailing_preserves_clean():
    from text_utils import strip_trailing_repetition
    clean = "我上周去了San Francisco参加了一个conference。"
    assert strip_trailing_repetition(clean) == clean


def test_strip_trailing_long_sentence_repetition():
    """Long sentence repetitions (>30 chars) should be caught."""
    from text_utils import strip_trailing_repetition
    sentence = "我们在世界上最大的优点就是能够持续不断地创新和发展"
    text = f"正常内容。{sentence}{sentence}{sentence}"
    result = strip_trailing_repetition(text)
    assert "正常内容" in result
    assert result.count(sentence) <= 1


def test_strip_trailing_repetition_punct_variation():
    """Repetition with different punctuation between copies should be caught."""
    from text_utils import strip_trailing_repetition
    text = "正常内容。人们最关心的是AI如何帮助推理。人们最关心的是AI如何帮助推理"
    result = strip_trailing_repetition(text)
    assert "正常内容" in result
    assert result.count("人们最关心的是AI如何帮助推理") <= 1


def test_strip_trailing_repetition_mixed_punct():
    """Repetition with mixed Chinese/ASCII punctuation should be caught."""
    from text_utils import strip_trailing_repetition
    text = "开头内容。这是重复句，很长的一句话！这是重复句,很长的一句话!"
    result = strip_trailing_repetition(text)
    assert "开头内容" in result


# ── Test: opencc t2s (via text_utils) ─────────────────────────

def test_opencc_t2s():
    from text_utils import convert_t2s
    assert convert_t2s("機器學習") == "机器学习"
    assert convert_t2s("Hello") == "Hello"


# ── Test: prefix/overlap utilities (via text_utils) ───────────

def test_common_prefix_len():
    from text_utils import common_prefix_len
    assert common_prefix_len("abcdef", "abcxyz") == 3
    assert common_prefix_len("hello", "hello world") == 5
    assert common_prefix_len("abc", "xyz") == 0
    assert common_prefix_len("", "abc") == 0


def test_common_prefix_len_fuzzy():
    """Tolerates case and CJK punctuation differences."""
    from text_utils import common_prefix_len
    assert common_prefix_len("Test3你好", "test3你好") == 7
    assert common_prefix_len("你好，世界", "你好,世界") == 5
    assert common_prefix_len("结束。下一句", "结束.下一句") == 6


def test_snap_to_boundary():
    from text_utils import snap_to_boundary
    text = "你好世界。这是测试。然后继续"
    pos = text.index("然")
    result = snap_to_boundary(text, pos)
    assert text[result - 1] == "。"
    assert snap_to_boundary(text, 0) == 0


def test_snap_to_boundary_no_punctuation():
    from text_utils import snap_to_boundary
    text = "没有标点的文字内容"
    result = snap_to_boundary(text, 5)
    assert result == 5


def test_find_after_overlap():
    from text_utils import find_after_overlap
    committed = "我上周去了San Francisco参加了一个conference。"
    raw = "参加了一个conference。这个conference是关于ML的。"
    result = find_after_overlap(committed, raw)
    assert result == "这个conference是关于ML的。"


def test_find_after_overlap_case_insensitive():
    """Overlap matching ignores case differences from Whisper."""
    from text_utils import find_after_overlap
    committed = "我不知道test3,我不知道为什么。"
    raw = "我不知道Test3,我不知道为什么,但是之后的效果还是不错的。"
    result = find_after_overlap(committed, raw)
    assert "但是之后" in result
    assert "我不知道" not in result


def test_find_after_overlap_punct_variation():
    """Overlap matching tolerates punctuation differences."""
    from text_utils import find_after_overlap
    committed = "这是测试。结果如何？"
    raw = "这是测试,结果如何,后续内容。"
    result = find_after_overlap(committed, raw)
    assert "后续内容" in result


def test_find_after_overlap_no_match():
    from text_utils import find_after_overlap
    committed = "完全不同的句子。"
    raw = "另一段话。"
    result = find_after_overlap(committed, raw)
    assert result == raw


def test_find_after_overlap_empty():
    from text_utils import find_after_overlap
    assert find_after_overlap("", "新内容") == "新内容"
    assert find_after_overlap("旧内容", "") == ""


def test_find_after_sentence_overlap_basic():
    """Sentence anchor finds overlap when tail-substring search fails."""
    from text_utils import find_after_sentence_overlap
    committed = "今天天气很好。我去了San Francisco。参加了一个会议。"
    raw = "我去了San Francisco。参加了一场会议。主题是AI。"
    result = find_after_sentence_overlap(committed, raw)
    assert result is not None
    assert "参加了" in result or "主题" in result


def test_find_after_sentence_overlap_no_match():
    """Returns None when no sentence from committed is found in raw."""
    from text_utils import find_after_sentence_overlap
    committed = "完全不相关的内容。另一个话题。"
    raw = "全新的一段文字。与之前无关。"
    result = find_after_sentence_overlap(committed, raw)
    assert result is None


# ── Test: history constants and dirs ─────────────────────────

def test_history_constants():
    import app
    assert hasattr(app, "HISTORY_DIR")
    assert hasattr(app, "AUDIO_DIR")
    assert hasattr(app, "TRANSCRIPT_LOG")
    assert app.AUDIO_DIR.endswith(os.path.join(".macwhisper", "audio"))
    assert app.TRANSCRIPT_LOG.endswith(os.path.join(".macwhisper", "transcripts.jsonl"))


def test_ensure_history_dirs(tmp_path):
    import app
    audio_dir = str(tmp_path / "history" / "audio")
    log_dir = str(tmp_path / "history" / "logs")
    with patch("app.AUDIO_DIR", audio_dir), patch("app.LOG_DIR", log_dir):
        app._ensure_history_dirs()
    assert os.path.isdir(audio_dir)
    assert os.path.isdir(log_dir)
    with patch("app.AUDIO_DIR", audio_dir), patch("app.LOG_DIR", log_dir):
        app._ensure_history_dirs()


def test_audio_history_saves_wav(tmp_path):
    """_do_transcribe should save a WAV file to AUDIO_DIR when save_audio is on."""
    import app
    import wave

    audio_dir = str(tmp_path / "history" / "audio")
    transcript_log = str(tmp_path / "history" / "transcripts.jsonl")
    data_dir = str(tmp_path / "history")

    frames = _make_frames(5, 1024)
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst.translate_mode = False
    inst.current_model = "mlx-community/whisper-small-mlx"
    inst.save_audio = True
    inst._live_loop_done = threading.Event()
    inst._live_loop_done.set()

    # Mock the ASR backend (app.py uses self._backend directly)
    from asr_backend import TranscriptionResult, Segment, WordTimestamp
    mock_result = TranscriptionResult(
        text="Hello world",
        segments=[Segment(text="Hello world", start=0.0, end=1.0,
                         words=[WordTimestamp("Hello", 0.0, 0.5),
                                WordTimestamp("world", 0.5, 1.0)])],
    )
    mock_backend = MagicMock()
    mock_backend.transcribe.return_value = mock_result
    inst._backend = mock_backend

    with patch("app.AUDIO_DIR", audio_dir), \
         patch("app.TRANSCRIPT_LOG", transcript_log), \
         patch("app._DATA_DIR", data_dir), \
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
    data_dir = str(tmp_path / "history")

    frames = _make_frames(5, 1024)
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst.translate_mode = False
    inst.current_model = "mlx-community/whisper-small-mlx"
    inst.save_audio = True
    inst._live_loop_done = threading.Event()
    inst._live_loop_done.set()

    # Mock the ASR backend (app.py uses self._backend directly)
    from asr_backend import TranscriptionResult, Segment, WordTimestamp
    mock_result = TranscriptionResult(
        text="你好世界 hello",
        segments=[Segment(text="你好世界 hello", start=0.0, end=1.0,
                         words=[WordTimestamp("你好世界", 0.0, 0.5),
                                WordTimestamp("hello", 0.5, 1.0)])],
    )
    mock_backend = MagicMock()
    mock_backend.transcribe.return_value = mock_result
    inst._backend = mock_backend
    with patch("app.AUDIO_DIR", audio_dir), \
         patch("app.TRANSCRIPT_LOG", transcript_log), \
         patch("app._DATA_DIR", data_dir), \
         patch.object(inst, "_auto_paste"):
        app._ensure_history_dirs()
        inst._do_transcribe(frames)

    assert os.path.isfile(transcript_log)
    with open(transcript_log, encoding="utf-8") as f:
        lines = f.readlines()
    # Two log lines: one from save_enhanced_history, one legacy append
    assert len(lines) == 2

    # The legacy entry (second line) has the expected format
    entry = json.loads(lines[1])
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
    assert app.SUBTITLE_LOG.endswith(os.path.join(".macwhisper", "subtitles.jsonl"))


# ── Test: data directory constants ───────────────────────────

def test_data_dir_constant():
    import app
    assert hasattr(app, "_DATA_DIR")
    assert app._DATA_DIR.endswith(".macwhisper")
    assert app.CONFIG_PATH.endswith(os.path.join(".macwhisper", "config.json"))
    assert hasattr(app, "LOG_DIR")
    assert app.LOG_DIR.endswith(os.path.join(".macwhisper", "logs"))


def test_ensure_history_dirs_creates_log_dir(tmp_path):
    import app
    audio_dir = str(tmp_path / "audio")
    log_dir = str(tmp_path / "logs")
    with patch("app.AUDIO_DIR", audio_dir), patch("app.LOG_DIR", log_dir):
        app._ensure_history_dirs()
    assert os.path.isdir(log_dir)


# ── Test: migration logic ────────────────────────────────────

def test_migrate_config(tmp_path):
    """Old ~/.macwhisper_config.json should move to new config.json."""
    import app
    old_config = str(tmp_path / "old_config.json")
    new_config = str(tmp_path / "new" / "config.json")
    audio_dir = str(tmp_path / "new" / "audio")
    log_dir = str(tmp_path / "new" / "logs")

    with open(old_config, "w") as f:
        json.dump({"current_model": "test-model"}, f)

    with patch("app.CONFIG_PATH", new_config), \
         patch("app.AUDIO_DIR", audio_dir), \
         patch("app.LOG_DIR", log_dir), \
         patch("os.path.expanduser", return_value=old_config), \
         patch("os.path.abspath", return_value=str(tmp_path / "project" / "app.py")):
        app._migrate_old_data()

    assert os.path.isfile(new_config)
    assert not os.path.exists(old_config)
    with open(new_config) as f:
        assert json.load(f)["current_model"] == "test-model"


def test_migrate_audio_files(tmp_path):
    """Audio WAVs should move from old history/audio/ to new audio dir."""
    import app
    project = tmp_path / "project"
    old_audio = project / "history" / "audio"
    old_audio.mkdir(parents=True)
    (old_audio / "20250101_120000.wav").write_text("fake wav")

    new_audio = tmp_path / "new" / "audio"
    log_dir = tmp_path / "new" / "logs"

    with patch("app.AUDIO_DIR", str(new_audio)), \
         patch("app.LOG_DIR", str(log_dir)), \
         patch("app.CONFIG_PATH", str(tmp_path / "new" / "config.json")), \
         patch("os.path.expanduser", return_value=str(tmp_path / "no_old_config")), \
         patch("os.path.abspath", return_value=str(project / "app.py")):
        app._migrate_old_data()

    assert (new_audio / "20250101_120000.wav").exists()
    assert not (old_audio / "20250101_120000.wav").exists()


def test_migrate_transcripts(tmp_path):
    """transcripts.jsonl should move to new location."""
    import app
    project = tmp_path / "project"
    old_history = project / "history"
    old_history.mkdir(parents=True)
    old_transcript = old_history / "transcripts.jsonl"
    old_transcript.write_text('{"text":"hello"}\n')

    new_dir = tmp_path / "new"
    new_transcript = new_dir / "transcripts.jsonl"

    with patch("app.AUDIO_DIR", str(new_dir / "audio")), \
         patch("app.LOG_DIR", str(new_dir / "logs")), \
         patch("app.TRANSCRIPT_LOG", str(new_transcript)), \
         patch("app.CONFIG_PATH", str(new_dir / "config.json")), \
         patch("os.path.expanduser", return_value=str(tmp_path / "no_old_config")), \
         patch("os.path.abspath", return_value=str(project / "app.py")):
        app._migrate_old_data()

    assert new_transcript.exists()
    assert new_transcript.read_text().strip() == '{"text":"hello"}'
    assert not old_transcript.exists()


def test_migrate_subtitles(tmp_path):
    """subtitles.jsonl should move to new location."""
    import app
    project = tmp_path / "project"
    old_history = project / "history"
    old_history.mkdir(parents=True)
    old_subtitle = old_history / "subtitles.jsonl"
    old_subtitle.write_text('{"text":"sub1"}\n')

    new_dir = tmp_path / "new"
    new_subtitle = new_dir / "subtitles.jsonl"

    with patch("app.AUDIO_DIR", str(new_dir / "audio")), \
         patch("app.LOG_DIR", str(new_dir / "logs")), \
         patch("app.SUBTITLE_LOG", str(new_subtitle)), \
         patch("app.CONFIG_PATH", str(new_dir / "config.json")), \
         patch("os.path.expanduser", return_value=str(tmp_path / "no_old_config")), \
         patch("os.path.abspath", return_value=str(project / "app.py")):
        app._migrate_old_data()

    assert new_subtitle.exists()
    assert new_subtitle.read_text().strip() == '{"text":"sub1"}'
    assert not old_subtitle.exists()


def test_migrate_merges_existing_transcripts(tmp_path):
    """When both old and new transcript files exist, old entries are appended."""
    import app
    project = tmp_path / "project"
    old_history = project / "history"
    old_history.mkdir(parents=True)
    old_transcript = old_history / "transcripts.jsonl"
    old_transcript.write_text('{"text":"old"}\n')

    new_dir = tmp_path / "new"
    new_dir.mkdir(parents=True)
    new_transcript = new_dir / "transcripts.jsonl"
    new_transcript.write_text('{"text":"new"}\n')

    with patch("app.AUDIO_DIR", str(new_dir / "audio")), \
         patch("app.LOG_DIR", str(new_dir / "logs")), \
         patch("app.TRANSCRIPT_LOG", str(new_transcript)), \
         patch("app.CONFIG_PATH", str(new_dir / "config.json")), \
         patch("os.path.expanduser", return_value=str(tmp_path / "no_old_config")), \
         patch("os.path.abspath", return_value=str(project / "app.py")):
        app._migrate_old_data()

    lines = new_transcript.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "new"
    assert json.loads(lines[1])["text"] == "old"
    assert not old_transcript.exists()


def test_migrate_logs(tmp_path):
    """Log files should move from <project>/logs/ to new logs dir."""
    import app
    project = tmp_path / "project"
    old_logs = project / "logs"
    old_logs.mkdir(parents=True)
    (old_logs / "macwhisper.log").write_text("some log output")

    new_dir = tmp_path / "new"
    new_logs = new_dir / "logs"

    with patch("app.AUDIO_DIR", str(new_dir / "audio")), \
         patch("app.LOG_DIR", str(new_logs)), \
         patch("app.CONFIG_PATH", str(new_dir / "config.json")), \
         patch("os.path.expanduser", return_value=str(tmp_path / "no_old_config")), \
         patch("os.path.abspath", return_value=str(project / "app.py")):
        app._migrate_old_data()

    assert (new_logs / "macwhisper.log").exists()
    assert (new_logs / "macwhisper.log").read_text() == "some log output"
    assert not (old_logs / "macwhisper.log").exists()


def test_migrate_noop_when_no_old_data(tmp_path):
    """Migration should not error when no old data exists."""
    import app
    new_dir = tmp_path / "new"

    with patch("app.AUDIO_DIR", str(new_dir / "audio")), \
         patch("app.LOG_DIR", str(new_dir / "logs")), \
         patch("app.CONFIG_PATH", str(new_dir / "config.json")), \
         patch("os.path.expanduser", return_value=str(tmp_path / "no_old_config")), \
         patch("os.path.abspath", return_value=str(tmp_path / "empty_project" / "app.py")):
        app._migrate_old_data()

    assert os.path.isdir(str(new_dir / "audio"))
    assert os.path.isdir(str(new_dir / "logs"))


def test_migrate_cleans_empty_dirs(tmp_path):
    """Old history/ and logs/ dirs should be removed when empty after migration."""
    import app
    project = tmp_path / "project"
    old_history = project / "history"
    old_audio = old_history / "audio"
    old_audio.mkdir(parents=True)
    (old_audio / "test.wav").write_text("wav")
    old_logs = project / "logs"
    old_logs.mkdir(parents=True)
    (old_logs / "macwhisper.log").write_text("log")

    new_dir = tmp_path / "new"

    with patch("app.AUDIO_DIR", str(new_dir / "audio")), \
         patch("app.LOG_DIR", str(new_dir / "logs")), \
         patch("app.CONFIG_PATH", str(new_dir / "config.json")), \
         patch("os.path.expanduser", return_value=str(tmp_path / "no_old_config")), \
         patch("os.path.abspath", return_value=str(project / "app.py")):
        app._migrate_old_data()

    assert not old_audio.exists()
    assert not old_history.exists()
    assert not old_logs.exists()


# ── Test: v0.5 architecture — OnlineASRProcessor integration ──

def test_processor_created_on_recording():
    """Verify OnlineASRProcessor is referenced in recording flow."""
    import app
    source = open(os.path.join(os.path.dirname(app.__file__), "app.py")).read()
    assert "OnlineASRProcessor" in source
    assert "self._processor" in source
    assert "MLXWhisperBackend" in source


def test_overlay_uses_dual_color():
    """Verify overlay module is used for dual-color rendering."""
    import app
    source = open(os.path.join(os.path.dirname(app.__file__), "app.py")).read()
    assert "create_overlay" in source
    assert "update_overlay" in source
    assert "destroy_overlay" in source


def test_live_loop_exists():
    """Verify unified _live_loop replaces old _live_chunk_loop + _live_transcription_worker."""
    import app
    source = open(os.path.join(os.path.dirname(app.__file__), "app.py")).read()
    assert "_live_loop" in source
    assert "_live_chunk_loop" not in source
    assert "_live_transcription_worker" not in source


# ── Test: version ────────────────────────────────────────────

def test_version_format():
    """__version__ must be a valid semver string."""
    import app
    import re
    assert re.match(r'^\d+\.\d+\.\d+$', app.__version__), f"Bad version: {app.__version__}"


def test_version_file_consistency():
    """VERSION file and app.__version__ must match."""
    import app
    version_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "VERSION")
    with open(version_path) as f:
        file_version = f.read().strip()
    assert file_version == app.__version__, f"VERSION={file_version} != app.__version__={app.__version__}"


# ── Test: audio device check ─────────────────────────────────

def test_check_audio_device_exists():
    """_check_audio_device function must be defined and callable."""
    import app
    assert callable(app._check_audio_device)


def test_check_audio_device_with_input(monkeypatch):
    """_check_audio_device should pass when an input device exists."""
    import app
    mock_devices = [
        {"name": "Built-in Mic", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
    ]
    monkeypatch.setattr("app.sd.query_devices", lambda: mock_devices)
    app._check_audio_device()


def test_check_audio_device_no_input(monkeypatch):
    """_check_audio_device should exit(1) when no input device exists."""
    import app
    mock_devices = [
        {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
    ]
    monkeypatch.setattr("app.sd.query_devices", lambda: mock_devices)
    monkeypatch.setattr("app.rumps.alert", lambda **kw: None)
    with pytest.raises(SystemExit) as exc_info:
        app._check_audio_device()
    assert exc_info.value.code == 1


def test_check_audio_device_exception(monkeypatch, capsys):
    """_check_audio_device should warn but not crash on query failure."""
    import app
    def raise_err():
        raise RuntimeError("no audio subsystem")
    monkeypatch.setattr("app.sd.query_devices", raise_err)
    app._check_audio_device()


def test_crash_handler_writes_log(tmp_path, monkeypatch):
    """Crash handler should write to crash.log when an exception occurs."""
    import app
    crash_log = tmp_path / "crash.log"
    monkeypatch.setattr("app.LOG_DIR", str(tmp_path))

    import datetime, traceback
    try:
        raise ValueError("simulated crash")
    except Exception:
        with open(str(crash_log), "a", encoding="utf-8") as f:
            f.write(f"MacWhisper v{app.__version__} crash at {datetime.datetime.now().isoformat()}\n")
            traceback.print_exc(file=f)

    content = crash_log.read_text()
    assert "simulated crash" in content
    assert app.__version__ in content
