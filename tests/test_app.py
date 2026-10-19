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

@pytest.fixture(autouse=True)
def _no_real_keyboard(request, monkeypatch):
    """Hotkey tests must not read the test machine's real modifier state."""
    if request.node.name == "test_system_modifiers_reads_quartz":
        return
    import app
    monkeypatch.setattr(app, "_system_modifiers", lambda: None)


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
    assert app.MIN_CHUNK_SIZE == 1.0
    assert app.MAX_BUFFER_S == 8.0
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
    inst.log_level = "WARNING"
    inst.meeting_subtitles = True
    inst.save_meeting_audio = False
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
         patch("app.MEETINGS_DIR", data_dir), \
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
         patch("app.MEETINGS_DIR", data_dir), \
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
    # CONFIG_PATH itself is redirected to a temp file for every test (conftest);
    # check the production definition instead.
    src = open(os.path.join(os.path.dirname(app.__file__), "app.py")).read()
    assert 'CONFIG_PATH = os.path.join(_DATA_DIR, "config.json")' in src
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


# ── Test: Meeting Subtitles toggle (v0.6.2) ───────────────────

def _meeting_app(meeting_subtitles=True, in_meeting=False, panel=None):
    import app
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst.translate_mode = False
    inst.current_model = "m"
    inst.live_mode = False
    inst.save_audio = False
    inst.log_level = "WARNING"
    inst.meeting_subtitles = meeting_subtitles
    inst.item_meeting_subtitles = MagicMock()
    inst.save_meeting_audio = False
    inst.status_item = MagicMock()
    inst._meeting_session = MagicMock() if in_meeting else None
    inst._overlay_panel = panel
    inst._overlay_text = None
    inst._last_overlay_key = ("old", "")
    inst._reporter = MagicMock()
    return inst


def test_meeting_subtitles_saved_to_config(tmp_config):
    inst = _meeting_app(meeting_subtitles=False)
    inst._save_config()
    assert json.loads(tmp_config.read_text())["meeting_subtitles"] is False


def test_toggle_meeting_subtitles_idle_only_flips_pref(tmp_config):
    import app
    inst = _meeting_app(meeting_subtitles=True)
    with patch.object(app.AppHelper, "callAfter") as call_after:
        inst._toggle_meeting_subtitles(None)
    assert inst.meeting_subtitles is False
    assert inst.item_meeting_subtitles.title == "   Meeting Subtitles: Off"
    call_after.assert_not_called()


def test_toggle_meeting_subtitles_off_during_meeting_hides_overlay(tmp_config):
    import app
    inst = _meeting_app(meeting_subtitles=True, in_meeting=True, panel=MagicMock())
    with patch.object(app.AppHelper, "callAfter") as call_after:
        inst._toggle_meeting_subtitles(None)
    call_after.assert_called_once_with(inst._destroy_overlay)


def test_toggle_meeting_subtitles_on_during_meeting_shows_overlay(tmp_config):
    import app
    inst = _meeting_app(meeting_subtitles=False, in_meeting=True, panel=None)
    with patch.object(app.AppHelper, "callAfter") as call_after:
        inst._toggle_meeting_subtitles(None)
    assert inst.meeting_subtitles is True
    assert inst.item_meeting_subtitles.title == "✅ Meeting Subtitles: On"
    assert inst._last_overlay_key is None  # next update must re-render
    call_after.assert_called_once()


def test_meeting_start_respects_subtitles_pref():
    import app
    source = open(os.path.join(os.path.dirname(app.__file__), "app.py")).read()
    assert "if self.meeting_subtitles:" in source


# ── Test: crash safety (fix/crash-alerts) ─────────────────────

def _bare_app():
    import app
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst.status_item = MagicMock()
    inst.recording = False
    inst.frames = []
    inst.stream = None
    inst._stream_lock = threading.Lock()
    inst._rec_gen = 0
    inst._reporter = MagicMock()
    inst._hotkey_stop = threading.Event()
    return inst


def test_set_status_from_background_thread_is_marshalled():
    import app
    inst = _bare_app()
    with patch.object(app.AppHelper, "callAfter") as call_after:
        t = threading.Thread(target=inst._set_status, args=("Ready",))
        t.start()
        t.join()
    call_after.assert_called_once()
    inst.status_item.title  # untouched by the background thread
    call_after.call_args[0][0]()  # run it "on main"
    assert inst.status_item.title == "Status: Ready"


def test_set_title_from_background_thread_is_marshalled():
    import app
    inst = _bare_app()
    with patch.object(app.AppHelper, "callAfter") as call_after:
        t = threading.Thread(target=inst._set_title, args=("🔴",))
        t.start()
        t.join()
    call_after.assert_called_once()


def test_no_direct_title_writes_left_in_app():
    """All menu-bar icon writes must go through _set_title (main thread)."""
    import re
    import app
    source = open(os.path.join(os.path.dirname(app.__file__), "app.py")).read()
    direct = re.findall(r"^\s*self\.title\s*=", source, flags=re.M)
    assert len(direct) == 1  # only inside _set_title itself


def test_guarded_hotkey_handler_reports_instead_of_raising():
    inst = _bare_app()

    def boom(_key):
        raise RuntimeError("layout engine")

    wrapped = inst._guarded(boom, "hotkey (press)")
    wrapped("key")  # must NOT raise — raising kills the pynput listener
    inst._reporter.report.assert_called_once()
    assert inst._reporter.report.call_args[0][0] == "hotkey (press)"


def test_hotkey_listener_restarts_after_it_dies():
    import app
    inst = _bare_app()
    starts = []

    class FakeListener:
        def __init__(self, **kw):
            starts.append(kw)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def join(self):
            if len(starts) >= 2:
                inst._hotkey_stop.set()
                return
            raise RuntimeError("listener crashed")

    with patch.object(app.keyboard, "Listener", FakeListener), \
         patch.object(app, "HOTKEY_RESTART_DELAY_S", 0):
        inst._start_hotkey_listener()
    assert len(starts) == 2
    inst._reporter.report.assert_called_once()


def _hotkey_app():
    import app
    import shortcuts
    inst = _bare_app()
    inst._held_mods = set()
    inst.shortcuts = shortcuts.merged({})
    inst._meeting_session = None
    inst.recording = False
    inst._ptt_cancelled = False
    inst._toggle_meeting_mode = MagicMock()
    inst._cycle_model = MagicMock()
    inst._start_recording = MagicMock()
    inst._stop_and_transcribe = MagicMock()
    inst._cancel_recording = MagicMock()
    return inst


def _press(inst, *keys):
    for k in keys:
        inst._on_press(k)


def _release(inst, *keys):
    for k in keys:
        inst._on_release(k)


def _run_sync_threads(monkeypatch):
    import app
    # Event-logic tests must not depend on the real keyboard of the test machine
    monkeypatch.setattr(app, "_system_modifiers", lambda: None)

    class SyncThread:
        def __init__(self, target, args=(), daemon=None, **kw):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)
    monkeypatch.setattr(app.threading, "Thread", SyncThread)


def test_combo_hotkeys_run_on_main_thread():
    import app
    from pynput.keyboard import Key, KeyCode
    inst = _hotkey_app()
    with patch.object(app.AppHelper, "callAfter") as call_after:
        t = threading.Thread(target=_press, args=(inst, Key.ctrl, Key.alt, KeyCode.from_vk(46, char="µ")))
        t.start()
        t.join()
    inst._toggle_meeting_mode.assert_not_called()  # not on the listener thread
    call_after.assert_called_once()


def test_default_meeting_shortcuts_and_old_ctrl_shift_r_is_gone(monkeypatch):
    import app
    from pynput.keyboard import Key, KeyCode
    monkeypatch.setattr(app.AppHelper, "callAfter", lambda fn, *a: fn(*a))
    inst = _hotkey_app()
    _press(inst, Key.ctrl, Key.alt, KeyCode.from_vk(46, char="µ")); _release(inst, Key.alt, Key.ctrl)
    _press(inst, Key.ctrl, Key.shift, Key.f8); _release(inst, Key.shift, Key.ctrl)
    _press(inst, Key.ctrl, Key.shift, KeyCode.from_vk(15, char="R")); _release(inst, Key.shift, Key.ctrl)
    assert inst._toggle_meeting_mode.call_count == 2


def test_custom_shortcut_from_config(monkeypatch):
    import app
    import shortcuts
    from pynput.keyboard import Key, KeyCode
    monkeypatch.setattr(app.AppHelper, "callAfter", lambda fn, *a: fn(*a))
    inst = _hotkey_app()
    inst.shortcuts = shortcuts.merged({"cycle_model": ["ctrl+cmd+vk46"]})
    _press(inst, Key.ctrl, Key.shift, KeyCode.from_vk(46)); _release(inst, Key.shift, Key.ctrl)
    inst._cycle_model.assert_not_called()                      # old default no longer bound
    _press(inst, Key.ctrl, Key.cmd, KeyCode.from_vk(46)); _release(inst, Key.cmd, Key.ctrl)
    inst._cycle_model.assert_called_once()


def test_push_to_talk_custom_key(monkeypatch):
    import shortcuts
    from pynput.keyboard import Key
    _run_sync_threads(monkeypatch)
    inst = _hotkey_app()
    inst.shortcuts = shortcuts.merged({"push_to_talk": ["f13"]})
    _press(inst, Key.alt_r)
    inst._start_recording.assert_not_called()          # Right ⌥ no longer records
    _press(inst, Key.f13)
    inst._start_recording.assert_not_called()          # ⌥ still held: not a plain F13
    _release(inst, Key.f13, Key.alt_r)
    _press(inst, Key.f13)
    inst._start_recording.assert_called_once()
    inst.recording = True
    _release(inst, Key.f13)
    inst._stop_and_transcribe.assert_called_once()


def test_ptt_not_started_while_ctrl_held(monkeypatch):
    """⌃ then Right ⌥ (start of ⌃⌥M) must not start a recording."""
    from pynput.keyboard import Key
    _run_sync_threads(monkeypatch)
    inst = _hotkey_app()
    _press(inst, Key.ctrl, Key.alt_r)
    inst._start_recording.assert_not_called()


def test_combo_with_right_option_cancels_started_recording(monkeypatch):
    """Right ⌥ pressed first (recording starts), then ⌃+M: cancel, toggle meeting, no paste."""
    import app
    from pynput.keyboard import Key, KeyCode
    _run_sync_threads(monkeypatch)
    monkeypatch.setattr(app.AppHelper, "callAfter", lambda fn, *a: fn(*a))
    inst = _hotkey_app()
    _press(inst, Key.alt_r)
    inst._start_recording.assert_called_once()
    inst.recording = True
    _press(inst, Key.ctrl, KeyCode.from_vk(46, char="µ"))
    inst._cancel_recording.assert_called_once()
    inst._toggle_meeting_mode.assert_called_once()
    _release(inst, Key.ctrl, Key.alt_r)
    inst._stop_and_transcribe.assert_not_called()             # nothing transcribed/pasted


def test_ptt_ignored_during_meeting(monkeypatch):
    from pynput.keyboard import Key
    _run_sync_threads(monkeypatch)
    inst = _hotkey_app()
    inst._meeting_session = MagicMock(is_recording=True)
    _press(inst, Key.alt_r)
    inst._start_recording.assert_not_called()


def test_modifier_state_is_tracked_left_and_right():
    from pynput.keyboard import Key
    inst = _hotkey_app()
    _press(inst, Key.ctrl_r, Key.shift)
    assert inst._held_mods == {"ctrl", "shift"}
    _release(inst, Key.ctrl_r)
    assert inst._held_mods == {"shift"}


def test_shortcuts_saved_to_config(tmp_config):
    import shortcuts
    inst = _engine_app()
    inst.shortcuts = shortcuts.merged({"meeting": ["ctrl+cmd+vk46"]})
    inst._save_config()
    cfg = json.loads(tmp_config.read_text())
    assert cfg["shortcuts"]["meeting"] == ["ctrl+cmd+vk46"]
    assert cfg["shortcuts"]["push_to_talk"] == ["alt_r"]


def test_fast_release_closes_late_stream():
    """Option released before the mic stream opened → stream must not leak."""
    import app
    inst = _bare_app()
    fake_stream = MagicMock()
    inst.recording = False  # already released
    with patch.object(app.sd, "InputStream", return_value=fake_stream):
        inst._open_mic_stream(gen=inst._rec_gen)
    fake_stream.start.assert_not_called()
    fake_stream.close.assert_called_once()
    assert inst.stream is None


def test_stale_generation_stream_is_closed():
    import app
    inst = _bare_app()
    inst.recording = True
    inst._rec_gen = 2  # a newer recording already started
    fake_stream = MagicMock()
    with patch.object(app.sd, "InputStream", return_value=fake_stream):
        inst._open_mic_stream(gen=1)
    fake_stream.close.assert_called_once()
    assert inst.stream is None


def test_open_and_close_mic_stream():
    import app
    inst = _bare_app()
    inst.recording = True
    fake_stream = MagicMock()
    with patch.object(app.sd, "InputStream", return_value=fake_stream):
        inst._open_mic_stream(gen=inst._rec_gen)
    fake_stream.start.assert_called_once()
    assert inst.stream is fake_stream
    inst._close_mic_stream()
    fake_stream.stop.assert_called_once()
    fake_stream.close.assert_called_once()
    assert inst.stream is None
    inst._close_mic_stream()  # idempotent, no "NoneType has no close"


def test_report_error_shows_persistent_alert_on_main():
    import app
    inst = _bare_app()
    with patch.object(app.AppHelper, "callAfter") as call_after, \
         patch.object(app.error_alert, "show_error_alert") as show, \
         patch.object(app.rumps, "notification"):
        t = threading.Thread(target=inst._show_error, args=("T", "line1\nline2"))
        t.start()
        t.join()
        call_after.assert_called_once()
        call_after.call_args[0][0]()
    show.assert_called_once()
    assert show.call_args[0][:2] == ("T", "line1\nline2")


def test_guarded_callbacks_work_through_real_pynput_wrapping():
    """Regression: pynput passes (key, injected) to *args callbacks.

    pynput's AbstractListener._wrap inspects the callback's signature; a
    bare *args wrapper made it pass 2 args to _on_press(key) → TypeError on
    every Option press.
    """
    from pynput._util import AbstractListener
    import app
    inst = _bare_app()
    seen = []
    inst._on_press = lambda key: seen.append(("press", key))
    inst._on_release = lambda key: seen.append(("release", key))
    for name, fn in (("press", inst._on_press), ("release", inst._on_release)):
        guarded = inst._guarded(fn, f"hotkey ({name})")
        cb = AbstractListener._wrap(None, guarded, 2)
        cb("KEY", False)  # exactly how pynput darwin calls it
    assert seen == [("press", "KEY"), ("release", "KEY")]
    inst._reporter.report.assert_not_called()


def test_guarded_real_method_signature():
    """Same check against the real bound methods' signatures."""
    import inspect
    import app
    inst = _bare_app()
    for meth in (app.TranscriberApp._on_press, app.TranscriberApp._on_release):
        bound = meth.__get__(inst)
        n = len(inspect.signature(bound).parameters)
        assert n == 1


# ── Test: meeting stop copies transcript file path ────────────

def _stopping_app(transcript_path, transcript="full meeting text"):
    import app
    inst = _bare_app()
    inst._overlay_panel = None
    inst._idle_icon = "🎙"
    inst.item_meeting = MagicMock()
    session = MagicMock()
    session.stop.return_value = transcript
    session.transcript_path = transcript_path
    session.segments = [1, 2, 3]
    inst._meeting_session = session
    inst._set_title = MagicMock()
    inst._notify = MagicMock()
    return inst


def test_meeting_stop_copies_file_path_not_text():
    import app
    inst = _stopping_app("/Users/x/.macwhisper/meetings/meeting_1.md")
    with patch.object(app.pyperclip, "copy") as copy:
        inst._stop_meeting()
    copy.assert_called_once_with("/Users/x/.macwhisper/meetings/meeting_1.md")
    inst._reporter.report.assert_not_called()


def test_meeting_stop_falls_back_to_text_if_save_failed():
    import app
    inst = _stopping_app(None)
    with patch.object(app.pyperclip, "copy") as copy:
        inst._stop_meeting()
    copy.assert_called_once_with("full meeting text")
    inst._reporter.report.assert_called_once()  # user is told the file is missing


def test_meeting_stop_empty_meeting_copies_nothing():
    import app
    inst = _stopping_app(None, transcript="")
    with patch.object(app.pyperclip, "copy") as copy:
        inst._stop_meeting()
    copy.assert_not_called()


def test_meeting_transcript_path_logged_at_default_level(caplog):
    """Path must reach macwhisper.log even with the default WARNING level."""
    import logging
    import app
    inst = _stopping_app("/Users/x/.macwhisper/meetings/meeting 1.md")
    with patch.object(app.pyperclip, "copy"), caplog.at_level(logging.WARNING):
        inst._stop_meeting()
    line = next(r.getMessage() for r in caplog.records if "MEETING SAVED" in r.getMessage())
    assert "/Users/x/.macwhisper/meetings/meeting 1.md" in line
    assert "file:///Users/x/.macwhisper/meetings/meeting%201.md" in line
    assert caplog.records[-1].levelno >= logging.WARNING


# ── Test: Save Meeting Audio toggle + grouped menu ────────────

def test_save_meeting_audio_defaults_off_and_persists(tmp_config):
    inst = _meeting_app()
    inst.save_meeting_audio = False
    inst._save_config()
    assert json.loads(tmp_config.read_text())["save_meeting_audio"] is False


def test_toggle_save_meeting_audio(tmp_config):
    inst = _meeting_app()
    inst.save_meeting_audio = False
    inst.item_save_meeting_audio = MagicMock()
    inst._toggle_save_meeting_audio(None)
    assert inst.save_meeting_audio is True
    assert inst.item_save_meeting_audio.title == "✅ Save Meeting Audio: On"
    assert json.loads(tmp_config.read_text())["save_meeting_audio"] is True
    inst._toggle_save_meeting_audio(None)
    assert inst.item_save_meeting_audio.title == "   Save Meeting Audio: Off"


def test_toggle_save_meeting_audio_mid_meeting_applies_at_stop(tmp_config):
    inst = _meeting_app(in_meeting=True)
    inst.save_meeting_audio = False
    inst.item_save_meeting_audio = MagicMock()
    inst._toggle_save_meeting_audio(None)
    assert inst._meeting_session.save_audio is True


def test_meeting_session_gets_save_audio_flag():
    import app
    source = open(os.path.join(os.path.dirname(app.__file__), "app.py")).read()
    assert "save_audio=self.save_meeting_audio" in source


def _menu_app():
    import app
    inst = _engine_app()
    inst.model_items = {k: MagicMock(title=k) for k in app.ALL_MODEL_KEYS}
    inst.item_meeting = MagicMock(title="   Meeting Mode: Off")
    return inst


def test_main_menu_is_slim_models_meeting_settings():
    """v0.7.1: no submenus; options live in the Settings window."""
    import app
    inst = _menu_app()
    menu = inst._build_menu()
    assert not any(isinstance(entry, (list, tuple)) for entry in menu)   # no submenus
    titles = [getattr(e, "title", None) for e in menu if e is not app.rumps.separator]
    assert titles[0] == "Status: Ready"
    for label in app.ALL_MODEL_KEYS:
        assert inst.model_items[label] in menu
    assert inst.item_meeting in menu
    assert any(t and t.startswith("Settings") for t in titles)


def test_settings_menu_item_has_cmd_comma():
    inst = _menu_app()
    settings = [e for e in inst._build_menu() if getattr(e, "title", "").startswith("Settings")][0]
    assert settings.key == ","


# ── Test: Qwen3-ASR engine wiring (v0.7.0) ────────────────────

class _FakeQwen:
    def __init__(self, ready=True, text="千问 result", fail=False):
        self.is_ready = ready
        self.started_ok = ready
        self.text = text
        self.fail = fail
        self.calls = 0
        self.model_repo = "mlx-community/Qwen3-ASR-0.6B-bf16"
        self.last_error = ""

    def transcribe_final(self, audio):
        self.calls += 1
        if self.fail:
            from qwen_backend import QwenUnavailable
            raise QwenUnavailable("sidecar died")
        return self.text


def _engine_app(engine="qwen", qwen=None, translate=False):
    inst = _meeting_app()
    inst.engine = engine
    inst.qwen_model = "mlx-community/Qwen3-ASR-0.6B-bf16"
    inst.translate_mode = translate
    inst._qwen = qwen if qwen is not None else _FakeQwen()
    inst._qwen_alerted = False
    return inst


def test_use_qwen_only_when_selected_ready_and_not_translating():
    assert _engine_app()._use_qwen() is True
    assert _engine_app(engine="whisper")._use_qwen() is False
    assert _engine_app(qwen=_FakeQwen(ready=False))._use_qwen() is False
    assert _engine_app(translate=True)._use_qwen() is False


def test_final_text_uses_qwen_when_available():
    inst = _engine_app()
    inst._whisper_final = MagicMock(return_value=("whisper text", None))
    text, engine = inst._final_text(np.zeros(16000, dtype=np.float32), np.zeros(16000, dtype=np.int16))
    assert text == "千问 result" and "Qwen3" in engine
    inst._whisper_final.assert_not_called()


def test_final_text_falls_back_to_whisper_on_same_audio_and_alerts():
    inst = _engine_app(qwen=_FakeQwen(fail=True))
    audio = np.ones(16000, dtype=np.float32) * 0.1
    inst._whisper_final = MagicMock(return_value=("whisper text", None))
    text, engine = inst._final_text(audio, (audio * 32768).astype(np.int16))
    assert text == "whisper text" and engine == inst.current_model
    assert inst._whisper_final.call_args[0][0] is audio      # same audio, nothing lost
    inst._reporter.report.assert_called_once()
    assert "Whisper" in inst._reporter.report.call_args[0][0]


def test_final_text_whisper_engine_never_touches_qwen():
    q = _FakeQwen()
    inst = _engine_app(engine="whisper", qwen=q)
    inst._whisper_final = MagicMock(return_value=("w", None))
    inst._final_text(np.zeros(16000, dtype=np.float32), np.zeros(16000, dtype=np.int16))
    assert q.calls == 0


def test_engine_and_qwen_settings_saved(tmp_config):
    inst = _engine_app()
    inst.qwen_strip_fillers = True
    inst.qwen_tight_spacing = True
    inst.meeting_final_pass = True
    inst._save_config()
    cfg = json.loads(tmp_config.read_text())
    assert cfg["engine"] == "qwen"
    assert cfg["qwen_model"] == "mlx-community/Qwen3-ASR-0.6B-bf16"
    assert cfg["meeting_final_pass"] is True


def test_initial_engine_defaults():
    import app
    assert app.initial_engine({}, qwen_env_exists=True) == "qwen"        # first upgrade
    assert app.initial_engine({}, qwen_env_exists=False) == "whisper"
    assert app.initial_engine({"engine": "whisper"}, qwen_env_exists=True) == "whisper"
    assert app.initial_engine({"engine": "qwen"}, qwen_env_exists=False) == "qwen"  # keeps choice


def test_model_menu_lists_whisper_and_qwen():
    import app
    labels = list(app.ALL_MODEL_OPTIONS)
    assert labels[:3] == list(app.MODEL_OPTIONS)
    assert any("Qwen3" in l and "0.6B" in l for l in labels)
    assert any("Qwen3" in l and "1.7B" in l for l in labels)


class _LiveBackend:
    def __init__(self, fail=False):
        self.fail = fail
        self.is_ready = True

    def transcribe(self, audio, **kw):
        from asr_backend import TranscriptionResult
        if self.fail:
            from qwen_backend import QwenUnavailable
            raise QwenUnavailable("live boom")
        return TranscriptionResult(text="我们开始 test")


def _qwen_live_app(backend, seconds=3):
    import app
    inst = _engine_app()
    inst._qwen = MagicMock(backend=backend, is_ready=True)
    inst.frames = [np.zeros((16000, 1), dtype=np.int16) for _ in range(seconds)]
    inst.recording = True
    inst._overlay_panel = MagicMock()
    inst._overlay_text = MagicMock()
    inst._last_overlay_key = None
    inst._live_loop_done = threading.Event()
    inst._log_subtitle_live = MagicMock()
    return inst


def test_qwen_live_loop_updates_overlay_and_signals_done(monkeypatch):
    import app
    calls = []
    monkeypatch.setattr(app.AppHelper, "callAfter", lambda fn, *a: calls.append(fn))
    inst = _qwen_live_app(_LiveBackend())
    threading.Timer(2.5, lambda: setattr(inst, "recording", False)).start()
    inst._qwen_live_loop()
    assert inst._live_loop_done.is_set()
    assert calls, "overlay never updated"
    assert "我们开始" in inst._qwen_live_text


def test_qwen_live_loop_failure_does_not_alert(monkeypatch):
    import app
    monkeypatch.setattr(app.AppHelper, "callAfter", lambda fn, *a: None)
    inst = _qwen_live_app(_LiveBackend(fail=True))
    threading.Timer(2.5, lambda: setattr(inst, "recording", False)).start()
    inst._qwen_live_loop()
    assert inst._live_loop_done.is_set()
    inst._reporter.report.assert_not_called()      # final pass handles fallback + alert


def test_qwen_live_loop_stops_when_next_recording_starts(monkeypatch):
    import app
    monkeypatch.setattr(app.AppHelper, "callAfter", lambda fn, *a: None)
    inst = _qwen_live_app(_LiveBackend())
    threading.Timer(1.5, lambda: setattr(inst, "frames", [])).start()   # new recording
    t = threading.Thread(target=inst._qwen_live_loop)
    t.start()
    t.join(5)
    assert not t.is_alive()


def _final_pass_app(qwen):
    inst = _engine_app(qwen=qwen)
    inst._notify = MagicMock()
    session = MagicMock()
    session.mixed_audio.return_value = np.zeros(16000 * 5, dtype=np.float32)
    session.write_final_transcript.return_value = "/tmp/m.md"
    return inst, session


def test_meeting_final_pass_rewrites_file_and_frees_audio():
    q = _FakeQwen(text="会议最终文稿")
    inst, session = _final_pass_app(q)
    inst._meeting_final_pass(session)
    text, engine = session.write_final_transcript.call_args[0]
    assert text == "会议最终文稿" and "Qwen3" in engine
    session.release_audio.assert_called_once()
    inst._notify.assert_called_once()


def test_meeting_final_pass_failure_keeps_live_transcript_and_alerts():
    inst, session = _final_pass_app(_FakeQwen(fail=True))
    inst._meeting_final_pass(session)
    session.write_final_transcript.assert_not_called()      # live .md untouched
    session.release_audio.assert_called_once()
    inst._reporter.report.assert_called_once()


def test_meeting_stop_whisper_engine_skips_final_pass_but_frees_audio():
    import app
    inst = _meeting_app(in_meeting=True)
    inst.engine = "whisper"
    inst._qwen = None
    inst._idle_icon = "🎙"
    inst.item_meeting = MagicMock()
    inst._notify = MagicMock()
    session = inst._meeting_session
    session.stop.return_value = "text"
    session.transcript_path = "/tmp/x.md"
    session.segments = []
    inst._meeting_final_pass = MagicMock()
    with patch.object(app.pyperclip, "copy"):
        inst._stop_meeting()
    inst._meeting_final_pass.assert_not_called()
    session.release_audio.assert_called_once()


class _CrashedQwen(_FakeQwen):
    """Was ready, sidecar died: is_ready False, and a retry fails too."""

    def __init__(self):
        super().__init__(ready=False, fail=True)
        self.started_ok = True
        self.last_error = "Qwen3 sidecar exited unexpectedly"


def test_crashed_sidecar_is_retried_then_alerts_instead_of_silent_whisper():
    """Regression (real-model verify): after a crash the app silently used Whisper."""
    inst = _engine_app(qwen=_CrashedQwen())
    inst._whisper_final = MagicMock(return_value=("whisper text", None))
    text, model = inst._final_text(np.zeros(16000, dtype=np.float32), np.zeros(16000, dtype=np.int16))
    assert text == "whisper text"
    assert inst._qwen.calls == 1                       # Qwen3 was actually tried
    inst._reporter.report.assert_called_once()         # and the user was told


def test_qwen_still_loading_uses_whisper_without_alert():
    q = _FakeQwen(ready=False)
    q.started_ok = False                                # never became ready yet
    inst = _engine_app(qwen=q)
    inst._whisper_final = MagicMock(return_value=("w", None))
    inst._final_text(np.zeros(16000, dtype=np.float32), np.zeros(16000, dtype=np.int16))
    assert q.calls == 0
    inst._reporter.report.assert_not_called()


def test_alert_rearms_after_qwen_recovers():
    q = _FakeQwen(fail=True)
    inst = _engine_app(qwen=q)
    inst._on_main = lambda fn, *a: fn(*a)
    inst._whisper_final = MagicMock(return_value=("w", None))
    audio = (np.zeros(16000, dtype=np.float32), np.zeros(16000, dtype=np.int16))
    inst._final_text(*audio)                 # fail -> alert 1
    q.fail = False
    inst._final_text(*audio)                 # recovered
    q.fail = True
    inst._final_text(*audio)                 # fails again -> alert 2 (not silent)
    assert inst._reporter.report.call_count == 2


def test_stale_modifier_from_lost_release_does_not_block_ptt(monkeypatch):
    """Lost Ctrl release (e.g. secure input) must not disable Push to Talk forever."""
    import app
    from pynput.keyboard import Key
    _run_sync_threads(monkeypatch)
    inst = _hotkey_app()
    inst._held_mods = {"ctrl"}                                  # stale: Ctrl is really up
    monkeypatch.setattr(app, "_system_modifiers", lambda: {"alt"})   # only Right ⌥ down now
    _press(inst, Key.alt_r)
    inst._start_recording.assert_called_once()
    assert inst._held_mods == {"alt"}


def test_real_ctrl_still_blocks_ptt(monkeypatch):
    import app
    from pynput.keyboard import Key
    _run_sync_threads(monkeypatch)
    inst = _hotkey_app()
    inst._held_mods = {"ctrl"}
    monkeypatch.setattr(app, "_system_modifiers", lambda: {"ctrl", "alt"})
    _press(inst, Key.alt_r)
    inst._start_recording.assert_not_called()


def test_system_modifier_query_failure_keeps_tracked_state(monkeypatch):
    import app
    from pynput.keyboard import Key
    _run_sync_threads(monkeypatch)
    inst = _hotkey_app()
    monkeypatch.setattr(app, "_system_modifiers", lambda: None)   # API unavailable
    _press(inst, Key.alt_r)
    inst._start_recording.assert_called_once()


def test_system_modifiers_reads_quartz():
    import app
    mods = app._system_modifiers()
    assert mods is None or mods <= {"ctrl", "alt", "shift", "cmd"}


def test_cancel_recording_discards_audio_without_transcribing(monkeypatch):
    import app
    monkeypatch.setattr(app.AppHelper, "callAfter", lambda fn, *a: fn(*a))
    inst = _bare_app()
    inst.recording = True
    old_frames = [np.zeros((1600, 1), dtype=np.int16)]
    inst.frames = old_frames
    inst._overlay_panel = MagicMock()
    inst._destroy_overlay = MagicMock()
    inst._close_mic_stream = MagicMock()
    inst._idle_icon = "🎙"
    inst.transcribe_queue = MagicMock()
    inst._cancel_recording()
    assert inst.recording is False
    assert inst.frames == [] and inst.frames is not old_frames      # live loop sees a new list
    inst._close_mic_stream.assert_called_once()
    inst._destroy_overlay.assert_called_once()
    inst.transcribe_queue.put.assert_not_called()


def test_overlay_stop_ends_running_meeting():
    inst = _meeting_app(in_meeting=True)
    inst._meeting_session.is_recording = True
    inst._stop_meeting = MagicMock()
    inst._start_meeting = MagicMock()
    inst._stop_meeting_from_overlay()
    inst._stop_meeting.assert_called_once()
    inst._start_meeting.assert_not_called()


def test_overlay_stop_after_meeting_already_ended_never_starts_a_new_one():
    """Confirm dialog open while the meeting was stopped by shortcut: do nothing."""
    inst = _meeting_app(in_meeting=False)
    inst._stop_meeting = MagicMock()
    inst._start_meeting = MagicMock()
    inst._stop_meeting_from_overlay()
    inst._stop_meeting.assert_not_called()
    inst._start_meeting.assert_not_called()


def test_meeting_overlay_gets_stop_callback(monkeypatch):
    import app
    seen = {}

    def fake_create(mode, frame_path, on_stop=None):
        seen[mode] = on_stop
        return MagicMock(), MagicMock()
    monkeypatch.setattr(app, "create_overlay", fake_create)
    inst = _meeting_app()
    inst._create_overlay(mode="meeting")
    inst._create_overlay(mode="talk")
    assert seen["meeting"] == inst._stop_meeting_from_overlay
    assert seen["talk"] is None


# ── Settings wiring (v0.7.1) ──────────────────────────────────

def test_shortcuts_paused_while_recording_one(monkeypatch):
    """Recording ⌃⌥M in Settings must not toggle the meeting."""
    import app
    from pynput.keyboard import Key, KeyCode
    _run_sync_threads(monkeypatch)
    monkeypatch.setattr(app.AppHelper, "callAfter", lambda fn, *a: fn(*a))
    inst = _hotkey_app()
    inst._shortcuts_paused = True
    _press(inst, Key.ctrl, Key.alt, KeyCode.from_vk(46, char="µ"))
    _press(inst, Key.alt_r)
    inst._toggle_meeting_mode.assert_not_called()
    inst._start_recording.assert_not_called()
    assert inst._held_mods == {"ctrl", "alt"}                  # still tracked
    _release(inst, Key.ctrl, Key.alt, Key.alt_r)
    assert inst._held_mods == set()
    inst._shortcuts_paused = False
    _press(inst, Key.ctrl, Key.alt, KeyCode.from_vk(46, char="µ"))
    inst._toggle_meeting_mode.assert_called_once()


def test_set_pref_saves_and_updates_running_qwen(tmp_config):
    inst = _engine_app()
    inst._qwen = MagicMock(strip_fillers=True, tight=True)
    inst._set_pref("qwen_strip_fillers", False)
    inst._set_pref("qwen_tight_spacing", False)
    inst._set_pref("meeting_final_pass", False)
    assert inst._qwen.strip_fillers is False and inst._qwen.tight is False
    cfg = json.loads(tmp_config.read_text())
    assert cfg["qwen_strip_fillers"] is False and cfg["meeting_final_pass"] is False


def test_set_pref_rejects_unknown_name():
    inst = _engine_app()
    with pytest.raises(ValueError):
        inst._set_pref("recording", True)


def test_set_shortcuts_saves_and_applies(tmp_config):
    import shortcuts
    inst = _engine_app()
    new = shortcuts.merged({"meeting": ["ctrl+cmd+vk46"]})
    inst._set_shortcuts(new)
    assert inst.shortcuts == new
    assert json.loads(tmp_config.read_text())["shortcuts"]["meeting"] == ["ctrl+cmd+vk46"]


def test_open_settings_builds_window_once(monkeypatch):
    import app
    inst = _menu_app()
    created = []

    class FakeWindow:
        def __init__(self, a, labels):
            created.append(labels)
            self.shown = 0

        def show(self):
            self.shown += 1
    monkeypatch.setattr(app.settings_window, "SettingsWindow", FakeWindow)
    inst._settings = None
    inst._open_settings(None)
    inst._open_settings(None)
    assert len(created) == 1 and created[0] == app.ALL_MODEL_KEYS
    assert inst._settings.shown == 2


def test_shortcut_action_refreshes_open_settings(monkeypatch):
    import app
    from pynput.keyboard import Key, KeyCode
    monkeypatch.setattr(app.AppHelper, "callAfter", lambda fn, *a: fn(*a))
    inst = _hotkey_app()
    inst._shortcuts_paused = False
    inst._settings = MagicMock()
    inst._toggle_translate = MagicMock()
    inst._on_main = lambda fn, *a: fn(*a)
    inst._guarded = lambda fn, where: fn
    _press(inst, Key.ctrl, Key.shift, KeyCode.from_vk(17, char="T"))
    inst._toggle_translate.assert_called_once()
    inst._settings.refresh.assert_called_once()


# ── v0.7.1 bug hunt ───────────────────────────────────────────

def test_right_cmd_ptt_cancelled_by_cmd_c(monkeypatch):
    """PTT = Right ⌘: ⌘C must not record-and-paste."""
    import shortcuts
    from pynput.keyboard import Key, KeyCode
    _run_sync_threads(monkeypatch)
    inst = _hotkey_app()
    inst.shortcuts = shortcuts.merged({"push_to_talk": ["cmd_r"]})
    _press(inst, Key.cmd_r)
    inst._start_recording.assert_called_once()
    inst.recording = True
    _press(inst, KeyCode.from_vk(8, char="c"))
    inst._cancel_recording.assert_called_once()
    _release(inst, KeyCode.from_vk(8), Key.cmd_r)
    inst._stop_and_transcribe.assert_not_called()


def test_right_option_ptt_unchanged_when_other_key_pressed(monkeypatch):
    """Default Right ⌥ keeps v0.7.0 behavior exactly."""
    from pynput.keyboard import Key, KeyCode
    _run_sync_threads(monkeypatch)
    inst = _hotkey_app()
    _press(inst, Key.alt_r)
    inst.recording = True
    _press(inst, KeyCode.from_vk(0, char="å"))
    inst._cancel_recording.assert_not_called()
    _release(inst, KeyCode.from_vk(0), Key.alt_r)
    inst._stop_and_transcribe.assert_called_once()


def test_meeting_toggle_debounced_after_overlay_stop(monkeypatch):
    """⌃⌥M pressed while the Stop confirm dialog was open must not start a new meeting."""
    import app
    inst = _meeting_app(in_meeting=True)
    inst._meeting_session.is_recording = True
    t = {"now": 100.0}
    monkeypatch.setattr(app.time, "monotonic", lambda: t["now"])
    inst._stop_meeting = MagicMock(side_effect=lambda: setattr(inst, "_meeting_session", None))
    inst._start_meeting = MagicMock()
    inst._stop_meeting_from_overlay()
    t["now"] += 0.3
    inst._toggle_meeting_mode(None)                  # queued shortcut
    inst._start_meeting.assert_not_called()
    t["now"] += 2.0
    inst._toggle_meeting_mode(None)                  # a deliberate press later works
    inst._start_meeting.assert_called_once()


def test_set_model_refreshes_settings_window():
    import app
    inst = _menu_app()
    inst._settings = MagicMock()
    inst._warmup_model = MagicMock()
    inst._set_status = MagicMock()
    inst._set_model("Large (Best)")
    inst._settings.refresh.assert_called_once()


def test_combo_ending_in_ptt_key_still_fires(monkeypatch):
    """Reviewer: PTT = F8 must not swallow ⌃⇧F8 (meeting)."""
    import app
    import shortcuts
    from pynput.keyboard import Key
    _run_sync_threads(monkeypatch)
    monkeypatch.setattr(app.AppHelper, "callAfter", lambda fn, *a: fn(*a))
    inst = _hotkey_app()
    inst.shortcuts = shortcuts.merged({"push_to_talk": ["f8"]})
    _press(inst, Key.ctrl, Key.shift, Key.f8)
    inst._start_recording.assert_not_called()
    inst._toggle_meeting_mode.assert_called_once()
    _release(inst, Key.f8)
    inst._stop_and_transcribe.assert_not_called()


def test_stale_modifier_does_not_fire_combo_while_typing(monkeypatch):
    """Reviewer: stale ⌃ + typing ⇧S must not toggle live subtitles."""
    import app
    from pynput.keyboard import Key, KeyCode
    _run_sync_threads(monkeypatch)
    monkeypatch.setattr(app.AppHelper, "callAfter", lambda fn, *a: fn(*a))
    inst = _hotkey_app()
    inst._toggle_live_mode = MagicMock()
    inst._held_mods = {"ctrl", "shift"}                        # ⌃ is stale
    monkeypatch.setattr(app, "_system_modifiers", lambda: {"shift"})
    _press(inst, KeyCode.from_vk(1, char="S"))
    inst._toggle_live_mode.assert_not_called()
    assert inst._held_mods == {"shift"}


def test_talk_overlay_not_created_after_recording_was_cancelled(monkeypatch):
    """Reviewer: queued overlay creation must not outlive a cancelled recording."""
    import app
    created = []
    monkeypatch.setattr(app, "create_overlay", lambda *a, **k: created.append(1) or (MagicMock(), MagicMock()))
    inst = _meeting_app()
    inst.recording = False
    inst._create_overlay()                                     # talk mode, recording over
    assert created == [] and inst._overlay_panel is None
    inst.recording = True
    inst._create_overlay()
    assert created == [1]


def test_meeting_stop_hint_follows_shortcuts():
    import overlay
    import shortcuts
    inst = _engine_app()
    inst._set_shortcuts(shortcuts.merged({"meeting": ["ctrl+cmd+vk46", "ctrl+shift+f8"]}))
    assert overlay.meeting_stop_hint == "⌃⌘M"
    inst._set_shortcuts(shortcuts.merged({"meeting": []}))
    assert overlay.meeting_stop_hint == ""
    inst._set_shortcuts(shortcuts.merged({}))
    assert overlay.meeting_stop_hint == "⌃⌥M"
