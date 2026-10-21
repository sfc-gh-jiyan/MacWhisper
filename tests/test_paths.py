"""paths: data dir and Qwen python can be overridden for the standalone app."""

import importlib
import os

import paths


def _reload(monkeypatch, **env):
    for k in ("MACWHISPER_DATA_DIR", "MACWHISPER_QWEN_PYTHON"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(paths)


def test_default_data_dir(monkeypatch):
    p = _reload(monkeypatch)
    assert p.DATA_DIR == os.path.expanduser("~/.macwhisper")
    assert p.LOG_DIR == os.path.join(p.DATA_DIR, "logs")


def test_env_overrides_data_dir(monkeypatch, tmp_path):
    p = _reload(monkeypatch, MACWHISPER_DATA_DIR=str(tmp_path))
    assert p.DATA_DIR == str(tmp_path)
    assert p.LOG_DIR == str(tmp_path / "logs")
    assert p.MAIN_LOG == str(tmp_path / "logs" / "macwhisper.log")


def test_qwen_python_override(monkeypatch):
    p = _reload(monkeypatch, MACWHISPER_QWEN_PYTHON="/x/bin/python3")
    assert p.qwen_python("/repo") == "/x/bin/python3"


def test_qwen_python_default(monkeypatch):
    p = _reload(monkeypatch)
    assert p.qwen_python("/repo") == "/repo/.venv-qwen/bin/python"


def teardown_module(_):
    os.environ.pop("MACWHISPER_DATA_DIR", None)
    os.environ.pop("MACWHISPER_QWEN_PYTHON", None)
    importlib.reload(paths)
