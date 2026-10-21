"""Filesystem locations shared by all modules.

Defaults match the developer setup (~/.macwhisper, repo .venv-qwen). The
standalone .app launcher sets MACWHISPER_DATA_DIR / MACWHISPER_QWEN_PYTHON so
the bundle never depends on the source checkout.
"""

import os

DATA_DIR = os.environ.get("MACWHISPER_DATA_DIR") or os.path.expanduser("~/.macwhisper")
LOG_DIR = os.path.join(DATA_DIR, "logs")
MAIN_LOG = os.path.join(LOG_DIR, "macwhisper.log")


def qwen_python(repo_root: str) -> str:
    """Python interpreter for the Qwen3 sidecar."""
    return os.environ.get("MACWHISPER_QWEN_PYTHON") or os.path.join(repo_root, ".venv-qwen", "bin", "python")
