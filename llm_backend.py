"""App-side client for the local text LLM sidecar (llm_sidecar.py in .venv-llm).

Reuses QwenSidecarBackend's process lifecycle (spawn, ready handshake, restart
budget, timeouts, clean shutdown); only the request differs.
"""

from __future__ import annotations

import json
import logging
import os
import time

from qwen_backend import QwenSidecarBackend, QwenUnavailable

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LLM_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


class LLMUnavailable(QwenUnavailable):
    """The local LLM could not answer (not installed, crashed, timed out)."""


def default_llm_python(repo_root: str = _REPO_ROOT) -> str:
    return os.environ.get("MACWHISPER_LLM_PYTHON") or os.path.join(repo_root, ".venv-llm", "bin", "python")


def llm_available(repo_root: str = _REPO_ROOT) -> bool:
    return os.path.isfile(default_llm_python(repo_root))


class LLMSidecarBackend(QwenSidecarBackend):
    def __init__(self, model_repo: str = DEFAULT_LLM_MODEL, python_path: str | None = None,
                 log_path: str | None = None, **kw):
        super().__init__(
            model_repo,
            python_path=python_path or default_llm_python(),
            sidecar_path=os.path.join(_REPO_ROOT, "llm_sidecar.py"),
            log_path=log_path or os.path.expanduser("~/.macwhisper/logs/llm_sidecar.log"),
            **kw,
        )

    def complete(self, prompt: str, system: str = "", max_tokens: int = 512,
                 timeout: float = 120.0) -> str:
        """Generate a reply. Raises LLMUnavailable on any failure."""
        try:
            with self._lock:
                self._ensure_running_locked()
                self._next_id += 1
                rid = self._next_id
                req = {"id": rid, "cmd": "complete", "system": system, "prompt": prompt,
                       "max_tokens": max_tokens}
                t0 = time.time()
                try:
                    self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
                    self._proc.stdin.flush()
                except (OSError, ValueError) as e:
                    self._fail_locked(f"LLM sidecar pipe broken: {e}")
                while True:
                    msg = self._wait_line(timeout)
                    if msg is None:
                        self._fail_locked(f"LLM request timed out after {timeout:.0f}s")
                    if msg.get("event") == "exit":
                        self._fail_locked("LLM sidecar exited unexpectedly")
                    if msg.get("id") == rid:
                        break
                if not msg.get("ok"):
                    self.last_error = msg.get("error", "unknown error")
                    raise LLMUnavailable(f"LLM error: {self.last_error}")
                logger.info("LLM completion: %d prompt chars in %.1fs", len(prompt), time.time() - t0)
                return (msg.get("text") or "").strip()
        except LLMUnavailable:
            raise
        except QwenUnavailable as e:
            raise LLMUnavailable(str(e)) from e
