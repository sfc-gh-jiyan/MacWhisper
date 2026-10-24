"""Qwen3-ASR backend for MacWhisper — client for the qwen_sidecar process.

The model runs in a separate process (isolated .venv-qwen, its own mlx), so
the app venv needs no ML stack. This client:
  - starts the sidecar and waits for its "ready" line,
  - serializes requests (one in flight; the sidecar is single-threaded),
  - applies a timeout that scales with audio length,
  - restarts a crashed/hung sidecar a limited number of times,
  - raises QwenUnavailable on ANY failure so the caller can keep the audio
    (retry later) — a request is never silently dropped.

Qwen3-ASR returns no word timestamps, so results carry text only.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import subprocess
import threading
import time

import numpy as np

from asr_backend import ASRBackend, TranscriptionResult

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


class QwenUnavailable(RuntimeError):
    """Qwen3-ASR could not serve this request (the caller keeps the audio)."""


def default_python_path(repo_root: str = _REPO_ROOT) -> str:
    return os.path.join(repo_root, ".venv-qwen", "bin", "python")


class QwenSidecarBackend(ASRBackend):
    def __init__(
        self,
        model_repo: str,
        python_path: str | None = None,
        sidecar_path: str | None = None,
        log_path: str | None = None,
        extra_args: list[str] | None = None,
        env: dict | None = None,
        base_timeout_s: float = 20.0,
        per_audio_s: float = 0.5,
        max_restarts: int = 2,
        healthy_reset_s: float = 300.0,
    ):
        self.model_repo = model_repo
        self.python_path = python_path or default_python_path()
        self.sidecar_path = sidecar_path or os.path.join(_REPO_ROOT, "qwen_sidecar.py")
        self.log_path = log_path or os.path.expanduser("~/.macwhisper/logs/qwen_sidecar.log")
        self.hotwords: list[str] = []  # vocabulary terms biasing recognition (set by the app)
        self.extra_args = list(extra_args or [])
        self.env = dict(env or {})
        self.base_timeout_s = base_timeout_s
        self.per_audio_s = per_audio_s
        self.max_restarts = max_restarts
        # The app runs for days: after this long without trouble the restart
        # budget refills, so occasional incidents never use it up for good.
        self.healthy_reset_s = healthy_reset_s
        self._ready_since: float | None = None

        self.restarts = 0
        self.last_error = ""
        self.load_s: float | None = None
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue | None = None
        self._ready = False
        self._started_once = False
        self._next_id = 0
        self._lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready and self._proc is not None and self._proc.poll() is None

    def start(self, timeout: float = 180.0) -> bool:
        """Spawn the sidecar and wait for "ready". Returns False on failure."""
        with self._lock:
            return self._start_locked(timeout)

    def restart(self, timeout: float = 180.0) -> bool:
        """Start a fresh sidecar now (meeting recovery after a failure)."""
        with self._lock:
            logger.warning("Restarting Qwen sidecar on request")
            return self._start_locked(timeout)

    def _start_locked(self, timeout: float) -> bool:
        self._kill_locked()
        self._started_once = True
        if not os.path.isfile(self.python_path):
            self.last_error = (f"Qwen3 environment not found ({self.python_path}). "
                               "Run: make qwen-setup")
            logger.warning("Qwen sidecar unavailable: %s", self.last_error)
            return False
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        log = open(self.log_path, "a", encoding="utf-8")
        log.write(f"\n=== sidecar start {time.strftime('%Y-%m-%d %H:%M:%S')} model={self.model_repo}\n")
        log.flush()
        cmd = [self.python_path, self.sidecar_path, "--model", self.model_repo, *self.extra_args]
        logger.info("Starting Qwen sidecar: %s", " ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                text=True, encoding="utf-8", bufsize=1,
                env={**os.environ, **self.env},
            )
        except OSError as e:
            self.last_error = f"cannot start sidecar: {e}"
            logger.error(self.last_error)
            return False
        finally:
            log.close()
        self._lines = queue.Queue()
        threading.Thread(target=self._read_stdout, args=(self._proc, self._lines),
                         daemon=True, name="qwen-sidecar-reader").start()
        msg = self._wait_line(timeout)
        if msg and msg.get("event") == "ready":
            self._ready = True
            self._ready_since = time.monotonic()
            self.load_s = msg.get("load_s")
            self.last_error = ""
            logger.info("Qwen sidecar ready (model=%s, load=%ss, pid=%s)",
                        msg.get("model"), self.load_s, msg.get("pid"))
            return True
        self.last_error = (msg or {}).get("error") or "sidecar did not become ready (see qwen_sidecar.log)"
        logger.error("Qwen sidecar failed to start: %s", self.last_error)
        self._kill_locked()
        return False

    @staticmethod
    def _read_stdout(proc, lines):
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)  # EOF: process exited

    def _wait_line(self, timeout: float) -> dict | None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                return None
            if line is None:
                return {"event": "exit", "error": "sidecar exited"}
            try:
                return json.loads(line)
            except ValueError:
                logger.warning("Qwen sidecar: non-protocol stdout line: %.120s", line)

    def _kill_locked(self) -> None:
        self._ready = False
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def close(self) -> None:
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is None and proc.stdin:
                try:
                    proc.stdin.write('{"cmd": "shutdown"}\n')
                    proc.stdin.flush()
                    proc.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired, ValueError):
                    pass
            self._kill_locked()

    # ── transcription ─────────────────────────────────────

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        task: str = "transcribe",
        hotwords: bool = True,
    ) -> TranscriptionResult:
        """Transcribe via the sidecar. Raises QwenUnavailable on any failure.

        `language`/`initial_prompt` are kept for the ASRBackend interface and ignored: the sidecar
        pins Chinese, which keeps code-switched English verbatim.
        hotwords=False sends this request without the vocabulary (meetings:
        the word list raised MER 0.185 -> 0.549 and was read out on silence).
        """
        if task != "transcribe":
            raise QwenUnavailable("Qwen3-ASR does not translate")
        with self._lock:
            self._ensure_running_locked()
            pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype(np.int16)
            self._next_id += 1
            rid = self._next_id
            req = {"id": rid, "cmd": "transcribe",
                   "pcm16_b64": base64.b64encode(pcm16.tobytes()).decode("ascii")}
            if hotwords and self.hotwords:
                req["hotwords"] = list(self.hotwords)
            duration = len(pcm16) / SAMPLE_RATE
            timeout = self.base_timeout_s + self.per_audio_s * duration
            t0 = time.time()
            try:
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError) as e:
                self._fail_locked(f"sidecar pipe broken: {e}")
            while True:
                msg = self._wait_line(timeout)
                if msg is None:
                    self._fail_locked(f"Qwen3 request timed out after {timeout:.0f}s")
                if msg.get("event") == "exit":
                    self._fail_locked("Qwen3 sidecar exited unexpectedly")
                if msg.get("id") == rid:
                    break
            if not msg.get("ok"):
                # Model error inside a live sidecar: keep the process, fail this request
                self.last_error = msg.get("error", "unknown error")
                raise QwenUnavailable(f"Qwen3 error: {self.last_error}")
            if (self.restarts and self._ready_since is not None
                    and time.monotonic() - self._ready_since >= self.healthy_reset_s):
                logger.info("Qwen sidecar healthy again — restart budget refilled")
                self.restarts = 0
            logger.debug("Qwen transcribe %.1fs audio in %.2fs (infer %.2fs)",
                         duration, time.time() - t0, msg.get("infer_s", 0))
            return TranscriptionResult(text=(msg.get("text") or "").strip(), language="zh")

    def _ensure_running_locked(self) -> None:
        if self.is_ready:
            return
        if self._proc is not None and self._proc.poll() is not None:
            # Died between requests (crash, OOM, killed): record why
            self.last_error = f"Qwen3 sidecar exited (code {self._proc.returncode})"
            logger.error("%s — see %s", self.last_error, self.log_path)
            self._kill_locked()
        if self._started_once:
            if self.restarts >= self.max_restarts:
                if not self.last_error.startswith("Qwen3 sidecar gave up"):
                    cause = self.last_error or "sidecar not running"
                    self.last_error = (f"Qwen3 sidecar gave up after {self.restarts} "
                                       f"restarts (last error: {cause})")
                raise QwenUnavailable(self.last_error)
            self.restarts += 1
            logger.warning("Restarting Qwen sidecar (restart %d/%d)", self.restarts, self.max_restarts)
        if not self._start_locked(timeout=180.0):
            raise QwenUnavailable(self.last_error)

    def _fail_locked(self, reason: str):
        self.last_error = reason
        logger.error("Qwen sidecar failure: %s", reason)
        self._kill_locked()
        raise QwenUnavailable(reason)
