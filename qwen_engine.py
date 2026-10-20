"""Qwen3-ASR engine for MacWhisper: lifecycle, live subtitles, final pass.

Keeps all Qwen3-specific logic out of app.py. The app keeps Whisper loaded
at all times; anything here that fails raises QwenUnavailable (or reports
through on_error) so the app can fall back to Whisper for the same audio.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

import qwen_text
from qwen_backend import QwenSidecarBackend, QwenUnavailable
from stream_text import StableText
from text_utils import convert_t2s

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SLOW_HINT = "（实时转录较慢，松开后仍会正常转录）"


class QwenEngine:
    """Owns the Qwen3 sidecar backend for one model at a time."""

    def __init__(self, backend_factory=None, strip_fillers: bool = True, tight: bool = True):
        self._factory = backend_factory or (lambda repo: QwenSidecarBackend(repo))
        self.strip_fillers = strip_fillers
        self.tight = tight
        self.backend = None
        self.model_repo: str | None = None
        self._ok = False
        self.started_ok = False   # became ready at least once for this model
        self.last_error = ""
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        b = self.backend
        return bool(self._ok and b is not None and getattr(b, "is_ready", False))

    def start(self, model_repo: str) -> bool:
        """(Re)start the sidecar for model_repo. Blocking; returns readiness."""
        with self._lock:
            if self.backend is not None:
                try:
                    self.backend.close()
                except Exception:
                    logger.debug("closing previous Qwen backend failed", exc_info=True)
            self.backend = self._factory(model_repo)
            self.model_repo = model_repo
            self.started_ok = False
            self._ok = bool(self.backend.start())
            self.started_ok = self._ok
            self.last_error = "" if self._ok else (getattr(self.backend, "last_error", "") or
                                                   "Qwen3 sidecar failed to start")
            logger.info("QwenEngine start %s -> ready=%s %s", model_repo, self._ok, self.last_error)
            return self._ok

    def start_async(self, model_repo: str, on_done=None) -> None:
        def run():
            ok = self.start(model_repo)
            if on_done:
                on_done(ok, self.last_error)
        threading.Thread(target=run, daemon=True, name="qwen-engine-start").start()

    def close(self) -> None:
        with self._lock:
            if self.backend is not None:
                self.backend.close()
            self._ok = False

    def transcribe_final(self, audio: np.ndarray) -> str:
        """Final text for a whole utterance. Raises QwenUnavailable on failure.

        After a sidecar crash (started_ok but not ready) the backend restarts
        it transparently, within its restart budget.
        """
        if not self.started_ok or self.backend is None:
            raise QwenUnavailable(self.last_error or "Qwen3 engine not started")
        result = self.backend.transcribe(np.asarray(audio, dtype=np.float32))
        return qwen_text.postprocess(result.text, strip_filler_words=self.strip_fillers,
                                     tight=self.tight)


def _frames_to_float(frames) -> np.ndarray:
    if not frames:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(frames, axis=0).squeeze()
    return audio.astype(np.float32) / 32768.0


def run_live_loop(backend, get_frames, is_recording, update, *, step: float = 1.0,
                  hint_after_s: float = 4.0, min_audio_s: float = 0.5, on_error=None,
                  now=time.monotonic, sleep=time.sleep, use_timer: bool = False) -> str:
    """Live subtitles by re-decoding the whole utterance every `step` seconds.

    get_frames() -> list of int16 frames recorded so far; is_recording() ->
    bool; update(confirmed, tail) pushes text to the overlay (caller handles
    the main thread). A backend failure is reported once via on_error and
    the loop stops — the final pass is unaffected. Returns the last text
    shown (confirmed + tail).

    use_timer=True adds a real-time watchdog that shows SLOW_HINT if nothing
    has been shown after hint_after_s — even while a first decode is stuck
    (e.g. the GPU is busy), which the loop itself cannot detect.
    """
    stable = StableText()
    start = now()
    next_tick = start + step
    last_shown = None
    shown_text = {"any": False}
    timer = None

    def show(confirmed, tail):
        nonlocal last_shown
        key = (confirmed, tail)
        if key != last_shown:
            last_shown = key
            if confirmed or (tail and tail != SLOW_HINT):
                shown_text["any"] = True
            update(confirmed, tail)

    if use_timer:
        def hint():
            if not shown_text["any"] and is_recording():
                logger.warning("Qwen live: no text after %.0fs — showing slow hint", hint_after_s)
                update("", SLOW_HINT)
        timer = threading.Timer(hint_after_s, hint)
        timer.daemon = True
        timer.start()

    try:
        while is_recording():
            t = now()
            if t < next_tick:
                sleep(min(0.05, next_tick - t))
                continue
            audio = _frames_to_float(get_frames())
            if len(audio) < min_audio_s * SAMPLE_RATE:
                next_tick = t + step
                continue
            t0 = now()
            try:
                text = convert_t2s(backend.transcribe(audio).text)
            except Exception as e:
                logger.warning("Qwen live decode failed: %s", e)
                if on_error:
                    on_error(e)
                return "".join(last_shown or ("", ""))
            took = now() - t0
            stable.update(text)
            confirmed = qwen_text.tighten_spacing(stable.confirmed_text)
            tail = qwen_text.tighten_spacing(stable.tail_text)
            if not confirmed and not tail and now() - start >= hint_after_s:
                tail = SLOW_HINT
            show(confirmed, tail)
            logger.debug("Qwen live: %.1fs audio decoded in %.2fs, confirmed=%d tail=%d rewrites=%d",
                         len(audio) / SAMPLE_RATE, took, len(stable.confirmed),
                         len(stable.tail), stable.rewrites)
            next_tick = now() + step if took >= step else next_tick + step
    finally:
        if timer is not None:
            timer.cancel()
    confirmed, tail = last_shown or ("", "")
    return confirmed + ("" if tail == SLOW_HINT else tail)
