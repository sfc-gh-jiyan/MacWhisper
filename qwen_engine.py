"""Qwen3-ASR engine for MacWhisper: lifecycle, live subtitles, final pass.

Keeps all Qwen3-specific logic out of app.py (Qwen3 is the only engine).
Anything here that fails raises QwenUnavailable (or reports through
on_error) so the app can keep the audio and alert instead of losing it.
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
        self.hotwords: list[str] = []
        self._lock = threading.Lock()
        self._settled = threading.Event()   # set when no start() is in flight
        self._settled.set()
        self._inflight = 0
        self._inflight_repo: str | None = None
        self._count_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        b = self.backend
        return bool(self._ok and b is not None and getattr(b, "is_ready", False))

    def wait_ready(self, timeout: float) -> bool:
        """Wait for a running start() to finish; True when the engine is ready."""
        self._settled.wait(timeout)
        return self._settled.is_set() and self.is_ready

    def start(self, model_repo: str) -> bool:
        """(Re)start the sidecar for model_repo. Blocking; returns readiness.

        A start for the same model already in flight is joined, not repeated
        (launch, a dictation and a meeting can all ask at once)."""
        if not self._enter(model_repo):
            self._settled.wait(240)           # same model already loading: join it
            return self.is_ready
        return self._run_start(model_repo)

    def _enter(self, model_repo: str) -> bool:
        """Register a start; False if one for the same model is already in flight."""
        with self._count_lock:
            if self._inflight > 0 and self._inflight_repo == model_repo:
                return False
            self._inflight += 1
            self._inflight_repo = model_repo
            self._settled.clear()
            return True

    def _run_start(self, model_repo: str) -> bool:
        try:
            return self._start(model_repo)
        finally:
            with self._count_lock:
                self._inflight -= 1
                if self._inflight == 0:
                    self._inflight_repo = None
                    self._settled.set()

    def _start(self, model_repo: str) -> bool:
        with self._lock:
            if self.backend is not None:
                try:
                    self.backend.close()
                except Exception:
                    logger.debug("closing previous Qwen backend failed", exc_info=True)
            self.backend = self._factory(model_repo)
            self.backend.hotwords = list(self.hotwords)
            self.model_repo = model_repo
            self.started_ok = False
            self._ok = bool(self.backend.start())
            self.started_ok = self._ok
            self.last_error = "" if self._ok else (getattr(self.backend, "last_error", "") or
                                                   "Qwen3 sidecar failed to start")
            logger.info("QwenEngine start %s -> ready=%s %s", model_repo, self._ok, self.last_error)
            return self._ok

    def set_hotwords(self, terms) -> None:
        """Vocabulary terms that bias recognition (kept across sidecar restarts)."""
        self.hotwords = list(terms)
        if self.backend is not None:
            self.backend.hotwords = list(self.hotwords)
        logger.debug("Qwen hotwords: %d terms", len(self.hotwords))

    def start_async(self, model_repo: str, on_done=None) -> None:
        # Registered now (not when the thread runs): a start() in between joins it
        joined = not self._enter(model_repo)

        def run():
            ok = self.start(model_repo) if joined else self._run_start(model_repo)
            if on_done:
                on_done(ok, self.last_error)
        threading.Thread(target=run, daemon=True, name="qwen-engine-start").start()

    def close(self) -> None:
        with self._lock:
            if self.backend is not None:
                self.backend.close()
            self._ok = False

    def transcribe_final(self, audio: np.ndarray, retry: bool = False,
                         hotwords: bool = True) -> str:
        """Final text for a whole utterance. Raises QwenUnavailable on failure.

        After a sidecar crash (started_ok but not ready) the backend restarts
        it transparently, within its restart budget. With retry=True a failed
        request (e.g. a timeout) restarts the sidecar and tries the same audio
        once more — there is no other engine to fall back to.
        hotwords=False: transcribe without the vocabulary (meetings).
        """
        if not self.started_ok or self.backend is None:
            raise QwenUnavailable(self.last_error or "Qwen3 engine not started")
        audio = np.asarray(audio, dtype=np.float32)
        try:
            result = self._transcribe(audio, hotwords)
        except QwenUnavailable as e:
            restart = getattr(self.backend, "restart", None)
            if not retry or restart is None:
                raise
            logger.warning("Qwen3 final pass failed (%s) — restarting and retrying once", e)
            if not restart():
                raise
            result = self._transcribe(audio, hotwords)
        return qwen_text.postprocess(result.text, strip_filler_words=self.strip_fillers,
                                     tight=self.tight)

    def _transcribe(self, audio: np.ndarray, hotwords: bool):
        # Only pass the switch when it is off: older/fake backends take audio alone
        if hotwords:
            return self.backend.transcribe(audio)
        return self.backend.transcribe(audio, hotwords=False)



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
