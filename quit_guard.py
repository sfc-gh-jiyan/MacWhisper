"""SIGTERM → quit, even when the main thread is stuck in AppKit.

Python runs a signal handler only when the MAIN thread next executes Python
code. A menu bar app's main thread sits in AppKit's C event loop, and a modal
alert (e.g. "crashed last time", a permission prompt) also stops the timers
that would wake it. So a plain signal.signal() handler could wait forever,
and logout/shutdown or a ./run.sh restart hung.

Here the C-level signal handler writes to a wakeup pipe (signal.set_wakeup_fd)
the moment SIGTERM arrives; a background thread reads it and:
  1. asks for a normal quit (same path as menu → Quit), then
  2. if the process is still alive after `deadline_s`, runs `cleanup` itself
     and ends the process with os._exit(0).
`cleanup` runs at most once, whichever path gets there first.
"""

from __future__ import annotations

import logging
import os
import signal
import threading

logger = logging.getLogger(__name__)


class QuitGuard:
    def __init__(self, request_quit, cleanup, deadline_s: float = 5.0, exit_fn=os._exit,
                 cleanup_wait_s: float = 8.0):
        self._request_quit = request_quit
        self._cleanup = cleanup
        self._deadline_s = deadline_s
        self._cleanup_wait_s = cleanup_wait_s
        self._exit = exit_fn
        self._lock = threading.Lock()
        self._signalled = False
        self._started = False
        self._done = threading.Event()   # set only when cleanup has FINISHED

    def run_cleanup(self, *_args, **_kwargs) -> None:
        """Clean up once (also used by the normal Quit path and atexit).

        Accepts and ignores arguments so any event emitter can call it.
        """
        with self._lock:
            already = self._started
            self._started = True
        if already:
            # Another path (normal Quit vs SIGTERM deadline) is cleaning up:
            # wait for it to FINISH, so the process never exits mid-write
            # (e.g. a half-written meeting file). Bounded if cleanup hangs.
            if not self._done.wait(self._cleanup_wait_s):
                logger.error("Quit cleanup still running after %.0f s — exiting anyway",
                             self._cleanup_wait_s)
            return
        try:
            self._cleanup()
        except Exception:
            logger.exception("Quit cleanup failed")
        finally:
            self._done.set()

    def on_signal(self) -> None:
        with self._lock:
            if self._signalled:
                return
            self._signalled = True
        logger.warning("SIGTERM received — quitting (forced after %.0f s)", self._deadline_s)
        try:
            self._request_quit()
        except Exception:
            logger.exception("Normal quit request failed; forcing")
        threading.Thread(target=self._force_after_deadline, name="quit-guard-deadline",
                         daemon=True).start()

    def _force_after_deadline(self) -> None:
        threading.Event().wait(self._deadline_s)
        logger.warning("Still running %.0f s after SIGTERM — cleaning up and exiting",
                       self._deadline_s)
        self.run_cleanup()
        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:
                pass
        self._exit(0)


def install(request_quit, cleanup, deadline_s: float = 5.0) -> QuitGuard:
    """Install the SIGTERM guard. Must be called from the main thread."""
    guard = QuitGuard(request_quit, cleanup, deadline_s)
    rfd, wfd = os.pipe()
    os.set_blocking(wfd, False)
    # A Python-level handler is required for the C handler (and the wakeup
    # write) to run; it does nothing itself — the reader thread acts.
    signal.signal(signal.SIGTERM, lambda signum, frame: None)
    signal.set_wakeup_fd(wfd, warn_on_full_buffer=False)

    def reader():
        while True:
            try:
                data = os.read(rfd, 64)
            except OSError:
                return
            if not data:
                return
            if signal.SIGTERM in data:
                guard.on_signal()

    threading.Thread(target=reader, name="quit-guard", daemon=True).start()
    return guard
