"""Crash evidence capture and user-facing error reporting.

Three layers so a failure is never silent:
  1. ErrorReporter — any caught exception (any thread) → log + alert callback.
  2. install_thread_excepthook — uncaught exceptions in threads → ErrorReporter.
  3. enable_faulthandler + RunMarker — hard crashes (SIGABRT/SIGSEGV) that kill
     the process are written to crash.log, and the next launch detects that
     the previous run did not exit cleanly.

Pure Python, no AppKit — the alert UI is injected as a callback.
"""

from __future__ import annotations

import datetime
import faulthandler
import logging
import os
import threading
import traceback

logger = logging.getLogger(__name__)

# faulthandler writes to this file object at crash time; it must stay open.
_fault_file = None


def enable_faulthandler(crash_log_path: str) -> None:
    """Dump native/fatal crashes (all threads' tracebacks) to crash_log_path."""
    global _fault_file
    os.makedirs(os.path.dirname(crash_log_path) or ".", exist_ok=True)
    _fault_file = open(crash_log_path, "a", encoding="utf-8", buffering=1)
    _fault_file.write(
        f"\n{'=' * 60}\nMacWhisper session started "
        f"{datetime.datetime.now().isoformat(timespec='seconds')} pid={os.getpid()}\n"
    )
    _fault_file.flush()
    faulthandler.enable(file=_fault_file, all_threads=True)
    logger.info("faulthandler enabled -> %s", crash_log_path)


def read_tail(path: str, max_lines: int = 40) -> str:
    """Return the last max_lines of a text file, or "" if unreadable."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunMarker:
    """File present while the app runs; left behind if the process dies."""

    def __init__(self, path: str):
        self.path = path

    def previous_run_crashed(self) -> bool:
        try:
            with open(self.path, encoding="utf-8") as f:
                content = f.read().strip()
        except FileNotFoundError:
            return False
        except OSError:
            return True
        try:
            pid = int(content)
        except ValueError:
            return True
        if pid == os.getpid():
            return False
        # Another instance still running is not a crash
        return not _pid_alive(pid)

    def create(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))

    def clear(self) -> None:
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("RunMarker.clear failed: %s", self.path, exc_info=True)


class ErrorReporter:
    """Turns exceptions into a log entry plus a user-visible alert.

    Args:
        alert: callable(title, message). Must be thread-safe or marshal to
            the main thread itself. Failures inside it are swallowed so error
            reporting can never cause a second crash.
    """

    TITLE = "⚠️ MacWhisper error — please stop talking"

    def __init__(self, alert):
        self._alert = alert
        self._lock = threading.Lock()

    def report(self, where: str, exc: BaseException, while_recording: bool = False) -> None:
        summary = f"{type(exc).__name__}: {exc}"
        logger.error("Error in %s: %s", where, summary,
                     exc_info=(type(exc), exc, exc.__traceback__))
        lines = [
            f"Something went wrong in: {where}",
            summary,
        ]
        if while_recording:
            lines.append("The current recording may not have been saved.")
        lines.append("Details: ~/.macwhisper/logs/macwhisper.log")
        message = "\n".join(lines)
        with self._lock:
            try:
                self._alert(self.TITLE, message)
            except Exception:
                logger.error("Error alert itself failed:\n%s", traceback.format_exc())


def install_thread_excepthook(reporter: ErrorReporter) -> None:
    """Route uncaught exceptions from any thread to the reporter."""
    previous = threading.excepthook

    def hook(args):
        if issubclass(args.exc_type, SystemExit):
            return previous(args)
        name = args.thread.name if args.thread else "unknown thread"
        exc = args.exc_value if args.exc_value is not None else args.exc_type()
        reporter.report(f"thread '{name}'", exc)

    threading.excepthook = hook
