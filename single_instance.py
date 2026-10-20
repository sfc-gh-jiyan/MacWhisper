"""Single running MacWhisper instance (double-click + ./run.sh must not both run).

Two instances would fight over the global hotkeys and the microphone. An
exclusive flock() on ~/.macwhisper/.instance.lock decides; the OS releases
it when the process exits in any way (quit, crash, kill -9), so a stale
lock can never block the next launch.
"""

from __future__ import annotations

import fcntl
import os


def acquire(path: str):
    """Return an open file holding the lock, or None if another instance has it.

    Keep the returned object alive for the life of the process.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    f.seek(0)
    f.truncate()
    f.write(f"{os.getpid()}\n")
    f.flush()
    return f


def release(f) -> None:
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


def holder_pid(path: str) -> int | None:
    """PID written by the current holder (informational; may be stale)."""
    try:
        with open(path) as f:
            return int(f.read().strip() or 0) or None
    except (OSError, ValueError):
        return None
