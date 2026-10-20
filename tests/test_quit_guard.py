"""Tests for quit_guard (SIGTERM must quit even when the main thread is stuck)."""

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import quit_guard

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _child(main_block: str, deadline: float, tmp_path) -> tuple:
    """Start a child that installs quit_guard then blocks its main thread."""
    done = tmp_path / "cleanup_done"
    code = textwrap.dedent(f"""
        import os, sys, time
        sys.path.insert(0, {REPO!r})
        import quit_guard

        def request_quit():
            open({str(tmp_path / 'requested')!r}, 'w').close()

        def cleanup():
            open({str(done)!r}, 'w').close()

        quit_guard.install(request_quit, cleanup, deadline_s={deadline})
        print("ready", flush=True)
    """) + textwrap.dedent(main_block)
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    return proc, done


def test_sigterm_quits_when_main_thread_is_stuck_in_a_run_loop(tmp_path):
    """Regression: the app sat in a modal NSAlert / C run loop, the Python
    signal handler never ran, and SIGTERM did nothing for 10+ s."""
    block = """
        from Foundation import NSRunLoop, NSDate, NSPort, NSDefaultRunLoopMode
        loop = NSRunLoop.currentRunLoop()
        loop.addPort_forMode_(NSPort.port(), NSDefaultRunLoopMode)
        while True:   # stays inside C, like AppKit's event loop
            loop.runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.distantFuture())
    """
    proc, done = _child(block, deadline=1.5, tmp_path=tmp_path)
    try:
        t0 = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
        elapsed = time.monotonic() - t0
    finally:
        if proc.poll() is None:
            proc.kill()
    assert elapsed < 5, elapsed
    assert (tmp_path / "requested").exists()      # normal quit was asked for first
    assert done.exists()                           # then the fallback cleaned up
    assert proc.returncode == 0


def test_sigterm_with_responsive_main_thread_still_cleans_up(tmp_path):
    proc, done = _child("import time\nwhile True: time.sleep(0.1)\n", deadline=1.0,
                        tmp_path=tmp_path)
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert done.exists() and proc.returncode == 0


def test_cleanup_runs_once_even_if_called_twice():
    calls = []
    guard = quit_guard.QuitGuard(request_quit=lambda: None, cleanup=lambda: calls.append(1),
                                 deadline_s=0.05, exit_fn=lambda code: None)
    guard.run_cleanup()
    guard.run_cleanup()
    assert calls == [1]


def test_cleanup_error_does_not_block_exit():
    exits = []

    def bad():
        raise RuntimeError("boom")
    guard = quit_guard.QuitGuard(request_quit=lambda: None, cleanup=bad,
                                 deadline_s=0.01, exit_fn=exits.append)
    guard.on_signal()
    deadline = time.monotonic() + 2
    while not exits and time.monotonic() < deadline:
        time.sleep(0.01)
    assert exits == [0]


def test_second_signal_is_ignored():
    requested = []
    guard = quit_guard.QuitGuard(request_quit=lambda: requested.append(1), cleanup=lambda: None,
                                 deadline_s=10, exit_fn=lambda c: None)
    guard.on_signal()
    guard.on_signal()
    assert requested == [1]


def test_request_quit_error_still_falls_back():
    exits = []

    def bad():
        raise RuntimeError("no run loop")
    guard = quit_guard.QuitGuard(request_quit=bad, cleanup=lambda: None,
                                 deadline_s=0.01, exit_fn=exits.append)
    guard.on_signal()
    deadline = time.monotonic() + 2
    while not exits and time.monotonic() < deadline:
        time.sleep(0.01)
    assert exits == [0]


def test_install_from_non_main_thread_raises_clearly():
    errors = []

    def worker():
        try:
            quit_guard.install(lambda: None, lambda: None)
        except Exception as e:  # signal handlers can only be set in the main thread
            errors.append(e)
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert errors and isinstance(errors[0], ValueError)


def test_run_cleanup_accepts_event_arguments():
    calls = []
    guard = quit_guard.QuitGuard(request_quit=lambda: None, cleanup=lambda: calls.append(1),
                                 deadline_s=1, exit_fn=lambda c: None)
    guard.run_cleanup("sender", reason="quit")
    assert calls == [1]


def test_deadline_waits_for_a_running_cleanup_to_finish():
    """Review CRITICAL: run_cleanup marked itself done before cleanup finished,
    so the deadline thread could os._exit mid-write (truncated meeting file)."""
    events = []
    started = threading.Event()

    def slow_cleanup():
        started.set()
        time.sleep(0.4)                  # e.g. writing the meeting files
        events.append("cleanup finished")

    def exit_fn(code):
        events.append("exit")
    guard = quit_guard.QuitGuard(request_quit=lambda: None, cleanup=slow_cleanup,
                                 deadline_s=0.05, exit_fn=exit_fn, cleanup_wait_s=5)
    t = threading.Thread(target=guard.run_cleanup)   # the normal Quit path
    t.start()
    started.wait(1)
    guard.on_signal()                                 # deadline fires during cleanup
    deadline = time.monotonic() + 3
    while "exit" not in events and time.monotonic() < deadline:
        time.sleep(0.01)
    t.join(2)
    assert events == ["cleanup finished", "exit"]


def test_deadline_does_not_wait_forever_for_a_stuck_cleanup():
    exits = []

    def stuck():
        time.sleep(30)
    guard = quit_guard.QuitGuard(request_quit=lambda: None, cleanup=stuck,
                                 deadline_s=0.01, exit_fn=exits.append, cleanup_wait_s=0.3)
    threading.Thread(target=guard.run_cleanup, daemon=True).start()
    time.sleep(0.05)
    t0 = time.monotonic()
    guard.on_signal()
    while not exits and time.monotonic() - t0 < 3:
        time.sleep(0.01)
    assert exits == [0] and time.monotonic() - t0 < 2
