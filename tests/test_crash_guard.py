"""Tests for crash_guard: crash evidence capture and error reporting."""

import os
import subprocess
import sys
import threading

import crash_guard


# ── RunMarker: detect unclean shutdown ────────────────────────

def test_marker_absent_means_no_previous_crash(tmp_path):
    m = crash_guard.RunMarker(str(tmp_path / "running"))
    assert m.previous_run_crashed() is False


def test_marker_left_behind_means_previous_crash(tmp_path):
    path = tmp_path / "running"
    path.write_text("99999999")  # PID that is not alive
    assert crash_guard.RunMarker(str(path)).previous_run_crashed() is True


def test_marker_of_live_process_is_not_a_crash(tmp_path):
    path = tmp_path / "running"
    path.write_text(str(os.getpid()))  # another live instance
    assert crash_guard.RunMarker(str(path)).previous_run_crashed() is False


def test_marker_create_and_clear(tmp_path):
    path = tmp_path / "running"
    m = crash_guard.RunMarker(str(path))
    m.create()
    assert path.read_text().strip() == str(os.getpid())
    m.clear()
    assert not path.exists()
    m.clear()  # idempotent


def test_marker_garbage_content_counts_as_crash(tmp_path):
    path = tmp_path / "running"
    path.write_text("not a pid")
    assert crash_guard.RunMarker(str(path)).previous_run_crashed() is True


# ── crash.log helpers ─────────────────────────────────────────

def test_read_tail_missing_file(tmp_path):
    assert crash_guard.read_tail(str(tmp_path / "nope.log")) == ""


def test_read_tail_returns_last_lines(tmp_path):
    p = tmp_path / "crash.log"
    p.write_text("\n".join(f"line{i}" for i in range(100)))
    tail = crash_guard.read_tail(str(p), max_lines=3)
    assert tail.splitlines() == ["line97", "line98", "line99"]


def test_faulthandler_writes_hard_crash_to_file(tmp_path):
    """A real SIGABRT in a child process must leave evidence in crash.log."""
    log = tmp_path / "crash.log"
    code = (
        "import sys, os, crash_guard;"
        f"crash_guard.enable_faulthandler({str(log)!r});"
        "os.abort()"
    )
    env = {**os.environ, "PYTHONPATH": os.getcwd()}
    subprocess.run([sys.executable, "-c", code], env=env, capture_output=True)
    text = log.read_text()
    assert "Fatal Python error" in text
    assert "MacWhisper session started" in text


# ── ErrorReporter: every thread's errors reach the user ───────

def test_reporter_calls_alert_with_summary():
    seen = []
    r = crash_guard.ErrorReporter(alert=lambda t, m: seen.append((t, m)))
    try:
        raise ValueError("boom")
    except ValueError as e:
        r.report("hotkey listener", e)
    assert len(seen) == 1
    assert "hotkey listener" in seen[0][1]
    assert "ValueError: boom" in seen[0][1]


def test_reporter_never_raises_even_if_alert_fails():
    def bad_alert(_t, _m):
        raise RuntimeError("alert broken")
    r = crash_guard.ErrorReporter(alert=bad_alert)
    r.report("x", ValueError("y"))  # must not raise


def test_reporter_recording_hint():
    seen = []
    r = crash_guard.ErrorReporter(alert=lambda t, m: seen.append(m))
    r.report("recording", ValueError("y"), while_recording=True)
    assert "may not have been saved" in seen[0]


def test_thread_excepthook_routes_to_reporter():
    seen = []
    r = crash_guard.ErrorReporter(alert=lambda t, m: seen.append(m))
    old = threading.excepthook
    try:
        crash_guard.install_thread_excepthook(r)
        t = threading.Thread(target=lambda: 1 / 0, name="worker-x")
        t.start()
        t.join()
    finally:
        threading.excepthook = old
    assert seen and "ZeroDivisionError" in seen[0]
    assert "worker-x" in seen[0]
