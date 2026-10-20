"""Tests for single_instance (real processes contend for the lock)."""

import os
import subprocess
import sys
import textwrap

import single_instance as si

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _other_process_tries(lock_path):
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {REPO!r})
        import single_instance as si
        print("got" if si.acquire({lock_path!r}) else "busy")
    """)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=30).stdout.strip()


def test_second_process_is_refused_while_first_holds_lock(tmp_path):
    path = str(tmp_path / "lock")
    held = si.acquire(path)
    assert held is not None
    try:
        assert _other_process_tries(path) == "busy"
    finally:
        si.release(held)
    assert _other_process_tries(path) == "got"


def test_lock_released_when_holder_dies(tmp_path):
    """A crashed/killed app must not block the next launch."""
    path = str(tmp_path / "lock")
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {REPO!r})
        import single_instance as si
        held = si.acquire({path!r})   # must stay referenced: GC would close + unlock it
        assert held
        print("held", flush=True)
        time.sleep(60)
    """)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert _other_process_tries(path) == "busy"
        holder.kill()
        holder.wait(10)
        assert _other_process_tries(path) == "got"
    finally:
        if holder.poll() is None:
            holder.kill()


def test_lock_file_records_pid(tmp_path):
    path = str(tmp_path / "lock")
    held = si.acquire(path)
    try:
        assert open(path).read().strip() == str(os.getpid())
        assert si.holder_pid(path) == os.getpid()
    finally:
        si.release(held)


def test_creates_parent_directory(tmp_path):
    path = str(tmp_path / "new" / "dir" / "lock")
    held = si.acquire(path)
    assert held is not None
    si.release(held)
