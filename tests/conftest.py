import sys
import os
import tempfile

import pytest

# Add project root to sys.path so tests can `import app`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pytest_addoption(parser):
    parser.addoption(
        "--generate", type=int, default=0, metavar="N",
        help="Generate N random WAV pairs for N-to-N dual-channel E2E testing",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that require real ML models (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "hardware: marks tests that require real audio hardware (deselect with '-m \"not hardware\"')")


@pytest.fixture(autouse=True)
def _redirect_meetings_dir(tmp_path):
    """Redirect meeting auto-save to a temp directory so tests don't pollute
    ~/.macwhisper/meetings/ with dozens of snippet files."""
    import meeting
    orig = meeting.MEETINGS_DIR
    meeting.MEETINGS_DIR = str(tmp_path / "meetings")
    yield
    meeting.MEETINGS_DIR = orig

_REAL_USER_FILES = ("config.json", "overlay.json")


@pytest.fixture(autouse=True)
def _never_touch_real_user_config(tmp_path, monkeypatch):
    """No test may read or write ~/.macwhisper/config.json or overlay.json.

    Regression guard: two v0.7.1 tests called _save_config() without the
    tmp_config fixture and overwrote the user's real settings. Every test
    now gets temp paths by default; tests that patch CONFIG_PATH themselves
    (tmp_config) still win because their patch is applied later.
    """
    import importlib
    app = sys.modules.get("app") or importlib.import_module("app")
    monkeypatch.setattr(app, "CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setattr(app, "OVERLAY_FRAME_PATH", str(tmp_path / "overlay.json"))
    yield


def pytest_sessionstart(session):
    data = os.path.expanduser("~/.macwhisper")
    session.config._mw_real_files = {
        name: _fingerprint(os.path.join(data, name)) for name in _REAL_USER_FILES
    }


def pytest_sessionfinish(session, exitstatus):
    before = getattr(session.config, "_mw_real_files", {})
    data = os.path.expanduser("~/.macwhisper")
    changed = [n for n, fp in before.items() if _fingerprint(os.path.join(data, n)) != fp]
    if changed:
        session.exitstatus = 1
        print(f"\n\nERROR: tests modified the user's real files in ~/.macwhisper: {changed}\n")


def _fingerprint(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None
