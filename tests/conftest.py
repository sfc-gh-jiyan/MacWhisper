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