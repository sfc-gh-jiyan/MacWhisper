import sys
import os

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
