"""Tests for permissions.py — the macOS calls are replaced by fakes."""

from unittest.mock import MagicMock

import pytest

import permissions as p


@pytest.fixture
def fake_os(monkeypatch):
    """Fake macOS: every permission missing until a test grants it."""
    state = {"mic": 0, "ax": False, "im": False}
    calls = {"opened": [], "mic_request": 0, "ax_prompt": 0, "im_request": 0}

    class Device:
        @staticmethod
        def authorizationStatusForMediaType_(media):
            assert media == "soun"
            return state["mic"]

        @staticmethod
        def requestAccessForMediaType_completionHandler_(media, handler):
            calls["mic_request"] += 1
            handler(True)

    monkeypatch.setattr(p, "_capture_device", lambda: Device)
    monkeypatch.setattr(p, "_ax_trusted", lambda prompt=False: (
        calls.__setitem__("ax_prompt", calls["ax_prompt"] + int(prompt)) or state["ax"]))
    monkeypatch.setattr(p, "_im_preflight", lambda: state["im"])
    monkeypatch.setattr(p, "_im_request", lambda: calls.__setitem__(
        "im_request", calls["im_request"] + 1))
    monkeypatch.setattr(p, "_open_url", lambda url: calls["opened"].append(url) or True)
    return state, calls


def test_the_three_required_permissions_in_order():
    assert [x.key for x in p.PERMISSIONS] == ["microphone", "accessibility", "input_monitoring"]
    for perm in p.PERMISSIONS:
        assert perm.name and perm.why and perm.pane.startswith("x-apple.systempreferences:")


@pytest.mark.parametrize("raw, expected", [
    (0, p.NOT_DETERMINED), (1, p.DENIED), (2, p.DENIED), (3, p.GRANTED)])
def test_microphone_status_mapping(fake_os, raw, expected):
    state, _ = fake_os
    state["mic"] = raw
    assert p.status("microphone") == expected


def test_accessibility_and_input_monitoring_status(fake_os):
    state, _ = fake_os
    assert p.status("accessibility") == p.DENIED
    assert p.status("input_monitoring") == p.DENIED
    state["ax"] = state["im"] = True
    assert p.status("accessibility") == p.GRANTED
    assert p.status("input_monitoring") == p.GRANTED


def test_a_failing_check_is_unknown_never_assumed_granted(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("API missing")
    monkeypatch.setattr(p, "_capture_device", boom)
    monkeypatch.setattr(p, "_ax_trusted", boom)
    monkeypatch.setattr(p, "_im_preflight", boom)
    for key in ("microphone", "accessibility", "input_monitoring"):
        assert p.status(key) == p.UNKNOWN


def test_missing_lists_only_what_is_not_granted(fake_os):
    state, _ = fake_os
    state["mic"] = 3
    assert [x.key for x in p.missing()] == ["accessibility", "input_monitoring"]
    state["ax"] = state["im"] = True
    assert p.missing() == []


def test_unknown_counts_as_missing(monkeypatch, fake_os):
    monkeypatch.setattr(p, "status", lambda key: p.UNKNOWN)
    assert len(p.missing()) == 3


def test_requests_use_the_system_prompts(fake_os):
    _, calls = fake_os
    p.request("microphone")
    p.request("accessibility")
    p.request("input_monitoring")
    assert calls["mic_request"] == 1 and calls["ax_prompt"] == 1 and calls["im_request"] == 1


def test_open_settings_opens_the_right_pane(fake_os):
    _, calls = fake_os
    assert p.open_settings("input_monitoring") is True
    assert calls["opened"] == [p.get("input_monitoring").pane]
    assert "Privacy_ListenEvent" in calls["opened"][0]


def test_microphone_check_never_opens_an_audio_stream(fake_os, monkeypatch):
    """Opening a stream is the call that hung on 2026-10-29 / 2026-11-06."""
    import sounddevice
    monkeypatch.setattr(sounddevice, "InputStream", MagicMock(side_effect=AssertionError))
    p.status("microphone")
    p.request("microphone")
