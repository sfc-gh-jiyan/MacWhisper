"""Tests for the persistent error alert panel."""

import pytest

import error_alert

SCREEN = (0, 0, 1728, 1117)


def test_alert_sits_above_talk_overlay():
    talk = (384, 40, 960, 160)
    x, y, w, h = error_alert.alert_frame(talk, SCREEN)
    assert y == 40 + 160 + error_alert._GAP
    assert x == pytest.approx(384 + (960 - w) / 2)
    assert h == error_alert.ALERT_HEIGHT


def test_alert_moves_down_when_talk_overlay_is_at_top():
    talk = (384, 1000, 960, 100)
    _, y, _, h = error_alert.alert_frame(talk, SCREEN)
    assert y + h <= SCREEN[3]


def test_alert_stays_on_screen_horizontally():
    x, _, w, _ = error_alert.alert_frame((1700, 40, 300, 100), SCREEN)
    assert x + w <= SCREEN[2]


def test_alert_is_persistent_and_dismissable(tmp_path):
    pytest.importorskip("AppKit")
    from AppKit import NSApplication
    NSApplication.sharedApplication()
    path = str(tmp_path / "overlay.json")
    error_alert.show_error_alert("T", "first", frame_path=path)
    panel = error_alert._state["panel"]
    assert panel is not None and panel.isVisible()
    error_alert.show_error_alert("T", "second", frame_path=path)
    assert error_alert._state["panel"] is panel  # reused, not stacked
    assert "2 errors" in error_alert._state["text"].stringValue()
    assert panel.ignoresMouseEvents() is False  # Dismiss must be clickable
    error_alert.dismiss_error_alert()
    assert not error_alert.is_showing()
    assert error_alert._state["count"] == 0


def test_alert_level_above_overlay(tmp_path):
    pytest.importorskip("AppKit")
    from AppKit import NSApplication
    NSApplication.sharedApplication()
    error_alert.show_error_alert("T", "m", frame_path=str(tmp_path / "o.json"))
    try:
        assert error_alert._state["panel"].level() == 1001
    finally:
        error_alert.dismiss_error_alert()
