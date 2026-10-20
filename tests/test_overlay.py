"""Tests for overlay window placement, persistence, and the no-auto-move rule.

Core requirement: once the overlay is shown, only the user (drag/resize)
may change its frame. Content updates must never move or resize it.
"""

import json
from unittest.mock import MagicMock

import pytest

import overlay

SCREEN_A = (0, 0, 1728, 1117)        # built-in display
SCREEN_B = (1728, 0, 2560, 1440)     # external display to the right


# ── default_frame ─────────────────────────────────────────────

def test_default_frame_centered_bottom_of_screen():
    x, y, w, h = overlay.default_frame(SCREEN_A)
    assert w == 960
    assert h == overlay.DEFAULT_HEIGHT
    assert x == pytest.approx((1728 - 960) / 2)
    assert y == 40


def test_default_frame_on_offset_screen():
    x, y, w, _ = overlay.default_frame(SCREEN_B)
    assert x == pytest.approx(1728 + (2560 - 960) / 2)


def test_default_frame_narrow_screen_uses_80_percent():
    _, _, w, _ = overlay.default_frame((0, 0, 1000, 800))
    assert w == pytest.approx(800)


# ── resolve_frame ─────────────────────────────────────────────

def test_resolve_uses_saved_frame_when_visible():
    saved = (2000, 300, 700, 200)
    assert overlay.resolve_frame(saved, [SCREEN_A, SCREEN_B], SCREEN_A) == saved


def test_resolve_falls_back_when_screen_disconnected():
    saved = (2000, 300, 700, 200)  # was on SCREEN_B, now unplugged
    assert overlay.resolve_frame(saved, [SCREEN_A], SCREEN_A) == overlay.default_frame(SCREEN_A)


def test_resolve_falls_back_when_no_saved_frame():
    assert overlay.resolve_frame(None, [SCREEN_A], SCREEN_A) == overlay.default_frame(SCREEN_A)


def test_resolve_falls_back_on_garbage():
    assert overlay.resolve_frame(("a", 1), [SCREEN_A], SCREEN_A) == overlay.default_frame(SCREEN_A)


def test_resolve_enforces_min_size():
    x, y, w, h = overlay.resolve_frame((100, 100, 10, 5), [SCREEN_A], SCREEN_A)
    assert (x, y) == (100, 100)
    assert w == overlay.MIN_WIDTH
    assert h == overlay.MIN_HEIGHT


# ── persistence ───────────────────────────────────────────────

def test_save_and_load_frame_per_mode(tmp_path):
    path = str(tmp_path / "overlay.json")
    overlay.save_frame(path, "meeting", (1, 2, 300, 400))
    overlay.save_frame(path, "talk", (5, 6, 700, 800))
    assert overlay.load_frame(path, "meeting") == (1, 2, 300, 400)
    assert overlay.load_frame(path, "talk") == (5, 6, 700, 800)


def test_load_frame_missing_or_corrupt(tmp_path):
    path = tmp_path / "overlay.json"
    assert overlay.load_frame(str(path), "talk") is None
    path.write_text("{not json")
    assert overlay.load_frame(str(path), "talk") is None
    path.write_text(json.dumps({"talk": [1, 2]}))
    assert overlay.load_frame(str(path), "talk") is None


def test_save_frame_without_path_is_noop():
    overlay.save_frame(None, "talk", (1, 2, 3, 4))  # must not raise


# ── the critical rule: updates never move/resize the window ──

@pytest.mark.parametrize("mode", ["talk", "meeting"])
def test_update_overlay_never_changes_frame(mode):
    pytest.importorskip("AppKit")
    panel = MagicMock()
    text_view = MagicMock()
    long_text = "这是一段很长的测试文字。" * 200
    overlay.update_overlay(panel, text_view, long_text, "unconfirmed tail", mode=mode)
    overlay.update_overlay(panel, text_view, "", "", mode=mode)
    panel.setFrame_display_.assert_not_called()
    panel.setFrame_display_animate_.assert_not_called()
    panel.setFrameOrigin_.assert_not_called()
    panel.setContentSize_.assert_not_called()
    panel.center.assert_not_called()


def test_overlay_window_level_stays_high():
    """Regression: setFloatingPanel_ silently reset the level to 3."""
    pytest.importorskip("AppKit")
    from AppKit import NSApplication
    NSApplication.sharedApplication()
    panel, _ = overlay.create_overlay("talk", None)
    try:
        assert panel.level() == 1000
    finally:
        overlay.destroy_overlay(panel)


# ── Meeting overlay Stop button (v0.7.1) ──────────────────────

def _real_panel(mode, on_stop=None):
    pytest.importorskip("AppKit")
    from AppKit import NSApplication
    NSApplication.sharedApplication()
    panel, _ = overlay.create_overlay(mode, None, on_stop=on_stop)
    return panel


def _stop_button(panel):
    from AppKit import NSButton
    return [v for v in panel.contentView().subviews() if isinstance(v, NSButton)]


def test_meeting_overlay_has_stop_button_on_top():
    panel = _real_panel("meeting", on_stop=lambda: None)
    try:
        buttons = _stop_button(panel)
        assert len(buttons) == 1
        assert "Stop" in buttons[0].title()
        # Must be ABOVE the full-size drag view, otherwise clicks never reach it
        assert panel.contentView().subviews()[-1] is buttons[0]
    finally:
        overlay.destroy_overlay(panel)


def test_no_stop_button_for_talk_or_without_callback():
    for mode, cb in (("talk", lambda: None), ("meeting", None)):
        panel = _real_panel(mode, on_stop=cb)
        try:
            assert _stop_button(panel) == []
        finally:
            overlay.destroy_overlay(panel)


@pytest.mark.parametrize("confirmed, expected_calls", [(True, 1), (False, 0)])
def test_stop_button_asks_for_confirmation(monkeypatch, confirmed, expected_calls):
    calls = []
    panel = _real_panel("meeting", on_stop=lambda: calls.append(1))
    try:
        monkeypatch.setattr(overlay, "_confirm_stop", lambda: confirmed)
        btn = _stop_button(panel)[0]
        btn.performClick_(None)
        assert len(calls) == expected_calls
    finally:
        overlay.destroy_overlay(panel)


def test_stop_button_stays_top_right_when_resized():
    from AppKit import NSMakeRect
    panel = _real_panel("meeting", on_stop=lambda: None)
    try:
        btn = _stop_button(panel)[0]
        f = panel.frame()
        panel.setFrame_display_(NSMakeRect(f.origin.x, f.origin.y, f.size.width + 200, f.size.height + 80), False)
        cf = panel.contentView().frame()
        bf = btn.frame()
        assert cf.size.width - (bf.origin.x + bf.size.width) < 20       # near right edge
        assert cf.size.height - (bf.origin.y + bf.size.height) < 20     # near top edge
    finally:
        overlay.destroy_overlay(panel)


def test_stop_callback_error_is_contained(monkeypatch, caplog):
    def boom():
        raise RuntimeError("stop failed")
    panel = _real_panel("meeting", on_stop=boom)
    try:
        monkeypatch.setattr(overlay, "_confirm_stop", lambda: True)
        _stop_button(panel)[0].performClick_(None)      # must not raise into AppKit
        assert "stop failed" in caplog.text
    finally:
        overlay.destroy_overlay(panel)


def test_destroy_detaches_stop_button_target():
    panel = _real_panel("meeting", on_stop=lambda: None)
    btn = _stop_button(panel)[0]
    assert btn.target() is not None
    overlay.destroy_overlay(panel)
    assert btn.target() is None
    assert not [k for k in overlay._delegates if isinstance(k, tuple)]


def test_confirm_stop_returns_focus_to_previous_app(monkeypatch):
    """After the dialog, the user's app must be frontmost again (auto-paste target)."""
    pytest.importorskip("AppKit")
    prev = MagicMock()
    monkeypatch.setattr(overlay, "_frontmost_app", lambda: prev)
    alert = MagicMock()
    alert.runModal.return_value = 1001          # Cancel
    monkeypatch.setattr(overlay, "_make_alert", lambda: alert)
    assert overlay._confirm_stop() is False
    prev.activateWithOptions_.assert_called_once()


def test_meeting_label_shows_current_stop_shortcut(monkeypatch):
    """Label must not advertise the removed ⌃⇧R."""
    assert "Ctrl+Shift+R" not in overlay.meeting_label()
    monkeypatch.setattr(overlay, "meeting_stop_hint", "⌃⌘M")
    assert "⌃⌘M" in overlay.meeting_label()
    monkeypatch.setattr(overlay, "meeting_stop_hint", "")
    assert overlay.meeting_label().startswith("● Meeting Recording")
    assert "to stop" not in overlay.meeting_label() or "Stop" in overlay.meeting_label()
