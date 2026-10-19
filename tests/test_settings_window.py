"""Tests for the Settings window model (no AppKit)."""

from unittest.mock import MagicMock

import pytest

import settings_window as sw
import shortcuts


class FakeApp:
    """Minimal stand-in exposing the same attributes/handlers as TranscriberApp."""

    def __init__(self):
        self.translate_mode = False
        self.live_mode = True
        self.save_audio = False
        self.meeting_subtitles = True
        self.save_meeting_audio = False
        self.meeting_final_pass = True
        self.qwen_strip_fillers = True
        self.qwen_tight_spacing = True
        self.shortcuts = shortcuts.merged({})
        self.engine = "qwen"
        self.calls = []
        self.saved = 0
        self.model_label = "Qwen3 0.6B (Fast, zh+en)"

    def _flip(self, attr):
        setattr(self, attr, not getattr(self, attr))
        self.calls.append(attr)

    def _toggle_translate(self, _):
        self._flip("translate_mode")

    def _toggle_live_mode(self, _):
        self._flip("live_mode")

    def _toggle_save_audio(self, _):
        self._flip("save_audio")

    def _toggle_meeting_subtitles(self, _):
        self._flip("meeting_subtitles")

    def _toggle_save_meeting_audio(self, _):
        self._flip("save_meeting_audio")

    def _set_pref(self, name, value):
        setattr(self, name, value)
        self.calls.append(name)

    def _set_model(self, label):
        self.model_label = label
        self.calls.append(("model", label))

    def _active_model_label(self):
        return self.model_label

    def _set_shortcuts(self, cfg):
        self.shortcuts = cfg
        self.calls.append("shortcuts")


def test_items_cover_every_setting():
    keys = [i.key for i in sw.TOGGLES]
    for k in ("translate_mode", "live_mode", "save_audio", "meeting_subtitles",
              "save_meeting_audio", "meeting_final_pass", "qwen_strip_fillers",
              "qwen_tight_spacing"):
        assert k in keys


def test_toggle_values_read_from_app():
    m = sw.SettingsModel(FakeApp())
    assert m.value("live_mode") is True
    assert m.value("translate_mode") is False


def test_set_same_value_does_not_flip():
    app = FakeApp()
    m = sw.SettingsModel(app)
    m.set_toggle("live_mode", True)            # already on
    assert app.calls == [] and app.live_mode is True


def test_set_toggle_uses_existing_handler_once():
    app = FakeApp()
    m = sw.SettingsModel(app)
    m.set_toggle("translate_mode", True)
    assert app.translate_mode is True and app.calls == ["translate_mode"]


def test_prefs_without_handler_use_set_pref():
    app = FakeApp()
    m = sw.SettingsModel(app)
    m.set_toggle("meeting_final_pass", False)
    assert app.meeting_final_pass is False and app.calls == ["meeting_final_pass"]


def test_model_choice():
    app = FakeApp()
    m = sw.SettingsModel(app)
    assert m.model_label() == "Qwen3 0.6B (Fast, zh+en)"
    m.set_model("Large (Best)")
    assert app.calls == [("model", "Large (Best)")]
    m.set_model("Large (Best)")                    # unchanged -> no reload
    assert len(app.calls) == 1


def test_record_shortcut_ok():
    app = FakeApp()
    m = sw.SettingsModel(app)
    err = m.record_shortcut("meeting", 0, "ctrl+cmd+vk46")
    assert err is None
    assert app.shortcuts["meeting"] == ["ctrl+cmd+vk46", "ctrl+shift+f8"]
    assert app.calls == ["shortcuts"]


def test_record_shortcut_rejects_invalid_and_keeps_old():
    app = FakeApp()
    m = sw.SettingsModel(app)
    assert m.record_shortcut("meeting", 0, "vk46")              # plain letter
    assert m.record_shortcut("push_to_talk", 0, "ctrl+alt+vk46")
    assert app.shortcuts == shortcuts.merged({}) and app.calls == []


def test_record_shortcut_rejects_conflict_with_other_action():
    app = FakeApp()
    m = sw.SettingsModel(app)
    err = m.record_shortcut("translate", 0, "ctrl+alt+vk46")    # = meeting
    assert err and "Start / stop meeting" in err
    assert app.calls == []


def test_record_same_shortcut_for_same_slot_is_fine():
    app = FakeApp()
    m = sw.SettingsModel(app)
    assert m.record_shortcut("meeting", 0, "ctrl+alt+vk46") is None


def test_record_second_slot_and_clear_it():
    app = FakeApp()
    m = sw.SettingsModel(app)
    m.record_shortcut("cycle_model", 1, "ctrl+cmd+vk46")
    assert app.shortcuts["cycle_model"] == ["ctrl+shift+vk46", "ctrl+cmd+vk46"]
    m.clear_shortcut("cycle_model", 1)
    assert app.shortcuts["cycle_model"] == ["ctrl+shift+vk46"]


def test_push_to_talk_cannot_be_cleared():
    app = FakeApp()
    m = sw.SettingsModel(app)
    assert m.clear_shortcut("push_to_talk", 0)
    assert app.shortcuts["push_to_talk"] == ["alt_r"]


def test_reset_shortcut_to_default():
    app = FakeApp()
    app.shortcuts = shortcuts.merged({"meeting": ["ctrl+cmd+vk46"]})
    m = sw.SettingsModel(app)
    m.reset_shortcut("meeting")
    assert app.shortcuts["meeting"] == shortcuts.DEFAULTS["meeting"]


def test_shortcut_labels():
    m = sw.SettingsModel(FakeApp())
    assert m.shortcut_labels("meeting") == ["⌃⌥M", "⌃⇧F8"]
    assert m.shortcut_labels("push_to_talk") == ["Right ⌥"]


def test_model_does_not_mutate_app_shortcuts_in_place():
    app = FakeApp()
    before = app.shortcuts
    snapshot = {k: list(v) for k, v in before.items()}
    sw.SettingsModel(app).record_shortcut("meeting", 0, "ctrl+cmd+vk46")
    assert before == snapshot              # old dict untouched; app got a new one


# ── recording flow (handle_key is plain Python; no real keyboard) ──

KEYDOWN, FLAGS = 10, 12
CTRL, ALT, SHIFT, CMD = 1 << 18, 1 << 19, 1 << 17, 1 << 20


def _recorder(action="meeting", slot=0):
    app = FakeApp()
    app._shortcuts_paused = False
    win = sw.SettingsWindow(app, ["Large (Best)"])
    win._recording = (action, slot)          # as start_recording() would set
    win._record_buttons = {}
    app._shortcuts_paused = True
    win._show_error = lambda text: setattr(win, "last_msg", text)
    win.refresh = lambda: None
    return app, win


def test_record_combo_saves_and_resumes_global_hotkeys():
    app, win = _recorder()
    assert win.handle_key(KEYDOWN, 46, CTRL | ALT) is None      # swallowed
    assert app.shortcuts["meeting"][0] == "ctrl+alt+vk46"
    assert win._recording is None and app._shortcuts_paused is False


def test_modifier_presses_alone_do_not_finish_a_combo():
    app, win = _recorder()
    win.handle_key(FLAGS, 59, CTRL)                              # ⌃ down
    win.handle_key(FLAGS, 58, CTRL | ALT)                        # ⌥ down
    assert win._recording == ("meeting", 0) and app.calls == []


def test_conflict_keeps_recording_and_shows_error():
    app, win = _recorder("translate", 0)
    win.handle_key(KEYDOWN, 46, CTRL | ALT)                      # = meeting
    assert win._recording == ("translate", 0)
    assert "already used" in win.last_msg
    assert app._shortcuts_paused is True


def test_escape_cancels():
    app, win = _recorder()
    win.handle_key(KEYDOWN, 53, 0)
    assert win._recording is None and app._shortcuts_paused is False and app.calls == []


def test_ptt_right_option_recorded_left_option_ignored():
    app, win = _recorder("push_to_talk", 0)
    win.handle_key(FLAGS, 58, ALT)                               # left ⌥: not allowed
    assert win._recording is not None and app.calls == []
    win.handle_key(FLAGS, 58, 0)                                 # released
    win.handle_key(FLAGS, 61, ALT)                               # right ⌥ down
    assert app.shortcuts["push_to_talk"] == ["alt_r"] and win._recording is None


def test_ptt_right_cmd_and_f13():
    app, win = _recorder("push_to_talk", 0)
    win.handle_key(FLAGS, 54, CMD)
    assert app.shortcuts["push_to_talk"] == ["cmd_r"]
    app, win = _recorder("push_to_talk", 0)
    win.handle_key(KEYDOWN, 105, 0)                              # F13
    assert app.shortcuts["push_to_talk"] == ["f13"]


def test_ptt_release_event_is_not_a_press():
    app, win = _recorder("push_to_talk", 0)
    win.handle_key(FLAGS, 61, 0)                                 # right ⌥ UP (flag cleared)
    assert win._recording is not None and app.calls == []


def test_plain_letter_rejected_for_combo():
    app, win = _recorder()
    win.handle_key(KEYDOWN, 46, 0)
    assert win._recording is not None and app.calls == []
    assert win.last_msg


def test_closing_window_while_recording_resumes_hotkeys():
    app, win = _recorder()
    win._monitor = None
    win.stop_recording()
    assert app._shortcuts_paused is False


def test_not_recording_passes_events_through():
    app, win = _recorder()
    win._recording = None
    assert win.handle_key(KEYDOWN, 46, CTRL | ALT) is True


# ── real AppKit window (catches selector / PyObjC mistakes) ──

LABELS = ["Small (Fast)", "Medium (Accurate)", "Large (Best)",
          "Qwen3 0.6B (Fast, zh+en)", "Qwen3 1.7B (Best, zh+en)"]


@pytest.fixture
def real_window():
    pytest.importorskip("AppKit")
    from AppKit import NSApplication
    NSApplication.sharedApplication()
    app = FakeApp()
    app._shortcuts_paused = False
    win = sw.SettingsWindow(app, LABELS)
    win._build()
    win.refresh()
    yield app, win
    win.window.close()


def test_real_window_has_all_controls(real_window):
    app, win = real_window
    assert len(win._checkboxes) == len(sw.TOGGLES)
    assert win._popup.numberOfItems() == 5
    assert win._popup.titleOfSelectedItem() == "Qwen3 0.6B (Fast, zh+en)"
    assert len(win._record_buttons) == 1 + 4 * sw.SLOTS
    assert win._record_buttons[("meeting", 0)].title() == "⌃⌥M"
    assert win._record_buttons[("meeting", 1)].title() == "⌃⇧F8"
    assert win._record_buttons[("cycle_model", 1)].title() == "Add…"


def test_real_checkbox_click_reaches_app(real_window):
    app, win = real_window
    box = win._checkboxes["translate_mode"]
    assert box.state() == 0
    box.performClick_(None)
    assert app.translate_mode is True and app.calls == ["translate_mode"]
    assert box.state() == 1


def test_real_checkbox_for_pref_without_handler(real_window):
    app, win = real_window
    win._checkboxes["qwen_tight_spacing"].performClick_(None)
    assert app.qwen_tight_spacing is False


def test_real_popup_selects_model(real_window):
    app, win = real_window
    win._popup.selectItemWithTitle_("Large (Best)")
    win.on_model(win._popup)       # NSPopUpButton fires its action on user selection only
    assert app.model_label == "Large (Best)"


def test_real_reset_button(real_window):
    from AppKit import NSButton
    app, win = real_window
    app.shortcuts = shortcuts.merged({"meeting": ["ctrl+cmd+vk46"]})
    win.refresh()
    assert win._record_buttons[("meeting", 0)].title() == "⌃⌘M"
    reset = [v for v in win.window.contentView().subviews()
             if isinstance(v, NSButton) and v.title() == "Default"
             and win._tags.get(v.tag()) == ("reset", "meeting")][0]
    reset.performClick_(None)
    assert app.shortcuts["meeting"] == shortcuts.DEFAULTS["meeting"]
    assert win._record_buttons[("meeting", 0)].title() == "⌃⌥M"


def test_real_record_button_starts_and_close_stops(real_window):
    app, win = real_window
    win._record_buttons[("translate", 0)].performClick_(None)
    assert win._recording == ("translate", 0) and app._shortcuts_paused is True
    assert win._monitor is not None
    win.window.performClose_(None)          # windowWillClose_ -> stop_recording
    assert win._recording is None and app._shortcuts_paused is False and win._monitor is None


def test_real_record_button_twice_cancels(real_window):
    app, win = real_window
    b = win._record_buttons[("translate", 0)]
    b.performClick_(None)
    b.performClick_(None)
    assert win._recording is None and app._shortcuts_paused is False
    assert b.title() == "⌃⇧T"


def test_window_reopens_after_close(real_window):
    app, win = real_window
    win.window.performClose_(None)
    win.show()
    assert win.window.isVisible()
