"""Settings window for MacWhisper: every option on one page.

Two layers:
  - SettingsModel: plain Python, no AppKit. Reads values from the app and
    applies changes through the app's EXISTING handlers (so a checkbox here
    behaves exactly like the old menu item). Unit-tested.
  - SettingsWindow: the AppKit view of the model (NSWindow + controls).

Recording a shortcut pauses the app's global hotkeys (app._shortcuts_paused)
so pressing e.g. ⌃⌥M to record it does not also start a meeting, and
pressing Right ⌥ does not start a recording.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import shortcuts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Toggle:
    key: str          # app attribute
    label: str
    section: str
    handler: str | None = None   # app method that flips it; None -> app._set_pref


TOGGLES = (
    Toggle("live_mode", "Live subtitles while recording", "Push to Talk", "_toggle_live_mode"),
    Toggle("translate_mode", "Translate to English (uses Whisper)", "Push to Talk", "_toggle_translate"),
    Toggle("save_audio", "Save audio of each recording", "Push to Talk", "_toggle_save_audio"),
    Toggle("meeting_subtitles", "Show meeting subtitles", "Meeting", "_toggle_meeting_subtitles"),
    Toggle("save_meeting_audio", "Save meeting audio (needs participants' consent)", "Meeting",
           "_toggle_save_meeting_audio"),
    Toggle("meeting_final_pass", "Final transcript after the meeting (Qwen3, in memory)", "Meeting"),
    Toggle("qwen_strip_fillers", "Remove fillers (嗯 / 呃 / um)", "Qwen3 text"),
    Toggle("qwen_tight_spacing", "No space between Chinese and English", "Qwen3 text"),
)
_TOGGLE_BY_KEY = {t.key: t for t in TOGGLES}
SLOTS = 2   # shortcuts per action (Push to Talk uses 1)


class SettingsModel:
    def __init__(self, app, model_labels=None):
        self.app = app
        self.model_labels = list(model_labels or getattr(app, "model_labels", []) or [])

    # ── toggles ──
    def value(self, key: str) -> bool:
        return bool(getattr(self.app, key, False))

    def set_toggle(self, key: str, value: bool) -> None:
        t = _TOGGLE_BY_KEY[key]
        if self.value(key) == bool(value):
            return
        logger.info("Settings: %s -> %s", key, value)
        if t.handler:
            getattr(self.app, t.handler)(None)
        else:
            self.app._set_pref(key, bool(value))

    # ── model ──
    def model_label(self) -> str:
        return self.app._active_model_label()

    def set_model(self, label: str) -> None:
        if label != self.model_label():
            logger.info("Settings: model -> %s", label)
            self.app._set_model(label)

    # ── shortcuts ──
    def specs(self, action: str) -> list:
        return list(self.app.shortcuts.get(action, []))

    def shortcut_labels(self, action: str) -> list:
        return [shortcuts.display(s) for s in self.specs(action)]

    def _apply(self, action: str, specs: list) -> None:
        cfg = {a: list(v) for a, v in self.app.shortcuts.items()}   # new dict, never mutate
        cfg[action] = specs
        self.app._set_shortcuts(shortcuts.merged(cfg))

    def record_shortcut(self, action: str, slot: int, spec: str) -> str | None:
        """Bind spec to action[slot]. Returns an error message, or None if saved."""
        try:
            spec = shortcuts.normalize(spec)
        except ValueError as e:
            return str(e)
        reason = shortcuts.validate(action, spec)
        if reason:
            return reason
        for other in shortcuts.ACTIONS:
            if other != action and spec in self.app.shortcuts.get(other, []):
                return (f"{shortcuts.display(spec)} is already used by "
                        f"“{shortcuts.ACTION_LABELS[other]}”")
        specs = self.specs(action)
        if action == "push_to_talk":
            specs = [spec]
        elif slot < len(specs):
            specs[slot] = spec
        else:
            specs.append(spec)
        specs = list(dict.fromkeys(specs))    # dedupe, keep order
        logger.info("Settings: shortcut %s[%d] -> %s", action, slot, spec)
        self._apply(action, specs)
        return None

    def clear_shortcut(self, action: str, slot: int) -> str | None:
        if action == "push_to_talk":
            return "Push to Talk always needs a key"
        specs = self.specs(action)
        if slot < len(specs):
            del specs[slot]
            self._apply(action, specs)
        return None

    def reset_shortcut(self, action: str) -> None:
        self._apply(action, list(shortcuts.DEFAULTS[action]))


# ── AppKit window ─────────────────────────────────────────────

_controller_cls = None
WIDTH = 560
ROW_H = 26
PAD = 20


def _get_controller_cls():
    global _controller_cls
    if _controller_cls is not None:
        return _controller_cls
    from Foundation import NSObject

    class MWSettingsController(NSObject):
        """Target/delegate for the Settings window. Logic lives in SettingsWindow."""

        def toggleChanged_(self, sender):
            self._mw_owner.on_toggle(sender)

        def modelChanged_(self, sender):
            self._mw_owner.on_model(sender)

        def recordClicked_(self, sender):
            self._mw_owner.on_record(sender)

        def resetClicked_(self, sender):
            self._mw_owner.on_reset(sender)

        def clearClicked_(self, sender):
            self._mw_owner.on_clear(sender)

        def windowWillClose_(self, _notification):
            self._mw_owner.stop_recording()

    _controller_cls = MWSettingsController
    return _controller_cls


class SettingsWindow:
    """One NSWindow; build once, show/refresh many times."""

    def __init__(self, app, model_labels):
        self.app = app
        self.model = SettingsModel(app, model_labels)
        self.window = None
        self._checkboxes = {}       # key -> NSButton
        self._record_buttons = {}   # (action, slot) -> NSButton
        self._tags = {}             # tag -> payload
        self._popup = None
        self._clear_buttons = {}    # action -> NSButton (clears the 2nd shortcut)
        self._error = None
        self._recording = None      # (action, slot) while waiting for keys
        self._monitor = None
        self._controller = None

    # ── public ──
    def show(self) -> None:
        from AppKit import NSApplication
        if self.window is None:
            self._build()
        self.refresh()
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)

    def is_visible(self) -> bool:
        return bool(self.window is not None and self.window.isVisible())

    def refresh(self) -> None:
        if self.window is None:
            return
        for key, box in self._checkboxes.items():
            box.setState_(1 if self.model.value(key) else 0)
        if self._popup is not None:
            self._popup.selectItemWithTitle_(self.model.model_label())
        for (action, slot), btn in self._record_buttons.items():
            if self._recording == (action, slot):
                continue
            labels = self.model.shortcut_labels(action)
            btn.setTitle_(labels[slot] if slot < len(labels) else "Add…")
        for action, btn in self._clear_buttons.items():
            btn.setHidden_(len(self.model.specs(action)) < 2)    # only when a 2nd exists

    # ── build ──
    def _tag(self, payload) -> int:
        tag = len(self._tags) + 1
        self._tags[tag] = payload
        return tag

    def _build(self) -> None:
        from AppKit import (NSWindow, NSMakeRect, NSBackingStoreBuffered, NSTextField,
                            NSButton, NSPopUpButton, NSFont, NSColor, NSBox)
        sections = 2 + len({t.section for t in TOGGLES})   # engine + toggles + shortcuts
        rows = 1 + len(TOGGLES) + len(shortcuts.ACTIONS) + 2   # model, toggles, shortcuts, msg+hint
        height = sections * (ROW_H + 6) + rows * ROW_H + PAD + 12   # 12: bottom margin
        style = 1 | 2 | 4  # titled | closable | miniaturizable
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, WIDTH, height), style, NSBackingStoreBuffered, False)
        win.setTitle_("MacWhisper Settings")
        win.setReleasedWhenClosed_(False)
        win.center()
        ctrl = _get_controller_cls().alloc().init()
        ctrl._mw_owner = self
        win.setDelegate_(ctrl)
        self._controller = ctrl
        content = win.contentView()
        y = height - PAD - ROW_H

        def label(text, x, yy, w, bold=False, size=13, color=None):
            f = NSTextField.labelWithString_(text)
            f.setFrame_(NSMakeRect(x, yy, w, ROW_H - 4))
            f.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
            if color is not None:
                f.setTextColor_(color)
            content.addSubview_(f)
            return f

        def section(title):
            nonlocal y
            y -= 6
            label(title, PAD, y, WIDTH - 2 * PAD, bold=True)
            y -= ROW_H

        # Engine
        section("Speech engine")
        label("Model", PAD + 10, y, 120)
        popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(PAD + 140, y - 2, 300, ROW_H), False)
        for lbl in self.model.model_labels:
            popup.addItemWithTitle_(lbl)
        popup.setTarget_(ctrl)
        popup.setAction_("modelChanged:")
        content.addSubview_(popup)
        self._popup = popup
        y -= ROW_H

        # Toggles grouped by section
        current = None
        for t in TOGGLES:
            if t.section != current:
                current = t.section
                section(current)
            box = NSButton.alloc().initWithFrame_(NSMakeRect(PAD + 10, y, WIDTH - 2 * PAD - 10, ROW_H))
            box.setButtonType_(3)  # NSButtonTypeSwitch
            box.setTitle_(t.label)
            box.setTag_(self._tag(("toggle", t.key)))
            box.setTarget_(ctrl)
            box.setAction_("toggleChanged:")
            content.addSubview_(box)
            self._checkboxes[t.key] = box
            y -= ROW_H

        # Shortcuts
        section("Keyboard shortcuts (click a shortcut, then press the new keys)")
        for action in shortcuts.ACTIONS:
            label(shortcuts.ACTION_LABELS[action], PAD + 10, y, 170)
            x = PAD + 185
            slots = 1 if action == "push_to_talk" else SLOTS
            for slot in range(slots):
                b = NSButton.alloc().initWithFrame_(NSMakeRect(x, y - 2, 110, ROW_H))
                b.setBezelStyle_(1)
                b.setTag_(self._tag(("record", action, slot)))
                b.setTarget_(ctrl)
                b.setAction_("recordClicked:")
                content.addSubview_(b)
                self._record_buttons[(action, slot)] = b
                x += 114
            if action != "push_to_talk":
                c = NSButton.alloc().initWithFrame_(NSMakeRect(x, y - 2, 26, ROW_H))
                c.setBordered_(False)          # rounded bezel is too wide for one glyph
                c.setFont_(NSFont.systemFontOfSize_(13))
                c.setTitle_("✕")
                c.setToolTip_("Remove this second shortcut")
                c.setTag_(self._tag(("clear", action, 1)))
                c.setTarget_(ctrl)
                c.setAction_("clearClicked:")
                content.addSubview_(c)
                self._clear_buttons[action] = c
            r = NSButton.alloc().initWithFrame_(NSMakeRect(WIDTH - PAD - 70, y - 2, 70, ROW_H))
            r.setBezelStyle_(1)
            r.setTitle_("Default")
            r.setTag_(self._tag(("reset", action)))
            r.setTarget_(ctrl)
            r.setAction_("resetClicked:")
            content.addSubview_(r)
            y -= ROW_H

        y -= 4
        self._error = label("", PAD + 10, y, WIDTH - 2 * PAD - 10, size=12,
                            color=NSColor.systemRedColor())
        y -= ROW_H
        label("Changes apply immediately and are saved.", PAD + 10, y, WIDTH - 2 * PAD - 10,
              size=11, color=NSColor.secondaryLabelColor())
        self.window = win

    # ── actions (called by the controller) ──
    def _payload(self, sender):
        return self._tags.get(sender.tag())

    def _show_error(self, text: str) -> None:
        if self._error is not None:
            self._error.setStringValue_(text or "")

    def on_toggle(self, sender) -> None:
        _, key = self._payload(sender)
        self._safe(lambda: self.model.set_toggle(key, sender.state() == 1), f"setting {key}")

    def on_model(self, sender) -> None:
        title = sender.titleOfSelectedItem()
        self._safe(lambda: self.model.set_model(title), "model change")

    def on_reset(self, sender) -> None:
        _, action = self._payload(sender)
        self.stop_recording()
        self._safe(lambda: self.model.reset_shortcut(action), "reset shortcut")
        self._show_error("")

    def on_clear(self, sender) -> None:
        _, action, slot = self._payload(sender)
        self.stop_recording()
        err = self.model.clear_shortcut(action, slot)
        self._show_error(err or "")
        self.refresh()

    def on_record(self, sender) -> None:
        _, action, slot = self._payload(sender)
        if self._recording == (action, slot):
            self.stop_recording()
            return
        self.stop_recording()
        self.start_recording(action, slot)

    def _safe(self, fn, what) -> None:
        try:
            fn()
        except Exception as e:
            logger.exception("Settings: %s failed", what)
            self._show_error(f"Could not apply {what}: {e}")
        self.refresh()

    # ── shortcut recording ──
    def start_recording(self, action: str, slot: int) -> None:
        from AppKit import NSEvent
        self._recording = (action, slot)
        self.app._shortcuts_paused = True
        hint = "Press a key…" if action == "push_to_talk" else "Press keys…"
        self._record_buttons[(action, slot)].setTitle_(hint)
        self._show_error("Esc cancels. Push to Talk: Right ⌥/⌘/⌃/⇧ or F1–F20."
                         if action == "push_to_talk" else "Esc cancels. Use ⌃, ⌥ or ⌘ with a key.")
        mask = (1 << 10) | (1 << 12)   # NSEventMaskKeyDown | NSEventMaskFlagsChanged
        self._monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(mask, self._on_key_event)
        logger.info("Settings: recording shortcut for %s[%d]", action, slot)

    def stop_recording(self) -> None:
        if self._monitor is not None:
            from AppKit import NSEvent
            NSEvent.removeMonitor_(self._monitor)
            self._monitor = None
        if self._recording is not None:
            self._recording = None
            self.app._shortcuts_paused = False
            self.refresh()

    def _on_key_event(self, event):
        """Local NSEvent monitor. Returning None swallows the event."""
        try:
            passthrough = self.handle_key(event.type(), event.keyCode(), int(event.modifierFlags()))
            return event if passthrough else None
        except Exception:
            logger.exception("Settings: shortcut recording failed")
            self.stop_recording()
            return None

    def handle_key(self, event_type: int, keycode: int, flags: int):
        """Recording logic (unit-tested). 10 = keyDown, 12 = flagsChanged.

        Returns True to let the event through, None to swallow it.
        """
        if self._recording is None:
            return True
        action, slot = self._recording
        mods = shortcuts.mods_from_flags(flags)
        if event_type == 12:                       # modifier went down/up
            side = shortcuts.side_modifier_name(keycode)
            own = shortcuts.modifier_name(side) if side else None
            if action == "push_to_talk" and side and own in mods:   # pressed, not released
                self._finish(action, slot, side)
            return None
        if keycode == 53 and not mods:             # Esc
            self._show_error("")
            self.stop_recording()
            return None
        spec = shortcuts.spec_from_keycode(keycode, mods)
        self._finish(action, slot, spec)
        return None

    def _finish(self, action, slot, spec) -> None:
        err = self.model.record_shortcut(action, slot, spec)
        if err:
            self._show_error(f"{shortcuts.display(spec)}: {err}")
            return                                 # keep recording; try another key
        self._show_error(f"Saved: {shortcuts.ACTION_LABELS[action]} = {shortcuts.display(spec)}")
        self.stop_recording()
