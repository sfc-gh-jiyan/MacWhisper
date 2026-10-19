"""Persistent red error alert panel for MacWhisper.

Shown when something fails (crashed thread, dead hotkey listener, previous
run crashed). Stays on screen until the user clicks "Dismiss" — never
auto-hides. Non-activating: it does not steal focus from the user's app.

Must be called on the main thread (use AppHelper.callAfter from others).
"""

from __future__ import annotations

import logging

import overlay

logger = logging.getLogger(__name__)

ALERT_WIDTH = 520
ALERT_HEIGHT = 170
_GAP = 12  # space above the Push to Talk overlay

_state = {"panel": None, "text": None, "count": 0, "target": None}
_target_cls = None


def alert_frame(talk_frame, screen_rect) -> tuple:
    """Place the alert just above where the Push to Talk overlay lives.

    Falls back to the screen's top area if that would go off-screen.
    """
    tx, ty, tw, th = talk_frame
    sx, sy, sw, sh = screen_rect
    w = min(ALERT_WIDTH, sw - 40)
    x = tx + (tw - w) / 2
    y = ty + th + _GAP
    if y + ALERT_HEIGHT > sy + sh:
        y = sy + sh - ALERT_HEIGHT - 60
    x = max(sx + 10, min(x, sx + sw - w - 10))
    return (x, y, w, ALERT_HEIGHT)


def _get_target_cls():
    global _target_cls
    if _target_cls is None:
        from Foundation import NSObject

        class MWAlertDismissTarget(NSObject):
            def dismiss_(self, _sender):
                dismiss_error_alert()

        _target_cls = MWAlertDismissTarget
    return _target_cls


def show_error_alert(title: str, message: str, frame_path: str | None = None) -> None:
    """Show (or update) the persistent error alert. Main thread only."""
    from AppKit import (
        NSPanel, NSColor, NSFont, NSScreen, NSMakeRect, NSBackingStoreBuffered,
        NSTextField, NSButton,
    )

    _state["count"] += 1
    count = _state["count"]
    body = message if count == 1 else f"{message}\n({count} errors since this alert appeared)"
    logger.warning("show_error_alert #%d: %s | %s", count, title, message.replace("\n", " | "))

    if _state["panel"] is not None:
        _state["text"].setStringValue_(f"{title}\n\n{body}")
        _state["panel"].orderFrontRegardless()
        return

    screen = NSScreen.mainScreen()
    if not screen:
        logger.error("show_error_alert: no screen")
        return
    screens = [overlay._rect_tuple(s.frame()) for s in NSScreen.screens()]
    talk = overlay.resolve_frame(
        overlay.load_frame(frame_path, "talk"), screens, overlay._rect_tuple(screen.frame())
    )
    # Use the screen containing the talk overlay
    host = next((s for s in screens
                 if s[0] <= talk[0] < s[0] + s[2] and s[1] <= talk[1] < s[1] + s[3]),
                overlay._rect_tuple(screen.frame()))
    x, y, w, h = alert_frame(talk, host)

    style = 1 << 0 | 1 << 7 | 1 << 15  # titled | nonactivating | full-size content
    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(x, y, w, h), style, NSBackingStoreBuffered, False
    )
    panel.setFrame_display_(NSMakeRect(x, y, w, h), False)
    panel.setTitlebarAppearsTransparent_(True)
    panel.setTitleVisibility_(1)  # NSWindowTitleHidden
    for btn in (0, 1, 2):  # close / miniaturize / zoom
        b = panel.standardWindowButton_(btn)
        if b:
            b.setHidden_(True)
    panel.setOpaque_(False)
    panel.setBackgroundColor_(
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.55, 0.05, 0.05, 0.96)
    )
    panel.setMovableByWindowBackground_(True)
    panel.setHidesOnDeactivate_(False)
    panel.setBecomesKeyOnlyIfNeeded_(True)
    panel.setFloatingPanel_(True)
    # setFloatingPanel_ resets the level to NSFloatingWindowLevel (3) — set level after it
    panel.setLevel_(1001)  # above the subtitle overlay
    panel.setRestorable_(False)
    panel.setCollectionBehavior_(1 << 0 | 1 << 4)

    content = panel.contentView()
    cf = content.frame()

    text = NSTextField.alloc().initWithFrame_(
        NSMakeRect(16, 44, cf.size.width - 32, cf.size.height - 54)
    )
    text.setEditable_(False)
    text.setSelectable_(False)
    text.setBezeled_(False)
    text.setDrawsBackground_(False)
    text.setTextColor_(NSColor.whiteColor())
    text.setFont_(NSFont.systemFontOfSize_(13))
    text.setStringValue_(f"{title}\n\n{body}")
    text.setAutoresizingMask_(0x12)
    content.addSubview_(text)

    target = _get_target_cls().alloc().init()
    button = NSButton.alloc().initWithFrame_(
        NSMakeRect(cf.size.width - 112, 10, 96, 28)
    )
    button.setTitle_("Dismiss")
    button.setBezelStyle_(1)  # rounded
    button.setTarget_(target)
    button.setAction_("dismiss:")
    button.setAutoresizingMask_(0x1)  # flexible left margin
    content.addSubview_(button)

    panel.orderFrontRegardless()
    _state.update(panel=panel, text=text, target=target)


def dismiss_error_alert() -> None:
    """Close the alert (user clicked Dismiss). Main thread only."""
    panel = _state["panel"]
    _state.update(panel=None, text=None, target=None, count=0)
    if panel is not None:
        logger.info("error alert dismissed by user")
        panel.orderOut_(None)
        panel.close()


def is_showing() -> bool:
    return _state["panel"] is not None
