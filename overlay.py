"""Subtitle overlay with dual-color rendering for MacWhisper.

Confirmed words are displayed in white, unconfirmed words in gray.
Uses NSTextView with NSAttributedString for rich text rendering.

Window placement rule: the overlay frame is ONLY changed by the user
(drag to move, drag edges to resize). Content updates never move or
resize the panel. The last user-chosen frame is persisted per mode and
restored on the next show, falling back to the main screen if the saved
position is no longer on any connected display.
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 160
MIN_WIDTH = 240
MIN_HEIGHT = 60
# A saved frame must overlap a screen by at least this much to be reused
_MIN_VISIBLE_W = 80
_MIN_VISIBLE_H = 40

# NSWindowStyleMask bits
_STYLE_TITLED = 1 << 0
_STYLE_RESIZABLE = 1 << 3
_STYLE_NONACTIVATING_PANEL = 1 << 7
_STYLE_FULL_SIZE_CONTENT = 1 << 15

# Keeps window delegates alive (PyObjC delegates are weak references)
_delegates: dict = {}
_drag_view_cls = None
_delegate_cls = None
_stop_target_cls = None

STOP_BUTTON_W = 68
STOP_BUTTON_H = 22
STOP_BUTTON_MARGIN = 8


# ── Pure geometry / persistence helpers (unit-tested) ─────────

def default_frame(screen_rect) -> tuple:
    """Default frame: horizontally centered, 40px above the screen bottom."""
    sx, sy, sw, _sh = screen_rect
    w = min(sw * 0.8, DEFAULT_WIDTH)
    h = DEFAULT_HEIGHT
    return (sx + (sw - w) / 2, sy + 40, w, h)


def _overlap(a, b) -> tuple:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ow = min(ax + aw, bx + bw) - max(ax, bx)
    oh = min(ay + ah, by + bh) - max(ay, by)
    return max(ow, 0), max(oh, 0)


def resolve_frame(saved, screens, fallback_screen) -> tuple:
    """Return the saved frame if it is valid and visible, else a default.

    Args:
        saved: (x, y, w, h) or None.
        screens: list of (x, y, w, h) for all connected screens.
        fallback_screen: (x, y, w, h) used for the default frame.
    """
    try:
        x, y, w, h = (float(v) for v in saved)
    except (TypeError, ValueError):
        logger.debug("resolve_frame: no usable saved frame (%r)", saved)
        return default_frame(fallback_screen)

    w = max(w, MIN_WIDTH)
    h = max(h, MIN_HEIGHT)
    frame = (x, y, w, h)
    for screen in screens:
        ow, oh = _overlap(frame, screen)
        if ow >= _MIN_VISIBLE_W and oh >= _MIN_VISIBLE_H:
            return frame
    logger.info("resolve_frame: saved frame %r is off-screen, using default", frame)
    return default_frame(fallback_screen)


def load_frame(path, mode: str):
    """Load the saved frame for a mode, or None."""
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        frame = data.get(mode)
        if isinstance(frame, list) and len(frame) == 4:
            return tuple(float(v) for v in frame)
    except (OSError, ValueError, TypeError, AttributeError):
        logger.debug("load_frame: cannot read %s", path, exc_info=True)
    return None


def save_frame(path, mode: str, frame) -> None:
    """Persist a frame for a mode (other modes are preserved)."""
    if not path:
        return
    data = {}
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            data = loaded
    except (OSError, ValueError):
        pass
    data = {**data, mode: [round(float(v), 1) for v in frame]}
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.debug("save_frame: %s -> %s", mode, data[mode])
    except OSError:
        logger.warning("save_frame: cannot write %s", path, exc_info=True)


def _rect_tuple(r) -> tuple:
    return (r.origin.x, r.origin.y, r.size.width, r.size.height)


def _persist_from_notification(delegate, notification) -> None:
    info = getattr(delegate, "_mw_info", None)
    if not info:
        return
    frame = _rect_tuple(notification.object().frame())
    save_frame(info["path"], info["mode"], frame)


def _frontmost_app():
    """Frontmost app if it is not MacWhisper itself, else None."""
    try:
        from AppKit import NSWorkspace, NSRunningApplication
        front = NSWorkspace.sharedWorkspace().frontmostApplication()
        me = NSRunningApplication.currentApplication()
        if front is None or front.processIdentifier() == me.processIdentifier():
            return None
        return front
    except Exception:
        return None


def _make_alert():
    from AppKit import NSAlert
    return NSAlert.alloc().init()


def _confirm_stop() -> bool:
    """Ask before ending the meeting (a stray click must not stop it).

    Activating MacWhisper is needed to show the alert in front; afterwards
    focus goes back to the user's app so the next Push to Talk pastes there.
    """
    from AppKit import NSApplication
    prev = _frontmost_app()
    NSApplication.sharedApplication().activateIgnoringOtherApps_(True)  # bring alert to front
    alert = _make_alert()
    alert.setMessageText_("End the meeting?")
    alert.setInformativeText_("The transcript is saved and its file path copied to the clipboard.")
    alert.addButtonWithTitle_("End Meeting")
    alert.addButtonWithTitle_("Cancel")
    confirmed = alert.runModal() == 1000  # NSAlertFirstButtonReturn
    if prev is not None:
        try:
            prev.activateWithOptions_(0)
        except Exception:
            logger.debug("restore focus failed", exc_info=True)
    return confirmed


def _handle_stop_click(target) -> None:
    callback = getattr(target, "_mw_on_stop", None)
    if callback is None:
        return
    logger.info("Overlay Stop clicked")
    if not _confirm_stop():
        logger.info("Overlay Stop cancelled by user")
        return
    try:
        callback()
    except Exception:
        # Never raise into AppKit's event loop
        logger.exception("Overlay Stop callback failed")


# ── AppKit classes (created lazily, defined once per process) ─

def _get_stop_target_cls():
    global _stop_target_cls
    if _stop_target_cls is None:
        from Foundation import NSObject

        class MWOverlayStopTarget(NSObject):
            """Target for the meeting overlay's Stop button."""

            def stopClicked_(self, _sender):
                _handle_stop_click(self)

        _stop_target_cls = MWOverlayStopTarget
    return _stop_target_cls


def _add_stop_button(panel, content, on_stop) -> None:
    from AppKit import NSButton, NSMakeRect, NSFont, NSColor
    cf = content.frame()
    x = cf.size.width - STOP_BUTTON_W - STOP_BUTTON_MARGIN
    y = cf.size.height - STOP_BUTTON_H - STOP_BUTTON_MARGIN
    btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, STOP_BUTTON_W, STOP_BUTTON_H))
    btn.setTitle_("■ Stop")
    btn.setBezelStyle_(1)               # NSBezelStyleRounded
    btn.setFont_(NSFont.systemFontOfSize_(12))
    btn.setBezelColor_(NSColor.systemRedColor())
    btn.setToolTip_("End the meeting (asks to confirm)")
    btn.setAutoresizingMask_(0x1 | 0x8)  # NSViewMinXMargin | NSViewMinYMargin: stick top-right
    target = _get_stop_target_cls().alloc().init()
    target._mw_on_stop = on_stop
    btn.setTarget_(target)
    btn.setAction_("stopClicked:")
    content.addSubview_(btn)            # added last: above the drag view
    _delegates[("stop", id(panel))] = target  # keep the target alive (weak ref in AppKit)


def _get_classes():
    global _drag_view_cls, _delegate_cls
    if _drag_view_cls is not None:
        return _drag_view_cls, _delegate_cls

    from AppKit import NSView
    from Foundation import NSObject

    class MWOverlayDragView(NSView):
        """Transparent top layer: any click-drag moves the window."""

        def mouseDownCanMoveWindow(self):
            return True

        def acceptsFirstMouse_(self, _event):
            return True

        def mouseDown_(self, event):
            self.window().performWindowDragWithEvent_(event)

    class MWOverlayDelegate(NSObject):
        """Persists the frame whenever the user moves or resizes it."""

        def windowDidMove_(self, notification):
            _persist_from_notification(self, notification)

        def windowDidEndLiveResize_(self, notification):
            _persist_from_notification(self, notification)

    _drag_view_cls, _delegate_cls = MWOverlayDragView, MWOverlayDelegate
    return _drag_view_cls, _delegate_cls


# ── Public API ────────────────────────────────────────────────

def create_overlay(mode: str = "talk", frame_path: str | None = None, on_stop=None):
    """Create a movable, resizable NSPanel for dual-color subtitles.

    Args:
        mode: "talk" for Push to Talk, "meeting" for Meeting Recording.
        frame_path: JSON file used to remember the frame per mode.
        on_stop: meeting mode only — adds a top-right Stop button that calls
            this (on the main thread) after the user confirms.

    Returns:
        (panel, text_view) tuple, or (None, None) if no screen available.
    """
    from AppKit import (
        NSPanel, NSColor, NSFont, NSScreen, NSMakeRect, NSMakeSize,
        NSBackingStoreBuffered, NSTextView, NSScrollView, NSView,
        NSWindowTitleHidden, NSWindowCloseButton,
        NSWindowMiniaturizeButton, NSWindowZoomButton,
    )

    screen = NSScreen.mainScreen()
    if not screen:
        print("[ERROR] No main screen found, cannot create overlay")
        return None, None

    screens = [_rect_tuple(s.frame()) for s in NSScreen.screens()]
    saved = load_frame(frame_path, mode)
    x, y, w, h = resolve_frame(saved, screens, _rect_tuple(screen.frame()))
    logger.info("create_overlay: mode=%s saved=%r frame=%r", mode, saved, (x, y, w, h))

    style = (_STYLE_TITLED | _STYLE_RESIZABLE
             | _STYLE_NONACTIVATING_PANEL | _STYLE_FULL_SIZE_CONTENT)
    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(x, y, w, h), style, NSBackingStoreBuffered, False
    )
    # Content rect != frame rect for titled windows; pin the exact frame
    panel.setFrame_display_(NSMakeRect(x, y, w, h), False)

    panel.setTitlebarAppearsTransparent_(True)
    panel.setTitleVisibility_(NSWindowTitleHidden)
    for btn in (NSWindowCloseButton, NSWindowMiniaturizeButton, NSWindowZoomButton):
        b = panel.standardWindowButton_(btn)
        if b:
            b.setHidden_(True)

    panel.setOpaque_(False)
    panel.setBackgroundColor_(
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0, 0, 0, 0.88)
    )
    panel.setIgnoresMouseEvents_(False)
    panel.setMovableByWindowBackground_(True)
    panel.setMinSize_(NSMakeSize(MIN_WIDTH, MIN_HEIGHT))
    panel.setHasShadow_(False)
    panel.setHidesOnDeactivate_(False)
    # Never steal keyboard focus (auto-paste targets the user's app)
    panel.setBecomesKeyOnlyIfNeeded_(True)
    panel.setFloatingPanel_(True)
    # setFloatingPanel_ resets the level to NSFloatingWindowLevel (3) — set level after it
    panel.setLevel_(1000)
    panel.setRestorable_(False)
    panel.setCollectionBehavior_(1 << 0 | 1 << 4)

    drag_cls, delegate_cls = _get_classes()
    delegate = delegate_cls.alloc().init()
    delegate._mw_info = {"path": frame_path, "mode": mode}
    panel.setDelegate_(delegate)
    _delegates[id(panel)] = delegate

    content = panel.contentView()
    cf = content.frame()
    flex_wh = 0x12  # NSViewWidthSizable | NSViewHeightSizable

    # Left accent stripe — orange-red for talk, red for meeting
    stripe_w = 3
    if mode == "meeting":
        stripe_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.9, 0.2, 0.2, 1.0)
    else:
        stripe_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.95, 0.5, 0.15, 1.0)
    stripe = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, stripe_w, cf.size.height))
    stripe.setWantsLayer_(True)
    stripe.layer().setBackgroundColor_(stripe_color.CGColor())
    stripe.setAutoresizingMask_(0x10)  # flexible height
    content.addSubview_(stripe)

    scroll_view = NSScrollView.alloc().initWithFrame_(
        NSMakeRect(stripe_w, 0, cf.size.width - stripe_w, cf.size.height)
    )
    scroll_view.setHasVerticalScroller_(False)
    scroll_view.setHasHorizontalScroller_(False)
    scroll_view.setDrawsBackground_(False)
    scroll_view.setBorderType_(0)  # NSNoBorder
    scroll_view.setAutoresizingMask_(flex_wh)

    text_view = NSTextView.alloc().initWithFrame_(scroll_view.contentView().bounds())
    text_view.setEditable_(False)
    text_view.setSelectable_(False)
    text_view.setDrawsBackground_(False)
    text_view.setRichText_(True)
    text_view.setFont_(NSFont.systemFontOfSize_(15))
    text_view.setBackgroundColor_(NSColor.clearColor())
    text_view.setTextContainerInset_(NSMakeSize(16, 10))
    text_view.setMinSize_(NSMakeSize(0, 0))
    text_view.setMaxSize_(NSMakeSize(1e7, 1e7))
    text_view.setVerticallyResizable_(True)
    text_view.setHorizontallyResizable_(False)
    text_view.setAutoresizingMask_(0x2)  # flexible width
    text_view.textContainer().setWidthTracksTextView_(True)
    scroll_view.setDocumentView_(text_view)
    content.addSubview_(scroll_view)

    # Topmost transparent layer captures clicks for window dragging
    drag_view = drag_cls.alloc().initWithFrame_(cf)
    drag_view.setAutoresizingMask_(flex_wh)
    content.addSubview_(drag_view)

    if mode == "meeting" and on_stop is not None:
        _add_stop_button(panel, content, on_stop)

    panel.orderFrontRegardless()

    # Render initial label immediately (don't wait for first transcription)
    update_overlay(panel, text_view, "", "", mode=mode)

    return panel, text_view


def update_overlay(panel, text_view, confirmed: str, unconfirmed: str, mode: str = "talk"):
    """Update overlay text. Never moves or resizes the panel.

    Args:
        panel: NSPanel instance.
        text_view: NSTextView instance.
        confirmed: Already confirmed text (displayed in white).
        unconfirmed: Not yet confirmed tail (displayed in gray).
        mode: "talk" or "meeting" — controls the label line.
    """
    from AppKit import (
        NSColor, NSFont,
        NSMutableAttributedString, NSAttributedString,
        NSForegroundColorAttributeName, NSFontAttributeName,
    )
    from Foundation import NSMakeRange

    if not text_view or not panel:
        return

    font = NSFont.systemFontOfSize_(15)
    white = NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.92)
    gray = NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.45)

    # Format text with sentence-break newlines
    def _format(text):
        if not text:
            return ""
        return re.sub(r'([。！？.!?])\s*', r'\1\n', text).rstrip('\n')

    confirmed_fmt = _format(confirmed)
    unconfirmed_fmt = _format(unconfirmed)

    attr_str = NSMutableAttributedString.alloc().init()

    def _append(text, color, fnt):
        attr_str.appendAttributedString_(
            NSAttributedString.alloc().initWithString_attributes_(
                text, {NSForegroundColorAttributeName: color, NSFontAttributeName: fnt}
            )
        )

    # Mode label line — orange-red for talk, red for meeting
    label_font = NSFont.systemFontOfSize_weight_(11, 0.4)  # semibold small
    if mode == "meeting":
        label_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.9, 0.25, 0.25, 0.9)
        label_text = "● Meeting Recording — Ctrl+Shift+R to stop\n"
    else:
        label_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.95, 0.5, 0.15, 0.9)
        label_text = "● Push to Talk\n"
    _append(label_text, label_color, label_font)

    if confirmed_fmt:
        _append(confirmed_fmt, white, font)
    if confirmed_fmt and unconfirmed_fmt:
        _append(" ", gray, font)
    if unconfirmed_fmt:
        _append(unconfirmed_fmt, gray, font)

    # Debug: log what the user sees on the overlay
    _conf_preview = confirmed[-60:] if confirmed else ""
    _unconf_preview = unconfirmed[-40:] if unconfirmed else ""
    print(f"[OVERLAY] conf=\"{_conf_preview}\" unconf=\"{_unconf_preview}\"")

    text_view.textStorage().setAttributedString_(attr_str)

    # Auto-scroll to bottom so newest text is always visible
    text_view.scrollRangeToVisible_(NSMakeRange(len(text_view.string()), 0))


def destroy_overlay(panel):
    """Persist the final frame, then hide and release the overlay panel."""
    if not panel:
        return
    if _delegates.pop(("stop", id(panel)), None) is not None:
        from AppKit import NSButton
        for v in panel.contentView().subviews():
            if isinstance(v, NSButton):
                v.setTarget_(None)      # no dangling target after the panel closes
    delegate = _delegates.pop(id(panel), None)
    info = getattr(delegate, "_mw_info", None) if delegate else None
    if info:
        save_frame(info["path"], info["mode"], _rect_tuple(panel.frame()))
    panel.setDelegate_(None)
    panel.orderOut_(None)
    panel.close()
