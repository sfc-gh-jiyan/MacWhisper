"""Subtitle overlay with dual-color rendering for MacWhisper.

Confirmed words are displayed in white, unconfirmed words in gray.
Uses NSTextView with NSAttributedString for rich text rendering.
"""

from __future__ import annotations

import re


def create_overlay():
    """Create an NSPanel with NSTextView for dual-color subtitle display.

    Returns:
        (panel, text_view) tuple, or (None, None) if no screen available.
    """
    from AppKit import (
        NSPanel, NSColor, NSFont, NSScreen, NSMakeRect,
        NSBackingStoreBuffered, NSTextView, NSScrollView,
    )

    NSWindowStyleMaskBorderless = 0

    screen = NSScreen.mainScreen()
    if not screen:
        print("[ERROR] No main screen found, cannot create overlay")
        return None, None

    screen_frame = screen.frame()
    panel_w = min(screen_frame.size.width * 0.8, 960)
    panel_h = 60
    # Center on the current screen (origin.x accounts for multi-monitor setups)
    panel_x = screen_frame.origin.x + (screen_frame.size.width - panel_w) / 2
    panel_y = screen_frame.origin.y + 40

    rect = NSMakeRect(panel_x, panel_y, panel_w, panel_h)
    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        rect, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
    )
    panel.setLevel_(1000)
    panel.setOpaque_(False)
    panel.setBackgroundColor_(
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0, 0, 0, 0.88)
    )
    panel.setIgnoresMouseEvents_(True)
    panel.setHasShadow_(False)
    panel.setHidesOnDeactivate_(False)
    panel.setCollectionBehavior_(1 << 0 | 1 << 4)

    content = panel.contentView()
    content_frame = content.frame()

    # Use NSTextView (instead of NSTextField) for attributed string support
    scroll_view = NSScrollView.alloc().initWithFrame_(
        NSMakeRect(0, 0, content_frame.size.width, content_frame.size.height)
    )
    scroll_view.setHasVerticalScroller_(False)
    scroll_view.setHasHorizontalScroller_(False)
    scroll_view.setDrawsBackground_(False)
    scroll_view.setBorderType_(0)  # NSNoBorder
    scroll_view.setAutoresizingMask_(0x12)  # width + height flexible

    text_view = NSTextView.alloc().initWithFrame_(
        NSMakeRect(20, 0, content_frame.size.width - 40, content_frame.size.height)
    )
    text_view.setEditable_(False)
    text_view.setSelectable_(False)
    text_view.setDrawsBackground_(False)
    text_view.setRichText_(True)
    text_view.setFont_(NSFont.systemFontOfSize_(15))
    # Make text view transparent background
    text_view.setBackgroundColor_(NSColor.clearColor())
    # Allow text view to grow vertically as content increases
    text_view.setVerticallyResizable_(True)
    text_view.setHorizontallyResizable_(False)
    text_view.textContainer().setWidthTracksTextView_(True)

    scroll_view.setDocumentView_(text_view)
    content.addSubview_(scroll_view)
    panel.orderFrontRegardless()

    return panel, text_view


def update_overlay(panel, text_view, confirmed: str, unconfirmed: str):
    """Update overlay with dual-color text.

    Args:
        panel: NSPanel instance.
        text_view: NSTextView instance.
        confirmed: Already confirmed text (displayed in white).
        unconfirmed: Not yet confirmed tail (displayed in gray).
    """
    from AppKit import (
        NSColor, NSFont, NSScreen, NSMakeRect,
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

    # Build attributed string
    attr_str = NSMutableAttributedString.alloc().init()

    if confirmed_fmt:
        confirmed_attrs = {
            NSForegroundColorAttributeName: white,
            NSFontAttributeName: font,
        }
        confirmed_as = NSAttributedString.alloc().initWithString_attributes_(
            confirmed_fmt, confirmed_attrs
        )
        attr_str.appendAttributedString_(confirmed_as)

    if confirmed_fmt and unconfirmed_fmt:
        # Separator between confirmed and unconfirmed
        sep_attrs = {NSForegroundColorAttributeName: gray, NSFontAttributeName: font}
        sep_as = NSAttributedString.alloc().initWithString_attributes_(" ", sep_attrs)
        attr_str.appendAttributedString_(sep_as)

    if unconfirmed_fmt:
        unconfirmed_attrs = {
            NSForegroundColorAttributeName: gray,
            NSFontAttributeName: font,
        }
        unconfirmed_as = NSAttributedString.alloc().initWithString_attributes_(
            unconfirmed_fmt, unconfirmed_attrs
        )
        attr_str.appendAttributedString_(unconfirmed_as)

    # Debug: log what the user sees on the overlay
    _conf_preview = confirmed[-60:] if confirmed else ""
    _unconf_preview = unconfirmed[-40:] if unconfirmed else ""
    print(f"[OVERLAY] conf=\"{_conf_preview}\" unconf=\"{_unconf_preview}\"")

    # Apply to text view
    text_view.textStorage().setAttributedString_(attr_str)

    # Resize panel height to fit content (only when height actually changes)
    screen = NSScreen.mainScreen()
    if not screen:
        return
    screen_frame = screen.frame()
    cur_frame = panel.frame()
    panel_w = cur_frame.size.width

    text_w = panel_w - 40
    # Use layoutManager to compute needed height
    layout_manager = text_view.layoutManager()
    text_container = text_view.textContainer()
    text_container.setContainerSize_(NSMakeRect(0, 0, text_w, 10000).size)
    layout_manager.ensureLayoutForTextContainer_(text_container)
    used_rect = layout_manager.usedRectForTextContainer_(text_container)
    needed_h = used_rect.size.height + 28
    min_h = 60  # minimum height to prevent collapse/flicker
    max_h = screen_frame.size.height * 0.4
    panel_h = max(min_h, min(needed_h, max_h))

    # Only reposition if height changed by more than 2px (avoids micro-jumps)
    if abs(panel_h - cur_frame.size.height) > 2:
        panel_x = screen_frame.origin.x + (screen_frame.size.width - panel_w) / 2
        panel_y = screen_frame.origin.y + 40
        panel.setFrame_display_(NSMakeRect(panel_x, panel_y, panel_w, panel_h), True)

        content_frame = panel.contentView().frame()
        # Resize scroll view to fill panel content area
        scroll_view = text_view.enclosingScrollView()
        if scroll_view:
            scroll_view.setFrame_(
                NSMakeRect(0, 0, content_frame.size.width, content_frame.size.height)
            )
        text_view.setFrame_(
            NSMakeRect(20, 0, content_frame.size.width - 40, content_frame.size.height)
        )

    # Auto-scroll to bottom so newest text is always visible
    text_view.scrollRangeToVisible_(NSMakeRange(len(text_view.string()), 0))


def destroy_overlay(panel):
    """Hide and release the overlay panel."""
    if panel:
        panel.orderOut_(None)
        panel.close()
