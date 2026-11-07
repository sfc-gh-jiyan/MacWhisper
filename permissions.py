"""The three permissions Push to Talk needs, checked without side effects.

Microphone status comes from AVFoundation (never by opening a stream: that is
the call that hung on 2026-10-29 and 2026-11-06). A check that fails reports
UNKNOWN, never "granted". The request_* calls trigger the macOS prompt, which
also adds MacWhisper to the list in System Settings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

GRANTED = "granted"
NOT_DETERMINED = "not_determined"   # microphone only: macOS has not asked yet
DENIED = "denied"
UNKNOWN = "unknown"

_PRIVACY = "x-apple.systempreferences:com.apple.preference.security?Privacy_"
_AUDIO = "soun"                      # AVMediaTypeAudio


@dataclass(frozen=True)
class Permission:
    key: str
    name: str
    why: str
    pane: str


PERMISSIONS = (
    Permission("microphone", "Microphone",
               "MacWhisper needs the microphone to record your voice.",
               _PRIVACY + "Microphone"),
    Permission("accessibility", "Accessibility",
               "Needed to paste the text into the app you are typing in.",
               _PRIVACY + "Accessibility"),
    Permission("input_monitoring", "Input Monitoring",
               "Needed to detect the Right Option key and your other shortcuts.",
               _PRIVACY + "ListenEvent"),
)
_BY_KEY = {perm.key: perm for perm in PERMISSIONS}


def get(key: str) -> Permission:
    return _BY_KEY[key]


# ── macOS calls (replaced by fakes in tests) ──────────────

_device = None


def _capture_device():
    """AVCaptureDevice, loaded from the system framework (no extra package)."""
    global _device
    if _device is None:
        import objc
        objc.loadBundle("AVFoundation", {},
                        bundle_path="/System/Library/Frameworks/AVFoundation.framework")
        _device = objc.lookUpClass("AVCaptureDevice")
    return _device


def _ax_trusted(prompt: bool = False) -> bool:
    from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
    return bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: prompt}))


def _im_preflight() -> bool:
    import Quartz
    return bool(Quartz.CGPreflightListenEventAccess())


def _im_request() -> None:
    import Quartz
    Quartz.CGRequestListenEventAccess()


def _open_url(url: str) -> bool:
    from AppKit import NSWorkspace
    from Foundation import NSURL
    return bool(NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(url)))


# ── Public API ────────────────────────────────────────────

def status(key: str) -> str:
    try:
        if key == "microphone":
            raw = _capture_device().authorizationStatusForMediaType_(_AUDIO)
            result = {0: NOT_DETERMINED, 3: GRANTED}.get(int(raw), DENIED)
        elif key == "accessibility":
            result = GRANTED if _ax_trusted() else DENIED
        elif key == "input_monitoring":
            result = GRANTED if _im_preflight() else DENIED
        else:
            raise KeyError(key)
    except Exception as e:
        logger.warning("Permission check for %s failed: %s", key, e)
        result = UNKNOWN
    logger.debug("Permission %s: %s", key, result)
    return result


def missing() -> list[Permission]:
    """Required permissions not granted (UNKNOWN counts as missing)."""
    return [perm for perm in PERMISSIONS if status(perm.key) != GRANTED]


def request(key: str) -> None:
    """Show the macOS prompt; it also adds MacWhisper to the System Settings list."""
    logger.info("Requesting permission: %s", key)
    try:
        if key == "microphone":
            _capture_device().requestAccessForMediaType_completionHandler_(
                _AUDIO, lambda granted: logger.info("Microphone prompt answered: %s", granted))
        elif key == "accessibility":
            _ax_trusted(prompt=True)
        elif key == "input_monitoring":
            _im_request()
    except Exception as e:
        logger.warning("Requesting %s failed: %s", key, e)


def open_settings(key: str) -> bool:
    url = get(key).pane
    logger.info("Opening System Settings: %s", url)
    try:
        return _open_url(url)
    except Exception as e:
        logger.warning("Opening System Settings for %s failed: %s", key, e)
        return False
