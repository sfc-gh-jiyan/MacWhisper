"""Launch MacWhisper at login via SMAppService (macOS 13+).

Uses the system ServiceManagement framework through objc, so the app venv
needs no extra package. SMAppService.mainAppService only works for a real
app bundle (the /Applications/MacWhisper.app built by scripts/build_app.sh);
from ./run.sh the status is "not found" and the option is unavailable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# SMAppServiceStatus
OFF, ON, NEEDS_APPROVAL, UNAVAILABLE = 0, 1, 2, 3

_HINTS = {
    OFF: "",
    ON: "",
    NEEDS_APPROVAL: "Approve MacWhisper in System Settings → General → Login Items",
    UNAVAILABLE: "Available when MacWhisper is started from /Applications (not ./run.sh)",
}


def _main_app_service():
    import objc
    # No official PyObjC binding: declare the NSError** argument as an OUT
    # parameter (type "o^@"), otherwise the call returns a bare bool and the
    # error can never be read. Mirrors how PyObjC's own bindings describe it.
    for sel in (b"registerAndReturnError:", b"unregisterAndReturnError:"):
        objc.registerMetaDataForSelector(
            b"SMAppService", sel,
            {"retval": {"type": b"Z"}, "arguments": {2: {"type_modifier": b"o"}}})
    objc.loadBundle("ServiceManagement", {},
                    bundle_path="/System/Library/Frameworks/ServiceManagement.framework")
    return objc.lookUpClass("SMAppService").mainAppService()


class LoginItem:
    def __init__(self, service):
        self._svc = service

    @classmethod
    def for_this_app(cls) -> "LoginItem":
        try:
            return cls(_main_app_service())
        except Exception:
            logger.warning("ServiceManagement unavailable; login item disabled", exc_info=True)
            return cls(None)

    def state(self) -> int:
        if self._svc is None:
            return UNAVAILABLE
        try:
            st = int(self._svc.status())
        except Exception:
            logger.warning("SMAppService.status failed", exc_info=True)
            return UNAVAILABLE
        return st if st in (OFF, ON, NEEDS_APPROVAL) else UNAVAILABLE

    def available(self) -> bool:
        return self.state() != UNAVAILABLE

    def enabled(self) -> bool:
        return self.state() in (ON, NEEDS_APPROVAL)

    def hint(self) -> str:
        return _HINTS[self.state()]

    def set_enabled(self, on: bool) -> tuple[bool, str]:
        """(ok, message). Errors are returned for display, never swallowed."""
        state = self.state()
        if state == UNAVAILABLE:
            return False, _HINTS[UNAVAILABLE]
        if bool(on) == (state in (ON, NEEDS_APPROVAL)):
            return True, ""
        method = "registerAndReturnError_" if on else "unregisterAndReturnError_"
        try:
            ok, err = getattr(self._svc, method)(None)
        except Exception as e:
            logger.error("Login item %s failed: %s", method, e)
            return False, str(e)
        if not ok:
            msg = err.localizedDescription() if err is not None else "unknown error"
            logger.error("Login item %s failed: %s", method, msg)
            return False, msg
        logger.info("Launch at login %s (status=%s)", "enabled" if on else "disabled", self.state())
        return True, ""
