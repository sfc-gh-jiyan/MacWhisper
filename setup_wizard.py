"""Step-by-step permission setup: one dialog per missing permission.

Flow per step: explain why → [Allow / Open Settings] or [Skip] → re-check the
live status; still off → [Check Again] [Open Settings] [Skip]. At the end, a
restart is offered when Accessibility or Input Monitoring was just granted
(macOS applies those to a running app only after it restarts).

The flow takes a `ui` with ask(title, text, buttons) -> index, so tests can
click through it without AppKit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import permissions

logger = logging.getLogger(__name__)

NEEDS_RESTART = ("accessibility", "input_monitoring")
MIC_PROMPT_WAIT_S = 30.0       # how long the macOS microphone prompt may stay open
POLL_S = 0.5


@dataclass
class Result:
    granted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    restart: bool = False


def run(ui, *, perms=permissions, can_restart: bool, from_menu: bool = False,
        sleep=time.sleep) -> Result:
    result = Result()
    todo = [perm for perm in perms.PERMISSIONS if perms.status(perm.key) != perms.GRANTED]
    logger.info("Permission setup: missing %s (from_menu=%s)", [x.key for x in todo], from_menu)
    if not todo:
        if from_menu:
            ui.ask("All set ✅", "MacWhisper has every permission it needs.", ["OK"])
        return result

    for n, perm in enumerate(todo, 1):
        title = f"Step {n} of {len(todo)}: {perm.name}"
        if _step(ui, perms, perm, title, sleep):
            result.granted.append(perm.key)
        else:
            result.skipped.append(perm.key)

    if any(key in NEEDS_RESTART for key in result.granted):
        if can_restart:
            choice = ui.ask("Restart MacWhisper to apply",
                            "macOS applies Accessibility and Input Monitoring after "
                            "MacWhisper restarts. It takes a few seconds.",
                            ["Restart Now", "Later"])
            result.restart = choice == 0
        else:
            ui.ask("Restart needed",
                   "Quit ./run.sh and start it again to apply the new permissions.", ["OK"])
    logger.info("Permission setup done: granted=%s skipped=%s restart=%s",
                result.granted, result.skipped, result.restart)
    return result


def _step(ui, perms, perm, title, sleep) -> bool:
    """One permission. True if it ends up granted."""
    first = perms.status(perm.key)
    ask_prompt = perm.key == "microphone" and first == perms.NOT_DETERMINED
    primary = "Allow" if ask_prompt else "Open Settings"
    if ui.ask(title, perm.why, [primary, "Skip"]) == 1:
        logger.info("Permission %s skipped", perm.key)
        return False

    if ask_prompt:
        perms.request(perm.key)                       # macOS asks; no Settings needed
        _wait_while_undecided(perms, perm.key, sleep)
    else:
        perms.request(perm.key)                       # adds MacWhisper to the list
        perms.open_settings(perm.key)
        sleep(1.0)

    while True:
        if perms.status(perm.key) == perms.GRANTED:
            logger.info("Permission %s granted", perm.key)
            return True
        choice = ui.ask(title,
                        f"{perm.name} is still off. In System Settings → Privacy & Security → "
                        f"{perm.name}, turn on MacWhisper, then click Check Again.",
                        ["Check Again", "Open Settings", "Skip"])
        if choice == 1:
            perms.open_settings(perm.key)
        elif choice == 2:
            logger.info("Permission %s skipped after trying", perm.key)
            return False


def _wait_while_undecided(perms, key, sleep) -> None:
    """The microphone prompt answers asynchronously; wait until it is answered."""
    waited = 0.0
    while perms.status(key) == perms.NOT_DETERMINED and waited < MIC_PROMPT_WAIT_S:
        sleep(POLL_S)
        waited += POLL_S


class AlertUI:
    """Native dialogs (NSAlert). Must run on the main thread."""

    def ask(self, title: str, text: str, buttons: list[str]) -> int:
        from AppKit import NSAlert, NSApp
        alert = NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(text)
        for label in buttons:
            alert.addButtonWithTitle_(label)
        NSApp.activateIgnoringOtherApps_(True)       # menu-bar app: bring the dialog forward
        index = int(alert.runModal()) - 1000         # NSAlertFirstButtonReturn
        logger.debug("Dialog %r -> %s", title, buttons[index] if 0 <= index < len(buttons) else index)
        return index
