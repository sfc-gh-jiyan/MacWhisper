"""Tests for setup_wizard.run — a fake UI clicks the buttons."""

import pytest

import permissions as p
import setup_wizard as w


class FakePerms:
    """Status per key; granting can happen after N checks (user flips the switch)."""

    PERMISSIONS = p.PERMISSIONS
    GRANTED = p.GRANTED
    NOT_DETERMINED = p.NOT_DETERMINED

    def __init__(self, statuses, grant_after=None):
        self.statuses = dict(statuses)
        self.grant_after = dict(grant_after or {})
        self.requested, self.opened = [], []

    def status(self, key):
        left = self.grant_after.get(key)
        acted = key in self.requested or key in self.opened
        if left is not None and acted:          # only after the user did something
            if left <= 0:
                self.statuses[key] = p.GRANTED
            self.grant_after[key] = left - 1
        return self.statuses[key]

    def request(self, key):
        self.requested.append(key)

    def open_settings(self, key):
        self.opened.append(key)
        return True


class FakeUI:
    def __init__(self, clicks):
        self.clicks = list(clicks)       # button labels, in order
        self.shown = []

    def ask(self, title, text, buttons):
        self.shown.append((title, text, buttons))
        label = self.clicks.pop(0)
        assert label in buttons, f"{label!r} not in {buttons} for {title!r}"
        return buttons.index(label)


ALL_MISSING = {"microphone": p.NOT_DETERMINED, "accessibility": p.DENIED,
               "input_monitoring": p.DENIED}


def _run(perms, clicks, **kw):
    ui = FakeUI(clicks)
    result = w.run(ui, perms=perms, sleep=lambda s: None, **kw)
    return result, ui


def test_only_missing_permissions_get_a_step():
    perms = FakePerms({"microphone": p.GRANTED, "accessibility": p.GRANTED,
                       "input_monitoring": p.DENIED}, grant_after={"input_monitoring": 0})
    result, ui = _run(perms, ["Open Settings", "Restart Now"], can_restart=True)
    assert ui.shown[0][0] == "Step 1 of 1: Input Monitoring"
    assert result.granted == ["input_monitoring"] and result.restart is True


def test_full_walkthrough_grants_all_three_and_offers_restart():
    perms = FakePerms(ALL_MISSING, grant_after={"microphone": 0, "accessibility": 0,
                                                "input_monitoring": 0})
    result, ui = _run(perms, ["Allow", "Open Settings", "Open Settings", "Restart Now"],
                      can_restart=True)
    titles = [s[0] for s in ui.shown]
    assert titles[:3] == ["Step 1 of 3: Microphone", "Step 2 of 3: Accessibility",
                          "Step 3 of 3: Input Monitoring"]
    assert result.granted == ["microphone", "accessibility", "input_monitoring"]
    assert result.restart is True
    assert perms.requested == ["microphone", "accessibility", "input_monitoring"]


def test_microphone_not_asked_yet_uses_the_prompt_not_settings():
    perms = FakePerms({**ALL_MISSING, "accessibility": p.GRANTED, "input_monitoring": p.GRANTED},
                      grant_after={"microphone": 0})
    result, ui = _run(perms, ["Allow"], can_restart=True)
    assert perms.requested == ["microphone"] and perms.opened == []
    assert result.restart is False                       # mic works without a restart


def test_denied_microphone_opens_settings():
    """After 'Don't Allow' macOS never prompts again: only Settings can fix it."""
    perms = FakePerms({**ALL_MISSING, "microphone": p.DENIED, "accessibility": p.GRANTED,
                       "input_monitoring": p.GRANTED}, grant_after={"microphone": 0})
    _, ui = _run(perms, ["Open Settings"], can_restart=True)
    assert perms.opened == ["microphone"]


def test_still_off_offers_check_again_until_it_is_on():
    perms = FakePerms({**ALL_MISSING, "microphone": p.GRANTED, "input_monitoring": p.GRANTED},
                      grant_after={"accessibility": 2})
    result, ui = _run(perms, ["Open Settings", "Check Again", "Check Again", "Restart Now"],
                      can_restart=True)
    assert ui.shown[1][2] == ["Check Again", "Open Settings", "Skip"]
    assert "System Settings" in ui.shown[1][1]
    assert result.granted == ["accessibility"]


def test_skip_moves_on_and_offers_no_restart():
    perms = FakePerms(ALL_MISSING)
    result, ui = _run(perms, ["Skip", "Skip", "Skip"], can_restart=True)
    assert result.skipped == ["microphone", "accessibility", "input_monitoring"]
    assert result.granted == [] and result.restart is False
    assert len(ui.shown) == 3                            # no restart question


def test_restart_later_does_not_restart():
    perms = FakePerms({**ALL_MISSING, "microphone": p.GRANTED, "input_monitoring": p.GRANTED},
                      grant_after={"accessibility": 0})
    result, _ = _run(perms, ["Open Settings", "Later"], can_restart=True)
    assert result.restart is False


def test_run_sh_says_to_restart_by_hand():
    perms = FakePerms({**ALL_MISSING, "microphone": p.GRANTED, "input_monitoring": p.GRANTED},
                      grant_after={"accessibility": 0})
    result, ui = _run(perms, ["Open Settings", "OK"], can_restart=False)
    assert "run.sh" in ui.shown[-1][1] and result.restart is False


def test_from_menu_with_everything_granted_says_all_set():
    perms = FakePerms({k: p.GRANTED for k in ALL_MISSING})
    result, ui = _run(perms, ["OK"], can_restart=True, from_menu=True)
    assert "All set" in ui.shown[0][0] and result.granted == []


def test_startup_with_everything_granted_shows_nothing():
    perms = FakePerms({k: p.GRANTED for k in ALL_MISSING})
    _, ui = _run(perms, [], can_restart=True)
    assert ui.shown == []
