"""Tests for shortcuts.py (pure logic, no pynput listener)."""

from types import SimpleNamespace

import pytest

import shortcuts as sc


def key(name):          # pynput Key-like (special key)
    return SimpleNamespace(name=name)


def char(vk, ch=None):  # pynput KeyCode-like
    return SimpleNamespace(vk=vk, char=ch)


def test_defaults():
    d = sc.DEFAULTS
    assert d["push_to_talk"] == ["alt_r"]
    assert d["meeting"] == ["ctrl+alt+vk46", "ctrl+shift+f8"]
    assert d["cycle_model"] == ["ctrl+shift+vk46"]
    assert "ctrl+shift+vk15" not in sum(d.values(), [])     # old ⌃⇧R is gone


def test_display():
    assert sc.display("alt_r") == "Right ⌥"
    assert sc.display("ctrl+alt+vk46") == "⌃⌥M"
    assert sc.display("ctrl+shift+f8") == "⌃⇧F8"
    assert sc.display("shift+cmd+ctrl+vk1") == "⌃⇧⌘S"       # canonical order


def test_parse_normalizes_order_and_rejects_garbage():
    assert sc.normalize("shift+ctrl+vk46") == "ctrl+shift+vk46"
    with pytest.raises(ValueError):
        sc.normalize("ctrl+")
    with pytest.raises(ValueError):
        sc.normalize("hyper+vk1")


def test_event_spec_uses_vk_not_char():
    # Option+M types 'µ' on macOS — must still match by key code
    assert sc.event_spec(char(46, "µ"), {"ctrl", "alt"}) == "ctrl+alt+vk46"
    assert sc.event_spec(key("f8"), {"shift", "ctrl"}) == "ctrl+shift+f8"
    assert sc.event_spec(key("alt_r"), set()) == "alt_r"


def test_modifier_of():
    assert sc.modifier_of(key("ctrl_l")) == "ctrl"
    assert sc.modifier_of(key("ctrl_r")) == "ctrl"
    assert sc.modifier_of(key("shift_r")) == "shift"
    assert sc.modifier_of(key("alt")) == "alt"
    assert sc.modifier_of(key("alt_r")) == "alt"
    assert sc.modifier_of(key("cmd")) == "cmd"
    assert sc.modifier_of(key("f8")) is None
    assert sc.modifier_of(char(46)) is None


def test_match_requires_exact_modifiers():
    cfg = sc.merged({})
    assert sc.match_action("ctrl+alt+vk46", cfg) == "meeting"
    assert sc.match_action("ctrl+shift+f8", cfg) == "meeting"
    assert sc.match_action("ctrl+shift+vk46", cfg) == "cycle_model"
    assert sc.match_action("ctrl+alt+shift+vk46", cfg) is None
    assert sc.match_action("ctrl+shift+vk15", cfg) is None          # ⌃⇧R no longer meeting


def test_merged_fills_missing_and_drops_invalid():
    cfg = sc.merged({"meeting": ["ctrl+cmd+vk46"], "translate": ["bogus+"], "nope": ["x"]})
    assert cfg["meeting"] == ["ctrl+cmd+vk46"]
    assert cfg["translate"] == sc.DEFAULTS["translate"]               # invalid -> default
    assert "nope" not in cfg
    assert cfg["push_to_talk"] == ["alt_r"]


def test_conflicts():
    cfg = sc.merged({"translate": ["ctrl+alt+vk46"]})               # same as meeting
    assert ("meeting", "translate", "ctrl+alt+vk46") in sc.conflicts(cfg)
    assert sc.conflicts(sc.merged({})) == []


def test_validate_for_action():
    assert sc.validate("push_to_talk", "alt_r") is None
    assert sc.validate("push_to_talk", "f13") is None
    assert sc.validate("push_to_talk", "ctrl+alt+vk46")             # PTT must be one key
    assert sc.validate("meeting", "vk46")                            # plain letter not allowed
    assert sc.validate("meeting", "shift+vk46")                      # shift alone types a capital
    assert sc.validate("meeting", "f8") is None                      # bare F-key is fine
    assert sc.validate("meeting", "alt_r")                           # combos need a non-modifier key


def test_ptt_blocked_when_other_modifiers_held():
    assert sc.ptt_allowed(held={"alt"}, ptt_spec="alt_r") is True
    assert sc.ptt_allowed(held={"alt", "ctrl"}, ptt_spec="alt_r") is False
    assert sc.ptt_allowed(held={"cmd"}, ptt_spec="f13") is False
    assert sc.ptt_allowed(held=set(), ptt_spec="f13") is True


def test_combo_cancels_ptt_when_ptt_key_is_part_of_it():
    assert sc.combo_uses_ptt_key("ctrl+alt+vk46", "alt_r") is True
    assert sc.combo_uses_ptt_key("ctrl+shift+f8", "alt_r") is False
    assert sc.combo_uses_ptt_key("ctrl+shift+f8", "f13") is False


# ── Settings recorder (NSEvent keyCode) must produce the SAME spec as pynput ──

def _pynput_spec(vk, mods):
    """What the global pynput listener produces for this physical key."""
    from pynput.keyboard import _darwin as d
    from pynput.keyboard import KeyCode
    key = d.Listener._SPECIAL_KEYS.get((vk, None)) or KeyCode.from_vk(vk)
    return sc.event_spec(key, mods)


@pytest.mark.parametrize("vk", [46, 15, 1, 17, 0, 49, 36, 48, 100, 122, 105, 90, 61, 54, 123])
@pytest.mark.parametrize("mods", [set(), {"ctrl", "alt"}, {"ctrl", "shift"}])
def test_recorder_spec_matches_global_listener(vk, mods):
    pytest.importorskip("pynput")
    assert sc.spec_from_keycode(vk, mods) == _pynput_spec(vk, mods)


def test_recorder_known_values():
    assert sc.spec_from_keycode(100, {"ctrl", "shift"}) == "ctrl+shift+f8"
    assert sc.spec_from_keycode(46, {"alt", "ctrl"}) == "ctrl+alt+vk46"
    assert sc.spec_from_keycode(61, set()) == "alt_r"


def test_mods_from_nsevent_flags():
    # NSEventModifierFlag: shift 1<<17, control 1<<18, option 1<<19, command 1<<20
    assert sc.mods_from_flags((1 << 18) | (1 << 19)) == {"ctrl", "alt"}
    assert sc.mods_from_flags(1 << 17) == {"shift"}
    assert sc.mods_from_flags(1 << 20 | 1 << 16) == {"cmd"}     # caps lock ignored
    assert sc.mods_from_flags(0) == set()


def test_right_side_modifier_keycodes_for_flags_changed():
    # flagsChanged events: which physical modifier key went down
    assert sc.side_modifier_name(61) == "alt_r"
    assert sc.side_modifier_name(54) == "cmd_r"
    assert sc.side_modifier_name(62) == "ctrl_r"
    assert sc.side_modifier_name(60) == "shift_r"
    assert sc.side_modifier_name(58) is None        # left Option: not a valid PTT key


def test_non_ascii_digits_rejected():
    """Reviewer: 'vk²'.isdigit() is True but int('²') fails in display()."""
    with pytest.raises(ValueError):
        sc.normalize("ctrl+vk²")
    assert sc.merged({"meeting": ["ctrl+vk²"]})["meeting"] == sc.DEFAULTS["meeting"]


def test_ptt_mode():
    assert sc.ptt_mode(set(), "alt_r") == "dictate"
    assert sc.ptt_mode({"alt"}, "alt_r") == "dictate"
    assert sc.ptt_mode({"alt", "shift"}, "alt_r") == "correct"
    assert sc.ptt_mode({"shift"}, "f13") == "correct"
    assert sc.ptt_mode({"alt", "ctrl"}, "alt_r") is None
    assert sc.ptt_mode({"shift", "ctrl"}, "alt_r") is None
    assert sc.ptt_mode({"shift"}, "shift_r") == "dictate"


def test_key_is_down_side_modifier_uses_hid_state(monkeypatch):
    calls = []
    monkeypatch.setattr(sc, "_hid_key_state", lambda vk: calls.append(vk) or True)
    assert sc.key_is_down("alt_r") is True and calls == [61]


def test_key_is_down_unknown_key_returns_none():
    assert sc.key_is_down("f13") is None
