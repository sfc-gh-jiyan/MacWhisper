"""Keyboard shortcut specs for MacWhisper (pure logic; no pynput listener).

A spec is a string of tokens joined by "+": zero or more modifiers in the
canonical order ctrl, alt, shift, cmd, then exactly one key:
  - "vkNN"   a regular key by macOS virtual key code (Option changes the
             typed character, e.g. Option+M -> 'µ', so chars are unreliable)
  - a pynput special key name: "f8", "alt_r", "space", ...

Examples: "alt_r" (hold Right Option), "ctrl+alt+vk46" (⌃⌥M), "ctrl+shift+f8".
"""

from __future__ import annotations

MODIFIERS = ("ctrl", "alt", "shift", "cmd")
_MOD_SYMBOL = {"ctrl": "⌃", "alt": "⌥", "shift": "⇧", "cmd": "⌘"}
_MOD_OF_NAME = {
    "ctrl": "ctrl", "ctrl_l": "ctrl", "ctrl_r": "ctrl",
    "alt": "alt", "alt_l": "alt", "alt_r": "alt", "alt_gr": "alt",
    "shift": "shift", "shift_l": "shift", "shift_r": "shift",
    "cmd": "cmd", "cmd_l": "cmd", "cmd_r": "cmd",
}
# Single keys allowed for hold-to-record (no modifiers held)
PTT_KEYS = ("alt_r", "cmd_r", "ctrl_r", "shift_r") + tuple(f"f{i}" for i in range(1, 21))

# macOS ANSI virtual key codes -> label
_VK_LABEL = {
    0: "A", 11: "B", 8: "C", 2: "D", 14: "E", 3: "F", 5: "G", 4: "H", 34: "I",
    38: "J", 40: "K", 37: "L", 46: "M", 45: "N", 31: "O", 35: "P", 12: "Q",
    15: "R", 1: "S", 17: "T", 32: "U", 9: "V", 13: "W", 7: "X", 16: "Y", 6: "Z",
    29: "0", 18: "1", 19: "2", 20: "3", 21: "4", 23: "5", 22: "6", 26: "7",
    28: "8", 25: "9", 49: "Space", 36: "Return", 48: "Tab", 53: "Esc",
    27: "-", 24: "=", 33: "[", 30: "]", 41: ";", 39: "'", 43: ",", 47: ".", 44: "/",
}
_SPECIAL_LABEL = {
    "alt_r": "Right ⌥", "cmd_r": "Right ⌘", "ctrl_r": "Right ⌃", "shift_r": "Right ⇧",
    "space": "Space", "enter": "Return", "tab": "Tab", "esc": "Esc",
}

# Action -> default specs. Order is the display order in Settings.
ACTIONS = ("push_to_talk", "meeting", "cycle_model", "translate", "live_subtitles")
ACTION_LABELS = {
    "push_to_talk": "Push to Talk (hold)",
    "meeting": "Start / stop meeting",
    "cycle_model": "Next model",
    "translate": "Translate mode",
    "live_subtitles": "Live subtitles",
}
DEFAULTS = {
    "push_to_talk": ["alt_r"],
    "meeting": ["ctrl+alt+vk46", "ctrl+shift+f8"],   # ⌃⌥M, ⌃⇧F8 (was ⌃⇧R)
    "cycle_model": ["ctrl+shift+vk46"],              # ⌃⇧M
    "translate": ["ctrl+shift+vk17"],                # ⌃⇧T
    "live_subtitles": ["ctrl+shift+vk1"],            # ⌃⇧S
}


def _is_key_token(tok: str) -> bool:
    if tok.startswith("vk"):
        digits = tok[2:]
        return digits.isascii() and digits.isdecimal()   # 'vk²'.isdigit() is True
    return tok.isidentifier() and tok not in MODIFIERS


def normalize(spec: str) -> str:
    """Canonical form of a spec. Raises ValueError if malformed."""
    if not isinstance(spec, str) or not spec:
        raise ValueError("empty shortcut")
    toks = [t.strip().lower() for t in spec.split("+")]
    if any(not t for t in toks):
        raise ValueError(f"malformed shortcut: {spec!r}")
    *mods, key = toks
    bad = [m for m in mods if m not in MODIFIERS]
    if bad:
        raise ValueError(f"unknown modifier {bad[0]!r} in {spec!r}")
    if not _is_key_token(key):
        raise ValueError(f"unknown key {key!r} in {spec!r}")
    ordered = [m for m in MODIFIERS if m in mods]
    return "+".join(ordered + [key])


def split(spec: str) -> tuple[set, str]:
    *mods, key = normalize(spec).split("+")
    return set(mods), key


def display(spec: str) -> str:
    mods, key = split(spec)
    if key.startswith("vk"):
        label = _VK_LABEL.get(int(key[2:]), f"Key {key[2:]}")
    elif key in _SPECIAL_LABEL:
        label = _SPECIAL_LABEL[key]
    elif key.startswith("f") and key[1:].isdigit():
        label = key.upper()
    else:
        label = key.replace("_", " ").title()
    return "".join(_MOD_SYMBOL[m] for m in MODIFIERS if m in mods) + label


def modifier_name(token: str) -> str | None:
    """'alt_r' -> 'alt'; None if the token is not a modifier key."""
    return _MOD_OF_NAME.get(token)


def modifier_of(key) -> str | None:
    """'ctrl'/'alt'/'shift'/'cmd' if this pynput key is a modifier, else None."""
    name = getattr(key, "name", None)
    return _MOD_OF_NAME.get(name) if name else None


def key_token(key) -> str | None:
    name = getattr(key, "name", None)
    if name:
        return name
    vk = getattr(key, "vk", None)
    return f"vk{vk}" if vk is not None else None


def event_spec(key, held: set) -> str | None:
    """Spec for pressing `key` while the modifiers in `held` are down."""
    tok = key_token(key)
    if tok is None:
        return None
    return "+".join([m for m in MODIFIERS if m in held] + [tok])


def merged(cfg: dict | None) -> dict:
    """Config shortcuts on top of DEFAULTS; invalid entries fall back to defaults."""
    cfg = cfg if isinstance(cfg, dict) else {}
    out = {}
    for action in ACTIONS:
        val = cfg.get(action)
        if not isinstance(val, list):
            out[action] = list(DEFAULTS[action])
            continue
        good = []
        for spec in val:
            try:
                s = normalize(spec)
            except ValueError:
                continue
            if validate(action, s) is None and s not in good:
                good.append(s)
        if action == "push_to_talk":
            out[action] = good[:1] or list(DEFAULTS[action])
        elif val and not good:
            out[action] = list(DEFAULTS[action])   # all invalid -> default
        else:
            out[action] = good                      # [] means intentionally unbound
    return out


def match_action(spec: str | None, cfg: dict) -> str | None:
    """Combo action bound to spec (push_to_talk is handled separately)."""
    if not spec:
        return None
    for action in ACTIONS:
        if action != "push_to_talk" and spec in cfg.get(action, []):
            return action
    return None


def conflicts(cfg: dict) -> list[tuple[str, str, str]]:
    seen: dict[str, str] = {}
    out = []
    for action in ACTIONS:
        for spec in cfg.get(action, []):
            if spec in seen and seen[spec] != action:
                out.append((seen[spec], action, spec))
            else:
                seen.setdefault(spec, action)
    return out


def validate(action: str, spec: str) -> str | None:
    """None if spec is usable for action, else a human-readable reason."""
    try:
        mods, key = split(spec)
    except ValueError as e:
        return str(e)
    if action == "push_to_talk":
        if mods or key not in PTT_KEYS:
            return "Push to Talk must be one key: Right ⌥/⌘/⌃/⇧ or F1–F20"
        return None
    if key in _MOD_OF_NAME:
        return "A shortcut needs a non-modifier key (for example ⌃⌥M)"
    is_fkey = key.startswith("f") and key[1:].isdigit()
    if not mods and not is_fkey:
        return "Add ⌃, ⌥ or ⌘ — a plain key would fire while you type"
    if mods == {"shift"} and not is_fkey:
        return "⇧ alone types a capital letter — add ⌃, ⌥ or ⌘"
    return None


def ptt_allowed(held: set, ptt_spec: str) -> bool:
    """Start hold-to-record only if no OTHER modifier is held (e.g. ⌃ of ⌃⌥M)."""
    _, key = split(ptt_spec)
    own = _MOD_OF_NAME.get(key)
    return not (set(held) - {own})


CORRECTION_MODIFIER = "shift"   # hold ⇧ + Push to Talk = teach a correction


def ptt_mode(held: set, ptt_spec: str) -> str | None:
    """"dictate", "correct" (only ⇧ held besides the PTT key's own modifier) or None."""
    _, key = split(ptt_spec)
    own = _MOD_OF_NAME.get(key)
    extra = set(held) - {own}
    if not extra:
        return "dictate"
    if extra == {CORRECTION_MODIFIER} and own != CORRECTION_MODIFIER:
        return "correct"
    return None


def combo_uses_ptt_key(combo_spec: str, ptt_spec: str) -> bool:
    """True if the combo includes the PTT key's modifier (⌃⌥M with Right ⌥)."""
    mods, _ = split(combo_spec)
    _, key = split(ptt_spec)
    own = _MOD_OF_NAME.get(key)
    return own is not None and own in mods


# ── Settings recorder: NSEvent -> spec (must match the pynput listener) ──

_NS_FLAG_MODS = (("shift", 1 << 17), ("ctrl", 1 << 18), ("alt", 1 << 19), ("cmd", 1 << 20))
_SIDE_MODIFIER_VK = {61: "alt_r", 54: "cmd_r", 62: "ctrl_r", 60: "shift_r"}
_special_by_vk: dict | None = None


def _special_names() -> dict:
    """macOS keyCode -> pynput special key name, from pynput's own table.

    Reusing pynput's table guarantees the Settings recorder and the global
    listener name every key identically (e.g. keyCode 100 -> "f8").
    """
    global _special_by_vk
    if _special_by_vk is None:
        try:
            from pynput.keyboard import _darwin
            table = _darwin.Listener._SPECIAL_KEYS
            _special_by_vk = {vk: key.name for (vk, media), key in table.items() if media is None}
        except Exception:
            _special_by_vk = {}
    return _special_by_vk


def spec_from_keycode(keycode: int, mods: set) -> str:
    """Spec for an NSEvent keyCode with modifiers — same string pynput yields."""
    token = _special_names().get(keycode) or f"vk{keycode}"
    return "+".join([m for m in MODIFIERS if m in mods] + [token])


def mods_from_flags(flags: int) -> set:
    """NSEvent modifierFlags -> {"ctrl", "alt", "shift", "cmd"} (caps lock ignored)."""
    return {name for name, bit in _NS_FLAG_MODS if flags & bit}


# Device-dependent modifier bits (IOKit NX_DEVICE*KEYMASK): one per physical side.
# Measured on this Mac (2026-10-23): CGEventSourceKeyState(61) stays 0 while
# Right ⌥ is held; the HID flag bit 0x40 follows the key exactly.
_SIDE_MODIFIER_FLAG = {"alt_r": 0x40, "cmd_r": 0x10, "ctrl_r": 0x2000, "shift_r": 0x04}


def _hid_flags() -> int:
    import Quartz
    return int(Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateHIDSystemState))


def key_is_down(token: str) -> bool | None:
    """Physical state of a side-modifier key from the HID system; None if not checkable."""
    bit = _SIDE_MODIFIER_FLAG.get(token)
    if bit is None:
        return None
    try:
        return bool(_hid_flags() & bit)
    except Exception:
        return None


def side_modifier_name(keycode: int) -> str | None:
    """Right-side modifier key name for a flagsChanged keyCode (PTT candidates)."""
    return _SIDE_MODIFIER_VK.get(keycode)
