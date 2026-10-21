"""Learn corrections from the user without manual editing.

Two signals:
  parse_command      — a dictation made with the correction hotkey, e.g.
                       "CIC 应该是 CSE" / "change CIC to CSE" -> ("CIC", "CSE").
  retype_corrections — the user immediately re-dictates nearly the same
                       sentence; the few spans that changed are candidate
                       corrections (stored as "learned", pending until seen twice).
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

MAX_TERM_LEN = 30
_SPELLED = re.compile(r"(?<![A-Za-z0-9])([A-Z](?: [A-Z]){1,7})(?![A-Za-z0-9])")
_QUOTES = "\"'“”‘’「」『』《》"
_EDGE = " \t\n,，.。!！?？:：;；、" + _QUOTES

_COMMANDS = [
    re.compile(r"^不是(?P<w>.+?)[，,、\s]*(?:而)?是(?P<r>.+)$"),
    re.compile(r"^把(?P<w>.+?)(?:改成|换成|改为|写成|替换成|纠正为|纠正成)(?P<r>.+)$"),
    re.compile(r"^(?P<w>.+?)(?:应该是|应该写成|应该叫|改成|换成|改为|写成|替换成|纠正为|纠正成)(?P<r>.+)$"),
    re.compile(r"^(?P<w>.+?)\s+should\s+be\s+(?P<r>.+)$", re.IGNORECASE),
    re.compile(r"^change\s+(?P<w>.+?)\s+(?:to|into)\s+(?P<r>.+)$", re.IGNORECASE),
    re.compile(r"^replace\s+(?P<w>.+?)\s+with\s+(?P<r>.+)$", re.IGNORECASE),
]


def _clean(s: str) -> str:
    return s.strip(_EDGE)


def parse_command(text: str) -> tuple[str, str] | None:
    """Parse a spoken correction. Spelled acronyms are joined first ("C S E")."""
    if not text:
        return None
    t = _SPELLED.sub(lambda m: m.group(1) if len(set(m.group(1).split())) == 1
                     else m.group(1).replace(" ", ""), _clean(text))
    for pat in _COMMANDS:
        m = pat.match(t)
        if not m:
            continue
        wrong, right = _clean(m.group("w")), _clean(m.group("r"))
        if not wrong or not right or wrong.lower() == right.lower():
            return None
        if len(wrong) > MAX_TERM_LEN or len(right) > MAX_TERM_LEN:
            return None
        logger.info("Correction command parsed: %r -> %r", wrong, right)
        return wrong, right
    return None


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*|[^\sA-Za-z0-9,，.。!！?？:：;；、" + re.escape(_QUOTES) + "]")
MIN_SIMILARITY = 0.6
MAX_SPANS = 2
MAX_SPAN_TOKENS = 6


def _tokens(s: str) -> list[str]:
    """English words are one token each; every other character is a token."""
    return _TOKEN.findall(s or "")


def _join(tokens: list[str]) -> str:
    out = ""
    for t in tokens:
        if out and out[-1].isascii() and out[-1].isalnum() and t[0].isascii() and t[0].isalnum():
            out += " "
        out += t
    return out


_NUMERIC = re.compile(r"^[\d零一二三四五六七八九十百千万两.]+$")


def _is_term_fix(wrong: str, right: str) -> bool:
    """Learn only misheard terms: the fix contains Latin text and nothing is numeric."""
    if _NUMERIC.match(wrong.replace(" ", "")) or _NUMERIC.match(right.replace(" ", "")):
        return False
    if not re.search(r"[A-Za-z]{2,}", right):
        return False
    if not re.search(r"[A-Za-z]", wrong) and len(wrong) < 2:
        return False
    return True


def retype_corrections(prev: str, cur: str) -> list[tuple[str, str]]:
    """Spans that changed between two near-identical dictations -> [(wrong, right)]."""
    a, b = _tokens(prev), _tokens(cur)
    if not a or not b or a == b:
        return []
    sm = SequenceMatcher(None, a, b, autojunk=False)
    if sm.ratio() < MIN_SIMILARITY:
        return []
    spans = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if len(spans) > MAX_SPANS or any(op[0] != "replace" for op in spans):
        return []
    out = []
    for _, i1, i2, j1, j2 in spans:
        if i2 - i1 > MAX_SPAN_TOKENS or j2 - j1 > MAX_SPAN_TOKENS:
            return []
        wrong, right = _join(a[i1:i2]), _join(b[j1:j2])
        if not _is_term_fix(wrong, right):
            return []   # rewording ("三点" -> "四点"), not a misheard term
        out.append((wrong, right))
    logger.debug("Retype corrections %r -> %r: %s", prev, cur, out)
    return out


_MANUAL_SEP = re.compile(r"\s*(?:=>|->|→|=)\s*")


def parse_manual(text: str) -> list[tuple[str, str]]:
    """Typed corrections, one per line or ';': "哈什 = hash", "漫威 → memory", "X 应该是 Y"."""
    pairs = []
    for part in re.split(r"[\n;；]+", text or ""):
        part = part.strip()
        if not part:
            continue
        bits = _MANUAL_SEP.split(part, maxsplit=1)
        if len(bits) == 2:
            wrong, right = _clean(bits[0]), _clean(bits[1])
            if wrong and right and wrong.lower() != right.lower() \
                    and len(wrong) <= MAX_TERM_LEN and len(right) <= MAX_TERM_LEN:
                pairs.append((wrong, right))
            continue
        parsed = parse_command(part)
        if parsed:
            pairs.append(parsed)
    return pairs


def is_same_sides(text: str) -> bool:
    """A spoken correction whose two sides were heard the same ("哈什应该是哈什")."""
    t = _clean(text or "")
    for pat in _COMMANDS:
        m = pat.match(t)
        if m:
            return _clean(m.group("w")).lower() == _clean(m.group("r")).lower()
    return False
