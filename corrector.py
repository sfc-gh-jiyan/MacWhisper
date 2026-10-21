"""Deterministic, model-free corrections applied to every final transcript.

Order:
  1. join spelled acronyms  ("C S E" -> "CSE")
  2. vocabulary aliases     ("CIC" -> "CSE", "漫威" -> "memory" in mixed text)
  3. canonical casing       ("snowflake" -> "Snowflake")

Latin aliases match on word boundaries, case-insensitively, with flexible
whitespace ("snow  flake"). CJK aliases match as substrings. Longer aliases
are applied first so "snow park" wins over "park".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from vocabulary import SCOPE_MIXED, Vocabulary

logger = logging.getLogger(__name__)

_SPELLED = re.compile(r"(?<![A-Za-z0-9])([A-Z](?: [A-Z]){1,7})(?![A-Za-z0-9])")
_LATIN = re.compile(r"[A-Za-z]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
MIXED_WINDOW = 12      # English must appear within this many chars of a "mixed" alias
MIN_CJK_ALIAS = 2      # single CJK characters are too common to replace
MIN_RECASE_LEN = 3     # "us" / "go" must not become "US" / "Go"


@dataclass
class Correction:
    text: str
    changes: list[tuple[str, str]] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)  # vocabulary terms present in the result


def _latin_pattern(word: str) -> re.Pattern:
    parts = [re.escape(p) for p in word.split()]
    return re.compile(r"(?<![A-Za-z0-9])" + r"\s+".join(parts) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _english_nearby(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - MIXED_WINDOW):start] + " " + text[end:end + MIXED_WINDOW]
    return bool(_LATIN_WORD.search(window))


def _replace_cjk(text: str, alias: str, term: str, mixed: bool, changes: list) -> str:
    out, i = [], 0
    while True:
        j = text.find(alias, i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        if not mixed or _english_nearby(text, j, j + len(alias)):
            out.append(term)
            changes.append((alias, term))
        else:
            out.append(alias)
        i = j + len(alias)


def _sub(pattern: re.Pattern, repl: str, text: str, changes: list) -> str:
    def f(m):
        if m.group(0) != repl:
            changes.append((m.group(0), repl))
        return repl
    return pattern.sub(f, text)


def correct(text: str, vocab: Vocabulary) -> Correction:
    if not text:
        return Correction(text="")
    changes: list[tuple[str, str]] = []

    def join(m):
        letters = m.group(1).split()
        if len(set(letters)) == 1:      # "I I saw" is a stutter, not an acronym
            return m.group(0)
        joined = "".join(letters)
        changes.append((m.group(1), joined))
        return joined

    out = _SPELLED.sub(join, text)

    entries = vocab.active()
    pairs = sorted(((a, e) for e in entries for a in e.aliases), key=lambda p: -len(p[0]))
    for alias, e in pairs:
        if _LATIN.search(alias):
            out = _sub(_latin_pattern(alias), e.term, out, changes)
        elif len(alias) >= MIN_CJK_ALIAS and alias in out:
            out = _replace_cjk(out, alias, e.term, e.scope == SCOPE_MIXED, changes)

    for e in entries:
        if len(e.term) >= MIN_RECASE_LEN and any(c.isupper() for c in e.term) and _LATIN.search(e.term):
            out = _sub(_latin_pattern(e.term), e.term, out, changes)

    terms = [e.term for e in entries if (_latin_pattern(e.term).search(out) if _LATIN.search(e.term)
                                         else e.term in out)]
    if changes:
        logger.debug("Corrections %s -> %r", changes, out)
    return Correction(text=out, changes=changes, terms=terms)
