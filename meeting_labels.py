"""Mic/Remote label helpers and echo-detection for Meeting Mode.

Does NOT import meeting.py — duck-types on objects with .text, .start_time,
.end_time, and .source attributes (meeting.py imports this module, not the
reverse).
"""

from __future__ import annotations

import difflib
import re
from typing import Any

ECHO_SIMILARITY = 0.8

_NORMALIZE_RE = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fff]")


def source_label(source: str | None) -> str | None:
    """Return display label for a source string, or None if source is None.

    Examples: "mic" -> "Mic", "remote" -> "Remote", "remote:2" -> "Remote 2".
    """
    if source is None:
        return None
    parts = source.split(":", 1)
    label = parts[0].capitalize()
    if len(parts) == 2:
        label += " " + parts[1]
    return label


def text_similarity(a: str, b: str) -> float:
    """Return 0..1 similarity ratio on normalised text.

    Normalisation: lowercase, remove all non-alphanumeric characters (including
    punctuation and whitespace). CJK characters are kept as-is.
    """
    def _norm(s: str) -> str:
        return _NORMALIZE_RE.sub("", s.lower())

    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def is_echo(mic: Any, remote: Any, threshold: float = ECHO_SIMILARITY) -> bool:
    """Return True when mic and remote segments overlap in time and share similar text.

    Both conditions must hold:
      - Time ranges overlap: mic.start_time < remote.end_time and
        remote.start_time < mic.end_time.
      - text_similarity(mic.text, remote.text) >= threshold.
    """
    overlap = mic.start_time < remote.end_time and remote.start_time < mic.end_time
    if not overlap:
        return False
    return text_similarity(mic.text, remote.text) >= threshold


def ordered(segments: list[Any]) -> list[Any]:
    """Return segments sorted stably by start_time ascending."""
    return sorted(segments, key=lambda s: s.start_time)
