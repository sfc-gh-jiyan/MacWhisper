"""Per-dictation correction and learning, kept out of app.py so it is testable.

    flow = CorrectionFlow(vocab, on_change=sync_hotwords)
    out = flow.process(raw_text)   # rules + retype learning -> out.text to paste
    flow.teach(raw_text)           # correction-hotkey dictation -> (wrong, right)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import corrector
import learn
from vocabulary import ACTIVE, Vocabulary

logger = logging.getLogger(__name__)

RETYPE_WINDOW_S = 30.0   # a re-dictation this soon after the last one is a retype
ECHO_MIN_TERMS = 3      # list terms in one output before it can be an echo
ECHO_MAX_REST = 0.2     # share of letters left once list terms are removed
HITS_SAVE_INTERVAL_S = 60.0  # hit counters are flushed at most this often


@dataclass
class Processed:
    raw_text: str
    text: str
    corrections: list[tuple[str, str]] = field(default_factory=list)
    learned: list[tuple[str, str]] = field(default_factory=list)


class CorrectionFlow:
    def __init__(self, vocab: Vocabulary, on_change=None, now=time.monotonic):
        self.vocab = vocab
        self._on_change = on_change or (lambda: None)
        self._now = now
        self._last: tuple[str, float] | None = None   # (final text, time)
        self._last_save = float("-inf")

    def hotwords(self, limit: int = 50) -> list[str]:
        return self.vocab.hotwords(limit)

    def _changed(self) -> None:
        # Never let vocabulary bookkeeping lose the user's dictation
        try:
            self.vocab.save()
            self._last_save = self._now()
        except Exception:
            logger.exception("Saving vocabulary failed")
        try:
            self._on_change()
        except Exception:
            logger.exception("Vocabulary change callback failed")

    def _undoes_rule(self, wrong: str, right: str) -> bool:
        e = self.vocab.find(wrong)
        return e is not None and right in e.aliases

    def is_hotword_echo(self, text: str) -> bool:
        """ASR on silence can return just the biasing list ("CSE, Snowflake, Cortex")."""
        terms = sorted({t for t in self.vocab.hotwords(200) if t.strip()}, key=len, reverse=True)
        if not terms:
            return False
        pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
        hits = len(pattern.findall(text))
        rest = pattern.sub(" ", text)
        letters = lambda s: len(re.findall(r"\w", s))
        total = letters(text)
        # Mostly list terms with little else (lists, repeats, a junk tail) — not speech
        return hits >= ECHO_MIN_TERMS and total > 0 and letters(rest) <= ECHO_MAX_REST * total

    def _reload(self) -> None:
        try:
            if self.vocab.reload_if_changed():
                self._on_change()
        except Exception:
            logger.exception("Reloading vocabulary failed")

    def add_manual(self, pairs) -> list[tuple[str, str]]:
        """Corrections typed by the user (menu "Add Correction…"); active at once."""
        self._reload()
        added = [(w, r) for w, r in pairs if self.vocab.add_correction(w, r, source="manual")]
        if added:
            self._changed()
        return added

    def only_terms(self, text: str) -> bool:
        """Text is nothing but vocabulary terms and punctuation (e.g. "会话。")."""
        terms = sorted({t for t in self.vocab.hotwords(200) if t.strip()}, key=len, reverse=True)
        if not terms or not text.strip():
            return False
        pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
        return bool(pattern.search(text)) and not re.search(r"\w", pattern.sub(" ", text))

    def process(self, raw_text: str, heard_speech: bool | None = None) -> Processed:
        self._reload()
        if not raw_text:
            return Processed(raw_text="", text="")
        if heard_speech is False and self.only_terms(raw_text):
            logger.warning("Dropped vocabulary echo on silence: %r", raw_text)
            return Processed(raw_text=raw_text, text="")
        if self.is_hotword_echo(raw_text):
            logger.warning("Dropped vocabulary echo: %r", raw_text)
            return Processed(raw_text=raw_text, text="")
        result = corrector.correct(raw_text, self.vocab)
        learned = []
        now = self._now()
        if self._last and now - self._last[1] <= RETYPE_WINDOW_S:
            for wrong, right in learn.retype_corrections(self._last[0], result.text):
                if self._undoes_rule(wrong, right):
                    continue
                before = self.vocab.find(right)
                entry = self.vocab.add_correction(wrong, right, source="learned")
                if entry is not None:
                    learned.append((wrong, right))
                    if entry.status == ACTIVE and (before is None or before.status != ACTIVE):
                        logger.info("Learned correction now active: %r -> %r", wrong, right)
        self._last = (result.text, now)
        if result.terms:
            self.vocab.record_hits(result.terms)
        if learned or (result.terms and now - self._last_save >= HITS_SAVE_INTERVAL_S):
            self._changed()
        return Processed(raw_text=raw_text, text=result.text,
                         corrections=result.changes, learned=learned)

    def teach(self, raw_text: str) -> tuple[str, str] | None:
        self._reload()
        parsed = learn.parse_command(raw_text)
        if parsed is None:
            logger.info("Correction not understood: %r", raw_text)
            return None
        self.vocab.add_correction(*parsed, source="voice")
        self._last = None
        self._changed()
        return parsed
