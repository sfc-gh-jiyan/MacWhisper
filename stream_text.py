"""Stable live text from repeated re-decoding (LocalAgreement-2 on text).

Used by the Qwen3-ASR live subtitles: every ~1 s the WHOLE utterance so far is
re-transcribed; the prefix on which two consecutive decodes agree is
confirmed (white), the rest is the tentative tail (gray). Agreement is on
normalized text tokens, so no word timestamps are needed (Qwen3 gives none).

Validated offline by tests/simulate_stream.py on saved Push to Talk clips.
"""

from __future__ import annotations

import re

# One CJK char, or a run of Latin letters/digits/apostrophes = one token.
_TOKEN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9']+")
# Hesitation fillers. Qwen3 emits them inconsistently (silence before speech
# decodes as "嗯。" and later decodes drop it), so they must never take part
# in agreement — a confirmed filler froze confirmation for a whole utterance.
FILLERS = {"嗯", "呃", "啊", "额", "唔", "um", "uh", "uhm", "erm", "hmm", "mm"}


def tokenize(text: str) -> list[tuple[str, str]]:
    """Split into (display, key) tokens. Display keeps trailing punctuation
    and spacing; key is lowercase content only, so punctuation changes
    between decodes ("，" -> "。") never block agreement."""
    text = text or ""
    matches = list(_TOKEN.finditer(text))
    tokens = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        key = m.group().lower()
        if key in FILLERS:
            continue
        tokens.append((text[m.start():end], key))
    return tokens


def common_prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class StableText:
    """LocalAgreement-2 on text tokens, confirmed prefix never shrinks."""

    def __init__(self):
        self.confirmed: list[tuple[str, str]] = []
        self.tail: list[tuple[str, str]] = []
        self._prev_keys: list[str] | None = None
        self.contradictions = 0  # newest decode disagreed with confirmed text
        self.rewrites = 0        # confirmed text replaced (model changed its mind twice)

    def update(self, hypothesis: str) -> None:
        toks = tokenize(hypothesis)
        keys = [k for _, k in toks]
        n_conf = len(self.confirmed)
        if keys[:n_conf] != [k for _, k in self.confirmed]:
            self.contradictions += 1
            agree = common_prefix_len(self._prev_keys or [], keys)
            if agree > 0:
                # Two consecutive decodes agree with each other but not with
                # what we confirmed: the model has settled on a new reading.
                # Rebase instead of freezing forever.
                self.confirmed = toks[:agree]
                self.rewrites += 1
                self.tail = toks[agree:]
            else:
                self.tail = []  # first disagreement: keep confirmed, wait
            self._prev_keys = keys
            return
        if self._prev_keys is not None:
            agree = common_prefix_len(self._prev_keys, keys)
            if agree > n_conf:
                self.confirmed.extend(toks[n_conf:agree])
        self.tail = toks[len(self.confirmed):]
        self._prev_keys = keys

    @property
    def confirmed_text(self) -> str:
        return "".join(d for d, _ in self.confirmed)

    @property
    def tail_text(self) -> str:
        return "".join(d for d, _ in self.tail)
