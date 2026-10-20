"""Post-processing for Qwen3-ASR output.

Kept separate from text_utils so the Whisper pipeline is unchanged. Whisper's
loop-detection rules (word_repeat, phrase_repeat, ...) are NOT applied: Qwen3
does not loop the way Whisper does, and those rules discarded a correct 67 s
English transcript ("... the Jira search ... the Jira search ...").

Defaults match the Whisper output style the user is used to:
  - standalone fillers removed (嗯 / 呃 / um ...), but never inside words
    (额外) and never sentence particles (好啊),
  - no space between CJK and Latin text ("这个commit的情况").
"""

from __future__ import annotations

import re

from text_utils import (
    _HALLUCINATION_PHRASES,
    _HALLUCINATION_SUBSTRINGS,
    convert_t2s,
    strip_trailing_repetition,
)

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_CJK_PUNCT = r"\u3000-\u303f\uff00-\uffef"
_PUNCT = r"，。、；：！？,.;:!?"

# Bare hesitation fillers only (not 啊 — usually a sentence particle).
_FILLER = r"(?:嗯|呃|额|唔|um|uh|uhm|erm|hmm)"
_FILLER_RE = re.compile(
    rf"(?:(?<=^)|(?<=[\s{_PUNCT}]))(?:{_FILLER})+(?=[\s{_PUNCT}]|$)[{_PUNCT}]?\s*",
    re.IGNORECASE,
)

_SPACE_CJK_LATIN = re.compile(rf"(?<=[{_CJK}{_CJK_PUNCT}])\s+(?=[A-Za-z0-9])")
_SPACE_LATIN_CJK = re.compile(rf"(?<=[A-Za-z0-9.,!?;:'\"%)\]])\s+(?=[{_CJK}{_CJK_PUNCT}])")
_SPACE_CJK_CJK = re.compile(rf"(?<=[{_CJK}{_CJK_PUNCT}])\s+(?=[{_CJK}{_CJK_PUNCT}])")

# Whisper's list misses this outro hallucination seen in real comparisons.
_EXTRA_PHRASES = {"谢谢大家", "谢谢观看", "感谢观看"}


def strip_fillers(text: str) -> str:
    return _FILLER_RE.sub("", text).strip()


def tighten_spacing(text: str) -> str:
    text = _SPACE_CJK_LATIN.sub("", text)
    text = _SPACE_LATIN_CJK.sub("", text)
    return _SPACE_CJK_CJK.sub("", text)


def is_qwen_hallucination(text: str) -> bool:
    bare = text.lower().strip(" .!,。，！?？")
    if bare in _HALLUCINATION_PHRASES or bare in _EXTRA_PHRASES:
        return True
    return any(sub in text for sub in _HALLUCINATION_SUBSTRINGS)


def postprocess(text: str, strip_filler_words: bool = True, tight: bool = True) -> str:
    text = convert_t2s((text or "").strip())
    text = strip_trailing_repetition(text)
    if strip_filler_words:
        text = strip_fillers(text)
    if tight:
        text = tighten_spacing(text)
    text = text.strip()
    if not text.strip(_PUNCT + " "):
        return ""
    if is_qwen_hallucination(text):
        return ""
    return text
