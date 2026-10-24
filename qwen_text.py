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


# ── Stutter repeats ("我我我们", "I I", "被占用，被占用") ──
# Only repeats that sit right next to each other. Chinese has many normal
# reduplications (看看, 找找, 整整, 慢慢, 大段大段, 什么什么), so 1- and 2-character
# repeats are collapsed only for words that are never doubled on purpose.
_STUTTER_CHARS = "我你他她它们这那会不没去在要就也都把被跟有让"
_STUTTER_WORDS = ("这个", "那个", "就是", "然后", "我们", "你们", "他们", "她们", "它们",
                  "我的", "你的", "他的", "给我", "可以", "因为", "所以", "但是", "如果",
                  "这样", "那样", "还是", "已经", "应该")
# grammatical ("had had", "that that") or emphasis ("very very important")
_KEEP_DOUBLED_EN = {"had", "that", "very", "really", "so", "bye", "ha"}
_NUMERALS = set("零一二三四五六七八九十百千万两")
_SEP = r"[，,、 ]"
_CLAUSE_END = rf"(?=[{_PUNCT}\s]|$)"
# Back to back: always a stutter. With a comma between: only when the clause
# ends after the repeat ("被占用，被占用，"), not "一个账户，一个账户对应…".
_REPEAT_PHRASE = re.compile(rf"([{_CJK}]{{3,8}})(?:\1)+")
_REPEAT_PHRASE_SEP = re.compile(rf"([{_CJK}]{{3,8}})(?:{_SEP}\1)+{_CLAUSE_END}")
_REPEAT_WORD = re.compile(rf"({'|'.join(_STUTTER_WORDS)})(?:{_SEP}?\1)+")
_REPEAT_CHAR = re.compile(rf"([{_STUTTER_CHARS}])\1+")
_REPEAT_EN = re.compile(r"\b([A-Za-z]+)(?:\s+\1\b)+", re.IGNORECASE)


def _keep_first_en(m: re.Match) -> str:
    word = m.group(1)
    if word.lower() in _KEEP_DOUBLED_EN or (len(word) == 1 and word.lower() not in "ai"):
        return m.group(0)                      # "A P P" spelled out is not a stutter
    return word


def _keep_first_cjk(m: re.Match) -> str:
    phrase = m.group(1)
    if sum(ch in _NUMERALS for ch in phrase) >= 2:
        return m.group(0)                      # 十月十九、十月二十 is a list, not a stutter
    return phrase


def collapse_repeats(text: str) -> str:
    """Remove stutter repeats from a final transcript; keep normal reduplication."""
    out = _REPEAT_PHRASE.sub(_keep_first_cjk, text)
    out = _REPEAT_PHRASE_SEP.sub(_keep_first_cjk, out)
    out = _REPEAT_WORD.sub(r"\1", out)
    out = _REPEAT_CHAR.sub(r"\1", out)
    return _REPEAT_EN.sub(_keep_first_en, out)


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


def postprocess(text: str, strip_filler_words: bool = True, tight: bool = True,
                collapse: bool = False) -> str:
    text = convert_t2s((text or "").strip())
    text = strip_trailing_repetition(text)
    if strip_filler_words:
        text = strip_fillers(text)
    if collapse:
        text = collapse_repeats(text)
    if tight:
        text = tighten_spacing(text)
    text = text.strip()
    if not text.strip(_PUNCT + " "):
        return ""
    if is_qwen_hallucination(text):
        return ""
    return text
