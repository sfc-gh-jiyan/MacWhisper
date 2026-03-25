"""Text processing utilities for MacWhisper.

Hallucination filtering, repetition stripping, overlap detection,
Traditional→Simplified Chinese conversion, and shared constants.
"""

import re
import unicodedata

import opencc

# ── Constants ─────────────────────────────────────────────

BILINGUAL_PROMPT = "以下是中英双语对话的转录。"

_PUNCT_NORMALIZE = str.maketrans('，。！？、', ',.!?,')
OVERLAP_STRIP_CHARS = set(' \t\n，。！？、,.!?-\u3000')

# English → Chinese punctuation for display consistency
_PUNCT_TO_CJK = str.maketrans({',': '，', '.': '。', '!': '！', '?': '？'})


def normalize_punctuation(text: str) -> str:
    """Normalize English punctuation to Chinese style for consistent display.

    Whisper freely mixes ，/, and 。/. in bilingual text. This normalizes
    to Chinese-style punctuation for visual consistency.
    """
    return text.translate(_PUNCT_TO_CJK)

_HALLUCINATION_PHRASES = {
    "thank you for watching", "thanks for watching", "thank you",
    "please subscribe", "中文字幕君", "字幕由amara", "字幕提供",
    "请不吝点赞", "订阅转发", "打赏支持", "明镜与点点栏目",
    "感谢观看", "欢迎订阅", "点赞关注", "支持明镜",
}

_HALLUCINATION_SUBSTRINGS = [
    "请不吝点赞", "打赏支持明镜", "字幕由amara", "字幕提供",
    "普通话与英语的混合", "中英双语对话的转录", "中英语对话的转录",
    "中文字幕组", "中英语对话",
]

# ── Traditional → Simplified Chinese ─────────────────────

_t2s = opencc.OpenCC('t2s')


def convert_t2s(text: str) -> str:
    """Convert Traditional Chinese to Simplified Chinese."""
    return _t2s.convert(text)


# ── Repetition stripping ─────────────────────────────────

def strip_trailing_repetition(text: str) -> str:
    """Remove repetitive tail that Whisper appends when decoder loops.

    Uses normalized matching (lowercase, strip punctuation/whitespace) so
    that repetitions with slightly different punctuation are still caught.
    A position map translates the cut point back to the original string.
    """
    stripped = text.rstrip()
    if len(stripped) < 4:
        return text

    raw_tail = stripped[-300:] if len(stripped) > 300 else stripped
    raw_tail_offset = len(stripped) - len(raw_tail)

    # Build normalized tail with position mapping
    norm_chars = []
    positions = []  # norm_index -> index in stripped
    for i, ch in enumerate(raw_tail):
        if ch not in OVERLAP_STRIP_CHARS:
            norm_chars.append(ch.lower())
            positions.append(raw_tail_offset + i)
    norm_tail = ''.join(norm_chars)

    if len(norm_tail) < 4:
        return text

    best_cut = None
    for unit_len in range(1, min(81, len(norm_tail) // 2 + 1)):
        unit = norm_tail[-unit_len:]
        count = 0
        pos = len(norm_tail)
        while pos >= unit_len and norm_tail[pos - unit_len:pos] == unit:
            count += 1
            pos -= unit_len
        min_repeats = 2 if unit_len > 10 else 3
        if count >= min_repeats:
            candidate = positions[pos] if pos < len(positions) else len(stripped)
            if best_cut is None or candidate < best_cut:
                best_cut = candidate

    if best_cut is not None:
        cleaned = text[:best_cut].rstrip(' ,，。.!！?？')
        return cleaned if cleaned else ""

    return text


# ── Prefix / overlap utilities ───────────────────────────

def common_prefix_len(a: str, b: str) -> int:
    """Return the length of the longest common prefix, tolerating case/punctuation."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] == b[i]:
            continue
        ca = a[i].lower().translate(_PUNCT_NORMALIZE)
        cb = b[i].lower().translate(_PUNCT_NORMALIZE)
        if ca != cb:
            return i
    return n


def prefix_overlap_ratio(a: str, b: str) -> float:
    """Character-bigram containment ratio after stripping punctuation.

    Computes |bigrams(a) ∩ bigrams(b)| / |bigrams(b)|, i.e. the fraction
    of b's bigrams that appear in a.
    """
    def _strip(s):
        return ''.join(c.lower() for c in s if c not in OVERLAP_STRIP_CHARS)

    sa, sb = _strip(a), _strip(b)
    if not sb:
        return 1.0
    if len(sb) < 2:
        return 1.0 if sb in sa else 0.0

    def _bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1))

    ba, bb = _bigrams(sa), _bigrams(sb)
    if not bb:
        return 1.0
    return len(ba & bb) / len(bb)


def snap_to_boundary(text: str, pos: int) -> int:
    """Snap a position back to the nearest sentence-ending punctuation."""
    if pos <= 0:
        return 0
    end = min(pos, len(text))
    for i in range(end - 1, max(0, end - 40) - 1, -1):
        if text[i] in '。！？.!?\n':
            return i + 1
    return end


def find_after_overlap(committed: str, raw: str, min_match: int = 6) -> str:
    """Find genuinely new content in *raw* that comes after *committed*.

    Uses aggressive normalization so Whisper's per-cycle wording variations
    don't break the overlap search.
    """
    if not committed or not raw:
        return raw

    norm_raw_chars = []
    raw_positions = []
    for i, ch in enumerate(raw):
        if ch not in OVERLAP_STRIP_CHARS:
            norm_raw_chars.append(ch.lower())
            raw_positions.append(i)
    norm_raw = ''.join(norm_raw_chars)

    norm_committed = ''.join(
        ch.lower() for ch in committed if ch not in OVERLAP_STRIP_CHARS
    )

    max_search = min(60, len(norm_committed))
    for tail_len in range(max_search, min_match - 1, -1):
        tail = norm_committed[-tail_len:]
        idx = norm_raw.find(tail)
        if idx >= 0:
            norm_end = idx + tail_len
            if norm_end >= len(raw_positions):
                return ""
            raw_end = raw_positions[norm_end]
            result = raw[raw_end:].lstrip(' ,，.。!！?？、')
            return result
    return raw


def find_after_sentence_overlap(committed: str, raw: str,
                                min_sent_len: int = 6) -> str | None:
    """Fallback overlap using individual sentence anchors from committed tail."""
    if not committed or not raw:
        return None

    sentences = re.split(r'(?<=[。！？.!?\n])', committed)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_sent_len]
    if not sentences:
        return None

    norm_raw_chars = []
    raw_positions = []
    for i, ch in enumerate(raw):
        if ch not in OVERLAP_STRIP_CHARS:
            norm_raw_chars.append(ch.lower())
            raw_positions.append(i)
    norm_raw = ''.join(norm_raw_chars)

    for sent in reversed(sentences[-5:]):
        norm_sent = ''.join(
            ch.lower() for ch in sent if ch not in OVERLAP_STRIP_CHARS
        )
        if len(norm_sent) < min_sent_len:
            continue
        idx = norm_raw.find(norm_sent)
        if idx >= 0:
            norm_end = idx + len(norm_sent)
            if norm_end >= len(raw_positions):
                return ""
            raw_end = raw_positions[norm_end]
            return raw[raw_end:].lstrip(' ,，.。!！?？、')

    return None


# ── Hallucination detection ──────────────────────────────

def hallucination_reason(text: str) -> str | None:
    """Return reason string if text is hallucination, else None."""
    lower = text.lower().strip(" .!,。，！")
    if lower in _HALLUCINATION_PHRASES:
        return "phrase"
    if text.lstrip().startswith('..'):
        return "garbled_prefix"
    for sub in _HALLUCINATION_SUBSTRINGS:
        if sub in text:
            return "substring"
    tokens = text.split()
    if len(tokens) >= 3:
        if len(set(tokens)) == 1:
            return "word_repeat"
        for i in range(len(tokens) - 2):
            if tokens[i] == tokens[i+1] == tokens[i+2]:
                return "word_repeat"
    if len(text) >= 6:
        for size in range(1, len(text) // 3 + 1):
            pat = text[:size]
            if pat * (len(text) // len(pat)) == text[:len(pat) * (len(text) // len(pat))] and len(text) // len(pat) >= 3:
                return "prefix_repeat"
    clean = ''.join(c for c in text if not c.isspace())
    if len(clean) >= 10:
        freq = {}
        for c in clean:
            freq[c] = freq.get(c, 0) + 1
        if max(freq.values()) / len(clean) > 0.4:
            return "dominant_char"
    if len(clean) >= 20:
        for n in (4, 6, 8):
            if len(clean) < n * 3:
                continue
            grams = {}
            for i in range(len(clean) - n + 1):
                gram = clean[i:i+n]
                grams[gram] = grams.get(gram, 0) + 1
            mx = max(grams.values())
            if mx >= 4 and mx * n > len(clean) * 0.4:
                return "phrase_repeat"
    if 5 <= len(clean) < 30:
        has_cjk = any(0x2E80 <= ord(c) <= 0x9FFF or 0xF900 <= ord(c) <= 0xFAFF
                       or 0x20000 <= ord(c) <= 0x2FA1F or 0x3040 <= ord(c) <= 0x30FF
                       for c in clean)
        if not has_cjk:
            return "no_cjk"
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith('L'):
            block = ord(ch)
            is_latin = block < 0x0250
            is_cjk = 0x2E80 <= block <= 0x9FFF or 0xF900 <= block <= 0xFAFF
            is_cjk_ext = 0x20000 <= block <= 0x2FA1F
            is_kana = 0x3040 <= block <= 0x30FF
            if not (is_latin or is_cjk or is_cjk_ext or is_kana):
                return "non_script"
    return None


def is_hallucination(text: str) -> bool:
    """Return True if text is detected as hallucination."""
    return hallucination_reason(text) is not None
