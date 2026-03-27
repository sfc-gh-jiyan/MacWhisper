"""Shared evaluation metrics for transcription quality.

Extracted from test_replay.py so both single-channel replay tests and
dual-channel meeting tests can reuse the same scoring functions.
"""

import re

from text_utils import OVERLAP_STRIP_CHARS


# ── Text normalisation ───────────────────────────────────────

def normalize(text):
    """Lowercase, strip punctuation/space for fuzzy comparison."""
    return ''.join(
        ch.lower() for ch in text
        if ch not in OVERLAP_STRIP_CHARS
    )


def tokenize_mixed(text):
    """Tokenize mixed CJK/Latin text: CJK chars individually, Latin words as tokens."""
    tokens = []
    current_latin = []
    for ch in text.lower():
        if ch in OVERLAP_STRIP_CHARS:
            if current_latin:
                tokens.append(''.join(current_latin))
                current_latin = []
            continue
        # CJK character ranges
        if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
            if current_latin:
                tokens.append(''.join(current_latin))
                current_latin = []
            tokens.append(ch)
        else:
            current_latin.append(ch)
    if current_latin:
        tokens.append(''.join(current_latin))
    return tokens


def split_sentences(text):
    """Split text into non-empty sentences."""
    parts = re.split(r'(?<=[。！？.!?\n])', text)
    return [s.strip() for s in parts if len(s.strip()) > 2]


# ── Core metrics ─────────────────────────────────────────────

def char_overlap(display, ground_truth):
    """Character-level overlap via LCS: ratio of GT chars found in display."""
    d = normalize(display)
    g = normalize(ground_truth)
    if not g:
        return 1.0
    m, n = len(g), len(d)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if g[i - 1] == d[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return round(prev[n] / m, 3)


def compute_wer(hypothesis, reference):
    """Compute WER using jiwer. Returns WER as float (0.0 = perfect)."""
    try:
        import jiwer
    except ImportError:
        return -1.0  # jiwer not installed

    ref_tokens = tokenize_mixed(reference)
    hyp_tokens = tokenize_mixed(hypothesis)

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    ref_str = ' '.join(ref_tokens)
    hyp_str = ' '.join(hyp_tokens)

    wer = jiwer.wer(ref_str, hyp_str)
    return round(wer, 3)


def compute_recall(hypothesis, reference):
    """Word-level recall: fraction of reference tokens found in hypothesis."""
    ref_tokens = tokenize_mixed(reference)
    hyp_tokens = set(tokenize_mixed(hypothesis))
    if not ref_tokens:
        return 1.0
    found = sum(1 for t in ref_tokens if t in hyp_tokens)
    return round(found / len(ref_tokens), 3)


def tier(value, fail_thresh, warn_thresh, pass_thresh, higher_is_better=True):
    """Classify a metric into FAIL/WARN/PASS/EXCELLENT tier."""
    if higher_is_better:
        if value < fail_thresh:
            return "FAIL"
        elif value < warn_thresh:
            return "WARN"
        elif value < pass_thresh:
            return "PASS"
        else:
            return "EXCELLENT"
    else:  # lower is better (e.g., WER, latency)
        if value > fail_thresh:
            return "FAIL"
        elif value > warn_thresh:
            return "WARN"
        elif value > pass_thresh:
            return "PASS"
        else:
            return "EXCELLENT"


def evaluate_dual_channel(transcript, ground_truth,
                          overlap_ratio=None, priority_stats=None):
    """Evaluate dual-channel transcription quality.

    Simplified version of test_replay.evaluate() — no snapshots needed,
    just transcript vs ground truth.

    Args:
        transcript: Transcribed text from MeetingSession.
        ground_truth: Combined ground truth from both WAV files.
        overlap_ratio: Optional float — fraction of total audio time where
            both sources were active simultaneously (0.0–1.0).
        priority_stats: Optional dict with keys "mic_wins", "sys_wins",
            "total_overlaps" — counts from RMS priority selection.

    Returns dict with metrics and tiered judgment.
    """
    if not transcript:
        return {
            "char_overlap": 0.0, "wer": 1.0, "recall": 0.0,
            "duplications": 0, "tier": "FAIL", "pass": False,
            "overlap_ratio": overlap_ratio,
            "priority_stats": priority_stats,
        }

    norm_display = normalize(transcript)

    # Character overlap (LCS-based)
    overlap = char_overlap(transcript, ground_truth)

    # WER
    wer = compute_wer(transcript, ground_truth)

    # Recall
    recall = compute_recall(transcript, ground_truth)

    # Completeness (sentence-level)
    gt_sents = split_sentences(ground_truth)
    found = sum(1 for s in gt_sents
                if len(normalize(s)) < 4 or normalize(s) in norm_display)
    completeness = round(found / len(gt_sents), 2) if gt_sents else 1.0

    # Duplications
    display_sents = split_sentences(transcript)
    seen = set()
    dup_count = 0
    for s in display_sents:
        ns = normalize(s)
        if len(ns) < 4:
            continue
        if ns in seen:
            dup_count += 1
        seen.add(ns)

    # Tiered judgment — relaxed for dual-channel
    tiers = {
        "wer":     tier(wer, 0.50, 0.30, 0.15, higher_is_better=False),
        "overlap": tier(overlap, 0.30, 0.50, 0.70, higher_is_better=True),
        "recall":  tier(recall, 0.30, 0.50, 0.70, higher_is_better=True),
        "dups":    tier(dup_count, 5, 3, 1, higher_is_better=False),
    }

    tier_order = ["FAIL", "WARN", "PASS", "EXCELLENT"]
    overall_tier = min(tiers.values(), key=lambda t: tier_order.index(t))
    passed = overall_tier in ("PASS", "EXCELLENT")

    return {
        "char_overlap": overlap,
        "wer": wer,
        "recall": recall,
        "completeness": completeness,
        "duplications": dup_count,
        "tiers": tiers,
        "tier": overall_tier,
        "pass": passed,
        "overlap_ratio": overlap_ratio,
        "priority_stats": priority_stats,
    }
