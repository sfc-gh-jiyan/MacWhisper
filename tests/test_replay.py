#!/usr/bin/env python3
"""Replay test for live subtitles — OnlineASRProcessor architecture.

Feeds saved WAV files through OnlineASRProcessor + MLXWhisperBackend
at real-time pace, collects confirmed/unconfirmed text over time,
and evaluates against ground truth transcripts.

Usage:
    venv/bin/python tests/test_replay.py                            # latest >31s WAV
    venv/bin/python tests/test_replay.py --wav 20260322_212421.wav  # specific file
    venv/bin/python tests/test_replay.py --top 5                    # 5 longest recordings
"""

import argparse
import json
import os
import re
import sys
import time
import wave

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asr_backend import MLXWhisperBackend
from online_processor import OnlineASRProcessor, SAMPLE_RATE
from text_utils import OVERLAP_STRIP_CHARS

# ── Constants ─────────────────────────────────────────────────

BLOCKSIZE = 1024
FRAME_DURATION = BLOCKSIZE / SAMPLE_RATE  # ~0.064s per frame

_DATA_DIR = os.path.expanduser("~/.macwhisper")
AUDIO_DIR = os.path.join(_DATA_DIR, "audio")
TRANSCRIPT_LOG = os.path.join(_DATA_DIR, "transcripts.jsonl")
REPLAY_RESULTS = os.path.join(_DATA_DIR, "replay_results.jsonl")

# Processor defaults (match app.py)
MIN_CHUNK_SIZE = 0.7
MAX_BUFFER_S = 20.0
DEFAULT_MODEL = "mlx-community/whisper-medium-mlx"


# ── WAV loader ────────────────────────────────────────────────

def load_wav_as_float32(wav_path):
    """Load a WAV file and return float32 array + metadata."""
    with wave.open(wav_path, "rb") as wf:
        assert wf.getsampwidth() == 2, f"Expected 16-bit WAV, got {wf.getsampwidth()*8}-bit"
        assert wf.getnchannels() == 1, f"Expected mono WAV, got {wf.getnchannels()} channels"
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)
        sr = wf.getframerate()
    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    duration = len(audio_float) / sr
    return audio_float, sr, duration


# ── Replay engine ─────────────────────────────────────────────

def replay_wav(audio_float, sample_rate, model=DEFAULT_MODEL,
               min_chunk_size=MIN_CHUNK_SIZE, max_buffer_s=MAX_BUFFER_S):
    """Feed audio through OnlineASRProcessor at real-time pace.

    Returns:
        snapshots: list of dicts with confirmed/unconfirmed text at each iteration
        final_confirmed: final confirmed text after segment_close()
        final_all: final confirmed + unconfirmed text
        elapsed: wall-clock time for replay
    """
    backend = MLXWhisperBackend(model_repo=model)
    proc = OnlineASRProcessor(
        backend=backend,
        vad=None,
        min_chunk_size=min_chunk_size,
        max_buffer_s=max_buffer_s,
        language=None,
    )

    snapshots = []
    chunk_samples = BLOCKSIZE
    total_samples = len(audio_float)
    pos = 0

    t0 = time.time()

    while pos < total_samples:
        # Feed one chunk
        end = min(pos + chunk_samples, total_samples)
        chunk = audio_float[pos:end]
        proc.insert_audio_chunk(chunk)
        pos = end

        # Run process_iter on each chunk (processor's own throttle handles timing)
        result = proc.process_iter()
        if result is not None:
            confirmed, unconfirmed = result
            audio_s = pos / sample_rate
            snapshots.append({
                "audio_s": round(audio_s, 2),
                "wall_s": round(time.time() - t0, 2),
                "confirmed": confirmed,
                "unconfirmed": unconfirmed,
                "confirmed_len": len(confirmed),
                "unconfirmed_len": len(unconfirmed),
                "debug": dict(proc.last_debug),
            })

        # Real-time pacing
        time.sleep(FRAME_DURATION)

    # Let last inference complete
    time.sleep(3)

    # Final: force-close segment to confirm all remaining words
    final_confirmed = proc.segment_close()
    final_all_words = proc.get_all_words()
    final_all = "".join(w[2] for w in final_all_words) if final_all_words else final_confirmed

    elapsed = time.time() - t0

    return snapshots, final_confirmed, final_all, elapsed


# ── Offline baseline ──────────────────────────────────────────

def generate_offline_baseline(audio_float, model=DEFAULT_MODEL):
    """Run a single offline transcription on entire audio as best-case reference."""
    backend = MLXWhisperBackend(model_repo=model)
    from text_utils import BILINGUAL_PROMPT
    result = backend.transcribe(audio_float, language=None,
                                initial_prompt=BILINGUAL_PROMPT, task="transcribe")
    return result.text.strip()


# ── Evaluation ────────────────────────────────────────────────

def _split_sentences(text):
    """Split text into non-empty sentences."""
    parts = re.split(r'(?<=[。！？.!?\n])', text)
    return [s.strip() for s in parts if len(s.strip()) > 2]


def _normalize(text):
    """Lowercase, strip punctuation/space for fuzzy comparison."""
    return ''.join(
        ch.lower() for ch in text
        if ch not in OVERLAP_STRIP_CHARS
    )


def _tokenize_mixed(text):
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


def _char_overlap(display, ground_truth):
    """Character-level overlap via LCS: ratio of GT chars found in display."""
    d = _normalize(display)
    g = _normalize(ground_truth)
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


def _compute_wer(hypothesis, reference):
    """Compute WER using jiwer. Returns WER as float (0.0 = perfect)."""
    try:
        import jiwer
    except ImportError:
        return -1.0  # jiwer not installed

    # Tokenize for mixed CJK/Latin
    ref_tokens = _tokenize_mixed(reference)
    hyp_tokens = _tokenize_mixed(hypothesis)

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    # jiwer expects space-separated strings
    ref_str = ' '.join(ref_tokens)
    hyp_str = ' '.join(hyp_tokens)

    wer = jiwer.wer(ref_str, hyp_str)
    return round(wer, 3)


def _compute_recall(hypothesis, reference):
    """Word-level recall: fraction of reference tokens found in hypothesis."""
    ref_tokens = _tokenize_mixed(reference)
    hyp_tokens = set(_tokenize_mixed(hypothesis))
    if not ref_tokens:
        return 1.0
    found = sum(1 for t in ref_tokens if t in hyp_tokens)
    return round(found / len(ref_tokens), 3)


def _compute_avg_latency(snapshots):
    """Estimate average confirmation latency from snapshots.

    Latency = how many seconds of additional audio were fed between when
    a chunk of text first appeared as unconfirmed and when it was confirmed.
    This measures the *audio-time* lag, not wall-clock processing delay.
    """
    latencies = []
    prev_conf_len = 0
    # Track when each confirmed-char-count was first seen as unconfirmed
    # We approximate: when confirmed grows, the new chars were likely first
    # available as unconfirmed ~2 iterations before (LocalAgreement needs 2 matches)
    first_audio_at_conf_len = {}  # conf_len -> audio_s when first reachable

    for s in snapshots:
        total_len = s["confirmed_len"] + s["unconfirmed_len"]
        # Record the earliest audio_s where we had at least this many total chars
        if total_len not in first_audio_at_conf_len:
            first_audio_at_conf_len[total_len] = s["audio_s"]
        # Also record for all lengths up to total_len that we haven't seen
        # (chars are confirmed in batches, so intermediate lengths may be skipped)

        conf_len = s["confirmed_len"]
        if conf_len > prev_conf_len:
            # New chars confirmed at this snapshot's audio_s
            # Find earliest audio_s where total chars >= conf_len
            earliest = s["audio_s"]  # fallback: this snapshot
            for tl, a_s in first_audio_at_conf_len.items():
                if tl >= conf_len and a_s < earliest:
                    earliest = a_s
            latency = s["audio_s"] - earliest
            if latency >= 0:
                latencies.append(latency)
        prev_conf_len = conf_len

    if not latencies:
        return 0.0, 0.0
    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
    return round(avg, 1), round(latencies[p95_idx], 1)


def _tier(value, fail_thresh, warn_thresh, pass_thresh, higher_is_better=True):
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


def evaluate(snapshots, final_confirmed, ground_truth, offline_baseline=None):
    """Evaluate replay quality against ground truth.

    Returns dict with all metrics from INTERNAL_subtitle_research.md §7.6.
    """
    if not snapshots and not final_confirmed:
        return {"total_iters": 0, "completeness": 0.0, "char_overlap": 0.0,
                "wer": 1.0, "recall": 0.0, "duplications": 0,
                "stability_jumps": 0, "avg_latency": 0, "p95_latency": 0,
                "offline_wer_delta": 0, "tier": "FAIL", "pass": False}

    display = final_confirmed
    norm_display = _normalize(display)

    # ── Completeness (sentence-level) ──
    gt_sents = _split_sentences(ground_truth)
    found = sum(1 for s in gt_sents
                if len(_normalize(s)) < 4 or _normalize(s) in norm_display)
    completeness = round(found / len(gt_sents), 2) if gt_sents else 1.0

    # ── Character overlap (LCS-based) ──
    overlap = _char_overlap(display, ground_truth)

    # ── WER/CER (standard metric via jiwer) ──
    wer = _compute_wer(display, ground_truth)

    # ── Recall (word-level coverage) ──
    recall = _compute_recall(display, ground_truth)

    # ── Duplications ──
    display_sents = _split_sentences(display)
    seen = set()
    dup_count = 0
    for s in display_sents:
        ns = _normalize(s)
        if len(ns) < 4:
            continue
        if ns in seen:
            dup_count += 1
        seen.add(ns)

    # ── Stability (confirmed text should only grow) ──
    jumps = 0
    for i in range(1, len(snapshots)):
        prev_c = snapshots[i - 1]["confirmed"]
        curr_c = snapshots[i]["confirmed"]
        check = min(20, len(prev_c))
        if check > 0 and not curr_c.startswith(prev_c[:check]):
            jumps += 1

    # ── Latency ──
    avg_latency, p95_latency = _compute_avg_latency(snapshots)

    # ── Offline baseline delta ──
    offline_wer_delta = 0.0
    if offline_baseline:
        offline_wer = _compute_wer(offline_baseline, ground_truth)
        if offline_wer >= 0 and wer >= 0:
            offline_wer_delta = round(wer - offline_wer, 3)

    # ── Tiered judgment (from research report §7.6) ──
    tiers = {
        "wer":       _tier(wer, 0.30, 0.15, 0.05, higher_is_better=False),
        "overlap":   _tier(overlap, 0.60, 0.75, 0.90, higher_is_better=True),
        "recall":    _tier(recall, 0.70, 0.85, 0.95, higher_is_better=True),
        "dups":      _tier(dup_count, 3, 1, 0.5, higher_is_better=False),
        "jumps":     _tier(jumps, 10, 5, 1, higher_is_better=False),
        "latency":   _tier(avg_latency, 5, 3, 1, higher_is_better=False),
    }
    # Overall tier: worst of all
    tier_order = ["FAIL", "WARN", "PASS", "EXCELLENT"]
    overall_tier = min(tiers.values(), key=lambda t: tier_order.index(t))

    passed = overall_tier in ("PASS", "EXCELLENT")
    return {
        "total_iters": len(snapshots),
        "completeness": completeness,
        "char_overlap": overlap,
        "wer": wer,
        "recall": recall,
        "duplications": dup_count,
        "stability_jumps": jumps,
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "offline_wer_delta": offline_wer_delta,
        "tiers": tiers,
        "tier": overall_tier,
        "pass": passed,
    }


# ── Load test corpus ──────────────────────────────────────────

def load_transcript_pairs(filter_wav=None, min_duration=0):
    """Load WAV + transcript pairs from history."""
    if not os.path.exists(TRANSCRIPT_LOG):
        print(f"[ERROR] No transcript log at {TRANSCRIPT_LOG}")
        return []

    pairs = []
    seen = set()
    with open(TRANSCRIPT_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            audio_file = entry.get("audio_file", "")
            text = entry.get("text", "")
            duration = entry.get("duration_s", 0)

            if not audio_file or not text:
                continue
            if audio_file in seen:
                continue
            if filter_wav and audio_file != filter_wav:
                continue
            if duration < min_duration:
                continue

            wav_path = os.path.join(AUDIO_DIR, audio_file)
            if not os.path.exists(wav_path):
                continue

            seen.add(audio_file)
            pairs.append({
                "audio_file": audio_file,
                "wav_path": wav_path,
                "ground_truth": text,
                "duration_s": duration,
            })

    return pairs


# ── Pretty printer ────────────────────────────────────────────

def print_report(snapshots, final_confirmed, ground_truth, scores,
                 audio_file, duration, elapsed):
    """Print a clean, readable report of the replay results."""
    print(f"\n{'=' * 70}")
    print(f"  Replay: {audio_file} ({duration:.1f}s audio, {elapsed:.0f}s wall)")
    print(f"{'=' * 70}")

    if not snapshots:
        print("  No iterations produced output.")
        return

    # Iteration summary table
    print(f"\n  {'#':>3}  {'Audio':>6}  {'Wall':>6}  {'Conf':>5}  {'Unconf':>6}  "
          f"{'Infer':>6}  Confirmed tail")
    print(f"  {'─'*3}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*30}")

    for i, s in enumerate(snapshots):
        conf_len = s["confirmed_len"]
        unconf_len = s["unconfirmed_len"]
        audio_s = s["audio_s"]
        wall_s = s["wall_s"]
        infer_ms = s["debug"].get("inference_ms", 0)

        # Show last 30 chars of confirmed text
        conf_tail = s["confirmed"][-30:] if s["confirmed"] else "(empty)"
        if len(s["confirmed"]) > 30:
            conf_tail = "..." + conf_tail

        print(f"  {i+1:3d}  {audio_s:5.1f}s  {wall_s:5.1f}s  {conf_len:4d}c  "
              f"{unconf_len:5d}c  {infer_ms:5d}ms  {conf_tail}")

    # Side-by-side: Ground truth vs Final confirmed
    print(f"\n  ┌─ Ground Truth ({len(ground_truth)} chars) ──")
    for j in range(0, len(ground_truth), 68):
        print(f"  │ {ground_truth[j:j+68]}")

    print(f"  ├─ Final Confirmed ({len(final_confirmed)} chars) ──")
    for j in range(0, len(final_confirmed), 68):
        print(f"  │ {final_confirmed[j:j+68]}")

    # Show last unconfirmed if any
    if snapshots and snapshots[-1]["unconfirmed"]:
        last_unconf = snapshots[-1]["unconfirmed"]
        print(f"  ├─ Last Unconfirmed ({len(last_unconf)} chars, gray in overlay) ──")
        for j in range(0, len(last_unconf), 68):
            print(f"  │ {last_unconf[j:j+68]}")

    # Scores
    _tier_icons = {"FAIL": "x", "WARN": "~", "PASS": "+", "EXCELLENT": "*"}
    tiers = scores.get("tiers", {})
    status = scores["tier"]

    gt_sents = _split_sentences(ground_truth)
    norm_display = _normalize(final_confirmed)
    found_count = sum(1 for s in gt_sents
                      if len(_normalize(s)) < 4 or _normalize(s) in norm_display)

    print(f"  {chr(9492)}{'─'*42}")
    print(f"\n  Quality Metrics (tiered per research report):")
    print(f"  {'Metric':<16} {'Value':>8}  {'Tier':<10}  Thresholds")
    print(f"  {'─'*16} {'─'*8}  {'─'*10}  {'─'*30}")
    print(f"  {'WER':<16} {scores['wer']:>7.1%}  {tiers.get('wer','?'):<10}  F>30% W>15% P>5% E<5%")
    print(f"  {'Char overlap':<16} {scores['char_overlap']:>7.1%}  {tiers.get('overlap','?'):<10}  F<60% W<75% P<90% E>90%")
    print(f"  {'Recall':<16} {scores['recall']:>7.1%}  {tiers.get('recall','?'):<10}  F<70% W<85% P<95% E>95%")
    print(f"  {'Duplications':<16} {scores['duplications']:>8d}  {tiers.get('dups','?'):<10}  F>=3 W>=1 P=0")
    print(f"  {'Stability':<16} {scores['stability_jumps']:>8d}  {tiers.get('jumps','?'):<10}  F>10 W>5 P>1 E<=1")
    print(f"  {'Avg latency':<16} {scores['avg_latency']:>6.1f}s  {tiers.get('latency','?'):<10}  F>5s W>3s P>1s E<1s")
    print(f"  {'P95 latency':<16} {scores['p95_latency']:>6.1f}s")
    print(f"  {'Completeness':<16} {scores['completeness']:>7.0%}  {'':10}  ({found_count}/{len(gt_sents)} sentences)")
    if scores.get('offline_wer_delta', 0) != 0:
        print(f"  {'Offline delta':<16} {scores['offline_wer_delta']:>+7.1%}  {'':10}  (streaming WER - offline WER)")
    print(f"\n  Overall: {status}")
    print(f"{'=' * 70}")


# ── Batch summary ─────────────────────────────────────────────

def print_batch_summary(all_results):
    """Print a summary table for batch runs."""
    print(f"\n{'=' * 80}")
    print(f"  BATCH SUMMARY -- {len(all_results)} recordings")
    print(f"{'=' * 80}")

    print(f"\n  {'File':<24}  {'Dur':>5}  {'WER':>6}  {'Overlap':>8}  {'Recall':>7}  "
          f"{'Dups':>4}  {'Jumps':>5}  {'Lat':>4}  {'Tier':<10}")
    print(f"  {'─'*24}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*4}  {'─'*5}  {'─'*4}  {'─'*10}")

    pass_count = 0
    total_overlap = 0
    total_wer = 0
    for r in all_results:
        af = r["audio_file"]
        dur = r["duration_s"]
        ov = r["char_overlap"]
        wer = r.get("wer", -1)
        recall = r.get("recall", 0)
        dups = r["duplications"]
        jumps = r["stability_jumps"]
        avg_lat = r.get("avg_latency", 0)
        tier = r.get("tier", "?")
        passed = r["pass"]
        if passed:
            pass_count += 1
        total_overlap += ov
        total_wer += max(wer, 0)

        print(f"  {af:<24}  {dur:4.0f}s  {wer:5.1%}  {ov:7.1%}  {recall:6.1%}  "
              f"{dups:4d}  {jumps:5d}  {avg_lat:3.0f}s  {tier:<10}")

    n = len(all_results)
    avg_overlap = total_overlap / n if n else 0
    avg_wer = total_wer / n if n else 0
    print(f"\n  Average WER:     {avg_wer:.1%}")
    print(f"  Average overlap: {avg_overlap:.1%}")
    print(f"  Pass rate:       {pass_count}/{n}")
    print(f"{'=' * 80}")

    return pass_count == n


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay test for live subtitles (OnlineASRProcessor)")
    parser.add_argument("--wav", help="Test a specific WAV file (filename only)")
    parser.add_argument("--top", type=int, default=0,
                        help="Run the N longest recordings (default: 1)")
    parser.add_argument("--min-duration", type=float, default=31,
                        help="Minimum recording duration in seconds (default: 31)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--min-chunk", type=float, default=MIN_CHUNK_SIZE,
                        help=f"min_chunk_size for processor (default: {MIN_CHUNK_SIZE})")
    parser.add_argument("--max-buffer", type=float, default=MAX_BUFFER_S,
                        help=f"max_buffer_s for processor (default: {MAX_BUFFER_S})")
    parser.add_argument("--no-offline", action="store_true",
                        help="Skip offline baseline generation (faster)")
    args = parser.parse_args()

    # Load pairs
    min_dur = 0 if args.wav else args.min_duration
    pairs = load_transcript_pairs(filter_wav=args.wav, min_duration=min_dur)

    if not pairs:
        print("[WARN] No matching WAV + transcript pairs found.")
        print(f"       Audio dir:    {AUDIO_DIR}")
        print(f"       Transcripts:  {TRANSCRIPT_LOG}")
        if args.wav:
            print(f"       Filter: --wav {args.wav}")
        return

    # Select recordings
    pairs.sort(key=lambda p: -p["duration_s"])
    if args.wav:
        selected = pairs[:1]
    elif args.top > 0:
        selected = pairs[:args.top]
    else:
        # Default: latest >31s recording
        selected = [max(pairs, key=lambda p: p["audio_file"])]

    print(f"Selected {len(selected)} recording(s) for replay test")
    print(f"Model: {args.model}")
    print(f"Params: min_chunk={args.min_chunk}s, max_buffer={args.max_buffer}s\n")

    all_results = []
    for idx, pair in enumerate(selected):
        audio_file = pair["audio_file"]
        duration = pair["duration_s"]

        print(f"[{idx+1}/{len(selected)}] Loading {audio_file} ({duration:.1f}s)...")
        audio_float, sr, actual_dur = load_wav_as_float32(pair["wav_path"])

        print(f"  Replaying at real-time pace ({actual_dur:.0f}s expected)...")
        print(f"  (Whisper inference logs below are from the REAL code path)\n")

        snapshots, final_confirmed, final_all, elapsed = replay_wav(
            audio_float, sr,
            model=args.model,
            min_chunk_size=args.min_chunk,
            max_buffer_s=args.max_buffer,
        )

        # Offline baseline (optional, for WER delta comparison)
        offline_baseline = None
        if not args.no_offline:
            print(f"  Generating offline baseline...")
            offline_baseline = generate_offline_baseline(audio_float, model=args.model)

        scores = evaluate(snapshots, final_confirmed, pair["ground_truth"],
                          offline_baseline=offline_baseline)
        scores["audio_file"] = audio_file
        scores["duration_s"] = duration
        scores["replay_time_s"] = round(elapsed, 1)
        scores["final_confirmed_len"] = len(final_confirmed)
        scores["ground_truth_len"] = len(pair["ground_truth"])

        print_report(snapshots, final_confirmed, pair["ground_truth"],
                     scores, audio_file, duration, elapsed)

        all_results.append(scores)

    # Batch summary
    if len(all_results) > 1:
        all_pass = print_batch_summary(all_results)
    else:
        all_pass = all_results[0]["pass"] if all_results else False

    # Save results
    with open(REPLAY_RESULTS, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {REPLAY_RESULTS}")

    # Exit code for CI
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
