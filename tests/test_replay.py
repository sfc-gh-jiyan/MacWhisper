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

# Shared metrics (canonical implementations live in eval_metrics.py)
from tests.eval_metrics import (
    normalize as _normalize,
    tokenize_mixed as _tokenize_mixed,
    split_sentences as _split_sentences,
    char_overlap as _char_overlap,
    compute_wer as _compute_wer,
    compute_recall as _compute_recall,
    tier as _tier,
)

# ── Constants ─────────────────────────────────────────────────

BLOCKSIZE = 1024
FRAME_DURATION = BLOCKSIZE / SAMPLE_RATE  # ~0.064s per frame

_DATA_DIR = os.path.expanduser("~/.macwhisper")
AUDIO_DIR = os.path.join(_DATA_DIR, "audio")
TRANSCRIPT_LOG = os.path.join(_DATA_DIR, "transcripts.jsonl")
REPLAY_RESULTS = os.path.join(_DATA_DIR, "replay_results.jsonl")

# Processor defaults (match app.py)
MIN_CHUNK_SIZE = 0.5
MAX_BUFFER_S = 5.0            # must match app.py
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

    # Warmup: first mlx inference compiles Metal shaders (~5-15s).
    # Do this once before timing so it doesn't pollute freeze metrics.
    print("  Warming up model (first inference compiles Metal shaders)...")
    warmup_audio = np.zeros(int(sample_rate * 1.0), dtype=np.float32)
    backend.transcribe(warmup_audio, language=None, task="transcribe")
    print("  Warmup complete.")

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


# NOTE: _split_sentences, _normalize, _tokenize_mixed, _char_overlap,
# _compute_wer, _compute_recall, _tier are now imported from eval_metrics.py


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

    # ── Freeze / inference metrics ──
    freeze_metrics = _compute_freeze_metrics(snapshots)
    max_freeze = freeze_metrics["max_freeze_s"]
    max_infer = freeze_metrics["max_inference_ms"]

    tiers["max_freeze"] = _tier(max_freeze, 5.0, 3.0, 1.5, higher_is_better=False)
    tiers["max_inference"] = _tier(max_infer, 3000, 1500, 800, higher_is_better=False)

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
        **freeze_metrics,
        "tiers": tiers,
        "tier": overall_tier,
        "pass": passed,
    }


# ── Freeze / inference diagnostics ───────────────────────────

def _analyze_freezes(snapshots, threshold_s=2.0):
    """Detect freeze gaps (periods >threshold_s between overlay updates).

    A 'freeze' is any period where the user sees no change on screen.
    In the live loop, this happens when process_iter() is blocked by
    a slow Whisper inference call.
    """
    freezes = []
    if len(snapshots) < 2:
        return freezes
    for i in range(1, len(snapshots)):
        gap = snapshots[i]["wall_s"] - snapshots[i - 1]["wall_s"]
        if gap > threshold_s:
            freezes.append({
                "idx": i,
                "from_wall_s": snapshots[i - 1]["wall_s"],
                "to_wall_s": snapshots[i]["wall_s"],
                "gap_s": round(gap, 1),
                "from_audio_s": snapshots[i - 1]["audio_s"],
                "to_audio_s": snapshots[i]["audio_s"],
                "buffer_at_start": snapshots[i - 1]["debug"].get("buffer_s", 0),
                "inference_ms": snapshots[i]["debug"].get("inference_ms", 0),
            })
    return freezes


def _analyze_inference(snapshots):
    """Extract inference timing statistics from snapshots."""
    times = [s["debug"].get("inference_ms", 0) for s in snapshots if s["debug"].get("inference_ms")]
    if not times:
        return {"avg_ms": 0, "max_ms": 0, "p95_ms": 0, "count": 0,
                "histogram": {}}
    times_sorted = sorted(times)
    p95_idx = min(int(len(times_sorted) * 0.95), len(times_sorted) - 1)
    # Histogram buckets
    buckets = {"0-500ms": 0, "500-1000ms": 0, "1000-2000ms": 0, "2000ms+": 0}
    for t in times:
        if t <= 500:
            buckets["0-500ms"] += 1
        elif t <= 1000:
            buckets["500-1000ms"] += 1
        elif t <= 2000:
            buckets["1000-2000ms"] += 1
        else:
            buckets["2000ms+"] += 1
    return {
        "avg_ms": int(sum(times) / len(times)),
        "max_ms": max(times),
        "p95_ms": times_sorted[p95_idx],
        "count": len(times),
        "histogram": buckets,
    }


def _analyze_trims(snapshots):
    """Count and detail buffer trim events."""
    trims = []
    for s in snapshots:
        trim_info = s["debug"].get("trim")
        if trim_info and trim_info.get("trimmed"):
            trims.append({
                "iter": s["debug"].get("iter", "?"),
                "audio_s": s["audio_s"],
                "from_s": trim_info.get("from_s", 0),
                "to_s": trim_info.get("to_s", 0),
                "effective_max": trim_info.get("effective_max", 0),
                "retained_s": trim_info.get("retained_s", 0),
            })
    return trims


def _compute_freeze_metrics(snapshots):
    """Compute freeze-related metrics for tiered judgment."""
    freezes = _analyze_freezes(snapshots, threshold_s=1.5)
    gaps = []
    if len(snapshots) >= 2:
        for i in range(1, len(snapshots)):
            gaps.append(snapshots[i]["wall_s"] - snapshots[i - 1]["wall_s"])
    max_freeze = max(gaps) if gaps else 0.0
    freeze_count = len(freezes)
    inf_stats = _analyze_inference(snapshots)
    trim_info = _analyze_trims(snapshots)
    return {
        "max_freeze_s": round(max_freeze, 1),
        "freeze_count": freeze_count,
        "avg_inference_ms": inf_stats["avg_ms"],
        "max_inference_ms": inf_stats["max_ms"],
        "p95_inference_ms": inf_stats["p95_ms"],
        "trim_count": len(trim_info),
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
    print(f"  {'Max freeze':<16} {scores.get('max_freeze_s',0):>5.1f}s  {tiers.get('max_freeze','?'):<10}  F>5s W>3s P>1.5s E<1.5s")
    print(f"  {'Freeze count':<16} {scores.get('freeze_count',0):>8d}  {'':10}  (gaps > 1.5s)")
    print(f"  {'Avg inference':<16} {scores.get('avg_inference_ms',0):>5d}ms  {'':10}")
    print(f"  {'Max inference':<16} {scores.get('max_inference_ms',0):>5d}ms  {tiers.get('max_inference','?'):<10}  F>3s W>1.5s P>800ms E<800ms")
    print(f"  {'Trim count':<16} {scores.get('trim_count',0):>8d}")
    print(f"  {'Completeness':<16} {scores['completeness']:>7.0%}  {'':10}  ({found_count}/{len(gt_sents)} sentences)")
    if scores.get('offline_wer_delta', 0) != 0:
        print(f"  {'Offline delta':<16} {scores['offline_wer_delta']:>+7.1%}  {'':10}  (streaming WER - offline WER)")
    print(f"\n  Overall: {status}")
    print(f"{'=' * 70}")


# ── Deep diagnostic report ───────────────────────────────────

def print_diagnostic_report(snapshots, audio_file, duration):
    """Print deep per-second diagnostic: timeline, freezes, inference, trims.

    This is the 'X-ray' view — shows exactly where the pipeline stalls.
    """
    if not snapshots:
        print("  [DIAGNOSE] No snapshots to analyze.")
        return

    print(f"\n{'=' * 70}")
    print(f"  DIAGNOSTIC: {audio_file}")
    print(f"{'=' * 70}")

    # ── 1. Per-second timeline ──
    print(f"\n  --- Per-second timeline ---")
    print(f"  {'Sec':>4}  {'Event':<14}  {'Buf':>5}  {'Infer':>7}  {'Conf':>5}  {'New':>4}  Note")
    print(f"  {'─'*4}  {'─'*14}  {'─'*5}  {'─'*7}  {'─'*5}  {'─'*4}  {'─'*30}")

    # Build a second-by-second view from snapshots
    snap_by_second = {}  # second -> list of snapshots in that second
    for s in snapshots:
        sec = int(s["wall_s"])
        snap_by_second.setdefault(sec, []).append(s)

    max_wall = int(snapshots[-1]["wall_s"]) + 1 if snapshots else int(duration)
    for sec in range(max_wall + 1):
        if sec in snap_by_second:
            for s in snap_by_second[sec]:
                buf = s["debug"].get("buffer_s", 0)
                inf = s["debug"].get("inference_ms", 0)
                conf = s["debug"].get("confirmed_words", 0)
                new_w = s["debug"].get("newly_confirmed", 0)
                trim = s["debug"].get("trim")
                note = ""
                if inf > 2000:
                    note = "<< SLOW INFERENCE"
                elif inf > 1000:
                    note = "< slow"
                if trim and trim.get("trimmed"):
                    note += f" TRIM->{trim['retained_s']}s"
                print(f"  {sec:4d}  {'inference':14}  {buf:4.1f}s  {inf:6d}ms  {conf:5d}  {new_w:4d}  {note}")
        else:
            # No snapshot this second — the loop was blocked
            # Find the surrounding snapshots
            prev_s = None
            next_s = None
            for s in snapshots:
                if s["wall_s"] <= sec:
                    prev_s = s
                if s["wall_s"] > sec and next_s is None:
                    next_s = s
            if prev_s:
                buf_est = prev_s["debug"].get("buffer_s", 0)
                print(f"  {sec:4d}  {'... blocked':14}  {buf_est:4.1f}s  {'---':>7}  {'':>5}  {'':>4}  (no overlay update)")
            else:
                print(f"  {sec:4d}  {'... waiting':14}  {'':>5}  {'---':>7}  {'':>5}  {'':>4}")

    # ── 2. Freeze events ──
    freezes = _analyze_freezes(snapshots, threshold_s=1.5)
    print(f"\n  --- Freeze events (gaps > 1.5s between updates) ---")
    if freezes:
        print(f"  {'#':>3}  {'From':>6}  {'To':>6}  {'Gap':>5}  {'Buffer':>6}  {'Inference':>9}")
        print(f"  {'─'*3}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*9}")
        for i, f in enumerate(freezes):
            print(f"  {i+1:3d}  {f['from_wall_s']:5.1f}s  {f['to_wall_s']:5.1f}s  "
                  f"{f['gap_s']:4.1f}s  {f['buffer_at_start']:5.1f}s  {f['inference_ms']:8d}ms")
    else:
        print(f"  None detected -- no gaps > 1.5s")

    # ── 3. Inference histogram ──
    inf_stats = _analyze_inference(snapshots)
    print(f"\n  --- Inference timing ---")
    print(f"  Avg: {inf_stats['avg_ms']}ms  Max: {inf_stats['max_ms']}ms  "
          f"P95: {inf_stats['p95_ms']}ms  ({inf_stats['count']} iterations)")
    if inf_stats["histogram"]:
        print(f"  Histogram:")
        total = inf_stats["count"]
        for bucket, count in inf_stats["histogram"].items():
            bar = "#" * int(count / max(total, 1) * 40)
            print(f"    {bucket:<12}  {count:3d}  ({count/total*100:4.0f}%)  {bar}")

    # ── 4. Buffer trim events ──
    trims = _analyze_trims(snapshots)
    print(f"\n  --- Buffer trims ---")
    if trims:
        print(f"  {'#':>3}  {'Audio':>6}  {'Trim from':>10}  {'Trim to':>8}  {'MaxEff':>7}  {'Retained':>8}")
        print(f"  {'─'*3}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*8}")
        for i, t in enumerate(trims):
            print(f"  {i+1:3d}  {t['audio_s']:5.1f}s  {t['from_s']:9.1f}s  "
                  f"{t['to_s']:7.1f}s  {t['effective_max']:6.1f}s  {t['retained_s']:7.1f}s")
    else:
        print(f"  No trims occurred (buffer stayed within limit)")

    # ── 5. Buffer growth over time ──
    print(f"\n  --- Buffer size over time ---")
    buf_points = [(s["wall_s"], s["debug"].get("buffer_s", 0)) for s in snapshots]
    # Show as ASCII sparkline
    if buf_points:
        max_buf = max(b for _, b in buf_points)
        scale = 40 / max(max_buf, 1)
        for wall_s, buf_s in buf_points:
            bar = "#" * int(buf_s * scale)
            print(f"  {wall_s:5.1f}s  {buf_s:4.1f}s  {bar}")

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
    parser.add_argument("--diagnose", action="store_true",
                        help="Deep diagnostic: per-second timeline, freeze detection, inference histogram")
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

        if args.diagnose:
            print_diagnostic_report(snapshots, audio_file, duration)

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
