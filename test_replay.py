#!/usr/bin/env python3
"""Automated replay test for live subtitle quality.

Replays saved WAV files through the live subtitle pipeline (streaming
simulation) and compares final display text against the ground-truth
full transcript. No microphone or GUI needed.

Usage:
    venv/bin/python test_replay.py                          # all WAVs with transcripts
    venv/bin/python test_replay.py --wav 30260322_183047.wav  # single file
    venv/bin/python test_replay.py --long-only               # only recordings > 30s
    venv/bin/python test_replay.py --min-duration 10         # only recordings > 10s
"""

import argparse
import json
import os
import re
import sys
import time
import wave

import numpy as np

# ── Mock macOS-only modules before importing app ──────────────

from unittest.mock import MagicMock

# rumps.App must be a real class so TranscriberApp can subclass it
class _FakeRumpsApp:
    def __init__(self, *args, **kwargs):
        self.title = ""
        self.menu = {}

_rumps_mock = MagicMock()
_rumps_mock.App = _FakeRumpsApp

_MOCK_MODULES = {
    "AppKit": MagicMock(),
    "PyObjCTools": MagicMock(),
    "PyObjCTools.AppHelper": MagicMock(),
    "rumps": _rumps_mock,
    "pynput": MagicMock(),
    "pynput.keyboard": MagicMock(),
    "sounddevice": MagicMock(),
    "pyperclip": MagicMock(),
    "ApplicationServices": MagicMock(),
}
for mod, mock in _MOCK_MODULES.items():
    sys.modules[mod] = mock

import app  # noqa: E402 — must come after mocks


# ── Constants ─────────────────────────────────────────────────

SAMPLE_RATE = app.SAMPLE_RATE          # 16000
BLOCKSIZE = 1024
CHUNK_SECONDS = app.LIVE_CHUNK_SECONDS  # 3
MAX_WINDOW = app.MAX_LIVE_WINDOW        # 30

AUDIO_DIR = app.AUDIO_DIR
TRANSCRIPT_LOG = app.TRANSCRIPT_LOG
REPLAY_RESULTS = os.path.join(os.path.dirname(TRANSCRIPT_LOG), "replay_results.jsonl")


# ── WAV loader ────────────────────────────────────────────────

def load_wav_as_frames(wav_path):
    """Load a WAV file and split into (BLOCKSIZE, 1) int16 frames."""
    with wave.open(wav_path, "rb") as wf:
        assert wf.getsampwidth() == 2, f"Expected 16-bit WAV, got {wf.getsampwidth()*8}-bit"
        assert wf.getnchannels() == 1, f"Expected mono WAV, got {wf.getnchannels()} channels"
        raw_bytes = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, 1)
    frames = []
    for i in range(0, len(audio), BLOCKSIZE):
        chunk = audio[i:i + BLOCKSIZE]
        if len(chunk) < BLOCKSIZE:
            # Pad last frame with silence
            padded = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            padded[:len(chunk)] = chunk
            chunk = padded
        frames.append(chunk)
    return frames


# ── Stream simulator ──────────────────────────────────────────

def create_headless_instance(model="mlx-community/whisper-medium-mlx"):
    """Create a minimal TranscriberApp with just the state needed for replay."""
    inst = app.TranscriberApp.__new__(app.TranscriberApp)
    inst.current_model = model
    inst._committed_text = ""
    inst._prev_raw_text = ""
    inst._stable_prefix_len = 0
    inst._stable_cycles = 0
    inst._segment_start_frame = 0
    inst._pause_silence_frames = 0
    inst._pause_detected = False
    inst._segment_committed_text = ""
    inst._last_live_result = ""
    return inst


def simulate_stream(frames, inst):
    """Simulate live subtitle streaming through a WAV's frames.

    Includes pause detection and segment commit logic matching _live_chunk_loop.
    Returns a list of dicts: [{cycle, audio_s, raw, display, elapsed_s}, ...]
    """
    max_frames_count = int(MAX_WINDOW * SAMPLE_RATE / BLOCKSIZE)
    frames_per_chunk = int(CHUNK_SECONDS * SAMPLE_RATE / BLOCKSIZE)
    pause_silence_threshold = int(app.PAUSE_MIN_DURATION * SAMPLE_RATE / BLOCKSIZE)

    entries = []
    accumulated = []
    cycle = 0

    for i, frame in enumerate(frames):
        accumulated.append(frame)

        # Pause detection per frame (mirrors audio callback)
        rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
        if rms < app.PAUSE_RMS_THRESHOLD:
            inst._pause_silence_frames += 1
        else:
            inst._pause_silence_frames = 0

        seg_frames = len(accumulated) - inst._segment_start_frame
        seg_secs = seg_frames * BLOCKSIZE / SAMPLE_RATE
        if (inst._pause_silence_frames >= pause_silence_threshold
                and seg_secs >= app.PAUSE_MIN_SEGMENT
                and not inst._pause_detected):
            inst._pause_detected = True

        is_chunk_boundary = (i + 1) % frames_per_chunk == 0
        is_last_frame = i == len(frames) - 1

        if is_chunk_boundary or is_last_frame:
            seg_start = inst._segment_start_frame
            n = len(accumulated)
            seg_secs = (n - seg_start) * BLOCKSIZE / SAMPLE_RATE

            # Check for pause commit or safety fallback
            needs_commit = inst._pause_detected or seg_secs >= app.PAUSE_SAFETY_FALLBACK

            if needs_commit and inst._last_live_result:
                # Direct assignment: _last_live_result already includes history prefix
                inst._segment_committed_text = inst._last_live_result
                reason = "pause" if inst._pause_detected else f"safety@{seg_secs:.0f}s"
                print(f"  [SEGMENT COMMIT ({reason})]: "
                      f"{len(inst._segment_committed_text)} chars total")
                inst._segment_start_frame = n
                inst._committed_text = ""
                inst._prev_raw_text = ""
                inst._stable_prefix_len = 0
                inst._stable_cycles = 0
                inst._last_live_result = ""
                inst._pause_detected = False
                inst._pause_silence_frames = 0
                continue

            # Take snapshot from current segment only, capped at max_frames
            seg_begin = max(seg_start, n - max_frames_count)
            snapshot = accumulated[seg_begin:n]
            audio_s = round(len(snapshot) * BLOCKSIZE / SAMPLE_RATE, 1)

            t0 = time.time()
            raw = inst._do_live_transcribe(snapshot)
            elapsed = round(time.time() - t0, 2)

            if raw:
                cycle += 1
                display = inst._build_display_text(raw)
                inst._last_live_result = display
                entries.append({
                    "cycle": cycle,
                    "audio_s": audio_s,
                    "raw": raw,
                    "display": display,
                    "elapsed_s": elapsed,
                })

    return entries


# ── Evaluation ────────────────────────────────────────────────

def _split_sentences(text):
    """Split text into non-empty sentences."""
    parts = re.split(r'(?<=[。！？.!?\n])', text)
    return [s.strip() for s in parts if len(s.strip()) > 2]


def _normalize_for_compare(text):
    """Lowercase, strip punctuation/space for fuzzy comparison."""
    return ''.join(
        ch.lower() for ch in text
        if ch not in app._OVERLAP_STRIP_CHARS
    )


def evaluate(entries, ground_truth):
    """Evaluate replay quality against ground truth transcript."""
    if not entries:
        return {
            "total_cycles": 0,
            "completeness": 0.0,
            "duplications": 0,
            "stability_jumps": 0,
            "pass": False,
            "notes": "No transcription cycles produced",
        }

    final_display = entries[-1]["display"]
    norm_display = _normalize_for_compare(final_display)

    # ── Completeness: how many ground truth sentences appear in final display
    gt_sentences = _split_sentences(ground_truth)
    found = 0
    for s in gt_sentences:
        norm_s = _normalize_for_compare(s)
        if len(norm_s) < 4:
            found += 1  # skip very short segments
            continue
        if norm_s in norm_display:
            found += 1
    completeness = round(found / len(gt_sentences), 2) if gt_sentences else 1.0

    # ── Duplication: count repeated sentences in final display
    display_sentences = _split_sentences(final_display)
    seen = set()
    dup_count = 0
    for s in display_sentences:
        norm_s = _normalize_for_compare(s)
        if len(norm_s) < 4:
            continue
        if norm_s in seen:
            dup_count += 1
        seen.add(norm_s)

    # ── Stability: count large display jumps between consecutive cycles
    jumps = 0
    for i in range(1, len(entries)):
        prev_d = entries[i - 1]["display"]
        curr_d = entries[i]["display"]
        # A "jump" is when current display doesn't start with previous display's first 20 chars
        check_len = min(20, len(prev_d))
        if check_len > 0 and not curr_d.startswith(prev_d[:check_len]):
            jumps += 1

    passed = completeness >= 0.6 and dup_count == 0

    return {
        "total_cycles": len(entries),
        "completeness": completeness,
        "duplications": dup_count,
        "stability_jumps": jumps,
        "pass": passed,
        "final_display_tail": final_display[-120:] if len(final_display) > 120 else final_display,
        "ground_truth_tail": ground_truth[-120:] if len(ground_truth) > 120 else ground_truth,
    }


# ── Load test corpus ──────────────────────────────────────────

def load_transcript_pairs(filter_wav=None, min_duration=0):
    """Load WAV + transcript pairs from history."""
    if not os.path.exists(TRANSCRIPT_LOG):
        print(f"[ERROR] No transcript log at {TRANSCRIPT_LOG}")
        return []

    pairs = []
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
            if filter_wav and audio_file != filter_wav:
                continue
            if duration < min_duration:
                continue

            wav_path = os.path.join(AUDIO_DIR, audio_file)
            if not os.path.exists(wav_path):
                continue

            pairs.append({
                "audio_file": audio_file,
                "wav_path": wav_path,
                "ground_truth": text,
                "duration_s": duration,
            })

    return pairs


# ── Main ──────────────────────────────────────────────────────

def run_replay(pair, verbose=True):
    """Run a single replay test and return results."""
    audio_file = pair["audio_file"]
    duration = pair["duration_s"]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Replay: {audio_file} ({duration:.1f}s)")
        print(f"{'=' * 60}")

    frames = load_wav_as_frames(pair["wav_path"])
    inst = create_headless_instance()

    t0 = time.time()
    entries = simulate_stream(frames, inst)
    total_time = round(time.time() - t0, 1)

    scores = evaluate(entries, pair["ground_truth"])
    scores["audio_file"] = audio_file
    scores["duration_s"] = duration
    scores["replay_time_s"] = total_time

    if verbose:
        status = "PASS" if scores["pass"] else "WARN"
        print(f"  Cycles:       {scores['total_cycles']}")
        print(f"  Completeness: {scores['completeness']}")
        print(f"  Duplications: {scores['duplications']}")
        print(f"  Stability:    {scores['stability_jumps']} jump(s)")
        print(f"  Replay time:  {total_time}s")
        print(f"  Final display: ...{scores.get('final_display_tail', '')}")
        print(f"  Ground truth:  ...{scores.get('ground_truth_tail', '')}")
        print(f"  Result: {status}")

    return scores, entries


def main():
    parser = argparse.ArgumentParser(description="Replay test for live subtitles")
    parser.add_argument("--wav", help="Test a specific WAV file (filename only)")
    parser.add_argument("--long-only", action="store_true", help="Only test recordings > 30s")
    parser.add_argument("--min-duration", type=float, default=31, help="Minimum duration in seconds (default: 31s)")
    parser.add_argument("--save", action="store_true", default=True, help="Save results to replay_results.jsonl")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print detailed output")
    args = parser.parse_args()

    min_dur = 30.0 if args.long_only else args.min_duration
    pairs = load_transcript_pairs(filter_wav=args.wav, min_duration=min_dur)

    if not pairs:
        print("[WARN] No matching WAV + transcript pairs found.")
        print(f"       Looked in: {AUDIO_DIR}")
        print(f"       Transcripts: {TRANSCRIPT_LOG}")
        if args.wav:
            print(f"       Filter: --wav {args.wav}")
        if min_dur > 0:
            print(f"       Min duration: {min_dur}s")
        return

    print(f"Found {len(pairs)} test pair(s)")

    all_results = []
    pass_count = 0

    for pair in pairs:
        scores, entries = run_replay(pair, verbose=args.verbose)
        all_results.append(scores)
        if scores["pass"]:
            pass_count += 1

    # Summary
    total = len(all_results)
    print(f"\n{'=' * 60}")
    print(f"Summary: {pass_count}/{total} PASS, {total - pass_count}/{total} WARN")
    print(f"{'=' * 60}")

    if args.save:
        with open(REPLAY_RESULTS, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Results saved to {REPLAY_RESULTS}")


if __name__ == "__main__":
    main()
