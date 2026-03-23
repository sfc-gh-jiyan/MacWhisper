#!/usr/bin/env python3
"""Parameter sweep for pause detection tuning.

Patches app constants, re-runs replay simulation for each combination,
and reports the best parameter set.

Usage:
    venv/bin/python sweep_params.py
"""

import itertools
import json
import os
import sys
import time

# Must mock before importing app/test_replay
from unittest.mock import MagicMock

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

import app
import test_replay

# ── Parameter grid ──────────────────────────────────────────

PARAM_GRID = {
    "PAUSE_RMS_THRESHOLD": [100, 150, 200],
    "PAUSE_MIN_DURATION": [0.8, 1.2, 1.5, 2.0],
    "PAUSE_MIN_SEGMENT": [2.0, 5.0, 8.0, 12.0],
}

# ── Load ONLY the pilot recording for Phase 1 ──────────────

MIN_DURATION = 31
pairs = test_replay.load_transcript_pairs(min_duration=MIN_DURATION)
if not pairs:
    print("[ERROR] No test pairs found >= 31s")
    sys.exit(1)

# Pick shortest recording as pilot (fast iteration, still >30s)
pairs.sort(key=lambda p: p["duration_s"])
pilot_pair = pairs[0]
print(f"Phase 1 pilot: {pilot_pair['audio_file']} ({pilot_pair['duration_s']:.1f}s)")
print(f"Loading pilot WAV...")
pilot_frames = test_replay.load_wav_as_frames(pilot_pair["wav_path"])
print("Loaded.\n")

# ── Helpers ─────────────────────────────────────────────────

def run_one(params, p, frames):
    """Run a single (params, recording) combo. Returns scores dict."""
    app.PAUSE_RMS_THRESHOLD = params["PAUSE_RMS_THRESHOLD"]
    app.PAUSE_MIN_DURATION = params["PAUSE_MIN_DURATION"]
    app.PAUSE_MIN_SEGMENT = params["PAUSE_MIN_SEGMENT"]
    inst = test_replay.create_headless_instance()
    entries = test_replay.simulate_stream(frames, inst)
    return test_replay.evaluate(entries, p["ground_truth"])


# ── Phase 1: 48 combos x 1 pilot recording ─────────────────

keys = list(PARAM_GRID.keys())
values = [PARAM_GRID[k] for k in keys]
combos = list(itertools.product(*values))

print(f"Phase 1: {len(combos)} combos x 1 recording "
      f"({pilot_pair['audio_file']}, {pilot_pair['duration_s']:.0f}s)\n")

phase1 = []
for ci, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    scores = run_one(params, pilot_pair, pilot_frames)
    status = "✓" if scores["duplications"] == 0 else "✗"
    print(f"  [{ci+1}/{len(combos)}] {status} RMS={params['PAUSE_RMS_THRESHOLD']:>3} "
          f"dur={params['PAUSE_MIN_DURATION']:.1f}s "
          f"seg={params['PAUSE_MIN_SEGMENT']:>4.1f}s → "
          f"comp={scores['completeness']:.2f} dup={scores['duplications']}")
    phase1.append({"params": params, **scores})

# Rank: zero dup, then best completeness
candidates = [r for r in phase1 if r["duplications"] == 0]
candidates.sort(key=lambda r: r["completeness"], reverse=True)

TOP_N = 5
shortlist = candidates[:TOP_N]

print(f"\nPhase 1 done: {len(candidates)}/{len(combos)} had zero dup")
print(f"Top {TOP_N} shortlisted for Phase 2:\n")
for i, r in enumerate(shortlist):
    p = r["params"]
    print(f"  #{i+1}: RMS={p['PAUSE_RMS_THRESHOLD']:>3} "
          f"dur={p['PAUSE_MIN_DURATION']:.1f}s "
          f"seg={p['PAUSE_MIN_SEGMENT']:>4.1f}s → comp={r['completeness']:.2f}")

if not shortlist:
    print("\n[ERROR] No zero-dup candidates found. Try wider parameter ranges.")
    sys.exit(1)

# ── Phase 2: Top 5 x all 6 recordings (lazy load) ──────────

print(f"\nPhase 2: {len(shortlist)} combos x {len(pairs)} recordings")
print("Loading WAV files one at a time...\n")

phase2 = []
for ci, candidate in enumerate(shortlist):
    params = candidate["params"]
    total_comp = 0.0
    total_dup = 0
    total_jumps = 0
    pass_count = 0
    file_results = []

    for p in pairs:
        frames = test_replay.load_wav_as_frames(p["wav_path"])
        scores = run_one(params, p, frames)
        total_comp += scores["completeness"]
        total_dup += scores["duplications"]
        total_jumps += scores["stability_jumps"]
        if scores["pass"]:
            pass_count += 1
        file_results.append({
            "file": p["audio_file"],
            "duration": p["duration_s"],
            "completeness": scores["completeness"],
            "duplications": scores["duplications"],
            "pass": scores["pass"],
        })
        del frames  # free memory

    n = len(pairs)
    avg_comp = round(total_comp / n, 3)
    record = {
        "params": params,
        "avg_completeness": avg_comp,
        "total_duplications": total_dup,
        "total_jumps": total_jumps,
        "pass_count": pass_count,
        "total": n,
        "files": file_results,
    }
    phase2.append(record)

    status = "✓" if total_dup == 0 else "✗"
    print(f"  [{ci+1}/{len(shortlist)}] {status} RMS={params['PAUSE_RMS_THRESHOLD']:>3} "
          f"dur={params['PAUSE_MIN_DURATION']:.1f}s "
          f"seg={params['PAUSE_MIN_SEGMENT']:>4.1f}s → "
          f"avg_comp={avg_comp:.3f} dup={total_dup} pass={pass_count}/{n}")

# ── Final ranking ───────────────────────────────────────────

phase2.sort(key=lambda r: (-int(r["total_duplications"] == 0), -r["avg_completeness"]))

print(f"\n{'='*70}")
print(f"FINAL RANKING")
print(f"{'='*70}\n")

for i, r in enumerate(phase2):
    p = r["params"]
    tag = "BEST" if i == 0 else ""
    print(f"  #{i+1} {tag}: RMS={p['PAUSE_RMS_THRESHOLD']:>3} "
          f"dur={p['PAUSE_MIN_DURATION']:.1f}s "
          f"seg={p['PAUSE_MIN_SEGMENT']:>4.1f}s → "
          f"avg_comp={r['avg_completeness']:.3f} "
          f"dup={r['total_duplications']} "
          f"pass={r['pass_count']}/{r['total']} "
          f"jumps={r['total_jumps']}")
    for f in r["files"]:
        ftag = "PASS" if f["pass"] else "WARN"
        print(f"       {f['file']} ({f['duration']:.0f}s): "
              f"comp={f['completeness']:.2f} dup={f['duplications']} [{ftag}]")
    print()

# Save
sweep_path = os.path.join(os.path.dirname(app.TRANSCRIPT_LOG), "sweep_results.json")
with open(sweep_path, "w", encoding="utf-8") as f:
    json.dump({"phase1": phase1, "phase2": phase2}, f, ensure_ascii=False, indent=2)
print(f"Full results saved to {sweep_path}")
