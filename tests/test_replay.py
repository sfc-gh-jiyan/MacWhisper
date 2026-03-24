#!/usr/bin/env python3
"""Replay test for live subtitles — uses the REAL code path.

Feeds a saved WAV file into the actual _live_chunk_loop and
_live_transcription_worker threads (same as a real recording),
just with frames injected from a file instead of a microphone.

Usage:
    venv/bin/python test_replay.py                            # latest >31s WAV
    venv/bin/python test_replay.py --wav 20260322_212421.wav  # specific file
"""

import argparse
import io
import json
import os
import queue
import re
import sys
import threading
import time
import wave

import numpy as np

# ── Mock macOS-only modules before importing app ──────────────

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app  # noqa: E402 — must come after mocks


# ── Constants ─────────────────────────────────────────────────

SAMPLE_RATE = app.SAMPLE_RATE
BLOCKSIZE = 1024
AUDIO_DIR = app.AUDIO_DIR
TRANSCRIPT_LOG = app.TRANSCRIPT_LOG
REPLAY_RESULTS = os.path.join(os.path.dirname(TRANSCRIPT_LOG), "replay_results.jsonl")

FRAME_DURATION = BLOCKSIZE / SAMPLE_RATE  # 0.064s per frame


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
            padded = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            padded[:len(chunk)] = chunk
            chunk = padded
        frames.append(chunk)
    return frames


# ── Headless instance ─────────────────────────────────────────

def create_headless_instance(model="mlx-community/whisper-medium-mlx"):
    """Create a TranscriberApp with real threads but no GUI/microphone."""
    inst = app.TranscriberApp.__new__(app.TranscriberApp)

    # Model
    inst.current_model = model

    # Recording state (mirrors _start_recording)
    inst.frames = []
    inst.recording = False
    inst.stream = None

    # Live subtitle state
    inst._live_queue = queue.Queue(maxsize=2)
    inst._inference_lock = threading.Lock()
    inst._last_live_result = ""
    inst._best_raw = ""
    inst._prev_raw = ""
    inst._frozen_prefix = ""
    inst._stale_count = 0
    inst._accept_count = 0
    inst._last_debug = {}
    inst._last_committed_raw = ""
    inst._last_display = ""
    inst._segment_gen = 0
    inst._segment_start_frame = 0
    inst._pause_silence_frames = 0
    inst._pause_detected = False
    inst._segment_committed_text = ""
    inst._segment_committed_display = ""
    inst._last_overlay_text = ""

    # GUI stubs
    inst._overlay_panel = None
    inst._overlay_text = None
    inst.live_mode = True

    # Start the real transcription worker thread
    threading.Thread(target=inst._live_transcription_worker, daemon=True).start()

    return inst


# ── Replay engine ─────────────────────────────────────────────

def replay_wav(inst, wav_frames):
    """Feed WAV frames through the real _live_chunk_loop at real-time pace.

    This exercises the exact same code path as a real recording:
    - Frames are appended to inst.frames (simulating audio callback)
    - Pause detection runs per-frame (same logic as audio callback)
    - _live_chunk_loop runs on its own thread (real code, real timing)
    - _live_transcription_worker processes chunks (real Whisper inference)

    Returns list of subtitle entries collected from the subtitle log.
    """
    pause_threshold = int(app.PAUSE_MIN_DURATION * SAMPLE_RATE / BLOCKSIZE)

    # Mark the subtitle log position so we can read only our entries
    app._ensure_history_dirs()
    log_pos = 0
    if os.path.exists(app.SUBTITLE_LOG):
        log_pos = os.path.getsize(app.SUBTITLE_LOG)

    # Init recording state (same as _start_recording, minus GUI/stream)
    inst.frames = []
    inst.recording = True
    inst._last_live_result = ""
    inst._best_raw = ""
    inst._prev_raw = ""
    inst._frozen_prefix = ""
    inst._stale_count = 0
    inst._accept_count = 0
    inst._last_debug = {}
    inst._last_committed_raw = ""
    inst._last_display = ""
    inst._segment_gen = 0
    inst._segment_start_frame = 0
    inst._pause_silence_frames = 0
    inst._pause_detected = False
    inst._segment_committed_text = ""
    inst._segment_committed_display = ""
    inst._last_overlay_text = ""

    # Start the real _live_chunk_loop thread
    chunk_thread = threading.Thread(target=inst._live_chunk_loop, daemon=True)
    chunk_thread.start()

    # Feed frames at real-time pace (simulating the audio callback)
    for frame in wav_frames:
        if not inst.recording:
            break
        inst.frames.append(frame)

        # Pause detection per frame — exact same logic as audio callback
        rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
        if rms < app.PAUSE_RMS_THRESHOLD:
            inst._pause_silence_frames += 1
        else:
            inst._pause_silence_frames = 0
        seg_frames = len(inst.frames) - inst._segment_start_frame
        seg_secs = seg_frames * BLOCKSIZE / SAMPLE_RATE
        if (inst._pause_silence_frames >= pause_threshold
                and seg_secs >= app.PAUSE_MIN_SEGMENT
                and not inst._pause_detected):
            inst._pause_detected = True

        time.sleep(FRAME_DURATION)

    # WAV finished — let pending inference complete
    time.sleep(5)
    inst.recording = False
    chunk_thread.join(timeout=3)

    # Drain queue
    while not inst._live_queue.empty():
        try:
            inst._live_queue.get_nowait()
        except queue.Empty:
            break

    # Read subtitle entries written during this replay
    entries = []
    if os.path.exists(app.SUBTITLE_LOG):
        with open(app.SUBTITLE_LOG, "r", encoding="utf-8") as f:
            f.seek(log_pos)
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

    return entries


# ── Evaluation ────────────────────────────────────────────────

def _split_sentences(text):
    """Split text into non-empty sentences."""
    parts = re.split(r'(?<=[。！？.!?\n])', text)
    return [s.strip() for s in parts if len(s.strip()) > 2]


def _normalize(text):
    """Lowercase, strip punctuation/space for fuzzy comparison."""
    return ''.join(
        ch.lower() for ch in text
        if ch not in app._OVERLAP_STRIP_CHARS
    )


def evaluate(entries, ground_truth):
    """Evaluate replay quality against ground truth."""
    if not entries:
        return {"total_cycles": 0, "completeness": 0.0, "duplications": 0,
                "stability_jumps": 0, "pass": False}

    final_display = entries[-1]["display"]
    norm_display = _normalize(final_display)

    # Completeness: how many GT sentences appear in final display
    gt_sents = _split_sentences(ground_truth)
    found = sum(1 for s in gt_sents
                if len(_normalize(s)) < 4 or _normalize(s) in norm_display)
    completeness = round(found / len(gt_sents), 2) if gt_sents else 1.0

    # Duplication: repeated sentences in final display
    display_sents = _split_sentences(final_display)
    seen = set()
    dup_count = 0
    for s in display_sents:
        ns = _normalize(s)
        if len(ns) < 4:
            continue
        if ns in seen:
            dup_count += 1
        seen.add(ns)

    # Stability: large display jumps between consecutive cycles
    jumps = 0
    for i in range(1, len(entries)):
        prev_d = entries[i - 1]["display"]
        curr_d = entries[i]["display"]
        check = min(20, len(prev_d))
        if check > 0 and not curr_d.startswith(prev_d[:check]):
            jumps += 1

    passed = completeness >= 0.6 and dup_count == 0
    return {
        "total_cycles": len(entries),
        "completeness": completeness,
        "duplications": dup_count,
        "stability_jumps": jumps,
        "pass": passed,
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


# ── Pretty printer ────────────────────────────────────────────

def print_report(entries, ground_truth, scores, audio_file, duration, elapsed):
    """Print a clean, readable report of the replay results."""
    print(f"\n{'=' * 70}")
    print(f"  Replay: {audio_file} ({duration:.1f}s audio, {elapsed:.0f}s real)")
    print(f"{'=' * 70}")

    if not entries:
        print("  No subtitle entries produced.")
        return

    # Per-cycle table
    print(f"\n  {'#':>3}  {'Time':>10}  {'Audio':>6}  {'Disp':>5}  {'Reason':<20}  Display text")
    print(f"  {'─'*3}  {'─'*10}  {'─'*6}  {'─'*5}  {'─'*20}  {'─'*30}")

    t0 = entries[0]["timestamp"] if "timestamp" in entries[0] else None
    prev_len = 0
    for i, e in enumerate(entries):
        ts = e.get("timestamp", "")
        if ts:
            ts_short = ts.split("T")[1][:8] if "T" in ts else ts[:8]
        else:
            ts_short = "?"
        disp = e.get("display", "")
        dlen = len(disp)

        # Show indicator
        if dlen < prev_len:
            mark = " <"
        elif dlen == prev_len:
            mark = " ="
        else:
            mark = ""

        # Build reason string from debug info
        dbg = e.get("debug", {})
        action = dbg.get("action", "?")
        reason = dbg.get("reason", "?")
        if action == "REJECT" and reason == "ratchet":
            reason_str = f"REJ ratchet {dbg.get('raw_len','')}< {dbg.get('best_len','')}"
        elif action == "REJECT" and reason == "guard1":
            reason_str = f"REJ guard1 g1={dbg.get('g1','?')}"
        elif action == "REJECT" and reason == "guard2":
            reason_str = f"REJ guard2 g2={dbg.get('g2','?')} f={dbg.get('frozen_len','')}"
        elif action == "REJECT" and reason == "post_commit_echo":
            reason_str = f"REJ echo e={dbg.get('echo','?')}"
        elif action == "ACCEPT" and reason == "stale_override":
            reason_str = f"OK  stale s={dbg.get('accept_n','')}"
        elif action == "ACCEPT":
            g1_str = f" g1={dbg['g1']}" if 'g1' in dbg else ""
            reason_str = f"OK {g1_str}"
        else:
            reason_str = f"{action} {reason}"

        # Truncate display for readability
        audio_s = e.get("audio_s", 0)
        if len(disp) > 40:
            show = disp[:15] + "..." + disp[-20:]
        else:
            show = disp
        print(f"  {i+1:3d}  {ts_short:>10}  {audio_s:5.1f}s  {dlen:4d}c{mark}  {reason_str:<20}  {show}")
        prev_len = dlen

    # Final display vs ground truth
    final = entries[-1].get("display", "")
    print(f"\n  Final display ({len(final)} chars):")
    # Print wrapped
    for j in range(0, len(final), 70):
        print(f"    {final[j:j+70]}")

    print(f"\n  Ground truth ({len(ground_truth)} chars):")
    for j in range(0, len(ground_truth), 70):
        print(f"    {ground_truth[j:j+70]}")

    # Scores
    status = "PASS" if scores["pass"] else "WARN"
    gt_sents = _split_sentences(ground_truth)
    norm_display = _normalize(final)
    found_count = sum(1 for s in gt_sents
                      if len(_normalize(s)) < 4 or _normalize(s) in norm_display)

    print(f"\n  Completeness: {scores['completeness']} ({found_count}/{len(gt_sents)} sentences)")
    print(f"  Duplications: {scores['duplications']}")
    print(f"  Stability:    {scores['stability_jumps']} jump(s)")
    print(f"  Result:       {status}")
    print(f"{'=' * 70}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Replay test for live subtitles (real code path)")
    parser.add_argument("--wav", help="Test a specific WAV file (filename only)")
    args = parser.parse_args()

    # Load pairs, filter to >31s
    pairs = load_transcript_pairs(filter_wav=args.wav, min_duration=31 if not args.wav else 0)

    if not pairs:
        print("[WARN] No matching WAV + transcript pairs found.")
        print(f"       Audio dir:    {AUDIO_DIR}")
        print(f"       Transcripts:  {TRANSCRIPT_LOG}")
        if args.wav:
            print(f"       Filter: --wav {args.wav}")
        return

    # Default: pick the latest >31s recording (by filename = timestamp)
    if not args.wav:
        pair = max(pairs, key=lambda p: p["audio_file"])
    else:
        pair = pairs[0]

    audio_file = pair["audio_file"]
    duration = pair["duration_s"]

    print(f"Loading {audio_file} ({duration:.1f}s)...")
    wav_frames = load_wav_as_frames(pair["wav_path"])
    inst = create_headless_instance()

    print(f"Replaying at real-time pace ({duration:.0f}s)...")
    print(f"(Whisper inference logs below are from the REAL code path)\n")

    t0 = time.time()
    entries = replay_wav(inst, wav_frames)
    elapsed = time.time() - t0

    scores = evaluate(entries, pair["ground_truth"])
    scores["audio_file"] = audio_file
    scores["duration_s"] = duration
    scores["replay_time_s"] = round(elapsed, 1)

    print_report(entries, pair["ground_truth"], scores, audio_file, duration, elapsed)

    # Save
    with open(REPLAY_RESULTS, "w", encoding="utf-8") as f:
        f.write(json.dumps(scores, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {REPLAY_RESULTS}")


if __name__ == "__main__":
    main()
