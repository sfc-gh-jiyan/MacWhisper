"""Dynamic dual-channel scenario generator.

Reads the WAV corpus from ~/.macwhisper and generates N mic×system
pairs in alternating / overlapping / gap modes.  Used by
test_meeting_dual.py for N-to-N parametrised E2E runs.

Usage from CLI (preview):
    venv/bin/python tests/dual_channel_gen.py            # default 5 pairs
    venv/bin/python tests/dual_channel_gen.py --pairs 10 # 10 pairs
"""

import json
import os
import random
import wave

AUDIO_DIR = os.path.expanduser("~/.macwhisper/audio")
TRANSCRIPT_LOG = os.path.expanduser("~/.macwhisper/transcripts.jsonl")

# Modes and their offset strategies
MODES = ("alternating", "overlapping", "gap")


def _load_wav_catalog(min_duration: float = 31.0):
    """Return list of {audio_file, duration_s} with GT and duration >= min_duration."""
    if not os.path.exists(TRANSCRIPT_LOG):
        return []

    gt_files = set()
    with open(TRANSCRIPT_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            af = entry.get("audio_file")
            text = entry.get("text")
            if af and text:
                gt_files.add(af)

    catalog = []
    for af in sorted(gt_files):
        path = os.path.join(AUDIO_DIR, af)
        if not os.path.isfile(path):
            continue
        try:
            with wave.open(path) as wf:
                dur = wf.getnframes() / wf.getframerate()
        except Exception:
            continue
        if dur >= min_duration:
            catalog.append({"audio_file": af, "duration_s": round(dur, 1)})

    return catalog


def _compute_offset(mic_dur: float, sys_dur: float, mode: str) -> float:
    """Compute system_offset_s for a given mode."""
    if mode == "alternating":
        # System starts 2s after mic finishes
        return round(mic_dur + 2.0, 1)
    elif mode == "overlapping":
        # System starts at 30% of mic duration → significant overlap
        return round(mic_dur * 0.3, 1)
    elif mode == "gap":
        # System starts 10s after mic finishes
        return round(mic_dur + 10.0, 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def generate_scenarios(
    n_pairs: int = 5,
    modes: list | None = None,
    min_duration: float = 31.0,
    seed: int | None = 42,
) -> list[dict]:
    """Generate N dual-channel scenarios from the WAV corpus.

    Args:
        n_pairs: Number of unique mic/system WAV pairs to create.
        modes: List of modes to generate per pair. Defaults to all three.
        min_duration: Minimum WAV duration in seconds.
        seed: Random seed for reproducibility. None = non-deterministic.

    Returns:
        List of scenario dicts compatible with dual_channel_scenarios.json.
    """
    if modes is None:
        modes = list(MODES)

    catalog = _load_wav_catalog(min_duration)
    if len(catalog) < 2:
        return []

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # Shuffle and pair up (no WAV paired with itself)
    files = list(catalog)
    rng.shuffle(files)

    # Generate pairs — take consecutive items, wrap around if needed
    pairs = []
    for i in range(min(n_pairs, len(files) // 2)):
        mic = files[i * 2]
        sys_ = files[i * 2 + 1]
        pairs.append((mic, sys_))

    # If we need more pairs than len//2, allow reuse but never same file as both
    while len(pairs) < n_pairs and len(files) >= 2:
        mic = rng.choice(files)
        sys_ = rng.choice(files)
        if mic["audio_file"] != sys_["audio_file"]:
            pairs.append((mic, sys_))

    scenarios = []
    for idx, (mic, sys_) in enumerate(pairs):
        for mode in modes:
            offset = _compute_offset(mic["duration_s"], sys_["duration_s"], mode)
            name = f"gen_{mode}_{idx:02d}"
            scenarios.append({
                "name": name,
                "description": f"Auto-generated: {mode} (mic={mic['audio_file']}, sys={sys_['audio_file']})",
                "mic_wav": mic["audio_file"],
                "system_wav": sys_["audio_file"],
                "mic_offset_s": 0,
                "system_offset_s": offset,
                "mic_duration_s": mic["duration_s"],
                "system_duration_s": sys_["duration_s"],
                "mode": mode,
                "tier": "e2e",
            })

    return scenarios


# ── CLI preview ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview generated dual-channel scenarios")
    parser.add_argument("--pairs", type=int, default=5, help="Number of WAV pairs")
    parser.add_argument("--modes", nargs="+", default=list(MODES),
                        choices=MODES, help="Modes to generate")
    parser.add_argument("--min-duration", type=float, default=31.0,
                        help="Minimum WAV duration (seconds)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    scenarios = generate_scenarios(
        n_pairs=args.pairs, modes=args.modes,
        min_duration=args.min_duration, seed=args.seed,
    )

    if not scenarios:
        print("No scenarios generated (not enough WAVs with GT)")
    else:
        print(f"Generated {len(scenarios)} scenarios from {args.pairs} pairs:\n")
        for s in scenarios:
            total = s["system_offset_s"] + s["system_duration_s"]
            print(f"  {s['name']:25s}  mic={s['mic_wav']:<24s}  sys={s['system_wav']:<24s}  "
                  f"offset={s['system_offset_s']:5.1f}s  total~{total:.0f}s  [{s['mode']}]")
