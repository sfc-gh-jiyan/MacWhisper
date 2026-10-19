"""Compare Push to Talk transcription: Whisper vs Qwen3-ASR on saved recordings.

Test-only tool — does not touch the app. Reads recordings the app saved
(~/.macwhisper/audio/*.wav) together with what the app actually pasted
(transcripts.jsonl incl. archive/), re-transcribes each with every model,
applies the same post-processing as app.py's final pass, and writes a
side-by-side Markdown report to ~/.macwhisper/eval/ (never to the repo).

There is no ground truth for these clips, so the report shows:
  - pairwise disagreement (MER between systems),
  - hallucination-filtered (empty) outputs per system,
  - "translated" outputs: Chinese in Whisper's text but none in the model's,
  - a stratified sample for the user to judge by reading.

Run with the isolated eval env (needs mlx-audio):
  .venv-eval/bin/python tests/compare_ptt.py [--limit N] [--sample 30]
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import random
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from eval_meeting import _read_wav  # noqa: E402
from eval_metrics import compute_wer  # noqa: E402

DATA_DIR = os.path.expanduser("~/.macwhisper")
OUT_DIR = os.path.join(DATA_DIR, "eval")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]{2,}")

SYSTEMS = {
    "whisper-large-v3": ("whisper", "mlx-community/whisper-large-v3-mlx"),
    "qwen3-0.6b": ("qwen3", "mlx-community/Qwen3-ASR-0.6B-bf16"),
    "qwen3-1.7b": ("qwen3", "mlx-community/Qwen3-ASR-1.7B-bf16"),
}


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(text or ""))


def is_code_switched(text: str) -> bool:
    return has_cjk(text) and bool(_LATIN.search(text or ""))


def looks_translated(reference: str, candidate: str) -> bool:
    """Chinese present in the reference output but absent in the candidate."""
    return has_cjk(reference) and bool(candidate) and not has_cjk(candidate)


def bucket(duration_s: float) -> str:
    if duration_s < 5:
        return "<5s"
    if duration_s < 15:
        return "5-15s"
    if duration_s < 31:
        return "15-31s"
    return ">=31s"


def load_clips(data_dir: str = DATA_DIR) -> list[dict]:
    """Recordings that have a saved WAV and the text the app pasted for it."""
    logs = glob.glob(os.path.join(data_dir, "archive", "**", "transcripts.jsonl"), recursive=True)
    logs.append(os.path.join(data_dir, "transcripts.jsonl"))
    clips, seen = [], set()
    for log in logs:
        if not os.path.isfile(log):
            continue
        with open(log, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                audio = r.get("audio_file")
                if not audio or r.get("translate") or not r.get("text"):
                    continue
                path = os.path.join(data_dir, "audio", os.path.basename(audio))
                if not os.path.isfile(path) or path in seen:
                    continue
                seen.add(path)
                clips.append({
                    "wav": path, "duration_s": float(r.get("duration_s") or 0),
                    "pasted": r["text"], "pasted_model": r.get("model", "?"),
                })
    return sorted(clips, key=lambda c: c["wav"])


def stratified_sample(clips: list[dict], n: int, seed: int = 7) -> list[dict]:
    """~n clips spread across duration buckets, code-switched clips first."""
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict]] = {}
    for c in clips:
        by_bucket.setdefault(bucket(c["duration_s"]), []).append(c)
    per = max(1, n // max(1, len(by_bucket)))
    picked = []
    for name in ("<5s", "5-15s", "15-31s", ">=31s"):
        group = by_bucket.get(name, [])
        mixed = [c for c in group if is_code_switched(c["pasted"])]
        rest = [c for c in group if c not in mixed]
        rng.shuffle(mixed)
        rng.shuffle(rest)
        picked.extend((mixed + rest)[:per])
    return picked[:n]


def postprocess(text: str) -> str:
    """Same final-pass cleanup app.py applies to Push to Talk results."""
    from text_utils import convert_t2s, hallucination_reason, strip_trailing_repetition
    text = convert_t2s((text or "").strip())
    text = strip_trailing_repetition(text)
    if text and hallucination_reason(text):
        return ""
    return text


def _transcriber(kind: str, repo: str):
    if kind == "whisper":
        from asr_backend import MLXWhisperBackend
        from text_utils import BILINGUAL_PROMPT
        b = MLXWhisperBackend(model_repo=repo)
        b.transcribe(np.zeros(16000, dtype=np.float32))

        def run(audio):
            text = b.transcribe(audio, initial_prompt=BILINGUAL_PROMPT, task="transcribe").text.strip()
            if text.startswith(BILINGUAL_PROMPT):
                text = text[len(BILINGUAL_PROMPT):].strip()
            return text
        return run
    from mlx_audio.stt.utils import load
    m = load(repo)
    m.generate(np.zeros(16000, dtype=np.float32), language="Chinese")
    return lambda audio: m.generate(audio, language="Chinese").text


def transcribe_all(clips: list[dict]) -> None:
    for name, (kind, repo) in SYSTEMS.items():
        print(f"[{name}] loading {repo}", flush=True)
        run = _transcriber(kind, repo)
        total = 0.0
        for i, c in enumerate(clips, 1):
            audio = _read_wav(c["wav"])
            t0 = time.time()
            try:
                c[name] = postprocess(run(audio))
            except Exception as e:  # keep going; record the failure
                c[name] = f"<ERROR {type(e).__name__}: {e}>"
            c[f"{name}_s"] = time.time() - t0
            total += c[f"{name}_s"]
            if i % 25 == 0:
                print(f"  {i}/{len(clips)}", flush=True)
        print(f"[{name}] done in {total:.0f}s", flush=True)


def summarize(clips: list[dict]) -> list[str]:
    audio_s = sum(c["duration_s"] for c in clips)
    lines = [f"Clips: {len(clips)} ({audio_s / 60:.1f} min of audio)", "",
             "| system | total time | RTF | empty (filtered) | translated vs Whisper | MER vs Whisper large-v3 |",
             "|---|---|---|---|---|---|"]
    base = "whisper-large-v3"
    for name in SYSTEMS:
        t = sum(c[f"{name}_s"] for c in clips)
        empty = sum(1 for c in clips if not c[name])
        trans = sum(1 for c in clips if name != base and looks_translated(c[base], c[name]))
        mers = [compute_wer(c[name], c[base]) for c in clips if c[base] and name != base]
        mer = f"{np.median(mers):.3f} (median)" if mers else "-"
        lines.append(f"| {name} | {t:.0f}s | {t / audio_s:.3f} | {empty} | "
                     f"{trans if name != base else '-'} | {mer} |")
    return lines


def write_report(clips: list[dict], sample: list[dict], path: str) -> None:
    out = ["# Push to Talk: Whisper vs Qwen3-ASR", "",
           f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. No ground truth — "
           "judge the side-by-side samples by reading.", "",
           "`pasted` = what the app actually pasted at the time "
           f"({clips[0]['pasted_model'] if clips else '?'}).", "", "## Summary", ""]
    out += summarize(clips)
    out += ["", "## Side-by-side sample", ""]
    for i, c in enumerate(sample, 1):
        out.append(f"### {i}. {os.path.basename(c['wav'])} — {c['duration_s']:.1f}s"
                   f"{' — mixed zh/en' if is_code_switched(c['pasted']) else ''}")
        out.append("")
        out.append(f"- **pasted**: {c['pasted']}")
        for name in SYSTEMS:
            flag = " ⚠️ translated?" if looks_translated(c["whisper-large-v3"], c[name]) else ""
            out.append(f"- **{name}** ({c[f'{name}_s']:.1f}s){flag}: {c[name] or '(empty)'}")
        out.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=0, help="Only the first N clips (0 = all)")
    p.add_argument("--sample", type=int, default=30, help="Clips shown side by side")
    args = p.parse_args(argv)

    clips = load_clips()
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        print("No recordings with matching pasted text found.")
        return 1
    print(f"{len(clips)} clips, {sum(c['duration_s'] for c in clips) / 60:.1f} min", flush=True)
    transcribe_all(clips)
    sample = stratified_sample(clips, args.sample)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    report = os.path.join(OUT_DIR, f"ptt_compare_{stamp}.md")
    write_report(clips, sample, report)
    with open(report.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump(clips, f, ensure_ascii=False, indent=1)
    print("\n".join(summarize(clips)))
    print(f"\nReport: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
