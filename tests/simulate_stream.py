"""Simulate live subtitles with Qwen3-ASR re-decoding the growing utterance.

Test-only tool — the app is not modified. This is the WhisprStream-style
approach: every `step` seconds the WHOLE utterance so far is re-transcribed,
and the stable prefix (agreeing across two consecutive decodes) is
confirmed. No word timestamps are needed — agreement is on text tokens.

Time is simulated with a virtual clock but every decode's REAL duration is
charged to it, so a slow decode delays the next one exactly as it would in
the app. Releasing the key triggers one final full decode.

Run with the isolated eval env (needs mlx-audio):
  .venv-eval/bin/python tests/simulate_stream.py [--clips 20] [--step 1.0]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

SAMPLE_RATE = 16000
DEFAULT_MODEL = "mlx-community/Qwen3-ASR-0.6B-bf16"
OUT_DIR = os.path.expanduser("~/.macwhisper/eval")

# One CJK char, or a run of Latin letters/digits/apostrophes = one token.
_TOKEN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9']+")
# Hesitation fillers. Qwen3 emits them inconsistently (silence before speech
# decodes as "嗯。" and later decodes drop it), so they must never take part
# in agreement — a confirmed filler froze confirmation for a whole utterance.
FILLERS = {"嗯", "呃", "啊", "额", "唔", "um", "uh", "uhm", "erm", "hmm", "mm"}


def tokenize(text: str) -> list[tuple[str, str]]:
    """Split into (display, key) tokens. Display keeps trailing punctuation
    and spacing; key is lowercase content only, so punctuation changes
    between decodes ("，" -> "。") never block agreement."""
    text = text or ""
    matches = list(_TOKEN.finditer(text))
    tokens = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        key = m.group().lower()
        if key in FILLERS:
            continue
        tokens.append((text[m.start():end], key))
    return tokens


def common_prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class StableText:
    """LocalAgreement-2 on text tokens, confirmed prefix never shrinks."""

    def __init__(self):
        self.confirmed: list[tuple[str, str]] = []
        self.tail: list[tuple[str, str]] = []
        self._prev_keys: list[str] | None = None
        self.contradictions = 0  # newest decode disagreed with confirmed text
        self.rewrites = 0        # confirmed text replaced (model changed its mind twice)

    def update(self, hypothesis: str) -> None:
        toks = tokenize(hypothesis)
        keys = [k for _, k in toks]
        n_conf = len(self.confirmed)
        if keys[:n_conf] != [k for _, k in self.confirmed]:
            self.contradictions += 1
            agree = common_prefix_len(self._prev_keys or [], keys)
            if agree > 0:
                # Two consecutive decodes agree with each other but not with
                # what we confirmed: the model has settled on a new reading.
                # Rebase instead of freezing forever.
                self.confirmed = toks[:agree]
                self.rewrites += 1
                self.tail = toks[agree:]
            else:
                self.tail = []  # first disagreement: keep confirmed, wait
            self._prev_keys = keys
            return
        if self._prev_keys is not None:
            agree = common_prefix_len(self._prev_keys, keys)
            if agree > n_conf:
                self.confirmed.extend(toks[n_conf:agree])
        self.tail = toks[len(self.confirmed):]
        self._prev_keys = keys

    @property
    def confirmed_text(self) -> str:
        return "".join(d for d, _ in self.confirmed)

    @property
    def tail_text(self) -> str:
        return "".join(d for d, _ in self.tail)


def simulate(audio: np.ndarray, decode, step: float = 1.0, min_audio: float = 0.5,
             clock=time.time) -> dict:
    """Replay `audio` as if spoken live; `decode(audio) -> text`.

    Virtual clock t (seconds since speech start). A decode may start at the
    next tick once the previous decode finished; it sees audio up to its
    start time and takes its real measured duration.
    """
    duration = len(audio) / SAMPLE_RATE
    stable = StableText()
    t = 0.0
    next_tick = max(step, min_audio)
    decodes, first_text_at = [], None
    while next_tick < duration:
        t = max(t, next_tick)
        if t >= duration:
            break
        chunk = audio[: int(t * SAMPLE_RATE)]
        t0 = clock()
        text = decode(chunk)
        took = clock() - t0
        t += took
        stable.update(text)
        decodes.append({"audio_s": round(len(chunk) / SAMPLE_RATE, 2), "took_s": round(took, 3)})
        if first_text_at is None and (stable.confirmed or stable.tail):
            first_text_at = t
        next_tick = t + step if took >= step else next_tick + step

    live_at_release = stable.confirmed_text + stable.tail_text
    t = max(t, duration)  # key released at end of audio
    t0 = clock()
    final = decode(audio)
    final_took = clock() - t0
    return {
        "duration_s": round(duration, 2),
        "decodes": len(decodes),
        "first_text_s": round(first_text_at, 2) if first_text_at is not None else None,
        "mean_decode_s": round(float(np.mean([d["took_s"] for d in decodes])), 3) if decodes else 0.0,
        "max_decode_s": round(max((d["took_s"] for d in decodes), default=0.0), 3),
        "last_decode_audio_s": decodes[-1]["audio_s"] if decodes else 0.0,
        "wait_after_release_s": round(t - duration + final_took, 2),
        "contradictions": stable.contradictions,
        "rewrites": stable.rewrites,
        "confirmed_at_release": stable.confirmed_text,
        "live_at_release": live_at_release,
        "final": final.strip(),
    }


def main(argv=None) -> int:
    from compare_ptt import is_code_switched, load_clips, postprocess, stratified_sample
    from eval_meeting import _read_wav
    from eval_metrics import compute_wer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", type=int, default=20)
    p.add_argument("--step", type=float, default=1.0, help="Seconds between re-decodes")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args(argv)

    from mlx_audio.stt.utils import load
    model = load(args.model)
    model.generate(np.zeros(SAMPLE_RATE, dtype=np.float32), language="Chinese")

    def decode(a):
        return model.generate(a, language="Chinese").text

    clips = load_clips()
    # Only clips where the app also logged its live (Whisper) subtitles
    clips = [c for c in clips if c.get("realtime")] or clips
    sample = stratified_sample(clips, args.clips)
    rows = []
    for i, c in enumerate(sample, 1):
        r = simulate(_read_wav(c["wav"]), decode, step=args.step)
        r["final"] = postprocess(r["final"])
        r["wav"] = os.path.basename(c["wav"])
        r["mixed"] = is_code_switched(c["pasted"])
        r["qwen_live_vs_final"] = compute_wer(r["live_at_release"], r["final"]) if r["final"] else None
        r["whisper_live"] = c.get("realtime", "")
        r["whisper_final"] = c["pasted"]
        r["whisper_live_vs_final"] = (compute_wer(c["realtime"], c["pasted"])
                                      if c.get("realtime") else None)
        rows.append(r)
        print(f"{i:2d}/{len(sample)} {r['wav']} {r['duration_s']:5.1f}s first={r['first_text_s']} "
              f"mean={r['mean_decode_s']}s max={r['max_decode_s']}s wait={r['wait_after_release_s']}s "
              f"contra={r['contradictions']} rewrite={r['rewrites']} qwenΔ={r['qwen_live_vs_final']} whisperΔ={r['whisper_live_vs_final']}",
              flush=True)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(OUT_DIR, f"stream_sim_{stamp}.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
