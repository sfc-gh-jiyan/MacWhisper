"""Meeting transcription evaluation harness.

Scores transcription strategies against a hand-checked reference for real
meeting recordings, so changes and model choices are judged by numbers.

Data stays OUT of the repo (recordings contain real voices):
  audio:      ~/.macwhisper/meetings/meeting_<id>_mic.wav (+ optional _sys.wav)
  reference:  ~/.macwhisper/eval/<id>.ref.txt   ('#' lines are comments)
  realtime:   ~/.macwhisper/meetings/meeting_<id>.json  ("transcript" field)

Usage:
  venv/bin/python tests/eval_meeting.py                     # all sessions
  venv/bin/python tests/eval_meeting.py --session 20261018_194440
  venv/bin/python tests/eval_meeting.py --model mlx-community/whisper-large-v3-turbo
  venv/bin/python tests/eval_meeting.py --strategies realtime fullpass --show-text

Metric: MER — mixed error rate (CJK per character, Latin per word), via
eval_metrics.compute_wer. 0.0 = perfect. Recall = share of reference tokens found.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root (text_utils, asr_backend)
sys.path.insert(0, _HERE)                   # tests/ (eval_metrics)

from eval_metrics import compute_recall, compute_wer  # noqa: E402

DATA_DIR = os.path.expanduser("~/.macwhisper")
MEETINGS_DIR = os.path.join(DATA_DIR, "meetings")
REF_DIR = os.path.join(DATA_DIR, "eval")
DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
SAMPLE_RATE = 16000


@dataclass
class Session:
    session_id: str
    reference: str
    mic_path: str
    sys_path: str | None
    json_path: str | None

    def load_audio(self) -> np.ndarray:
        """Mic + system audio mixed to one float32 track (clipped to [-1, 1])."""
        mic = _read_wav(self.mic_path)
        if self.sys_path:
            sys_a = _read_wav(self.sys_path)
            n = max(len(mic), len(sys_a))
            mix = np.zeros(n, dtype=np.float32)
            mix[:len(mic)] += mic
            mix[:len(sys_a)] += sys_a
            return np.clip(mix, -1.0, 1.0)
        return mic

    def realtime_text(self) -> str:
        if not self.json_path:
            return ""
        with open(self.json_path, encoding="utf-8") as f:
            return json.load(f).get("transcript", "")


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _read_reference(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    return "\n".join(lines)


def discover_sessions(meetings_dir: str = MEETINGS_DIR, ref_dir: str = REF_DIR) -> list[Session]:
    """Sessions that have BOTH a reference and a saved mic recording."""
    if not os.path.isdir(ref_dir):
        return []
    sessions = []
    for name in sorted(os.listdir(ref_dir)):
        if not name.endswith(".ref.txt"):
            continue
        sid = name[: -len(".ref.txt")]
        mic = os.path.join(meetings_dir, f"meeting_{sid}_mic.wav")
        if not os.path.isfile(mic):
            continue
        sys_p = os.path.join(meetings_dir, f"meeting_{sid}_sys.wav")
        js = os.path.join(meetings_dir, f"meeting_{sid}.json")
        sessions.append(Session(
            session_id=sid,
            reference=_read_reference(os.path.join(ref_dir, name)),
            mic_path=mic,
            sys_path=sys_p if os.path.isfile(sys_p) else None,
            json_path=js if os.path.isfile(js) else None,
        ))
    return sessions


# ── Strategies: (session, backend) -> transcript text ─────────

def _realtime(session: Session, backend) -> str:
    """What the app actually produced live (baseline, no inference)."""
    return session.realtime_text()


def _fullpass(session: Session, backend) -> str:
    """One Whisper call over the whole meeting (same settings as Push to Talk)."""
    from text_utils import BILINGUAL_PROMPT
    result = backend.transcribe(session.load_audio(), initial_prompt=BILINGUAL_PROMPT)
    return result.text.strip()


STRATEGIES = {
    "realtime": _realtime,
    "fullpass": _fullpass,
}
NEEDS_MODEL = {"fullpass"}


def score(hypothesis: str, reference: str) -> dict:
    return {
        "mer": compute_wer(hypothesis, reference),
        "recall": compute_recall(hypothesis, reference),
    }


def format_table(rows: list[dict]) -> str:
    head = f"{'session':18s} {'strategy':10s} {'model':34s} {'MER':>6s} {'recall':>7s} {'time':>7s} {'RTF':>6s}"
    lines = [head, "-" * len(head)]
    for r in rows:
        rtf = r["seconds"] / r["audio_s"] if r["audio_s"] else 0.0
        lines.append(
            f"{r['session']:18s} {r['strategy']:10s} {r['model'][-34:]:34s} "
            f"{r['mer']:6.3f} {r['recall']:7.3f} {r['seconds']:6.1f}s {rtf:6.2f}"
        )
    return "\n".join(lines)


def run(sessions: list[Session], strategies: list[str], model: str) -> list[dict]:
    backend = None
    if any(s in NEEDS_MODEL for s in strategies):
        from asr_backend import MLXWhisperBackend
        backend = MLXWhisperBackend(model_repo=model)
        backend.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))  # warm-up, not timed
    rows = []
    for sess in sessions:
        audio_s = len(_read_wav(sess.mic_path)) / SAMPLE_RATE
        for name in strategies:
            t0 = time.time()
            text = STRATEGIES[name](sess, backend)
            elapsed = time.time() - t0
            rows.append({
                "session": sess.session_id, "strategy": name,
                "model": model if name in NEEDS_MODEL else "-",
                "seconds": elapsed, "audio_s": audio_s, "text": text,
                **score(text, sess.reference),
            })
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", action="append", help="Session id (repeatable); default all")
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--show-text", action="store_true", help="Print each hypothesis")
    p.add_argument("--json", help="Also write rows to this JSON file")
    args = p.parse_args(argv)

    sessions = discover_sessions()
    if args.session:
        sessions = [s for s in sessions if s.session_id in args.session]
    if not sessions:
        print(f"No sessions found. Need {REF_DIR}/<id>.ref.txt and "
              f"{MEETINGS_DIR}/meeting_<id>_mic.wav")
        return 1

    rows = run(sessions, args.strategies, args.model)
    print(format_table(rows))
    if args.show_text:
        for r in rows:
            print(f"\n[{r['session']} / {r['strategy']}]\n{r['text']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
