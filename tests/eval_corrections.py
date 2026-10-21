"""Replay saved dictations through the deterministic corrector (text only, no model).

    venv/bin/python tests/eval_corrections.py [--log PATH] [--vocab PATH] [--show N]

Prints every transcript the rules would change, plus totals, so the rules can be
checked for false positives on real data before shipping. Read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import corrector  # noqa: E402
import vocabulary  # noqa: E402

DATA_DIR = os.environ.get("MACWHISPER_DATA_DIR") or os.path.expanduser("~/.macwhisper")


def replay(log_path: str, vocab: vocabulary.Vocabulary) -> list[dict]:
    rows = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            text = d.get("raw_text") or d.get("text") or ""
            if not text:
                continue
            r = corrector.correct(text, vocab)
            rows.append({"ts": d.get("timestamp", ""), "before": text, "after": r.text,
                         "changes": r.changes})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(DATA_DIR, "transcripts.jsonl"))
    ap.add_argument("--vocab", default=os.path.join(DATA_DIR, "vocabulary.json"))
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args(argv)
    vocab = vocabulary.Vocabulary.load(args.vocab)
    rows = replay(args.log, vocab)
    changed = [r for r in rows if r["changes"]]
    print(f"{len(rows)} transcripts, {len(changed)} changed "
          f"({sum(len(r['changes']) for r in changed)} edits) with {len(vocab.active())} active terms\n")
    for r in changed[: args.show]:
        print(f"[{r['ts'][:16]}] {r['changes']}")
        print(f"  - {r['before'][:160]}")
        print(f"  + {r['after'][:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
