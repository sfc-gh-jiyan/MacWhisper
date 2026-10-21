"""Human-readable view of the vocabulary and of what was corrected or learned.

Written to <data dir>/corrections.md and opened from the menu, so the user can
see exactly what MacWhisper changed instead of trusting it blindly.
"""

from __future__ import annotations

import json
import logging
import os

from vocabulary import ACTIVE, PENDING, Vocabulary

logger = logging.getLogger(__name__)


def _cell(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _read_log(path: str) -> list[dict]:
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def render(vocab: Vocabulary, log_path: str, vocab_path: str = "", limit: int = 50) -> str:
    out = ["# MacWhisper corrections", ""]
    out += ["How to teach: menu **Add Correction…** and type `哈什 = hash`, "
            "or hold **⇧ + Push to Talk** and say \"X 应该是 Y\".", ""]
    if vocab_path:
        out += [f"Vocabulary file (editable): `{vocab_path}`", ""]

    out += ["## Vocabulary (active)", "", "| Term | Heard as | Uses | Note |", "|---|---|---|---|"]
    for e in sorted((e for e in vocab.entries if e.status == ACTIVE), key=lambda e: -e.hits):
        out.append(f"| {_cell(e.term)} | {_cell(', '.join(e.aliases))} | {e.hits} | {_cell(e.note)} |")
    pending = [e for e in vocab.entries if e.status == PENDING]
    if pending:
        out += ["", "## Pending (learned once — becomes active when seen again)", "",
                "| Term | Heard as | Seen |", "|---|---|---|"]
        out += [f"| {_cell(e.term)} | {_cell(', '.join(e.aliases))} | {e.seen} |" for e in pending]

    rows = _read_log(log_path)
    teach = [r for r in rows if r.get("type") == "teach"][-limit:]
    fixed = [r for r in rows if r.get("corrections") or r.get("learned") and r.get("type") != "teach"]
    fixed = fixed[-limit:]
    if not teach and not fixed:
        out += ["", "No corrections yet."]
        return "\n".join(out) + "\n"

    if teach:
        out += ["", "## Taught", ""]
        for r in reversed(teach):
            ts = r.get("timestamp", "")[:16].replace("T", " ")
            got = r.get("learned")
            res = f"{got[0]} → {got[1]}" if got else "not understood"
            out.append(f"- {ts} — {res} (heard: “{_cell(r.get('raw_text', ''))}”)")
    if fixed:
        out += ["", "## Recent corrections", "", "| Time | Heard | Pasted | Changes |", "|---|---|---|---|"]
        for r in reversed(fixed):
            ts = r.get("timestamp", "")[:16].replace("T", " ")
            ch = "; ".join(f"{a} → {b}" for a, b in (r.get("corrections") or []))
            if r.get("learned"):
                ch += ("; " if ch else "") + "learned " + "; ".join(f"{a} → {b}" for a, b in r["learned"])
            out.append(f"| {ts} | {_cell(r.get('raw_text', '')[:80])} | {_cell(r.get('text', '')[:80])} | {_cell(ch)} |")
    return "\n".join(out) + "\n"


def write(vocab: Vocabulary, log_path: str, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render(vocab, log_path, vocab_path=vocab.path.replace(os.path.expanduser("~"), "~", 1)))
    return out_path
