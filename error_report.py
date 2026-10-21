"""Summarize errors from the MacWhisper log into a de-duplicated report.

Groups ERROR/CRITICAL lines plus a few notable WARNING patterns by a
normalized "signature" (numbers, thread ids and paths stripped) so the
same problem seen many times shows up as one row with a count.
"""

from __future__ import annotations

import logging
import os

import paths
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\S+) (ERROR|CRITICAL|WARNING) (.*)$")

# WARNING lines worth reporting (everything else at WARNING is routine).
_NOTABLE_WARNINGS = (
    "engine failed",
    "Restarting Qwen sidecar",
    "re-initializing PortAudio",
    "Excessive words",
    "did not exit cleanly",
    "timed out",
)

_TS_FMT = "%Y-%m-%d %H:%M:%S"
_SAMPLE_MAX = 220


@dataclass
class ErrorGroup:
    signature: str
    sample: str
    count: int
    first: str
    last: str


def _signature(module: str, msg: str) -> str:
    s = re.sub(r"Thread-\d+", "Thread-N", msg)
    s = re.sub(r"(/[\w.\-~]+)+", "<path>", s)
    s = re.sub(r"\d+(\.\d+)?", "N", s)
    return f"{module}: {s[:160]}"


def _is_reportable(level: str, msg: str) -> bool:
    if level in ("ERROR", "CRITICAL"):
        return True
    return any(p in msg for p in _NOTABLE_WARNINGS)


def summarize(log_path: str, days: int = 1, now: datetime | None = None) -> list[ErrorGroup]:
    """Return error groups from the last `days` days, most frequent first."""
    now = now or datetime.now()
    cutoff = now - timedelta(days=days)
    groups: dict[str, ErrorGroup] = {}
    try:
        f = open(log_path, encoding="utf-8", errors="replace")
    except OSError as e:
        logger.debug("No log to summarize (%s): %s", log_path, e)
        return []
    with f:
        for line in f:
            m = _LINE_RE.match(line.rstrip("\n"))
            if not m:
                continue
            ts, module, level, msg = m.groups()
            if not _is_reportable(level, msg):
                continue
            try:
                if datetime.strptime(ts, _TS_FMT) < cutoff:
                    continue
            except ValueError:
                continue
            sig = _signature(module, msg)
            g = groups.get(sig)
            if g is None:
                groups[sig] = ErrorGroup(sig, f"{module} {level} {msg}"[:_SAMPLE_MAX], 1, ts, ts)
            else:
                g.count += 1
                g.last = ts
    result = sorted(groups.values(), key=lambda g: (-g.count, g.first))
    logger.debug("Error summary: %d groups from %s (last %d day(s))", len(result), log_path, days)
    return result


def render(groups: list[ErrorGroup], days: int, now: datetime | None = None) -> str:
    now = now or datetime.now()
    lines = [
        "# MacWhisper error report",
        "",
        f"Generated {now.strftime(_TS_FMT)} — last {days} day(s), {sum(g.count for g in groups)} event(s).",
        "",
    ]
    if not groups:
        lines.append("No errors in this period.")
        return "\n".join(lines) + "\n"
    lines += ["| Count | First | Last | Message |", "|---|---|---|---|"]
    for g in groups:
        msg = g.sample.replace("|", "\\|")
        lines.append(f"| {g.count} | {g.first} | {g.last} | {msg} |")
    return "\n".join(lines) + "\n"


def write_report(log_path: str, out_path: str, days: int = 1, now: datetime | None = None) -> str:
    groups = summarize(log_path, days=days, now=now)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render(groups, days=days, now=now))
    logger.info("Wrote error report (%d groups) to %s", len(groups), out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Summarize MacWhisper log errors")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--log", default=paths.MAIN_LOG)
    args = ap.parse_args()
    print(render(summarize(args.log, days=args.days), days=args.days), end="")
