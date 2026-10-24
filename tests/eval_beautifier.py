"""Measure the AI beautifier on the user's real dictations (no app changes).

    venv/bin/python tests/eval_beautifier.py            # 30 dictations x 4 levels
    venv/bin/python tests/eval_beautifier.py --n 6      # quick run

Starts the local LLM sidecar (.venv-llm, uses the GPU). Writes a side-by-side
report to ~/.macwhisper/eval/beautify_<time>.md: per level the seconds per
dictation and how many would miss the 5 s paste budget.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import beautifier as bf  # noqa: E402
import llm_backend  # noqa: E402

DATA = os.path.expanduser("~/.macwhisper")
LEVELS = ("light", "medium", "heavy", "custom")
SAMPLE_CUSTOM = "改成简洁、礼貌的 Slack 消息，保留所有要点。"
BUDGET_S = 5.0


def sample(n: int, seed: int = 0) -> list[str]:
    rows = [json.loads(line) for line in open(os.path.join(DATA, "transcripts.jsonl"), encoding="utf-8")
            if line.strip()]
    texts = [r["text"] for r in rows if r.get("text") and r.get("timestamp", "") >= "2026-10-18"]
    texts = sorted(set(texts), key=len)
    rng = random.Random(seed)
    third = len(texts) // 3                      # short / medium / long, equally
    picks = []
    for part in (texts[:third], texts[third:2 * third], texts[2 * third:]):
        picks += rng.sample(part, min(len(part), n // 3 + (1 if len(picks) < n % 3 else 0)))
    return sorted(picks[:n], key=len)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=60.0, help="measure the true time")
    a = ap.parse_args()
    texts = sample(a.n)
    backend = llm_backend.LLMSidecarBackend(max_restarts=1)
    t0 = time.time()
    if not backend.start(timeout=300):
        print("LLM did not start:", backend.last_error)
        return 1
    print(f"LLM ready in {time.time() - t0:.0f}s", flush=True)
    bf.beautify("热身一下", "light", llm=backend.complete, timeout=60)
    results = []
    try:
        for i, text in enumerate(texts, 1):
            row = {"text": text, "chars": len(text)}
            for level in LEVELS:
                t1 = time.time()
                out = bf.beautify(text, level, llm=backend.complete, custom=SAMPLE_CUSTOM,
                                  timeout=a.timeout)
                row[level] = {"secs": round(time.time() - t1, 2), "text": out}
            results.append(row)
            print(f"[{i}/{len(texts)}] {len(text)} chars: " +
                  " ".join(f"{lv}={row[lv]['secs']}s" for lv in LEVELS), flush=True)
    finally:
        backend.close()
    stamp = time.strftime("%Y%m%d_%H%M")
    json.dump(results, open(os.path.join(DATA, "eval", f"beautify_{stamp}.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
    path = os.path.join(DATA, "eval", f"beautify_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(results))
    print("REPORT", path)
    return 0


def render(results) -> str:
    out = ["# AI 美化实测（你的真实听写）", "",
           f"{len(results)} 条听写，每条四档各跑一次。自定义档用的示例规则：「{SAMPLE_CUSTOM}」", "",
           "| 档位 | 平均秒数 | 最慢 | 超过 5 秒的条数 | 出错 |", "|---|---|---|---|---|"]
    for lv in LEVELS:
        secs = [r[lv]["secs"] for r in results]
        over = sum(s > BUDGET_S for s in secs)
        err = sum(r[lv]["text"] is None for r in results)
        out.append(f"| {lv} | {sum(secs) / len(secs):.1f} | {max(secs):.1f} | {over} | {err} |")
    out += ["", "## 按长度：多长的话能在 5 秒内完成", "", "| 字数 | light | medium | heavy | custom |",
            "|---|---|---|---|---|"]
    for r in results:
        out.append(f"| {r['chars']} | " + " | ".join(f"{r[lv]['secs']}" for lv in LEVELS) + " |")
    out += ["", "## 逐条对照", ""]
    for i, r in enumerate(results, 1):
        out += [f"### {i}. （{r['chars']} 字）", "", f"**原文**：{r['text']}", ""]
        for lv in LEVELS:
            out += [f"**{lv}**（{r[lv]['secs']}s）：{r[lv]['text'] if r[lv]['text'] is not None else '（失败，会贴原文）'}", ""]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
