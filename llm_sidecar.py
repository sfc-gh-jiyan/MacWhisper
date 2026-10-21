"""Local text LLM sidecar (mlx-lm), JSON lines over stdin/stdout.

Runs inside the isolated .venv-llm, so the app venv and the Qwen3-ASR
environment are never changed. Same protocol shape as qwen_sidecar.py:

  -> {"event": "ready", "model": ..., "load_s": ..., "pid": ...}
  <- {"id": 1, "cmd": "complete", "system": "...", "prompt": "...", "max_tokens": 512}
  -> {"id": 1, "ok": true, "text": "...", "infer_s": 1.2}
  <- {"cmd": "ping"} / {"cmd": "shutdown"}

Usage:
  .venv-llm/bin/python llm_sidecar.py --model mlx-community/Qwen3-4B-Instruct-2507-4bit
  .venv-llm/bin/python llm_sidecar.py --fake        # tests: echoes MW_LLM_FAKE_RESPONSE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MAX_TOKENS_CAP = 2048


def parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def handle(req: dict, complete_fn) -> dict:
    """Process one request. Never raises — errors become {"ok": false}."""
    rid = req.get("id")
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"id": rid, "ok": True, "pong": True}
    if cmd != "complete":
        return {"id": rid, "ok": False, "error": f"unknown cmd: {cmd!r}"}
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return {"id": rid, "ok": False, "error": "missing prompt"}
    try:
        max_tokens = max(1, min(int(req.get("max_tokens", 512)), MAX_TOKENS_CAP))
        t0 = time.time()
        text = complete_fn(str(req.get("system") or ""), prompt, max_tokens)
        return {"id": rid, "ok": True, "text": text or "", "infer_s": round(time.time() - t0, 3)}
    except Exception as e:  # report, keep serving
        return {"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"}


def load_complete_fn(model_repo: str):
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_repo)
    sampler = make_sampler(temp=0.0)

    def run(system: str, prompt: str, max_tokens: int) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return generate(model, tokenizer, prompt=text, max_tokens=max_tokens,
                        sampler=sampler, verbose=False)

    run("", "ok", 1)  # warm Metal kernels
    return run


def fake_complete_fn(system: str, prompt: str, max_tokens: int) -> str:
    delay = float(os.environ.get("MW_LLM_FAKE_DELAY_S", "0"))
    if delay:
        time.sleep(delay)
    return os.environ.get("MW_LLM_FAKE_RESPONSE", f"fake {len(prompt)} chars")


def watch_parent(interval: float = 2.0) -> None:
    """Exit if the app that started us is gone, so the model never lingers in GPU memory."""
    parent = os.getppid()

    def run():
        while True:
            time.sleep(interval)
            if os.getppid() != parent:
                print("[llm_sidecar] parent process gone — exiting", file=sys.stderr)
                os._exit(0)

    threading.Thread(target=run, daemon=True, name="parent-watch").start()


def serve(complete_fn, model_name: str, load_s: float, out, inp) -> None:
    def send(obj):
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out.flush()

    send({"event": "ready", "model": model_name, "load_s": round(load_s, 2), "pid": os.getpid()})
    for line in inp:
        req = parse_line(line)
        if req is None:
            send({"id": None, "ok": False, "error": "bad request line"})
            continue
        if req.get("cmd") == "shutdown":
            send({"id": req.get("id"), "ok": True, "bye": True})
            return
        send(handle(req, complete_fn))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--fake", action="store_true", help="No model (tests)")
    p.add_argument("--self-test", action="store_true", help="Load, generate a few tokens, exit")
    args = p.parse_args(argv)

    proto_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    t0 = time.time()
    try:
        fn = fake_complete_fn if args.fake else load_complete_fn(args.model)
    except Exception as e:
        proto_out.write(json.dumps({"event": "error", "error": f"{type(e).__name__}: {e}"}) + "\n")
        proto_out.flush()
        return 2
    load_s = time.time() - t0
    if args.self_test:
        print(f"[llm_sidecar] self-test ok: {fn('', 'Say OK.', 8)!r} load={load_s:.1f}s", file=sys.stderr)
        return 0
    watch_parent()
    serve(fn, "fake" if args.fake else args.model, load_s, proto_out, sys.stdin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
