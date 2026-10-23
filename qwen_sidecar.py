"""Qwen3-ASR sidecar process for MacWhisper.

Runs inside the isolated .venv-qwen (mlx-audio + newer mlx) so the app's own
venv and mlx-whisper stay untouched. The app talks to it over stdin/stdout
with one JSON object per line:

  app -> sidecar   {"id": 1, "cmd": "transcribe", "pcm16_b64": "...", "language": "Chinese"}
                   {"id": 2, "cmd": "ping"}
                   {"id": 3, "cmd": "shutdown"}
  sidecar -> app   {"event": "ready", "model": "...", "load_s": 1.2}
                   {"id": 1, "ok": true, "text": "...", "infer_s": 0.12, "audio_s": 3.4}
                   {"id": 1, "ok": false, "error": "..."}

Audio is 16 kHz mono int16 PCM, base64 encoded. stdout carries ONLY protocol
lines; everything else (library logs, progress bars) goes to stderr.

Usage:
  .venv-qwen/bin/python qwen_sidecar.py --model mlx-community/Qwen3-ASR-0.6B-bf16
  .venv-qwen/bin/python qwen_sidecar.py --model ... --self-test   # load + 1 s decode, exit
  python qwen_sidecar.py --fake                                    # no model (tests)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_LANGUAGE = "Chinese"  # keeps code-switched English verbatim (no translation)
MAX_HOTWORDS = 64      # keep the prompt (and decode latency) small


def decode_audio(pcm16_b64: str) -> np.ndarray:
    raw = base64.b64decode(pcm16_b64)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def parse_line(line: str) -> dict | None:
    try:
        req = json.loads(line)
    except ValueError:
        return None
    return req if isinstance(req, dict) else None


def handle(req: dict, model_fn) -> dict:
    """Process one request. Never raises — errors become {"ok": false}."""
    rid = req.get("id")
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"id": rid, "ok": True, "pong": True}
    if cmd != "transcribe":
        return {"id": rid, "ok": False, "error": f"unknown cmd: {cmd!r}"}
    if not req.get("pcm16_b64"):
        return {"id": rid, "ok": False, "error": "missing pcm16_b64"}
    try:
        audio = decode_audio(req["pcm16_b64"])
        t0 = time.time()
        language = req.get("language") or DEFAULT_LANGUAGE
        hotwords = req.get("hotwords")
        if isinstance(hotwords, list) and hotwords:
            text = model_fn(audio, language, [str(h) for h in hotwords][:MAX_HOTWORDS])
        else:
            text = model_fn(audio, language)
        return {"id": rid, "ok": True, "text": text or "",
                "infer_s": round(time.time() - t0, 3),
                "audio_s": round(len(audio) / SAMPLE_RATE, 2)}
    except Exception as e:  # report, keep serving
        return {"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"}


# Generation budget. Normal speech is ~4-5 tokens per second of audio (real
# meetings, 2026-10-23); a repetition loop ("那个那个那个…", the hotword list read
# back on silence) otherwise runs to mlx-audio's default 8192 tokens and times out.
# No upper ceiling: mlx-audio's max_tokens is a TOTAL over all 20-min chunks, so
# the old default 8192 also cut off the final pass of meetings over ~30 min.
TOKENS_PER_AUDIO_S = 25
MIN_TOKENS = 128


def token_cap(audio_s: float) -> int:
    return int(max(MIN_TOKENS, audio_s * TOKENS_PER_AUDIO_S))


def make_run(model):
    """model_fn for a loaded mlx-audio model, with a token budget per request."""
    def run(audio: np.ndarray, language: str, hotwords=None) -> str:
        cap = token_cap(len(audio) / SAMPLE_RATE)
        if hotwords and not getattr(run, "no_hotwords", False):
            try:
                return model.generate(audio, language=language, hotwords=hotwords,
                                      max_tokens=cap).text
            except TypeError as e:   # older mlx-audio: no hotwords argument
                run.no_hotwords = True
                print(f"hotwords unsupported ({e}); continuing without", file=sys.stderr, flush=True)
        return model.generate(audio, language=language, max_tokens=cap).text
    return run


def load_model_fn(model_repo: str):
    from mlx_audio.stt.utils import load
    run = make_run(load(model_repo))
    run(np.zeros(SAMPLE_RATE, dtype=np.float32), DEFAULT_LANGUAGE)  # warm Metal kernels
    return run


def fake_model_fn(audio: np.ndarray, language: str, hotwords=None) -> str:
    """Deterministic stand-in for tests: text encodes length and language."""
    if os.environ.get("QWEN_FAKE_CRASH_AFTER"):
        fake_model_fn.calls = getattr(fake_model_fn, "calls", 0) + 1
        if fake_model_fn.calls > int(os.environ["QWEN_FAKE_CRASH_AFTER"]):
            os._exit(3)
    delay = float(os.environ.get("QWEN_FAKE_DELAY_S", "0"))
    if delay:
        time.sleep(delay)
    text = f"fake {len(audio) / SAMPLE_RATE:.1f}s {language}"
    return f"{text} hw={len(hotwords)}" if hotwords else text


def watch_parent(interval: float = 2.0) -> None:
    """Exit if the app that started us is gone (killed, crashed, kill -9).

    Normally a closed stdin ends serve(); this also covers the case where the
    app dies while a long decode is running, so the model never lingers in
    GPU memory.
    """
    import threading

    parent = os.getppid()

    def run():
        while True:
            time.sleep(interval)
            if os.getppid() != parent:
                print("[qwen_sidecar] parent process gone — exiting", file=sys.stderr)
                os._exit(0)

    threading.Thread(target=run, daemon=True, name="parent-watch").start()


def serve(model_fn, model_name: str, load_s: float, out, inp) -> None:
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
        send(handle(req, model_fn))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen3-ASR-0.6B-bf16")
    p.add_argument("--fake", action="store_true", help="No model (tests)")
    p.add_argument("--self-test", action="store_true", help="Load, decode 1 s, exit")
    args = p.parse_args(argv)

    # Reserve the real stdout for protocol lines; route all other output to stderr.
    proto_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    t0 = time.time()
    try:
        model_fn = fake_model_fn if args.fake else load_model_fn(args.model)
    except Exception as e:
        proto_out.write(json.dumps({"event": "error", "error": f"{type(e).__name__}: {e}"}) + "\n")
        proto_out.flush()
        return 2
    load_s = time.time() - t0

    if args.self_test:
        text = model_fn(np.zeros(SAMPLE_RATE, dtype=np.float32), DEFAULT_LANGUAGE)
        print(f"[qwen_sidecar] self-test ok: model={args.model} load={load_s:.1f}s "
              f"text={text!r}", file=sys.stderr)
        return 0

    watch_parent()
    serve(model_fn, "fake" if args.fake else args.model, load_s, proto_out, sys.stdin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
