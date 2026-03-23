# Comment for ml-explore/mlx-examples#1373

> Post to: https://github.com/ml-explore/mlx-examples/issues/1373

---

Hi @esphoenixc, I'm experiencing the exact same issue in a live subtitle application using `whisper-medium-mlx`.

## My scenario

I'm building a macOS menu bar app that provides **real-time live subtitles** during hold-to-record. The app uses a growing buffer architecture — as the user speaks, accumulated audio is periodically re-transcribed via `mlx_whisper.transcribe()` and displayed in a floating overlay.

## Observed behavior

With `whisper-medium-mlx` on 30s audio segments, `transcribe()` occasionally takes **14s** (vs the normal ~1.2s):

```
[INFO] Live transcribing 30.0s chunk (RMS=1057)...
[INFO] Live inference took 14.0s for 30.0s audio
[INFO] Live result (hallucination filtered): 想想想想想想想想想想想想想想想想想想想想想想想想想想...
```

The decoder gets stuck in a repetition loop, generating 448 identical tokens at ~30ms each (448 × 30ms ≈ 13.4s — matches the observed 14s perfectly).

During this time, the `_inference_lock` blocks all subtitle updates, causing the entire live subtitle system to **freeze for 14 seconds**.

I also observed a similar case with arrow symbols:

```
[INFO] Live inference took 13.6s for 0.8s audio
[INFO] Live result (hallucination filtered): ➞ ➞ ➞ ➞ ➞ ➞ ➞ ➞ ➞ ➞ ➞ ➞ ➞...
```

## Current workarounds (insufficient)

- `condition_on_previous_text=False` — reduces repetition in output but does NOT prevent the decoder loop itself
- Post-processing hallucination filter — catches the bad output but only AFTER the 14s delay
- Pause-based segmentation to keep segments shorter — reduces the probability but doesn't eliminate it

## What would help

A `max_tokens_per_segment` (or `max_new_tokens`) parameter exposed in `mlx_whisper.transcribe()` would allow us to cap decoder output and prevent runaway loops. For 30s audio, normal speech produces ~100-150 tokens. A limit of 200-256 would cover legitimate output while preventing 448-token decoder loops.

Alternatively, a callback or cancellation mechanism during `model.decode()` would let applications implement their own timeout logic.

## Environment

- `mlx-whisper` 0.4.2
- Model: `mlx-community/whisper-medium-mlx`
- macOS 15.x, Apple Silicon
- Python 3.13

Thanks for filing this — it's a real pain point for any real-time/streaming use case.
