# Plan: Growing Buffer + Full Replace (Live Subtitle v2)

**Branch:** `feature/live-subtitles` (continue on current branch)
**Prerequisite:** Commit current quality fixes first (hallucination filter, opencc, silence threshold)

---

## Problem

Current live subtitles use **fixed-interval isolated chunking** — each 3-second chunk is
transcribed independently with zero context from previous audio. This causes:

- Words split at chunk boundaries get lost ("San Fran" | "cisco" → neither chunk recognises it)
- No sentence context → Whisper mistranslates or hallucinates
- English terms translated to Chinese due to lack of surrounding context

## Solution: Growing Buffer + Full Replace

Inspired by Zoom's local transcription and the LocalAgreement algorithm
(ufal/whisper_streaming, INTERSPEECH 2023).

**Core idea:** Every N seconds, re-transcribe ALL accumulated audio (not just the new chunk),
then replace the entire overlay display with the latest result.

```
Current (broken):
  t=3s: transcribe frames[0:3]  → "我上周去了"
  t=6s: transcribe frames[3:6]  → "参加了一个"        ← no context
  t=9s: transcribe frames[6:9]  → "关于深度思考"       ← broken

New (correct):
  t=3s: transcribe frames[0:3]  → "我上周去了San Francisco参"
  t=6s: transcribe frames[0:6]  → "我上周去了San Francisco参加了一个conference。"
  t=9s: transcribe frames[0:9]  → "我上周去了San Francisco参加了一个conference。这个conference是有关于AI的。"
```

## Implementation Steps

### Step 1: Modify `_live_chunk_loop` — Growing Buffer

**Before:** `snapshot = self.frames[chunk_start:chunk_end]` (isolated 3s chunk)
**After:** `snapshot = self.frames[:]` (all accumulated audio)

- Remove `self._live_chunk_idx` (no longer needed)
- Each iteration sends the complete audio buffer to the queue
- Add window cap: if audio > `MAX_LIVE_WINDOW` (20s), only send the last 20s
  and preserve the confirmed prefix text

### Step 2: Modify `_update_overlay` → `_replace_overlay` — Full Replace

**Before:** Append new line to `_live_lines`, display all lines joined by `\n`
**After:** Replace entire overlay text with the latest transcription result

- Remove `self._live_lines` (no longer needed)
- New method `_replace_overlay(text)` simply sets the entire text
- Panel auto-resizes to fit the content (existing logic reused)

### Step 3: Add Window Cap for Long Recordings

For recordings exceeding 20 seconds:
- Keep `_committed_prefix`: text confirmed from earlier transcriptions
- Only transcribe the last 20s of audio
- Display = `_committed_prefix` + latest transcription of the window
- This prevents inference time from exceeding the chunk interval

### Step 4: Update `_do_live_transcribe`

- No changes to inference logic (hallucination filter, opencc, silence check all stay)
- Silence check applies to the TAIL of the buffer (last 3s) to detect if user stopped speaking
- Add `initial_prompt` with `_committed_prefix[-50:]` for continuity when window cap activates

### Step 5: Clean Up State Variables

Remove:
- `self._live_chunk_idx`
- `self._live_lines`

Add:
- `self._committed_prefix = ""`  (for window cap continuity)
- `MAX_LIVE_WINDOW = 20`  (seconds, max audio to transcribe at once)

### Step 6: Update Tests

- Update `test_app.py` to reflect new growing buffer behavior
- Add test for window cap logic
- Verify overlay replace (not append) behavior

---

## Performance Budget

| Audio Length | Inference Time (Small, M-series) | Within 3s interval? |
|---|---|---|
| 3s  | ~0.3s | Yes |
| 10s | ~0.8s | Yes |
| 15s | ~1.2s | Yes |
| 20s | ~2.0s | Yes (cap here) |
| 30s | ~3.0s | Borderline (window cap prevents this) |

## Risk & Mitigation

| Risk | Mitigation |
|---|---|
| Inference slower as buffer grows | Window cap at 20s |
| Queue congestion from larger payloads | Keep queue maxsize=4, drop oldest |
| Text "jumps" between iterations | Whisper is stable with growing context; acceptable for preview |

## Not In Scope

- LocalAgreement confirmed/unconfirmed text styling (future enhancement)
- Separate partial vs committed display (future enhancement)
- These are polish items; the growing buffer alone solves 90% of accuracy issues
