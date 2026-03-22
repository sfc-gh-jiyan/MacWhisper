---
name: Live Subtitle Quality Fix
overview: "Fix three core issues with live subtitles: hallucinations/accuracy, bilingual support (English words being translated to Chinese), and queue congestion causing dropped chunks. Stay on feature/live-subtitles branch until stable."
todos:
  - id: fix-a
    content: Remove language='zh', add opencc t2s post-processing in _do_live_transcribe
    status: pending
  - id: fix-b
    content: Add hallucination filter (repetition, script validation, known phrases)
    status: pending
  - id: fix-c
    content: Increase live queue maxsize from 2 to 4
    status: pending
  - id: fix-d
    content: Run tests + manual validation with conference speech sample
    status: pending
isProject: false
---

# Live Subtitle Quality Improvement Plan

## Problem Analysis (from terminal logs)

Three categories of issues observed:

### Issue 1: Hallucinations (fabricated text)

- `"中文字幕君s"` (lines 624, 658) -- never spoken, Whisper invented this
- `"between between"` (lines 654, 664) -- repeated nonsense
- `"技术技术技术"` (line 650) -- repetition hallucination
- `"இ இ"` (line 630) -- Tamil script, completely wrong language

**Root cause**: `language="zh"` forces Whisper to interpret all audio as Chinese, so English words get mangled. The 2-second chunks lack context, making Whisper prone to hallucinations. Additionally, there is no hallucination filter -- repeated/nonsensical output goes straight to the overlay.

### Issue 2: English words translated instead of preserved

- "San Francisco" became "三番西" or "Disco" 
- "Artificial Intelligence" became "技术" or "智慧技术"
- "Machine Learning" disappeared entirely in some chunks

**Root cause**: `language="zh"` explicitly tells Whisper the spoken language is Chinese, so it translates English words. But removing it causes traditional/simplified inconsistency.

### Issue 3: Queue congestion (dropped chunks)

- `[WARN] Live queue full, dropped oldest chunk` appears frequently (lines 610-611, 573)
- This means inference is slower than the 2-second chunk interval -- chunks pile up and get discarded

**Root cause**: `queue.Queue(maxsize=2)` is too small. When inference takes 3+ seconds (especially first run with model loading), multiple chunks queue up and get dropped.

## Planned Fixes

### Fix A: Remove `language="zh"`, add post-processing for traditional-to-simplified

Change [app.py](app.py) `_do_live_transcribe` (line 495-507):

- Remove `language="zh"` and `initial_prompt` entirely -- let Whisper auto-detect language per chunk
- After Whisper returns text, run `opencc` (already installed) to convert any traditional characters to simplified
- This preserves English words as-is while normalizing Chinese to simplified

```python
import opencc
_t2s = opencc.OpenCC('t2s')
```

### Fix B: Add hallucination filter

Add a post-inference filter in `_do_live_transcribe` that rejects obvious hallucinations before sending to overlay:

1. **Repetition detector**: If a single word/phrase repeats 3+ times consecutively, discard (catches "Ok Ok Ok", "between between between", "技术技术技术")
2. **Script validator**: If output contains characters outside CJK + Latin + common punctuation, discard (catches Tamil "இ", Russian "Спасибо")
3. **Prompt leakage check**: If output matches known Whisper hallucination phrases ("Thank you for watching", "中文字幕君", etc.), discard

### Fix C: Increase queue size and add overlap

In [app.py](app.py):

- Increase `_live_queue` maxsize from 2 to 4 -- gives more buffer during model loading / first inference
- Optionally increase `LIVE_CHUNK_SECONDS` from 2 to 3 -- slightly longer chunks improve Whisper accuracy on short segments while keeping latency acceptable

### Fix D: Stay on feature branch

Confirmed: all work stays on `feature/live-subtitles`. Only merge to `main` after these fixes are validated by manual testing.

## Execution Order

1. Fix A (remove language lock + opencc) -- biggest impact, fixes bilingual + reduces hallucinations
2. Fix B (hallucination filter) -- catches remaining edge cases
3. Fix C (queue size) -- reduces dropped chunks
4. Run automated tests + manual test with the conference speech
5. Commit as separate commit on `feature/live-subtitles`

