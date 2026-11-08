# Meeting Mode: label every line Mic or Remote

Date: 2026-11-07 · Status: draft for Jackie's review

## Intent

Jackie records customer calls (phone apps, customer-hosted meetings with no
Zoom transcript) and reads the transcript afterwards to remember details. Today
the transcript is one undifferentiated stream, so she cannot tell what she said
from what the customer said.

Qwen3-ASR returns text only; it has no speaker information. But Meeting Mode
already captures two separate sources: the microphone (Jackie: she nearly
always speaks through it) and system audio (everyone on the far end, via
ScreenCaptureKit). Labelling by source is reliable and needs no speaker
recognition model.

## Goals

- Every transcript line says where it came from: `Mic` or `Remote`.
- Nothing is lost when both sides talk at once (today the quieter channel's
  audio is dropped every ~1 s).
- Fix negative paragraph timestamps (`[-1:56]` in the 2026-11-07 meeting).
- Leave room to split `Remote` into `Remote 1`, `Remote 2` later.

## Non-goals

- Telling individual people apart (diarization). Later, separate work.
- Echo cancellation in the audio driver.
- Any change to Hold-to-Record / Push to Talk.

## Anti-goals (a failure even if it "works")

- Lost meeting content, in any path (live, outage buffering, final pass, quit).
- The same sentence appearing twice because the mic picked up the speakers.
- A noticeably slower live transcript, or Push to Talk blocked longer.

## Approaches considered

| | How | Trade-off |
|---|---|---|
| A | Keep "louder channel wins each second"; tag each paragraph with the winner | Smallest change; still drops crosstalk; mixed paragraphs get one label |
| **B (chosen)** | Transcribe mic and system audio as two independent tracks; merge by time | Nothing lost in crosstalk, labels are exact; more Qwen work when both talk |
| C | B, plus a diarization model on the Remote track after the meeting | Can tell customers apart; model quality on mixed Chinese/English unverified |

Jackie picked **B** and the labels **Mic / Remote** (2026-11-07). C stays
open: installed `mlx-audio` 0.5.9 ships `moss_transcribe_diarize`,
`vibevoice_asr` and an ECAPA speaker-embedding model, none tried yet.

## Design

### Output (decided)

Markdown, one paragraph per segment, ordered by start time:

```
**[00:12] Mic:** 我们先看一下这个 query profile
**[00:18] Remote:** OK, which warehouse is this?
```

Labels are English (the app UI is English). SRT/VTT cues carry the same
prefix (`Mic: ...`). The plain-text transcript (clipboard, `.txt`) uses
`Mic: ...` / `Remote: ...` lines. Single-channel meetings (system audio
unavailable or off) keep today's unlabelled format.

### Two tracks

A new `ChannelTrack` holds the per-source state that `MeetingSession` keeps
once today: live processor, its own Silero VAD instance (stateful RNN: one per
stream, each chunk fed exactly once), segment start time, VAD skip count, and
the source label. `MeetingSession` owns one track (single channel) or two
(dual channel); the meeting loop feeds each track only its own chunks. The
RMS "overlap priority" selection is removed.

Segments become `MeetingSegment(text, start_time, end_time, words, source)`,
`source` in {`"mic"`, `"remote"`, `None`}; `None` = single channel, rendered
without a label. A later diarizer only needs to set a finer `source`
(`"remote:1"`), rendered `Remote 1`.

### Qwen load

Both tracks share the single sidecar (its lock already serialises requests).
Each track decodes only while its VAD says speech, so the extra work is mainly
during crosstalk. The live loop must not starve one track: tracks take turns
per loop tick.

### Echo (speakers instead of headphones)

When the mic hears the speakers, the same words arrive on both tracks. Both
tracks transcribe normally; duplicates are removed **by text, after commit**:
a Mic segment whose time range overlaps a Remote segment and whose text is
mostly the same (normalized similarity >= `ECHO_SIMILARITY`, start at 0.8) is
dropped. Logged (both texts, score) so the threshold can be tuned. An
audio-level rule was rejected: with headphones it would delete Jackie's soft
speech during crosstalk (lost content). *(Builder-check call.)*

Raw audio is never altered: `_mic.wav` and `_sys.wav` stay exactly as
captured. The final pass applies the same text rule to its two blocks only
as a note, never by deleting text from the whole-meeting result.

### Timestamps

Segment start = wall-clock time of the **first speech chunk** of that segment
on that track (set on the first VAD speech chunk, and in the outage catch-up
path), not the time the previous segment closed, and not the processor's word
timestamps (which reset when the buffer is trimmed and produced `[-1:56]`).
Mic and Remote lines are ordered by this time. *(Builder-check call.)*

### Live overlay

Shows the latest line of each track, labelled, newest at the bottom:

```
Mic: 我们先看一下这个 query profile
Remote: OK, which warehouse is
```

### Engine outage and fallback

The existing outage path (buffer audio, retry the sidecar, catch up) runs per
track; one restart attempt serves both. `had_engine_outage` is set if either
track lost the engine.

### Final pass (off for Jackie, on by default)

Today it mixes both channels into one request. Dual channel: transcribe the
mic and system recordings as two requests, then label each result. Because a
single whole-meeting request has no timestamps, the final version lists the
**Mic** text and the **Remote** text as two blocks, above the labelled,
time-ordered live transcript, which stays in the file. Single channel:
unchanged.

Two whole-meeting requests take about twice as long, so Push to Talk is paused
about twice as long. Accepted: Jackie has the final pass off; it only runs
forced after an engine outage. `FINAL_PASS_S_PER_AUDIO_S`-based estimates use
the total audio of both tracks. *(Builder-check call.)*

If the engine gave up, the kept audio for "Retry Last Failed Recording" is
saved **per track** (two files), so a retry still produces labels.
*(Builder-check call.)*

### Logging

Per track: segment commits (`[MEETING] Mic segment #n ...`), echo drops,
outage start/end. Session start logs whether it is single or dual channel.

## Constraints

- One Qwen sidecar; its lock is held for a whole request.
- `make smoke` after every change; `make qa` applies (`meeting.py` changes).
  Note: `make qa` scores raw sidecar output and the saved live transcript,
  so it does not measure the new labels; the measurable check is replaying
  the two eval recordings (they have `_mic.wav` + `_sys.wav`) through
  `MeetingSession` with a fake clock and checking labels and order.
- Never lose content; the live transcript file is written before any final
  pass.

## Left to the builder

- Exact `ChannelTrack` API and how the outage code is split between session
  and track.
- How tracks take turns in the loop.
- Test doubles for two sources.
