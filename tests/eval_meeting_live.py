"""Meeting live-subtitle evaluation: replay recordings through MeetingSession.

Unlike eval_meeting.py (whole-file strategies), this runs the REAL meeting
loop in real time — dual-channel selection, Silero VAD, segmenting — with
the Qwen3 live engine, and measures what users see during a meeting:

  - MER of the live transcript vs the reference (when one exists)
  - decode time per Qwen3 call and the segment length it decoded
    (each call re-decodes the whole segment, bounded by 15 s)
  - first-text latency: first speech chunk -> first subtitle text

Data stays out of the repo (see eval_meeting.py). Meeting files are written
to a temp dir, never to ~/.macwhisper/meetings.

  venv/bin/python tests/eval_meeting_live.py --engine qwen3
  venv/bin/python tests/eval_meeting_live.py --engine qwen3 --audio-only 20261019_003606

This loads a model and uses the GPU: ask the user before running it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from audio_capture import AudioSource  # noqa: E402
from eval_meeting import MEETINGS_DIR, _read_wav, discover_sessions  # noqa: E402
from eval_metrics import compute_wer  # noqa: E402

SAMPLE_RATE = 16000
CHUNK = 1024


class RealtimeWavSource(AudioSource):
    """Feeds a float32 track as int16 chunks at real-time speed."""

    def __init__(self, audio: np.ndarray, speed: float = 1.0, on_done=None):
        self._pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        self._speed = speed
        self._on_done = on_done
        self._active = False
        self._thread = None

    def start(self, callback):
        self._active = True

        def run():
            t0 = time.monotonic()
            for i in range(0, len(self._pcm), CHUNK):
                if not self._active:
                    return
                due = t0 + (i / SAMPLE_RATE) / self._speed
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                callback(self._pcm[i:i + CHUNK].reshape(-1, 1))
            if self._on_done:
                self._on_done()
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._active = False

    @property
    def is_active(self):
        return self._active


class TimingBackend:
    """Wraps a backend; records (audio seconds, decode seconds) per call."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[tuple[float, float]] = []

    def transcribe(self, audio, *a, **k):
        t0 = time.monotonic()
        try:
            return self.inner.transcribe(audio, *a, **k)
        finally:
            self.calls.append((len(audio) / SAMPLE_RATE, time.monotonic() - t0))

    def __getattr__(self, name):
        return getattr(self.inner, name)


@dataclass
class LiveResult:
    session_id: str
    engine: str
    audio_s: float
    transcript: str
    segments: int
    first_text_s: float | None
    decode_calls: list = field(default_factory=list)
    mer: float | None = None

    def summary(self) -> dict:
        decodes = [d for _, d in self.decode_calls]
        longest = max((a for a, _ in self.decode_calls), default=0.0)
        return {
            "session": self.session_id, "engine": self.engine,
            "audio_s": round(self.audio_s, 1), "segments": self.segments,
            "mer": None if self.mer is None else round(self.mer, 3),
            "first_text_s": None if self.first_text_s is None else round(self.first_text_s, 2),
            "decodes": len(decodes),
            "decode_p50_s": round(float(np.median(decodes)), 2) if decodes else None,
            "decode_max_s": round(max(decodes), 2) if decodes else None,
            "longest_segment_decoded_s": round(longest, 1),
            "decode_time_over_1s": sum(1 for d in decodes if d > 1.0),
        }


def replay(session_id: str, mic: np.ndarray, sys_a: np.ndarray | None, *, engine: str,
           backend, vad, reference: str = "", speed: float = 1.0) -> LiveResult:
    """Run one recording through a real MeetingSession and collect metrics."""
    import meeting
    from meeting import MeetingSession

    timing = TimingBackend(backend)
    import qwen_meeting
    kwargs = {"processor_factory": lambda: qwen_meeting.QwenSegmentProcessor(timing),
              "extended_silence_ms": 800}

    done = {"n": 0}
    finished = threading.Event()
    expected = 2 if sys_a is not None else 1

    def source_done():
        done["n"] += 1
        if done["n"] >= expected:
            finished.set()

    first = {"speech": None, "text": None}
    t_start = time.monotonic()

    def on_update(confirmed, unconfirmed, segments, debug=None):
        if first["text"] is None and (confirmed.strip() or unconfirmed.strip()):
            first["text"] = time.monotonic() - t_start

    tmp = tempfile.mkdtemp(prefix="mw_eval_live_")
    old_dir = meeting.MEETINGS_DIR
    meeting.MEETINGS_DIR = tmp            # never write into the user's meetings
    try:
        session = MeetingSession(backend=backend, vad=vad,
                                 audio_source=RealtimeWavSource(mic, speed, source_done),
                                 min_chunk_size=1.0, max_buffer_s=8.0,
                                 on_update=on_update, **kwargs)
        if sys_a is not None:
            session._dual_channel = True
            session._audio_source = None
            session._mic_source = RealtimeWavSource(mic, speed, source_done)
            session._sys_source = RealtimeWavSource(sys_a, speed, source_done)
        session.start()
        finished.wait(timeout=len(mic) / SAMPLE_RATE / speed + 60)
        time.sleep(1.5)                   # let the last segment close on silence
        transcript = session.stop()
    finally:
        meeting.MEETINGS_DIR = old_dir

    result = LiveResult(session_id=session_id, engine=engine, audio_s=len(mic) / SAMPLE_RATE,
                        transcript=transcript, segments=len(session.segments),
                        first_text_s=first["text"],
                        decode_calls=list(timing.calls))
    if reference:
        result.mer = compute_wer(transcript, reference)
    return result


def _load_backend(qwen_model: str):
    from qwen_backend import QwenSidecarBackend
    b = QwenSidecarBackend(model_repo=qwen_model)
    if not b.start():
        raise SystemExit(f"Qwen3 sidecar did not start: {getattr(b, 'last_error', '')}")
    return b


def main(argv=None) -> int:
    from eval_meeting import DEFAULT_QWEN_MODEL
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--engine", choices=["qwen3"], default="qwen3")
    p.add_argument("--session", action="append", help="Session with a reference (repeatable)")
    p.add_argument("--audio-only", action="append", default=[],
                   help="Session without a reference: speed/latency only")
    p.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    p.add_argument("--json", help="Write summaries to this JSON file")
    p.add_argument("--show-text", action="store_true")
    args = p.parse_args(argv)

    jobs = []
    for s in discover_sessions():
        if args.session and s.session_id not in args.session:
            continue
        sys_a = _read_wav(s.sys_path) if s.sys_path else None
        jobs.append((s.session_id, _read_wav(s.mic_path), sys_a, s.reference))
    for sid in args.audio_only:
        mic_p = os.path.join(MEETINGS_DIR, f"meeting_{sid}_mic.wav")
        sys_p = os.path.join(MEETINGS_DIR, f"meeting_{sid}_sys.wav")
        jobs.append((sid, _read_wav(mic_p),
                     _read_wav(sys_p) if os.path.isfile(sys_p) else None, ""))
    if not jobs:
        print("No recordings found — nothing to measure.")
        return 1

    from vad import VoiceActivityDetector
    backend = _load_backend(args.qwen_model)
    rows = []
    try:
        for sid, mic, sys_a, ref in jobs:
            print(f"[eval] {sid} ({len(mic)/SAMPLE_RATE:.0f}s) engine={args.engine} ...", flush=True)
            r = replay(sid, mic, sys_a, engine=args.engine, backend=backend,
                       vad=VoiceActivityDetector(), reference=ref)
            rows.append(r.summary())
            print("  ", json.dumps(rows[-1], ensure_ascii=False), flush=True)
            if args.show_text:
                print("   text:", r.transcript.replace("\n", " ")[:600])
    finally:
        if hasattr(backend, "close"):
            backend.close()
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
