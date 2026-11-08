"""Meeting Mode session manager for MacWhisper.

Provides continuous recording with automatic segmentation via VAD,
real-time Qwen3-ASR transcription (qwen_meeting.QwenSegmentProcessor), and transcript
export (Markdown, SRT, VTT).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import time
import wave
from dataclasses import dataclass, field

import numpy as np

from audio_capture import AudioSource, MicrophoneSource, SystemAudioSource
from asr_backend import ASRBackend, WordTimestamp
from vad import VoiceActivityDetector
from meeting_labels import is_echo, ordered, source_label, text_similarity
from meeting_track import ChannelTrack
from subtitle_export import export_cues, export_srt, export_vtt, word_cues  # noqa: F401

logger = logging.getLogger(__name__)

# Meeting data directory
SAMPLE_RATE = 16000

MEETINGS_DIR = os.path.join(os.path.expanduser("~/.macwhisper"), "meetings")


@dataclass
class MeetingSegment:
    """A paragraph-level segment committed by VAD silence detection."""
    text: str
    start_time: float  # seconds from meeting start
    end_time: float    # seconds from meeting start
    words: list[tuple[float, float, str]] = field(default_factory=list)
    source: str | None = None


def _labelled(source: str | None, text: str) -> str:
    """'Mic: text' for a labelled source; unlabelled text unchanged."""
    label = source_label(source)
    return f"{label}: {text}" if label else text


class MeetingSession:
    """Manages a single meeting recording session.

    Lifecycle: idle -> recording -> (paused -> recording)* -> stopped

    Usage:
        session = MeetingSession(backend, vad)
        session.start()           # begin recording
        # ... session runs in background ...
        session.pause() / session.resume()
        transcript = session.stop()  # returns full transcript
        session.export("~/meeting.md")
    """

    def __init__(
        self,
        backend: ASRBackend,
        vad: VoiceActivityDetector | None = None,
        audio_source: AudioSource | None = None,
        min_chunk_size: float = 0.5,
        max_buffer_s: float = 5.0,
        language: str | None = None,
        extended_silence_ms: int = 2000,
        on_update: callable | None = None,
        capture_system_audio: bool = False,
        save_audio: bool = False,
        processor_factory=None,
        on_engine_failed=None,
        engine_restart=None,
        engine_retry_s: float = 10.0,
        engine_max_attempts: int = 5,
        on_engine_recovered=None,
        engine_ready=None,
        vad_factory: callable | None = None,
    ):
        """processor_factory() builds the live processor (default: a Qwen3
        QwenSegmentProcessor over `backend`).

        If it raises QwenUnavailable, on_engine_failed(exc) is called and
        audio keeps collecting in the same processor while engine_restart()
        (optional, -> bool) is retried in the background every engine_retry_s.
        Once it is back the buffered audio is decoded in chunks and
        on_engine_recovered() is called; the meeting final pass covers
        anything still missing. engine_ready() -> bool (optional) reports
        whether the engine has loaded; if not, the meeting starts buffering.
        vad_factory() (optional) builds the Remote track's own VAD in dual
        channel (Silero is stateful: never share one between tracks)."""
        self.backend = backend
        self._processor_factory = processor_factory
        self._on_engine_failed = on_engine_failed
        self._engine_restart = engine_restart
        self._engine_retry_s = engine_retry_s
        self._engine_max_attempts = engine_max_attempts
        self._on_engine_recovered = on_engine_recovered
        self._engine_back = threading.Event()   # set by the restart thread
        self._engine_down = False               # Qwen3-only: engine restarting, audio buffered
        self.had_engine_outage = False          # app forces the final pass if True
        self._engine_gave_up = False
        self._engine_ready = engine_ready       # () -> bool: engine loaded (Qwen3 only)
        # Raw audio is only written to disk when explicitly enabled —
        # recording other participants requires their consent.
        self.save_audio = save_audio
        self.audio_paths: list[str] = []
        self.vad = vad
        self._vad_factory = vad_factory
        self._tick = 0                          # alternates track order per loop tick
        self.min_chunk_size = min_chunk_size
        self.max_buffer_s = max_buffer_s
        self.language = language
        self.extended_silence_ms = extended_silence_ms

        # Callback: on_update(confirmed_text, unconfirmed_text, segments)
        self._on_update = on_update

        # Audio source setup
        # Dual-channel mode: separate mic + sys sources for overlap handling
        # Single-channel mode: mic only (or caller-supplied audio_source)
        self._dual_channel = False
        self._mic_source: AudioSource | None = None
        self._sys_source: AudioSource | None = None
        self._audio_source: AudioSource | None = None

        if audio_source is not None:
            # Caller-supplied source (e.g. tests with WavFileSource)
            self._audio_source = audio_source
        elif capture_system_audio:
            mic = MicrophoneSource(sample_rate=SAMPLE_RATE)
            sys_audio = SystemAudioSource(sample_rate=SAMPLE_RATE)
            if sys_audio.available:
                # Dual-channel: keep sources separate for overlap priority
                self._dual_channel = True
                self._mic_source = mic
                self._sys_source = sys_audio
            else:
                # System audio unavailable — fall back to mic only
                self._audio_source = mic
        else:
            self._audio_source = MicrophoneSource(sample_rate=SAMPLE_RATE)

        # State
        self._state: str = "idle"  # idle, recording, paused, stopped
        self._state_lock = threading.Lock()
        self._loop_thread: threading.Thread | None = None
        self._loop_done = threading.Event()
        self._loop_done.set()

        # Audio frames — dual buffers for overlap handling
        self._mic_frames: list[np.ndarray] = []
        self._sys_frames: list[np.ndarray] = []
        self._frames_lock = threading.Lock()
        # Serializes processor use between the loop thread and stop():
        # a Qwen3 decode can outlast stop()'s wait (review CRITICAL).
        self._proc_lock = threading.RLock()

        # Meeting data
        self._segments: list[MeetingSegment] = []
        self.transcript_path: str | None = None  # set by _auto_save()
        self._meeting_start: float = 0.0
        self._tracks: list[ChannelTrack] = []   # per-source live state

        # Session ID for file naming
        self._session_id: str = ""

    # ── Track-0 compatibility (single channel; tests and app.py use these) ──

    def _track0(self) -> ChannelTrack:
        if not self._tracks:
            self._tracks = [ChannelTrack(None, None, self.vad)]
        return self._tracks[0]

    @property
    def _processor(self):
        return self._tracks[0].processor if self._tracks else None

    @_processor.setter
    def _processor(self, proc) -> None:
        self._track0().processor = proc

    @property
    def _outage_offset(self) -> float:
        return self._track0().outage_offset

    @_outage_offset.setter
    def _outage_offset(self, value: float) -> None:
        self._track0().outage_offset = value

    @property
    def _segment_start_time(self) -> float | None:
        return self._track0().segment_start

    @_segment_start_time.setter
    def _segment_start_time(self, value: float | None) -> None:
        self._track0().segment_start = value

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def segments(self) -> list[MeetingSegment]:
        return list(self._segments)

    @property
    def is_recording(self) -> bool:
        return self.state == "recording"

    def start(self) -> None:
        """Start meeting recording."""
        with self._state_lock:
            if self._state not in ("idle", "stopped"):
                return
            self._state = "recording"

        self._session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._meeting_start = time.time()
        self._segments = []
        self._mic_frames = []
        self._sys_frames = []

        self._build_tracks()
        self._engine_down = self._engine_gave_up = False
        self._fallback_used = False
        self.had_engine_outage = False
        if (self._engine_ready is not None
                and not self._engine_ready()):
            # Started right after launch: Qwen3 still loading. Not an error —
            # buffer audio and catch up once it is ready (no alert).
            logger.warning("Meeting started while the live engine is loading — buffering")
            self._engine_down = True
            self.had_engine_outage = True
            for track in self._tracks:
                track.outage_offset = 0.0
            self._fallback_used = True
            if self._engine_restart is not None:
                self._engine_back.clear()
                threading.Thread(target=self._restart_engine_loop, daemon=True,
                                 name="meeting-engine-wait").start()

        # Reset VADs
        for track in self._tracks:
            if track.vad:
                track.vad.reset()

        # Start audio capture
        self._start_sources()

        # Start processing loop
        self._loop_done.clear()
        self._loop_thread = threading.Thread(
            target=self._meeting_loop, daemon=True
        )
        self._loop_thread.start()

        print(f"[MEETING] Started session {self._session_id}")

    def _build_tracks(self) -> None:
        """One unlabelled track (single channel) or Mic + Remote tracks, each
        with its own processor and VAD."""
        if self._dual_channel:
            remote_vad = self._vad_factory() if self._vad_factory else None
            self._tracks = [ChannelTrack("mic", self._new_processor(), self.vad),
                            ChannelTrack("remote", self._new_processor(), remote_vad)]
            logger.warning("Meeting started: dual channel (Mic + Remote)")
        else:
            self._tracks = [ChannelTrack(None, self._new_processor(), self.vad)]
            logger.warning("Meeting started: single channel")
        for track in self._tracks:
            track.buffer_start = self._meeting_start
        self._tick = 0

    def pause(self) -> None:
        """Pause recording (keeps session alive)."""
        with self._state_lock:
            if self._state != "recording":
                return
            self._state = "paused"

        self._stop_sources()

        # Commit current segment
        for track in self._tracks:
            if track.processor:
                text = track.processor.segment_close()
                if text.strip():
                    self._commit_segment(track, text)

        print("[MEETING] Paused")

    def resume(self) -> None:
        """Resume recording after pause."""
        with self._state_lock:
            if self._state != "paused":
                return
            self._state = "recording"

        for track in self._tracks:
            if track.vad:
                track.vad.reset()

        self._start_sources()
        print("[MEETING] Resumed")

    def _new_processor(self):
        if self._processor_factory is not None:
            return self._processor_factory()
        from qwen_meeting import QwenSegmentProcessor
        return QwenSegmentProcessor(self.backend)

    def _switch_to_fallback(self, exc: Exception) -> None:
        """Qwen3 failed mid-meeting: keep the processor (and its audio) and
        buffer until the engine is back; saved segments stay."""
        logger.warning("Live meeting engine failed (%s) — buffering audio while it restarts (%s)",
                       exc, self._track_names())
        if not self._engine_down:   # buffered audio starts with the current segment
            now = time.time()
            for track in self._tracks:
                # catch-up decodes the whole buffer, which starts at the last
                # trim (not at the first speech)
                track.outage_offset = max(0.0, (track.buffer_start or now) - self._meeting_start)
        self._engine_down = True
        self.had_engine_outage = True
        self._engine_gave_up = False
        self._fallback_used = True
        if self._engine_restart is not None:
            self._engine_back.clear()
            threading.Thread(target=self._restart_engine_loop, daemon=True,
                             name="meeting-engine-restart").start()
        if self._on_engine_failed:
            try:
                self._on_engine_failed(exc)
            except Exception:
                logger.exception("on_engine_failed callback failed")

    def _restart_engine_loop(self) -> None:
        """Background: restart the failed engine until it works, the meeting ends,
        or engine_max_attempts fail (then the meeting final pass covers the rest).
        Waits double after each failure (10 s, 20 s, 40 s… up to 2 min): a model
        load is heavy and a broken engine must not be reloaded every few seconds."""
        wait = self._engine_retry_s
        for attempt in range(1, self._engine_max_attempts + 1):
            if not self.is_recording:
                return
            try:
                ok = bool(self._engine_restart())
            except Exception:
                logger.exception("Restarting the live meeting engine failed")
                ok = False
            if ok:
                logger.warning("Live meeting engine restarted — switching back at once")
                self._engine_back.set()
                return
            logger.warning("Live meeting engine restart %d/%d failed — retry in %.0fs",
                           attempt, self._engine_max_attempts, wait)
            deadline = time.time() + wait
            while self.is_recording and time.time() < deadline:
                time.sleep(min(0.05, wait))
            wait = min(wait * 2, 120.0)
        self._engine_gave_up = True
        logger.error("Live meeting engine did not come back — no live subtitles; "
                     "the meeting final pass will transcribe it")

    def _switch_back_to_engine(self) -> None:
        """Engine is back: decode the audio buffered while it was down."""
        self._engine_back.clear()
        if self._engine_down:
            self._catch_up_buffered()

    CATCH_UP_CHUNK_S = 15.0     # = Qwen3 max segment: bounded decode, real timestamps

    def _catch_up_buffered(self) -> None:
        """Engine back: decode the audio each track buffered during the outage
        in 15 s chunks, each committed with the meeting time it was spoken
        and the track's label. If the engine fails again, the rest goes back
        into the buffer (tracks not reached yet keep theirs)."""
        self._engine_down = False
        self._fallback_used = False
        logger.warning("Live meeting engine back — catching up buffered audio (%s)",
                       self._track_names())
        for track in self._tracks:
            if not self._catch_up_track(track):
                return
        if self._on_engine_recovered:
            try:
                self._on_engine_recovered()
            except Exception:
                logger.exception("on_engine_recovered callback failed")

    def _catch_up_track(self, track: ChannelTrack) -> bool:
        """Catch up one track; False if the engine failed again."""
        from qwen_backend import QwenUnavailable
        pending = np.asarray(getattr(track.processor, "audio_buffer", np.zeros(0)), dtype=np.float32)
        step = int(self.CATCH_UP_CHUNK_S * SAMPLE_RATE)
        offset = track.outage_offset
        logger.warning("Catching up %.0fs of buffered audio (%s)",
                       len(pending) / SAMPLE_RATE, source_label(track.source) or "single channel")
        for start in range(0, len(pending), step):
            chunk = pending[start:start + step]
            t0 = offset + start / SAMPLE_RATE
            if len(chunk) < 0.3 * SAMPLE_RATE:
                break
            proc = self._new_processor()
            proc.insert_audio_chunk(chunk)
            mark = getattr(proc, "mark_speech", None)
            if mark is not None:
                mark()                     # buffered audio may hold speech
            try:
                text = proc.segment_close()
            except QwenUnavailable as e:   # failed again: keep the rest buffered
                track.processor = self._new_processor()
                track.processor.insert_audio_chunk(pending[start:])
                track.outage_offset = t0
                self._engine_down = True   # keeps this offset (not the segment start)
                self._switch_to_fallback(e)
                return False
            if text.strip():
                self._add_segment(MeetingSegment(
                    text=text.strip(), start_time=t0, end_time=t0 + len(chunk) / SAMPLE_RATE,
                    source=track.source))
        track.processor = self._new_processor()
        track.reset_segment(time.time())
        # a later outage keeps this offset (_engine_down guard): date it from the new buffer
        track.outage_offset = max(0.0, track.buffer_start - self._meeting_start)
        track.live_confirmed = track.live_unconfirmed = ""
        return True

    def _track_names(self) -> str:
        return ", ".join(source_label(t.source) or "single channel" for t in self._tracks)

    def _guarded(self, fn):
        """Run a processor call; on QwenUnavailable start buffering (once)."""
        from qwen_backend import QwenUnavailable
        try:
            return fn()
        except QwenUnavailable as e:
            if getattr(self, "_fallback_used", False):
                raise
            self._switch_to_fallback(e)
            return None

    def stop(self, wait_s: float = 5.0) -> str:
        """Stop meeting and return full transcript text.

        wait_s: how long to wait for the transcription loop to finish. Quit /
        logout uses a short wait so the meeting is saved before the process
        must exit.
        """
        with self._state_lock:
            if self._state in ("idle", "stopped"):
                return ""
            self._state = "stopped"

        # Stop audio
        self._stop_sources()

        # Wait for loop to finish
        self._loop_done.wait(timeout=wait_s)

        # Commit final segment. Under the processor lock: if the loop thread
        # is still inside a decode (wait timed out), wait for it to finish.
        with self._proc_lock:
            for track in self._tracks:
                if track.processor and self._engine_down:
                    logger.warning("Meeting stopped while the live engine was down — "
                                   "%.0fs of audio left to the final pass",
                                   len(getattr(track.processor, "audio_buffer", ())) / SAMPLE_RATE)
                    track.processor.reset()
                    track.processor = None
                if track.processor:
                    if track.has_pending():
                        self._close_and_trim_segment(track)
                    if track.processor:
                        track.processor.reset()
                    track.processor = None
                track.live_confirmed = track.live_unconfirmed = ""

        # Build full transcript
        full_text = self._build_transcript()
        duration = time.time() - self._meeting_start

        print(f"[MEETING] Stopped — {len(self._segments)} segments, "
              f"{len(full_text)} chars, {duration:.0f}s")

        # Auto-save
        self._auto_save()

        return full_text

    def get_transcript(self) -> str:
        """Return current transcript text (while still recording)."""
        return self._build_transcript()

    def export(self, path: str, fmt: str = "md") -> str:
        """Export meeting transcript to file.

        Args:
            path: Output file path (directory or full path).
            fmt: Format — "md" (Markdown), "srt", "vtt", "txt".

        Returns:
            Path of the exported file.
        """
        if os.path.isdir(path):
            ext = fmt if fmt != "md" else "md"
            path = os.path.join(path, f"meeting_{self._session_id}.{ext}")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if fmt == "md":
            content = self._format_markdown()
        elif fmt == "txt":
            content = self._build_transcript()
        elif fmt in ("srt", "vtt"):
            return export_cues(self._subtitle_cues(), path, fmt)
        else:
            content = self._build_transcript()

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    # ── Internal: audio source helpers ─────────────────────────

    def _start_sources(self) -> None:
        """Start audio capture source(s)."""
        if self._dual_channel:
            self._mic_source.start(self._on_mic_chunk)
            self._sys_source.start(self._on_sys_chunk)
        else:
            self._audio_source.start(self._on_mic_chunk)

    def _stop_sources(self) -> None:
        """Stop audio capture source(s)."""
        if self._dual_channel:
            self._mic_source.stop()
            self._sys_source.stop()
        else:
            self._audio_source.stop()

    # ── Internal: audio callbacks ────────────────────────────

    def _on_mic_chunk(self, chunk: np.ndarray) -> None:
        """Called by mic AudioSource when new audio arrives."""
        with self._frames_lock:
            self._mic_frames.append(chunk)

    def _on_sys_chunk(self, chunk: np.ndarray) -> None:
        """Called by system AudioSource when new audio arrives."""
        with self._frames_lock:
            self._sys_frames.append(chunk)

    # ── Internal: main processing loop ───────────────────────

    def _meeting_loop(self) -> None:
        """Main meeting processing loop (runs in background thread)."""
        try:
            self._meeting_loop_impl()
        except Exception as e:
            # Live transcription stopped unexpectedly. Say so — the meeting can still be stopped and saved.
            logger.exception("Meeting transcription loop stopped")
            print(f"[MEETING] Loop error: {e}")
            if self._on_engine_failed:
                try:
                    self._on_engine_failed(e)
                except Exception:
                    logger.exception("on_engine_failed callback failed")
        finally:
            self._loop_done.set()

    def _meeting_loop_impl(self) -> None:
        """Inner meeting loop — feeds audio to processor, handles VAD.

        In dual-channel mode mic audio goes only to the Mic track and system
        audio only to the Remote track, so crosstalk keeps both sides.
        """
        last_mic_idx = 0
        last_sys_idx = 0
        for track in self._tracks:
            track.vad_skip_count = 0

        while self.state == "recording":
            time.sleep(0.05)

            if self.state != "recording":
                break

            # Collect new frames from both buffers
            with self._frames_lock:
                mic_n = len(self._mic_frames)
                sys_n = len(self._sys_frames)
                mic_new = self._mic_frames[last_mic_idx:mic_n] if mic_n > last_mic_idx else []
                sys_new = self._sys_frames[last_sys_idx:sys_n] if sys_n > last_sys_idx else []
                last_mic_idx = mic_n
                last_sys_idx = sys_n

            if not mic_new and not sys_new:
                continue

            # Feed + transcribe under the processor lock: stop() takes the
            # same lock, so it never uses the processor during a decode.
            with self._proc_lock:
                self._process_new_audio(mic_new, sys_new)


    def _process_new_audio(self, mic_new, sys_new) -> None:
        """One loop iteration on new audio (caller holds _proc_lock)."""
        # stop() may have timed out and released the processors: nothing to feed
        if self._state != "recording":
            return
        if self._engine_back.is_set() and getattr(self, "_fallback_used", False):
            self._switch_back_to_engine()
        # Build chunks from each source
        mic_chunk = np.concatenate(mic_new, axis=0).squeeze() if mic_new else None
        sys_chunk = np.concatenate(sys_new, axis=0).squeeze() if sys_new else None

        if self._dual_channel and len(self._tracks) > 1:
            # Each source feeds only its own track; alternate which goes
            # first so neither track always waits on the other's decode.
            pairs = [(self._tracks[0], mic_chunk), (self._tracks[1], sys_chunk)]
            if self._tick % 2:
                pairs.reverse()
            self._tick += 1
            for track, chunk in pairs:
                if chunk is not None:
                    self._feed_track(track, chunk)
            return

        chosen = mic_chunk if mic_chunk is not None else sys_chunk
        self._feed_track(self._track0(), chosen)

    def _feed_track(self, track: ChannelTrack, chosen) -> None:
        """Feed one chunk to a track: VAD, segment breaks, transcription."""
        proc = track.processor
        now = time.time()
        audio_float = chosen.astype(np.float32) / 32768.0
        if self._engine_down:
            # Qwen3 only, engine restarting: collect audio, decode nothing
            if self._engine_gave_up:
                return               # the meeting final pass will transcribe it
            proc.insert_audio_chunk(audio_float)
            if not track.vad or track.vad.process_chunk(chosen)["is_speech"]:
                track.mark_speech_start(now)
                if track.vad:
                    mark = getattr(proc, "mark_speech", None)
                    if mark is not None:
                        mark()
            return
        proc.insert_audio_chunk(audio_float)

        if not track.vad:
            track.mark_speech_start(now)   # no VAD: every chunk counts as speech
        else:
            # Silero is a stateful RNN: each chunk must reach it exactly
            # ONCE. Re-evaluating the same chunk corrupted its hidden state
            # and misclassified over half of real speech as silence.
            vad_state = track.vad.process_chunk(chosen)
            chunk_is_speech = vad_state["is_speech"]
            if hasattr(proc, "uses_vad"):
                proc.uses_vad = True
            if chunk_is_speech:
                track.mark_speech_start(now)
                mark = getattr(proc, "mark_speech", None)
                if mark is not None:
                    mark()            # Qwen3: this segment has had real speech

            # Safety: force segment break if current segment exceeds 30s.
            # Natural speech rarely has 2s clean silence, so the extended-
            # silence trigger alone can produce unbounded segments (56s+).
            # Measured from the segment's first speech (wall clock, NOT
            # audio_buffer length — the buffer is capped by pre-inference trim).
            MAX_SEGMENT_DURATION_S = 30.0
            seg_elapsed = now - (track.segment_start or now)
            if seg_elapsed > MAX_SEGMENT_DURATION_S and track.has_text():
                self._close_and_trim_segment(track)
                return
            # Extended silence -> paragraph break (segment commit)
            if track.vad.is_extended_silence(self.extended_silence_ms):
                seg_duration = len(proc.audio_buffer) / SAMPLE_RATE
                silence_ms = getattr(track.vad, "silence_ms", self.extended_silence_ms)
                if (seg_duration > 1.0 and track.has_text()
                        and track.may_close_on_silence(silence_ms)):
                    self._close_and_trim_segment(track)
                    return

            # VAD pre-filter: skip transcription during pure silence
            if not chunk_is_speech:
                track.vad_skip_count += 1
                if track.vad_skip_count % 20 == 0:
                    print(f"[MEETING] VAD skip (silence): {track.vad_skip_count} chunks")
                return
            track.vad_skip_count = 0

        # Qwen3: bound each segment's audio (every decode re-reads it all).
        # Outside the VAD block so it holds with or without VAD.
        seg_full = getattr(proc, "segment_full", None)
        if seg_full is not None and seg_full():
            self._close_and_trim_segment(track)
            return

        # Run transcription iteration
        result = self._guarded(proc.process_iter)
        if result is None:
            return

        confirmed, unconfirmed = result

        # Store live text on this track (used by the dual overlay)
        track.live_confirmed = confirmed
        track.live_unconfirmed = unconfirmed

        # Notify overlay
        if self._on_update:
            try:
                debug = track.processor.last_debug if track.processor else {}
                if self._dual_channel:
                    ov_c, ov_u = self._overlay_text_for(track, confirmed, unconfirmed)
                else:
                    ov_c, ov_u = confirmed, unconfirmed
                self._on_update(ov_c, ov_u, self._segments, debug)
            except Exception:
                pass


    # ── Internal: segment management ─────────────────────────

    def _overlay_text_for(
        self,
        active_track: ChannelTrack,
        confirmed: str,
        unconfirmed: str,
    ) -> tuple[str, str]:
        """Build combined (confirmed, unconfirmed) for the overlay in dual-channel.

        confirmed  — "Label: text" line per track that has confirmed text, joined by \\n.
        unconfirmed — active track's unconfirmed; prefixed with "Label: " only when
                      that track has no confirmed text yet (label not yet visible).
        """
        # _feed_track stores live_confirmed on every track (the active one
        # included) before calling this, so all tracks read the same field.
        ov_confirmed = "\n".join(_labelled(t.source, t.live_confirmed)
                                 for t in self._tracks if t.live_confirmed)

        if confirmed or not unconfirmed:
            ov_unconfirmed = unconfirmed
        else:
            ov_unconfirmed = _labelled(active_track.source, unconfirmed)

        return ov_confirmed, ov_unconfirmed


    def _commit_segment(self, track: ChannelTrack, text: str) -> None:
        """Add a committed segment to the meeting record.

        start_time is the wall clock of the segment's first speech chunk:
        word timestamps are relative to a trimmed buffer (gave negative or
        previous-close starts), so they are never used for it.
        """
        now = time.time()
        elapsed = now - self._meeting_start
        start_time = max(0.0, (track.segment_start or now) - self._meeting_start)
        # committed still holds exactly this segment's words (not yet cleared)
        words = list(track.processor.committed) if track.processor else []

        segment = MeetingSegment(
            text=text.strip(),
            start_time=start_time,
            end_time=elapsed,
            words=list(words),
            source=track.source,
        )
        if not self._add_segment(segment):
            return
        label = source_label(track.source)
        prefix = f"{label} segment" if label else "Segment"
        print(f"[MEETING] {prefix} #{len(self._segments)}: "
              f"{len(text)} chars at {elapsed:.0f}s")

    def _add_segment(self, segment: MeetingSegment) -> bool:
        """Append a segment, dropping Mic echoes of Remote speech (by text,
        after commit; audio is never touched). False if the new one was dropped."""
        if segment.source == "mic":
            for other in self._segments:
                if other.source == "remote" and is_echo(segment, other):
                    self._log_echo(segment, other)
                    return False
        elif segment.source == "remote":
            kept = []
            for other in self._segments:
                if other.source == "mic" and is_echo(other, segment):
                    self._log_echo(other, segment)
                else:
                    kept.append(other)
            self._segments = kept
        self._segments.append(segment)
        return True

    @staticmethod
    def _log_echo(mic: MeetingSegment, remote: MeetingSegment) -> None:
        logger.warning("Echo dropped (Mic): %r ~ %r (%.2f)",
                    mic.text, remote.text, text_similarity(mic.text, remote.text))

    def _close_and_trim_segment(self, track: ChannelTrack | None = None) -> None:
        """Close the track's segment, trim its audio buffer, start the next.

        Used by extended-silence detection, the max-duration safety net,
        segment_full and stop. Trimming stops the next process_iter() from
        re-transcribing old audio (content loss / duplication).
        """
        track = track or self._track0()
        text = self._guarded(track.processor.segment_close)
        if text is None:          # Qwen3 failed; the processor keeps buffering the audio
            if track.vad:
                track.vad.reset()
            return
        if text.strip():
            self._commit_segment(track, text)
        track.reset_segment(time.time())

    # ── Internal: transcript formatting ──────────────────────

    def _build_transcript(self) -> str:
        """Build plain text transcript from all segments."""
        parts = []
        for seg in ordered(self._segments):
            parts.append(_labelled(seg.source, seg.text))

        # Add any uncommitted text from the live processors
        for track in self._tracks:
            if track.processor and track.processor.committed_history:
                current = "".join(w[2] for w in track.processor.committed_history)
                if current.strip():
                    parts.append(_labelled(track.source, current.strip()))

        return "\n\n".join(parts)

    def _format_markdown(self) -> str:
        """Format transcript as Markdown with timestamps."""
        lines = []
        lines.append("# Meeting Transcript")
        lines.append("")
        lines.append(f"- **Date**: {self._session_id[:8]}")
        lines.append(f"- **Segments**: {len(self._segments)}")

        duration = 0
        if self._segments:
            duration = max(s.end_time for s in self._segments)
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        lines.append(f"- **Duration**: {minutes}m {seconds}s")
        for path in self.audio_paths:
            lines.append(f"- **Audio**: {os.path.basename(path)}")
        lines.append("")
        lines.append("---")
        lines.append("")
        ai = getattr(self, "ai_transcript", None)
        if ai:
            lines.append(f"## AI version ({getattr(self, 'ai_style', None) or 'AI'})")
            lines.append("")
            lines.append(ai)
            lines.append("")
            lines.append("---")
            lines.append("")
        final = getattr(self, "final_transcript", None)
        if final is not None:
            lines.append(f"## Final transcript ({self.final_engine})")
            lines.append("")
            lines.append(final)
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append("## Live transcript (real-time subtitles)")
            lines.append("")

        for seg in ordered(self._segments):
            ts_min = int(seg.start_time // 60)
            ts_sec = int(seg.start_time % 60)
            label = source_label(seg.source)
            if label:
                lines.append(f"**[{ts_min:02d}:{ts_sec:02d}] {label}:** {seg.text}")
            else:
                lines.append(f"**[{ts_min:02d}:{ts_sec:02d}]** {seg.text}")
            lines.append("")

        return "\n".join(lines)

    def _subtitle_cues(self) -> list[tuple[float, float, str]]:
        """SRT/VTT cues: word-timed lines where words exist (Whisper), else one
        cue per segment (Qwen3 segments carry no word timestamps)."""
        cues = []
        for seg in ordered(self._segments):
            if seg.words:
                cues += [(s, e, _labelled(seg.source, t)) for s, e, t in
                         word_cues([WordTimestamp(word=t, start=s, end=e, probability=1.0)
                                    for s, e, t in seg.words])]
            elif seg.text.strip():
                cues.append((seg.start_time, seg.end_time, _labelled(seg.source, seg.text)))
        return cues

    def _all_word_timestamps(self) -> list[WordTimestamp]:
        """Collect all word timestamps across segments for SRT/VTT export."""
        words = []
        for seg in self._segments:
            for start, end, text in seg.words:
                words.append(WordTimestamp(
                    word=text, start=start, end=end, probability=1.0,
                ))
        return words

    # ── Internal: auto-save ──────────────────────────────────

    def _auto_save(self) -> str | None:
        """Save meeting data to ~/.macwhisper/meetings/.

        Returns the Markdown transcript path (also stored on
        ``self.transcript_path``), or None if saving failed.
        """
        self.transcript_path = None
        self.audio_paths = []
        if self.save_audio:
            try:
                self.audio_paths = self._write_audio()
            except OSError as e:
                logger.error("Meeting audio save failed: %s", e, exc_info=True)
        try:
            return self._write_files()
        except OSError as e:
            logger.error("Meeting auto-save failed: %s", e, exc_info=True)
            return None

    # ── Final (post-meeting) transcript ──────────────────────

    def mixed_audio(self) -> np.ndarray:
        """Mic + system audio from memory, mixed to one float32 track.

        Used for the post-meeting final pass. Nothing is written to disk.
        """
        with self._frames_lock:
            mic = list(self._mic_frames)
            sys_ = list(self._sys_frames)
        tracks = [np.concatenate(f, axis=0).squeeze().astype(np.float32) / 32768.0
                  for f in (mic, sys_) if f]
        if not tracks:
            return np.zeros(0, dtype=np.float32)
        n = max(len(t) for t in tracks)
        mix = np.zeros(n, dtype=np.float32)
        for t in tracks:
            mix[:len(t)] += t
        return np.clip(mix, -1.0, 1.0)

    def channel_audio(self) -> list[tuple[str | None, np.ndarray]]:
        """Audio for the final pass, one entry per track (float32).

        Dual channel: [("mic", ...), ("remote", ...)]; single: [(None, mixed)].
        """
        if not self._dual_channel:
            return [(None, self.mixed_audio())]
        with self._frames_lock:
            mic = list(self._mic_frames)
            sys_ = list(self._sys_frames)
        return [(source, np.concatenate(f, axis=0).squeeze().astype(np.float32) / 32768.0
                 if f else np.zeros(0, dtype=np.float32))
                for source, f in (("mic", mic), ("remote", sys_))]

    def active_channel_count(self) -> int:
        """Tracks with audio (no copy): the final pass makes one request each."""
        if not self._dual_channel:
            return 1
        with self._frames_lock:
            return max(1, int(bool(self._mic_frames)) + int(bool(self._sys_frames)))

    def release_audio(self) -> None:
        """Drop in-memory audio once the final pass is done."""
        with self._frames_lock:
            self._mic_frames = []
            self._sys_frames = []

    def write_final_transcript(self, text: str, engine: str, ai_text: str | None = None,
                               ai_style: str | None = None) -> str | None:
        """Rewrite the saved .md/.json with the final transcript first.

        The live (real-time) transcript stays below it for comparison. An AI
        tidied version (optional) goes on top; nothing below it is removed.
        Returns the .md path, or None if the meeting was never saved.
        """
        if not self.transcript_path:
            return None
        self.final_transcript = text
        self.final_engine = engine
        self.ai_transcript = ai_text
        self.ai_style = ai_style
        md = self._format_markdown()
        with open(self.transcript_path, "w", encoding="utf-8") as f:
            f.write(md)
        json_path = self.transcript_path[: -len(".md")] + ".json"
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {"session_id": self._session_id}
        data = {**data, "final_transcript": text, "final_engine": engine}
        if ai_text is not None:
            data = {**data, "ai_transcript": ai_text, "ai_style": ai_style}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Meeting final transcript written (%s, %d chars)", engine, len(text))
        return self.transcript_path

    def _write_audio(self) -> list[str]:
        """Write each captured channel to its own 16 kHz mono WAV.

        Channels are kept separate so a replay can reproduce the live loop's
        per-chunk mic/system selection exactly.
        """
        os.makedirs(MEETINGS_DIR, exist_ok=True)
        base = os.path.join(MEETINGS_DIR, f"meeting_{self._session_id}")
        with self._frames_lock:
            channels = [("mic", list(self._mic_frames)), ("sys", list(self._sys_frames))]
        paths = []
        for name, frames in channels:
            if not frames:
                continue
            audio = np.concatenate(frames, axis=0).astype(np.int16)
            path = f"{base}_{name}.wav"
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
            logger.info("Meeting audio saved: %s (%.1fs)", path, len(audio) / SAMPLE_RATE)
            paths.append(path)
        return paths

    def _write_files(self) -> str:
        os.makedirs(MEETINGS_DIR, exist_ok=True)
        base = os.path.join(MEETINGS_DIR, f"meeting_{self._session_id}")

        # Save Markdown transcript
        md_path = f"{base}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._format_markdown())

        # Save raw segments as JSON
        json_path = f"{base}.json"
        data = {
            "session_id": self._session_id,
            "segments": [
                {
                    "text": s.text,
                    "start_time": round(s.start_time, 2),
                    "end_time": round(s.end_time, 2),
                    "source": s.source,
                }
                for s in ordered(self._segments)
            ],
            "total_segments": len(self._segments),
            "transcript": self._build_transcript(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[MEETING] Saved to {base}.*")
        self.transcript_path = md_path
        return md_path
