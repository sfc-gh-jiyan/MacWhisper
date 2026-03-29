"""Meeting Mode session manager for MacWhisper.

Provides continuous recording with automatic segmentation via VAD,
real-time transcription using OnlineASRProcessor, and transcript
export (Markdown, SRT, VTT).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from audio_capture import AudioSource, MicrophoneSource, MixedAudioSource, SystemAudioSource
from asr_backend import ASRBackend, WordTimestamp
from online_processor import OnlineASRProcessor, SAMPLE_RATE
from vad import VoiceActivityDetector
from subtitle_export import export_srt, export_vtt

logger = logging.getLogger(__name__)

# Meeting data directory
MEETINGS_DIR = os.path.join(os.path.expanduser("~/.macwhisper"), "meetings")


@dataclass
class MeetingSegment:
    """A paragraph-level segment committed by VAD silence detection."""
    text: str
    start_time: float  # seconds from meeting start
    end_time: float    # seconds from meeting start
    words: list[tuple[float, float, str]] = field(default_factory=list)


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
    ):
        self.backend = backend
        self.vad = vad
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

        # Meeting data
        self._segments: list[MeetingSegment] = []
        self._meeting_start: float = 0.0
        self._processor: OnlineASRProcessor | None = None

        # Session ID for file naming
        self._session_id: str = ""

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
        self._segment_start_time = time.time()  # wall-clock start of current segment

        # Create processor
        self._processor = OnlineASRProcessor(
            backend=self.backend,
            vad=None,  # VAD is managed here, not inside processor
            min_chunk_size=self.min_chunk_size,
            max_buffer_s=self.max_buffer_s,
            language=self.language,
        )

        # Reset VAD
        if self.vad:
            self.vad.reset()

        # Start audio capture
        self._start_sources()

        # Start processing loop
        self._loop_done.clear()
        self._loop_thread = threading.Thread(
            target=self._meeting_loop, daemon=True
        )
        self._loop_thread.start()

        print(f"[MEETING] Started session {self._session_id}")

    def pause(self) -> None:
        """Pause recording (keeps session alive)."""
        with self._state_lock:
            if self._state != "recording":
                return
            self._state = "paused"

        self._stop_sources()

        # Commit current segment
        if self._processor:
            text = self._processor.segment_close()
            if text.strip():
                self._commit_segment(text)

        print("[MEETING] Paused")

    def resume(self) -> None:
        """Resume recording after pause."""
        with self._state_lock:
            if self._state != "paused":
                return
            self._state = "recording"

        if self.vad:
            self.vad.reset()

        self._start_sources()
        print("[MEETING] Resumed")

    def stop(self) -> str:
        """Stop meeting and return full transcript text."""
        with self._state_lock:
            if self._state in ("idle", "stopped"):
                return ""
            self._state = "stopped"

        # Stop audio
        self._stop_sources()

        # Wait for loop to finish
        self._loop_done.wait(timeout=5)

        # Commit final segment
        if self._processor:
            if self._processor.committed or self._processor.transcript_buffer.peek_unconfirmed():
                self._close_and_trim_segment()
            self._processor.reset()
            self._processor = None

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
        elif fmt == "srt":
            words = self._all_word_timestamps()
            return export_srt(words, path)
        elif fmt == "vtt":
            words = self._all_word_timestamps()
            return export_vtt(words, path)
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
            print(f"[MEETING] Loop error: {e}")
        finally:
            self._loop_done.set()

    def _meeting_loop_impl(self) -> None:
        """Inner meeting loop — feeds audio to processor, handles VAD.

        In dual-channel mode, compares mic and system audio RMS energy
        each iteration.  When both sources have data (overlap), only the
        louder source is fed to Whisper; the quieter one is discarded.
        Tie-breaks favour the microphone (local speaker priority).
        """
        last_mic_idx = 0
        last_sys_idx = 0
        vad_skip_count = 0

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

            # Build chunks from each source
            mic_chunk = np.concatenate(mic_new, axis=0).squeeze() if mic_new else None
            sys_chunk = np.concatenate(sys_new, axis=0).squeeze() if sys_new else None

            # Overlap priority selection
            if mic_chunk is not None and sys_chunk is not None:
                mic_rms = np.sqrt(np.mean(mic_chunk.astype(np.float32) ** 2))
                sys_rms = np.sqrt(np.mean(sys_chunk.astype(np.float32) ** 2))
                if sys_rms > mic_rms:
                    chosen = sys_chunk
                    logger.debug("overlap: sys wins (mic=%.1f sys=%.1f)", mic_rms, sys_rms)
                else:
                    # mic wins on tie (local speaker priority)
                    chosen = mic_chunk
                    logger.debug("overlap: mic wins (mic=%.1f sys=%.1f)", mic_rms, sys_rms)
            elif mic_chunk is not None:
                chosen = mic_chunk
            else:
                chosen = sys_chunk

            # Convert and feed to processor
            audio_float = chosen.astype(np.float32) / 32768.0
            self._processor.insert_audio_chunk(audio_float)

            # VAD processing
            if self.vad:
                self.vad.process_chunk(chosen)

                # Safety: force segment break if current segment exceeds 30s.
                # Natural speech rarely has 2s clean silence, so the extended-
                # silence trigger alone can produce unbounded segments (56s+).
                # A 30s cap keeps segments manageable and limits accumulation
                # of transcription errors within a single segment.
                # NOTE: use wall-clock time, NOT audio_buffer length — the
                # buffer is capped at max_buffer_s (8s) by pre-inference trim.
                MAX_SEGMENT_DURATION_S = 30.0
                seg_elapsed = time.time() - self._segment_start_time
                if seg_elapsed > MAX_SEGMENT_DURATION_S and self._processor.committed:
                    self._close_and_trim_segment()
                    continue

                # Extended silence -> paragraph break (segment commit)
                if self.vad.is_extended_silence(self.extended_silence_ms):
                    seg_duration = len(self._processor.audio_buffer) / SAMPLE_RATE
                    if seg_duration > 1.0 and self._processor.committed:
                        self._close_and_trim_segment()
                        continue

                # VAD pre-filter: skip transcription during pure silence
                if not self.vad.is_speech(chosen):
                    vad_skip_count += 1
                    if vad_skip_count % 20 == 0:
                        print(f"[MEETING] VAD skip (silence): {vad_skip_count} chunks")
                    continue
                else:
                    vad_skip_count = 0

            # Run transcription iteration
            result = self._processor.process_iter()
            if result is None:
                continue

            confirmed, unconfirmed = result

            # Notify overlay
            if self._on_update:
                try:
                    debug = self._processor.last_debug if self._processor else {}
                    self._on_update(confirmed, unconfirmed, self._segments, debug)
                except Exception:
                    pass

    # ── Internal: segment management ─────────────────────────

    def _commit_segment(self, text: str) -> None:
        """Add a committed segment to the meeting record."""
        elapsed = time.time() - self._meeting_start
        # Estimate segment start from word timestamps
        # Use committed (current segment words), NOT get_confirmed_words()
        # which returns committed_history (all words since meeting start).
        # At this point segment_close() has been called but committed hasn't
        # been cleared yet, so it contains exactly this segment's words.
        words = list(self._processor.committed) if self._processor else []

        start_time = elapsed - len(text) * 0.1  # rough estimate
        if words:
            start_time = words[0][0]

        segment = MeetingSegment(
            text=text.strip(),
            start_time=start_time,
            end_time=elapsed,
            words=list(words),
        )
        self._segments.append(segment)
        print(f"[MEETING] Segment #{len(self._segments)}: "
              f"{len(text)} chars at {elapsed:.0f}s")

    def _close_and_trim_segment(self) -> None:
        """Close current segment, trim audio buffer, and reset for next segment.

        This is the standard segment-break sequence used by both extended-silence
        detection and the max-segment-duration safety net.

        After segment_close(), the audio buffer still contains all old audio.
        Without trimming, the next process_iter() would re-transcribe it,
        causing: (1) echo detection dropping legitimate new confirmations
        (content loss), and (2) near-duplicate text slipping past echo
        detection (duplication).  Trimming the buffer to the segment's end
        time eliminates both problems at the source.
        """
        text = self._processor.segment_close()
        if text.strip():
            self._commit_segment(text)

        # Trim audio buffer up to the last committed word's end time.
        # Keep only audio that arrived after the segment ended.
        if self._processor.committed:
            seg_end = self._processor.committed[-1][1]
            trim_samples = int(
                (seg_end - self._processor.buffer_time_offset) * SAMPLE_RATE
            )
            trim_samples = min(trim_samples, len(self._processor.audio_buffer))
            if trim_samples > 0:
                self._processor.audio_buffer = self._processor.audio_buffer[trim_samples:]
                self._processor.buffer_time_offset = seg_end

        # Clear committed words so next segment starts fresh.
        self._processor.committed = []
        # Clear echo state: old audio is trimmed, so there's nothing to
        # echo-detect against. Without this, the retained _last_committed_raw
        # would cause echo detection to drop legitimate new words.
        self._processor._last_committed_raw = ""

        if self.vad:
            self.vad.reset()

        # Reset segment timer for the next segment
        self._segment_start_time = time.time()

    # ── Internal: transcript formatting ──────────────────────

    def _build_transcript(self) -> str:
        """Build plain text transcript from all segments."""
        parts = []
        for seg in self._segments:
            parts.append(seg.text)

        # Add any uncommitted text from current processor
        if self._processor and self._processor.committed_history:
            current = "".join(w[2] for w in self._processor.committed_history)
            if current.strip():
                parts.append(current.strip())

        return "\n\n".join(parts)

    def _format_markdown(self) -> str:
        """Format transcript as Markdown with timestamps."""
        lines = []
        lines.append(f"# Meeting Transcript")
        lines.append(f"")
        lines.append(f"- **Date**: {self._session_id[:8]}")
        lines.append(f"- **Segments**: {len(self._segments)}")

        duration = 0
        if self._segments:
            duration = self._segments[-1].end_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        lines.append(f"- **Duration**: {minutes}m {seconds}s")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        for i, seg in enumerate(self._segments, 1):
            ts_min = int(seg.start_time // 60)
            ts_sec = int(seg.start_time % 60)
            lines.append(f"**[{ts_min:02d}:{ts_sec:02d}]** {seg.text}")
            lines.append(f"")

        return "\n".join(lines)

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

    def _auto_save(self) -> None:
        """Save meeting data to ~/.macwhisper/meetings/."""
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
                }
                for s in self._segments
            ],
            "total_segments": len(self._segments),
            "transcript": self._build_transcript(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[MEETING] Saved to {base}.*")
