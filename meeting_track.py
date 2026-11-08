"""Per-source live state of a meeting (one track per audio channel).

MeetingSession keeps the engine-level state (outage, restart thread); each
ChannelTrack owns its processor, VAD and current-segment state.
"""

import logging

SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)


class ChannelTrack:
    """Live state of one audio source: processor, VAD and the open segment."""

    def __init__(self, source: str | None, processor, vad):
        self.source = source              # "mic" / "remote", None = unlabelled
        self.processor = processor
        self.vad = vad
        self.segment_start: float | None = None   # wall clock of first speech chunk
        self.buffer_start: float | None = None    # wall clock where audio_buffer starts
        self.outage_offset = 0.0          # meeting time where the buffered audio starts
        self.vad_skip_count = 0
        self.live_confirmed: str = ""     # latest confirmed text from process_iter
        self.live_unconfirmed: str = ""   # latest unconfirmed text from process_iter
        logger.debug("ChannelTrack created (source=%s)", source)

    def mark_speech_start(self, now: float) -> None:
        """Remember when the current segment's speech began (first call wins)."""
        if self.segment_start is None:
            self.segment_start = now

    def has_text(self) -> bool:
        """True if the current segment has confirmed OR unconfirmed words.

        segment_close force-confirms the gray tail, so it is safe to close
        on either (with unstable recognition nothing ever confirms).
        """
        p = self.processor
        if p is None:
            return False
        return bool(p.committed or p.transcript_buffer.peek_unconfirmed())

    def has_pending(self) -> bool:
        """Text OR untranscribed audio in the current segment (at stop:
        a Qwen3 segment can have audio but no decode result yet)."""
        if self.has_text():
            return True
        pending = getattr(self.processor, "has_pending_speech", None)
        return bool(pending()) if pending is not None else False

    def may_close_on_silence(self, silence_ms: int) -> bool:
        """Let the processor veto a silence close (Qwen3: a tiny fragment
        waits for a longer pause). Processors without a veto always close."""
        ready = getattr(self.processor, "ready_to_close", None)
        return True if ready is None else bool(ready(silence_ms))

    def reset_segment(self, now: float) -> None:
        """After a segment close: trim the buffer and start a fresh segment.

        After segment_close() the audio buffer still holds all old audio.
        Without trimming, the next process_iter() would re-transcribe it
        (echo detection drops new words; near-duplicates slip through).
        """
        p = self.processor
        if p is not None:
            if p.committed:
                seg_end = p.committed[-1][1]
                trim_samples = int((seg_end - p.buffer_time_offset) * SAMPLE_RATE)
                trim_samples = min(trim_samples, len(p.audio_buffer))
                if trim_samples > 0:
                    p.audio_buffer = p.audio_buffer[trim_samples:]
                    p.buffer_time_offset = seg_end
            # Next segment starts fresh; old audio is trimmed, so there is
            # nothing to echo-detect against.
            p.committed = []
            p._last_committed_raw = ""
        if self.vad:
            self.vad.reset()
        self.segment_start = None
        # Committed text is in the segments now; stale live text would show it twice
        self.live_confirmed = self.live_unconfirmed = ""
        self.buffer_start = now           # the trimmed buffer holds audio from here on
