"""Fixes from the final review of the Mic / Remote labels branch."""
import numpy as np

from meeting import MeetingSession, MeetingSegment
from meeting_track import ChannelTrack


class _Backend:
    def transcribe(self, *a, **k):
        return []


def _dual_session(live):
    session = MeetingSession(backend=_Backend())
    session._dual_channel = True
    session._tracks = [ChannelTrack(s, None, None) for s in ("mic", "remote")]
    for track, (c, u) in zip(session._tracks, live):
        track.live_confirmed, track.live_unconfirmed = c, u
    return session


def test_reset_segment_clears_live_text():
    track = ChannelTrack("remote", None, None)
    track.live_confirmed, track.live_unconfirmed = "old words", "gray"
    track.reset_segment(0.0)
    assert (track.live_confirmed, track.live_unconfirmed) == ("", "")


def test_committed_remote_text_shown_once_in_overlay():
    """After Remote commits, a Mic update must not repeat the Remote sentence."""
    import app
    session = _dual_session([("", ""), ("remote said this", "")])
    session._segments = [MeetingSegment(text="remote said this", start_time=0.0,
                                        end_time=1.0, source="remote")]
    session._tracks[1].reset_segment(1.0)             # segment closed
    session._tracks[0].live_confirmed = "mic now"
    c, u = session._overlay_text_for(session._tracks[0], "mic now", "")
    full = "\n".join(f"{app._seg_source_label(s.source)}: {s.text}"
                     for s in session._segments[-2:]) + "\n" + c
    assert full.count("remote said this") == 1


def test_duration_uses_latest_end_time():
    session = MeetingSession(backend=_Backend())
    session._segments = [
        MeetingSegment(text="b", start_time=60.0, end_time=125.0, source="remote"),
        MeetingSegment(text="a", start_time=0.0, end_time=10.0, source="mic"),
    ]
    assert "- **Duration**: 2m 5s" in session._format_markdown()


def test_no_engine_failure_after_stop():
    failed = []
    session = MeetingSession(backend=_Backend(), on_engine_failed=failed.append)
    session._state = "stopped"
    for track in session._tracks:
        track.processor = None
    session._process_new_audio([np.zeros((1600, 1), dtype=np.float32)], [])
    assert failed == []
