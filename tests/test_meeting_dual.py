"""Dual-channel Meeting Mode tests.

Tests the full mic + system-audio pipeline using WavFileSource to inject
real WAV recordings into MeetingSession. Three layers:

- Layer 1 (integration): MockBackend, fast, CI-friendly
- Layer 2 (e2e quality): Real MLXWhisperBackend, @pytest.mark.slow
- Layer 3 (hardware):    Real SystemAudioHelper, @pytest.mark.hardware
"""

import json
import os
import time
import threading

import numpy as np
import pytest

from audio_capture import (
    AudioSource,
    MixedAudioSource,
    SystemAudioSource,
    WavFileSource,
    SAMPLE_RATE,
)
from meeting import MeetingSession, MeetingSegment


# ── Paths ────────────────────────────────────────────────────

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SCENARIOS_FILE = os.path.join(FIXTURES_DIR, "dual_channel_scenarios.json")
AUDIO_DIR = os.path.expanduser("~/.macwhisper/audio")
TRANSCRIPT_LOG = os.path.expanduser("~/.macwhisper/transcripts.jsonl")


# ── Helpers ──────────────────────────────────────────────────

def _load_scenarios(tier=None):
    """Load test scenarios from fixtures JSON."""
    with open(SCENARIOS_FILE) as f:
        scenarios = json.load(f)
    if tier:
        scenarios = [s for s in scenarios if s.get("tier") == tier]
    return scenarios


def _wav_exists(filename):
    return os.path.isfile(os.path.join(AUDIO_DIR, filename))


def _load_ground_truth():
    """Load audio_file -> text mapping from transcripts.jsonl."""
    gt = {}
    if not os.path.exists(TRANSCRIPT_LOG):
        return gt
    with open(TRANSCRIPT_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            af = entry.get("audio_file")
            text = entry.get("text")
            if af and text:
                gt[af] = text
    return gt


class MockBackend:
    """Mock ASR backend returning predictable text."""

    def __init__(self, responses=None):
        self._responses = responses or []
        self._call_count = 0

    def transcribe(self, audio, *, language=None, initial_prompt=None, task="transcribe"):
        from asr_backend import TranscriptionResult, Segment, WordTimestamp

        if self._call_count < len(self._responses):
            text = self._responses[self._call_count]
        else:
            text = f"chunk{self._call_count}"
        self._call_count += 1

        words_text = text.split() if text else []
        words = []
        t = 0.0
        for w in words_text:
            words.append(WordTimestamp(word=w + " ", start=t, end=t + 0.3, probability=0.9))
            t += 0.4

        seg = Segment(text=text, start=0.0, end=t, words=words)
        return TranscriptionResult(text=text, segments=[seg], language="zh")


# ── Layer 1: Integration tests (MockBackend, fast) ───────────

class TestWavFileSource:
    """Verify WavFileSource reads WAV and delivers chunks correctly."""

    def _find_short_wav(self):
        """Find any WAV file for testing."""
        scenarios = _load_scenarios("integration")
        for s in scenarios:
            path = os.path.join(AUDIO_DIR, s["mic_wav"])
            if os.path.isfile(path):
                return path
        # Fallback: find any WAV in the audio dir
        if os.path.isdir(AUDIO_DIR):
            for f in os.listdir(AUDIO_DIR):
                if f.endswith(".wav"):
                    return os.path.join(AUDIO_DIR, f)
        pytest.skip("No WAV files available in ~/.macwhisper/audio")

    def test_delivers_int16_chunks(self):
        wav_path = self._find_short_wav()
        received = []
        src = WavFileSource(wav_path, blocksize=1024, speed=0)  # speed=0: no sleep
        src.start(lambda chunk: received.append(chunk))
        # Wait for playback to finish (no sleep mode = instant)
        time.sleep(1.0)
        src.stop()

        assert len(received) > 0, "WavFileSource delivered no chunks"
        for chunk in received:
            assert chunk.dtype == np.int16
            assert chunk.ndim == 2 and chunk.shape[1] == 1

    def test_stop_interrupts_playback(self):
        wav_path = self._find_short_wav()
        received = []
        src = WavFileSource(wav_path, blocksize=1024, speed=1.0)
        src.start(lambda chunk: received.append(chunk))
        time.sleep(0.2)
        src.stop()
        assert not src.is_active

    def test_offset_delays_first_chunk(self):
        wav_path = self._find_short_wav()
        received = []
        t0 = time.time()
        src = WavFileSource(wav_path, blocksize=1024, offset_s=0.5, speed=1.0)
        src.start(lambda chunk: received.append((time.time() - t0, chunk)))
        time.sleep(1.0)
        src.stop()

        assert len(received) > 0
        first_time = received[0][0]
        # First chunk should arrive after ~0.5s offset
        assert first_time >= 0.4, f"First chunk arrived too early: {first_time:.2f}s"

    def test_is_active_lifecycle(self):
        wav_path = self._find_short_wav()
        src = WavFileSource(wav_path, speed=0)
        assert not src.is_active
        src.start(lambda x: None)
        time.sleep(0.1)
        # Should be active while playing or should have finished (speed=0, instant)
        # Either way, stop should work cleanly
        src.stop()
        assert not src.is_active


class TestDualChannelPipeline:
    """Test two WavFileSources through MixedAudioSource + MeetingSession."""

    def _get_scenario(self, name="integration_alternating"):
        scenarios = _load_scenarios("integration")
        for s in scenarios:
            if s["name"] == name:
                mic_path = os.path.join(AUDIO_DIR, s["mic_wav"])
                sys_path = os.path.join(AUDIO_DIR, s["system_wav"])
                if os.path.isfile(mic_path) and os.path.isfile(sys_path):
                    return s, mic_path, sys_path
        pytest.skip("Integration WAV files not available")

    def test_both_channels_deliver_audio(self):
        """Verify MixedAudioSource forwards chunks from both WAV sources."""
        s, mic_path, sys_path = self._get_scenario()

        received = {"count": 0}
        lock = threading.Lock()

        def on_chunk(chunk):
            with lock:
                received["count"] += 1

        mic_src = WavFileSource(mic_path, speed=0)
        sys_src = WavFileSource(sys_path, speed=0)
        mixed = MixedAudioSource([mic_src, sys_src])

        mixed.start(on_chunk)
        time.sleep(2.0)  # Both sources run in parallel at max speed
        mixed.stop()

        assert received["count"] > 10, f"Only got {received['count']} chunks from dual sources"

    def test_meeting_session_with_dual_wav(self):
        """MeetingSession processes dual-channel WAV input and produces transcript."""
        s, mic_path, sys_path = self._get_scenario()

        backend = MockBackend(responses=[f"segment{i}" for i in range(200)])
        mic_src = WavFileSource(mic_path, speed=0)
        sys_src = WavFileSource(sys_path, offset_s=s["system_offset_s"], speed=0)
        mixed = MixedAudioSource([mic_src, sys_src])

        session = MeetingSession(
            backend=backend,
            audio_source=mixed,
        )

        session.start()
        assert session.state == "recording"

        # Let it process (speed=0 means audio arrives instantly,
        # but meeting loop has 50ms sleep per iteration)
        time.sleep(2.0)

        transcript = session.stop()
        assert session.state == "stopped"
        assert isinstance(transcript, str)
        assert len(transcript) > 0, "Dual-channel session produced empty transcript"

    def test_meeting_session_state_machine(self):
        """Verify start -> pause -> resume -> stop works with dual sources."""
        s, mic_path, sys_path = self._get_scenario()

        backend = MockBackend(responses=["hello world"] * 100)
        mic_src = WavFileSource(mic_path, speed=0)
        sys_src = WavFileSource(sys_path, speed=0)
        mixed = MixedAudioSource([mic_src, sys_src])

        session = MeetingSession(backend=backend, audio_source=mixed)

        session.start()
        assert session.state == "recording"

        time.sleep(0.5)
        session.pause()
        assert session.state == "paused"

        session.resume()
        assert session.state == "recording"

        time.sleep(0.3)
        transcript = session.stop()
        assert session.state == "stopped"
        assert isinstance(transcript, str)


class TestAlternatingChannels:
    """Test sequential (non-overlapping) dual-channel playback."""

    def _get_scenario(self):
        scenarios = _load_scenarios("integration")
        for s in scenarios:
            if s["name"] == "integration_alternating":
                mic_path = os.path.join(AUDIO_DIR, s["mic_wav"])
                sys_path = os.path.join(AUDIO_DIR, s["system_wav"])
                if os.path.isfile(mic_path) and os.path.isfile(sys_path):
                    return s, mic_path, sys_path
        pytest.skip("Alternating scenario WAVs not available")

    def test_both_sources_contribute_chunks(self):
        """Each source delivers chunks in its own time window."""
        s, mic_path, sys_path = self._get_scenario()

        mic_chunks = []
        sys_chunks = []

        def mic_cb(chunk):
            mic_chunks.append(chunk)

        def sys_cb(chunk):
            sys_chunks.append(chunk)

        # Run each source separately to count their contributions
        mic_src = WavFileSource(mic_path, speed=0)
        mic_src.start(mic_cb)
        time.sleep(1.0)
        mic_src.stop()

        sys_src = WavFileSource(sys_path, speed=0)
        sys_src.start(sys_cb)
        time.sleep(1.0)
        sys_src.stop()

        assert len(mic_chunks) > 0, "Mic source produced no chunks"
        assert len(sys_chunks) > 0, "System source produced no chunks"


class TestOverlappingChannels:
    """Test overlapping dual-channel playback (both active simultaneously)."""

    def _get_scenario(self):
        scenarios = _load_scenarios("integration")
        for s in scenarios:
            if s["name"] == "integration_overlapping":
                mic_path = os.path.join(AUDIO_DIR, s["mic_wav"])
                sys_path = os.path.join(AUDIO_DIR, s["system_wav"])
                if os.path.isfile(mic_path) and os.path.isfile(sys_path):
                    return s, mic_path, sys_path
        pytest.skip("Overlapping scenario WAVs not available")

    def test_no_crash_with_concurrent_sources(self):
        """MeetingSession doesn't crash when both sources deliver audio simultaneously."""
        s, mic_path, sys_path = self._get_scenario()

        backend = MockBackend(responses=[f"word{i}" for i in range(300)])
        mic_src = WavFileSource(mic_path, speed=0)
        # Small offset so they overlap significantly
        sys_src = WavFileSource(sys_path, offset_s=s["system_offset_s"], speed=0)
        mixed = MixedAudioSource([mic_src, sys_src])

        session = MeetingSession(backend=backend, audio_source=mixed)

        session.start()
        time.sleep(2.0)
        transcript = session.stop()

        assert isinstance(transcript, str)
        assert len(transcript) > 0, "Overlapping session produced empty transcript"

    def test_callback_thread_safety(self):
        """Multiple sources calling callback concurrently should not corrupt frames."""
        s, mic_path, sys_path = self._get_scenario()

        received = []
        lock = threading.Lock()

        def safe_callback(chunk):
            with lock:
                received.append(chunk.copy())

        mic_src = WavFileSource(mic_path, speed=0)
        sys_src = WavFileSource(sys_path, speed=0)
        mixed = MixedAudioSource([mic_src, sys_src])

        mixed.start(safe_callback)
        time.sleep(1.5)
        mixed.stop()

        assert len(received) > 0
        # All chunks should be valid int16 arrays
        for chunk in received:
            assert chunk.dtype == np.int16
            assert chunk.ndim == 2


# ── Layer 2: E2E Quality tests (real Whisper) ────────────────

@pytest.mark.slow
class TestDualChannelQuality:
    """End-to-end quality evaluation with real MLXWhisperBackend.

    Uses real WAV recordings and evaluates transcription quality
    by reusing metrics from test_replay.py.
    """

    def _get_e2e_scenario(self, name="e2e_alternating"):
        scenarios = _load_scenarios("e2e")
        for s in scenarios:
            if s["name"] == name:
                mic_path = os.path.join(AUDIO_DIR, s["mic_wav"])
                sys_path = os.path.join(AUDIO_DIR, s["system_wav"])
                if os.path.isfile(mic_path) and os.path.isfile(sys_path):
                    return s, mic_path, sys_path
        pytest.skip("E2E scenario WAV files not available")

    def _get_combined_ground_truth(self, scenario):
        """Combine ground truth from both WAV files."""
        gt = _load_ground_truth()
        mic_gt = gt.get(scenario["mic_wav"], "")
        sys_gt = gt.get(scenario["system_wav"], "")
        return mic_gt + " " + sys_gt

    def test_alternating_quality(self):
        """Alternating dual-channel produces reasonable transcription quality."""
        from asr_backend import MLXWhisperBackend

        s, mic_path, sys_path = self._get_e2e_scenario("e2e_alternating")

        backend = MLXWhisperBackend()

        # Warmup
        warmup = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)
        backend.transcribe(warmup, language=None, task="transcribe")

        mic_src = WavFileSource(mic_path, speed=1.0)
        sys_src = WavFileSource(sys_path, offset_s=s["system_offset_s"], speed=1.0)
        mixed = MixedAudioSource([mic_src, sys_src])

        session = MeetingSession(
            backend=backend,
            audio_source=mixed,
            max_buffer_s=5.0,
        )

        session.start()

        # Wait for both sources to finish (mic ~36s + system starts at 37s, ~33s)
        # Total ~70s of audio time
        total_duration = s["system_offset_s"] + 40  # generous buffer
        time.sleep(total_duration)

        transcript = session.stop()

        assert len(transcript) > 0, "E2E alternating produced empty transcript"

        # Quality check using replay evaluation metrics
        combined_gt = self._get_combined_ground_truth(s)
        if combined_gt.strip():
            from tests.test_replay import _char_overlap, _compute_recall

            overlap = _char_overlap(transcript, combined_gt)
            recall = _compute_recall(transcript, combined_gt)

            print(f"\n  Dual-channel alternating quality:")
            print(f"    Transcript length: {len(transcript)} chars")
            print(f"    Char overlap: {overlap:.1%}")
            print(f"    Recall: {recall:.1%}")

            # Relaxed thresholds for dual-channel (mixing degrades quality)
            assert overlap >= 0.3, f"Char overlap too low: {overlap:.1%}"
            assert recall >= 0.4, f"Recall too low: {recall:.1%}"

    def test_overlapping_quality(self):
        """Overlapping dual-channel still produces usable transcription."""
        from asr_backend import MLXWhisperBackend

        s, mic_path, sys_path = self._get_e2e_scenario("e2e_overlapping")

        backend = MLXWhisperBackend()
        warmup = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)
        backend.transcribe(warmup, language=None, task="transcribe")

        mic_src = WavFileSource(mic_path, speed=1.0)
        sys_src = WavFileSource(sys_path, offset_s=s["system_offset_s"], speed=1.0)
        mixed = MixedAudioSource([mic_src, sys_src])

        session = MeetingSession(
            backend=backend,
            audio_source=mixed,
            max_buffer_s=5.0,
        )

        session.start()

        # mic ~51s, system starts at 15s and runs ~36s, so total ~51s
        total_duration = 55
        time.sleep(total_duration)

        transcript = session.stop()

        assert len(transcript) > 0, "E2E overlapping produced empty transcript"

        combined_gt = self._get_combined_ground_truth(s)
        if combined_gt.strip():
            from tests.test_replay import _char_overlap, _compute_recall

            overlap = _char_overlap(transcript, combined_gt)
            recall = _compute_recall(transcript, combined_gt)

            print(f"\n  Dual-channel overlapping quality:")
            print(f"    Transcript length: {len(transcript)} chars")
            print(f"    Char overlap: {overlap:.1%}")
            print(f"    Recall: {recall:.1%}")

            # Even more relaxed for overlapping speech
            assert overlap >= 0.2, f"Char overlap too low: {overlap:.1%}"
            assert recall >= 0.3, f"Recall too low: {recall:.1%}"


# ── Layer 3: Hardware tests (real SystemAudioHelper) ─────────

@pytest.mark.hardware
class TestSystemAudioSourceReal:
    """Tests requiring real SystemAudioHelper binary and macOS audio hardware."""

    def test_helper_binary_available(self):
        """SystemAudioHelper binary can be found."""
        src = SystemAudioSource()
        if not src.available:
            pytest.skip(
                "SystemAudioHelper not built. "
                "Run: cd swift/SystemAudioHelper && swift build -c release"
            )
        assert "SystemAudioHelper" in src._helper_path

    def test_captures_system_audio(self):
        """SystemAudioSource can start, receive audio, and stop."""
        src = SystemAudioSource()
        if not src.available:
            pytest.skip("SystemAudioHelper not built")

        received = []
        src.start(lambda chunk: received.append(chunk))

        # Let it capture for 2 seconds
        time.sleep(2.0)
        src.stop()

        assert not src.is_active
        # May or may not receive audio depending on whether anything is playing
        # The key assertion is: no crash, clean lifecycle
        print(f"  SystemAudioSource captured {len(received)} chunks in 2s")


@pytest.mark.hardware
class TestMeetingSessionWithRealSystem:
    """Full MeetingSession with real system audio capture."""

    def test_capture_system_audio_flag_starts(self):
        """MeetingSession with capture_system_audio=True starts without error."""
        sys_src = SystemAudioSource()
        if not sys_src.available:
            pytest.skip("SystemAudioHelper not built")

        from tests.test_meeting import MockBackend as MeetingMockBackend

        backend = MeetingMockBackend(responses=["test"] * 50)
        session = MeetingSession(
            backend=backend,
            capture_system_audio=True,
        )

        session.start()
        assert session.state == "recording"

        time.sleep(2.0)
        transcript = session.stop()
        assert session.state == "stopped"
        assert isinstance(transcript, str)
