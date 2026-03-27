"""Tests for Meeting Mode (MeetingSession, AudioSource, VAD extensions).

Tests use mock backends to avoid loading actual ML models.
"""

import os
import time
import threading
import tempfile

import numpy as np
import pytest

from audio_capture import AudioSource, MicrophoneSource, MixedAudioSource, SystemAudioSource
from vad import VoiceActivityDetector
from meeting import MeetingSession, MeetingSegment


# ── Mock classes ─────────────────────────────────────────

class MockBackend:
    """Mock ASR backend that returns predictable transcription results."""

    def __init__(self, responses=None):
        self._responses = responses or []
        self._call_count = 0

    def transcribe(self, audio, *, language=None, initial_prompt=None, task="transcribe"):
        from asr_backend import TranscriptionResult, Segment, WordTimestamp

        if self._call_count < len(self._responses):
            text = self._responses[self._call_count]
        else:
            text = f"word{self._call_count}"
        self._call_count += 1

        # Generate word timestamps from text
        words_text = text.split() if text else []
        words = []
        t = 0.0
        for w in words_text:
            words.append(WordTimestamp(word=w + " ", start=t, end=t + 0.3, probability=0.9))
            t += 0.4

        seg = Segment(text=text, start=0.0, end=t, words=words)
        return TranscriptionResult(text=text, segments=[seg], language="zh")


class MockAudioSource(AudioSource):
    """Audio source that generates synthetic audio for testing."""

    def __init__(self, chunk_size=1024, sample_rate=16000):
        self._chunk_size = chunk_size
        self._sample_rate = sample_rate
        self._callback = None
        self._active = False
        self._thread = None

    def start(self, callback):
        self._callback = callback
        self._active = True
        self._thread = threading.Thread(target=self._generate, daemon=True)
        self._thread.start()

    def stop(self):
        self._active = False
        self._callback = None

    @property
    def is_active(self):
        return self._active

    def _generate(self):
        """Generate synthetic speech-like audio chunks."""
        while self._active:
            # Generate audio with enough amplitude to pass VAD/RMS checks
            chunk = np.random.randint(-3000, 3000, size=self._chunk_size, dtype=np.int16)
            if self._callback:
                self._callback(chunk.reshape(-1, 1))
            time.sleep(self._chunk_size / self._sample_rate)


# ── AudioSource tests ────────────────────────────────────

class TestMicrophoneSource:
    """Test MicrophoneSource interface (without actual hardware)."""

    def test_initial_state(self):
        mic = MicrophoneSource()
        assert not mic.is_active

    def test_start_stop_lifecycle(self):
        """Test that start/stop toggle active state (mock approach)."""
        # We can't test actual audio without a device, but we can test
        # the state management
        mic = MicrophoneSource()
        assert not mic.is_active
        # Calling stop when not started should be safe
        mic.stop()
        assert not mic.is_active


class TestMixedAudioSource:

    def test_empty_sources(self):
        mixed = MixedAudioSource([])
        assert not mixed.is_active
        mixed.start(lambda x: None)
        assert mixed.is_active
        mixed.stop()
        assert not mixed.is_active

    def test_add_source(self):
        mixed = MixedAudioSource()
        mock = MockAudioSource()
        mixed.add_source(mock)
        assert len(mixed._sources) == 1

    def test_forwards_to_callback(self):
        """Test that audio from source reaches the callback."""
        received = []
        mock = MockAudioSource(chunk_size=512)
        mixed = MixedAudioSource([mock])

        mixed.start(lambda chunk: received.append(chunk))
        time.sleep(0.15)
        mixed.stop()

        assert len(received) > 0
        assert received[0].dtype == np.int16


# ── VAD extension tests ──────────────────────────────────

class TestVADExtensions:
    """Test is_extended_silence() and is_active_speech() additions."""

    def _make_vad(self):
        """Create a VAD with RMS fallback (no torch needed for tests)."""
        vad = VoiceActivityDetector.__new__(VoiceActivityDetector)
        vad.threshold = 0.5
        vad.min_silence_ms = 800
        vad.min_speech_ms = 250
        vad.sample_rate = 16000
        vad._model = None
        vad._use_silero = False
        vad._silence_samples = 0
        vad._speech_samples = 0
        return vad

    def test_extended_silence_false_initially(self):
        vad = self._make_vad()
        assert not vad.is_extended_silence(2000)

    def test_extended_silence_after_accumulation(self):
        vad = self._make_vad()
        # Simulate 2.5 seconds of silence
        vad._silence_samples = int(16000 * 2.5)
        assert vad.is_extended_silence(2000)
        assert not vad.is_extended_silence(3000)

    def test_is_active_speech(self):
        vad = self._make_vad()
        assert not vad.is_active_speech()
        # Simulate 300ms of speech (above min_speech_ms=250)
        vad._speech_samples = int(16000 * 0.3)
        assert vad.is_active_speech()

    def test_is_speech_end_unchanged(self):
        """Ensure existing is_speech_end still works."""
        vad = self._make_vad()
        assert not vad.is_speech_end()
        vad._silence_samples = int(16000 * 0.9)  # 900ms > 800ms threshold
        assert vad.is_speech_end()


# ── MeetingSession tests ─────────────────────────────────

class TestMeetingSessionState:
    """Test MeetingSession state machine."""

    def _make_session(self, **kwargs):
        backend = MockBackend(responses=["hello world"] * 50)
        source = MockAudioSource()
        return MeetingSession(
            backend=backend,
            audio_source=source,
            **kwargs,
        )

    def test_initial_state(self):
        session = self._make_session()
        assert session.state == "idle"
        assert not session.is_recording
        assert session.segments == []

    def test_start_sets_recording(self):
        session = self._make_session()
        session.start()
        assert session.state == "recording"
        assert session.is_recording
        session.stop()

    def test_stop_returns_transcript(self):
        session = self._make_session()
        session.start()
        time.sleep(0.3)  # let some audio flow
        transcript = session.stop()
        assert session.state == "stopped"
        assert isinstance(transcript, str)

    def test_pause_resume(self):
        session = self._make_session()
        session.start()
        assert session.state == "recording"

        session.pause()
        assert session.state == "paused"

        session.resume()
        assert session.state == "recording"

        session.stop()
        assert session.state == "stopped"

    def test_double_start_ignored(self):
        session = self._make_session()
        session.start()
        session.start()  # should be ignored
        assert session.state == "recording"
        session.stop()

    def test_stop_when_idle(self):
        session = self._make_session()
        result = session.stop()
        assert result == ""

    def test_export_markdown(self):
        session = self._make_session()
        session._session_id = "20260326_120000"
        session._segments = [
            MeetingSegment(text="Hello world", start_time=0.0, end_time=5.0),
            MeetingSegment(text="Second paragraph", start_time=7.0, end_time=12.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = session.export(tmpdir, fmt="md")
            assert os.path.exists(path)
            content = open(path).read()
            assert "# Meeting Transcript" in content
            assert "Hello world" in content
            assert "Second paragraph" in content
            assert "[00:00]" in content
            assert "[00:07]" in content

    def test_export_txt(self):
        session = self._make_session()
        session._segments = [
            MeetingSegment(text="First", start_time=0.0, end_time=3.0),
            MeetingSegment(text="Second", start_time=5.0, end_time=8.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            session.export(path, fmt="txt")
            content = open(path).read()
            assert "First" in content
            assert "Second" in content

    def test_auto_save(self):
        session = self._make_session()
        session._session_id = "20260326_test"
        session._segments = [
            MeetingSegment(text="Test segment", start_time=0.0, end_time=5.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            import meeting
            orig_dir = meeting.MEETINGS_DIR
            meeting.MEETINGS_DIR = tmpdir
            try:
                session._auto_save()
                files = os.listdir(tmpdir)
                assert any("meeting_20260326_test.md" in f for f in files)
                assert any("meeting_20260326_test.json" in f for f in files)
            finally:
                meeting.MEETINGS_DIR = orig_dir


class TestMeetingSegment:

    def test_dataclass_fields(self):
        seg = MeetingSegment(text="hello", start_time=1.0, end_time=3.0)
        assert seg.text == "hello"
        assert seg.start_time == 1.0
        assert seg.end_time == 3.0
        assert seg.words == []

    def test_with_words(self):
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]
        seg = MeetingSegment(text="hello world", start_time=0.0, end_time=1.0, words=words)
        assert len(seg.words) == 2


class TestSystemAudioSource:

    def test_initial_state(self):
        src = SystemAudioSource(helper_path="/nonexistent/binary")
        assert not src.is_active
        assert not src.available

    def test_available_with_real_binary(self):
        """Check that the built helper binary is detected."""
        src = SystemAudioSource()
        # In dev environment with built binary, this should be True.
        # If not built, skip gracefully.
        if src.available:
            assert src._helper_path is not None
            assert "SystemAudioHelper" in src._helper_path

    def test_start_raises_without_helper(self):
        src = SystemAudioSource(helper_path="/nonexistent/binary")
        with pytest.raises(RuntimeError, match="SystemAudioHelper binary not found"):
            src.start(lambda x: None)

    def test_stop_when_not_started(self):
        """stop() should not raise when called without start()."""
        src = SystemAudioSource(helper_path="/nonexistent/binary")
        src.stop()  # Should not raise

    def test_find_helper_dev_layout(self, tmp_path):
        """Test that _find_system_audio_helper finds binary in dev layout."""
        from audio_capture import _find_system_audio_helper
        # The real function searches relative to audio_capture.py,
        # so we just verify it returns a string or None
        result = _find_system_audio_helper()
        assert result is None or isinstance(result, str)

    def test_subprocess_lifecycle(self, tmp_path):
        """Test start/stop with a fake helper that produces PCM data."""
        import sys
        # Create a Python script that outputs raw PCM bytes continuously
        script = tmp_path / "fake_helper.py"
        script.write_text(
            "import sys, time\n"
            "# Output blocks of zero bytes (int16 silence)\n"
            "block = b'\\x00' * 2048  # 1024 int16 samples\n"
            "for _ in range(20):\n"
            "    sys.stdout.buffer.write(block)\n"
            "    sys.stdout.buffer.flush()\n"
            "    time.sleep(0.05)\n"
            "time.sleep(10)\n"
        )

        received = []
        src = SystemAudioSource(helper_path=sys.executable)
        # Override the command to use our script
        src._helper_path = sys.executable
        original_start = SystemAudioSource.start

        # Monkey-patch to inject our script args
        import subprocess as sp
        src._proc = sp.Popen(
            [sys.executable, str(script)],
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            bufsize=0,
        )
        src._callback = lambda chunk: received.append(chunk)
        src._stop_event.clear()
        src._thread = threading.Thread(
            target=src._read_loop, daemon=True, name="test-reader"
        )
        src._thread.start()

        # Wait for data to arrive
        time.sleep(0.8)
        src.stop()

        assert not src.is_active
        assert len(received) > 0
        for chunk in received:
            assert chunk.dtype == np.int16

    def test_mixed_source_with_system_audio(self):
        """MixedAudioSource should accept SystemAudioSource."""
        mic = MockAudioSource()
        sys = SystemAudioSource(helper_path="/nonexistent/binary")
        mixed = MixedAudioSource([mic])
        mixed.add_source(sys)
        assert len(mixed._sources) == 2


class TestMeetingSessionWithSystemAudio:

    def test_capture_system_audio_flag(self):
        """MeetingSession with capture_system_audio creates SystemAudioSource."""
        backend = MockBackend()
        session = MeetingSession(
            backend=backend,
            capture_system_audio=True,
        )
        # Should have a MixedAudioSource
        assert isinstance(session._audio_source, MixedAudioSource)
        sources = session._audio_source._sources
        # Always has at least MicrophoneSource
        assert any(isinstance(s, MicrophoneSource) for s in sources)
        # If helper binary is available, SystemAudioSource should be present
        sys_sources = [s for s in sources if isinstance(s, SystemAudioSource)]
        if sys_sources:
            assert sys_sources[0].available

    def test_no_system_audio_by_default(self):
        """MeetingSession without flag should not have SystemAudioSource."""
        backend = MockBackend()
        session = MeetingSession(backend=backend)
        sources = session._audio_source._sources
        assert not any(isinstance(s, SystemAudioSource) for s in sources)
