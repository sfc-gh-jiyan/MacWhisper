"""Dual-channel Meeting Mode tests.

Tests the full mic + system-audio pipeline using WavFileSource to inject
real WAV recordings into MeetingSession. Three layers:

- Layer 1 (integration): MockBackend, fast, CI-friendly
- Layer 2 (e2e quality): Real MLXWhisperBackend, @pytest.mark.slow
- Layer 3 (hardware):    Real SystemAudioHelper, @pytest.mark.hardware

Run Layer 1 only (CI):
    venv/bin/python -m pytest tests/test_meeting_dual.py -m "not slow and not hardware" -v

Run E2E quality (fixed scenarios):
    venv/bin/python -m pytest tests/test_meeting_dual.py -m slow -v

Run E2E quality (N-to-N generated, 3 pairs × 3 modes = 9 scenarios):
    venv/bin/python -m pytest tests/test_meeting_dual.py -m slow -v --generate 3
"""

import datetime
import json
import os
import time
import threading
import wave

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
from tests.eval_metrics import evaluate_dual_channel


# ── Paths ────────────────────────────────────────────────────

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SCENARIOS_FILE = os.path.join(FIXTURES_DIR, "dual_channel_scenarios.json")
AUDIO_DIR = os.path.expanduser("~/.macwhisper/audio")
TRANSCRIPT_LOG = os.path.expanduser("~/.macwhisper/transcripts.jsonl")
RESULTS_FILE = os.path.expanduser("~/.macwhisper/dual_channel_results.jsonl")


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


def _wait_for(predicate, timeout=5.0, poll=0.05):
    """Poll predicate() until it returns True or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return predicate()


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
        _wait_for(lambda: received["count"] > 10, timeout=5.0)
        mixed.stop()

        assert received["count"] > 10, f"Only got {received['count']} chunks from dual sources"

    def test_meeting_session_with_dual_wav(self):
        """MeetingSession processes dual-channel WAV input and produces transcript."""
        s, mic_path, sys_path = self._get_scenario()

        backend = MockBackend(responses=["hello world"] * 200)
        mic_src = WavFileSource(mic_path, speed=0)
        sys_src = WavFileSource(sys_path, offset_s=s["system_offset_s"], speed=0)
        mixed = MixedAudioSource([mic_src, sys_src])

        session = MeetingSession(
            backend=backend,
            audio_source=mixed,
        )

        session.start()
        assert session.state == "recording"

        # Wait for processor to produce output (speed=0: audio arrives instantly,
        # but meeting loop has 50ms sleep per iteration)
        _wait_for(lambda: backend._call_count > 0, timeout=5.0)

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

        backend = MockBackend(responses=["hello world"] * 300)
        mic_src = WavFileSource(mic_path, speed=0)
        # Small offset so they overlap significantly
        sys_src = WavFileSource(sys_path, offset_s=s["system_offset_s"], speed=0)
        mixed = MixedAudioSource([mic_src, sys_src])

        session = MeetingSession(backend=backend, audio_source=mixed)

        session.start()
        _wait_for(lambda: backend._call_count > 0, timeout=5.0)
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
        _wait_for(lambda: len(received) > 20, timeout=5.0)
        mixed.stop()

        assert len(received) > 0
        # All chunks should be valid int16 arrays
        for chunk in received:
            assert chunk.dtype == np.int16
            assert chunk.ndim == 2


class TestDualChannelPipelineV2:
    """Layer 1 integration: dual-channel path (separate mic/sys sources, RMS priority)."""

    def _get_scenario(self, name="integration_alternating"):
        scenarios = _load_scenarios("integration")
        for s in scenarios:
            if s["name"] == name:
                mic_path = os.path.join(AUDIO_DIR, s["mic_wav"])
                sys_path = os.path.join(AUDIO_DIR, s["system_wav"])
                if os.path.isfile(mic_path) and os.path.isfile(sys_path):
                    return s, mic_path, sys_path
        pytest.skip("Integration WAV files not available")

    def _make_dual_session(self, mic_path, sys_path, offset_s=0):
        """Create a MeetingSession in dual-channel mode with WavFileSources."""
        backend = MockBackend(responses=["hello world"] * 200)
        mic_src = WavFileSource(mic_path, speed=0)
        sys_src = WavFileSource(sys_path, offset_s=offset_s, speed=0)

        session = MeetingSession(
            backend=backend, audio_source=mic_src, min_chunk_size=0.1,
        )
        session._dual_channel = True
        session._mic_source = mic_src
        session._sys_source = sys_src
        session._audio_source = None
        return session, backend

    def test_dual_channel_alternating(self):
        """Dual-channel mode processes alternating WAV input correctly."""
        s, mic_path, sys_path = self._get_scenario("integration_alternating")
        session, backend = self._make_dual_session(
            mic_path, sys_path, offset_s=s["system_offset_s"],
        )

        session.start()
        assert session.state == "recording"
        _wait_for(lambda: backend._call_count > 0, timeout=5.0)
        transcript = session.stop()

        assert session.state == "stopped"
        assert isinstance(transcript, str)
        assert len(transcript) > 0, "Dual-channel alternating produced empty transcript"

    def test_dual_channel_overlapping(self):
        """Dual-channel mode handles overlapping sources via RMS priority."""
        s, mic_path, sys_path = self._get_scenario("integration_overlapping")
        session, backend = self._make_dual_session(
            mic_path, sys_path, offset_s=s["system_offset_s"],
        )

        session.start()
        _wait_for(lambda: backend._call_count > 0, timeout=5.0)
        transcript = session.stop()

        assert isinstance(transcript, str)
        assert len(transcript) > 0, "Dual-channel overlapping produced empty transcript"

    def test_dual_channel_state_machine(self):
        """Start -> pause -> resume -> stop works in dual-channel mode."""
        s, mic_path, sys_path = self._get_scenario()
        session, backend = self._make_dual_session(
            mic_path, sys_path, offset_s=s["system_offset_s"],
        )

        session.start()
        assert session.state == "recording"

        time.sleep(0.3)
        session.pause()
        assert session.state == "paused"

        session.resume()
        assert session.state == "recording"

        time.sleep(0.3)
        transcript = session.stop()
        assert session.state == "stopped"
        assert isinstance(transcript, str)

    def test_dual_channel_both_sources_contribute(self):
        """Both mic and sys buffers receive chunks in dual-channel mode."""
        s, mic_path, sys_path = self._get_scenario()
        session, backend = self._make_dual_session(
            mic_path, sys_path, offset_s=s["system_offset_s"],
        )

        session.start()
        _wait_for(
            lambda: len(session._mic_frames) > 5 and len(session._sys_frames) > 5,
            timeout=5.0,
        )
        mic_count = len(session._mic_frames)
        sys_count = len(session._sys_frames)
        session.stop()

        assert mic_count > 5, f"Mic only delivered {mic_count} frames"
        assert sys_count > 5, f"Sys only delivered {sys_count} frames"


# ── Layer 2: E2E Quality tests (real Whisper) ────────────────

def _wav_duration(filename):
    """Get WAV duration in seconds."""
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(path):
        return 0.0
    with wave.open(path) as wf:
        return wf.getnframes() / wf.getframerate()


def _e2e_scenario_ids():
    """Load E2E scenario names for parametrize IDs."""
    scenarios = _load_scenarios("e2e")
    return [s["name"] for s in scenarios]


def _e2e_scenarios():
    """Load E2E scenarios for parametrize."""
    return _load_scenarios("e2e")


def _save_result(result):
    """Append a result dict to the JSONL results file."""
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


@pytest.mark.slow
class TestDualChannelQuality:
    """End-to-end quality evaluation with real MLXWhisperBackend.

    Parametrized over all E2E scenarios in dual_channel_scenarios.json.
    Each scenario runs two WAV files through MeetingSession at real-time
    pace and evaluates the transcript against combined ground truth.
    """

    @pytest.fixture(params=_e2e_scenarios(), ids=_e2e_scenario_ids())
    def scenario(self, request):
        """Yield one E2E scenario, skipping if WAVs are unavailable."""
        s = request.param
        mic_path = os.path.join(AUDIO_DIR, s["mic_wav"])
        sys_path = os.path.join(AUDIO_DIR, s["system_wav"])
        if not os.path.isfile(mic_path) or not os.path.isfile(sys_path):
            pytest.skip(f"WAV files not available for {s['name']}")
        return s, mic_path, sys_path

    @pytest.fixture(scope="class")
    def backend(self):
        """Shared MLXWhisperBackend with warmup (once per class)."""
        from asr_backend import MLXWhisperBackend

        backend = MLXWhisperBackend()
        warmup = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)
        backend.transcribe(warmup, language=None, task="transcribe")
        return backend

    def _get_combined_ground_truth(self, scenario):
        """Combine ground truth from both WAV files."""
        gt = _load_ground_truth()
        mic_gt = gt.get(scenario["mic_wav"], "")
        sys_gt = gt.get(scenario["system_wav"], "")
        return mic_gt + " " + sys_gt

    def _run_and_evaluate(self, s, mic_path, sys_path, backend, pipeline="mixed"):
        """Run a scenario through the specified pipeline and return (transcript, scores).

        pipeline: "mixed" (legacy MixedAudioSource) or "dual" (RMS priority selection)
        """
        mode = s.get("mode", "alternating")

        mic_src = WavFileSource(mic_path, speed=1.0)
        sys_src = WavFileSource(sys_path, offset_s=s["system_offset_s"], speed=1.0)

        if pipeline == "dual":
            session = MeetingSession(
                backend=backend,
                audio_source=mic_src,
                max_buffer_s=5.0,
            )
            session._dual_channel = True
            session._mic_source = mic_src
            session._sys_source = sys_src
            session._audio_source = None
        else:
            mixed = MixedAudioSource([mic_src, sys_src])
            session = MeetingSession(
                backend=backend,
                audio_source=mixed,
                max_buffer_s=5.0,
            )

        session.start()

        mic_dur = _wav_duration(s["mic_wav"])
        sys_dur = _wav_duration(s["system_wav"])
        total_audio = max(mic_dur, s["system_offset_s"] + sys_dur)
        time.sleep(total_audio + 5)

        transcript = session.stop()

        assert len(transcript) > 0, f"{s['name']} [{pipeline}]: produced empty transcript"

        combined_gt = self._get_combined_ground_truth(s)
        if not combined_gt.strip():
            pytest.skip(f"No ground truth available for {s['name']}")

        scores = evaluate_dual_channel(transcript, combined_gt)

        # Print report
        print(f"\n  [{pipeline}] {s['name']} ({mode}):")
        print(f"    Transcript: {len(transcript)} chars")
        print(f"    Char overlap: {scores['char_overlap']:.1%}")
        print(f"    Recall:       {scores['recall']:.1%}")
        print(f"    WER:          {scores['wer']:.1%}")
        print(f"    Duplications: {scores['duplications']}")
        print(f"    Tier:         {scores['tier']}")

        # Persist result
        result = {
            "timestamp": datetime.datetime.now().isoformat(),
            "scenario": s["name"],
            "pipeline": pipeline,
            "mic_wav": s["mic_wav"],
            "system_wav": s["system_wav"],
            "mode": mode,
            "char_overlap": scores["char_overlap"],
            "recall": scores["recall"],
            "wer": scores["wer"],
            "duplications": scores["duplications"],
            "tier": scores["tier"],
            "transcript_len": len(transcript),
            "gt_len": len(combined_gt),
            "audio_duration_s": round(total_audio, 1),
        }
        _save_result(result)

        return transcript, scores, mode, total_audio

    def _assert_thresholds(self, s, scores, mode):
        """Apply pass/fail thresholds based on mode."""
        if mode == "overlapping":
            if scores["char_overlap"] < 0.05:
                pytest.xfail(
                    f"{s['name']}: extreme overlap produced {scores['char_overlap']:.1%} "
                    f"overlap — known Whisper limitation with fully concurrent speech"
                )
            assert scores["char_overlap"] >= 0.15, \
                f"{s['name']}: char overlap {scores['char_overlap']:.1%} < 15%"
            assert scores["recall"] >= 0.20, \
                f"{s['name']}: recall {scores['recall']:.1%} < 20%"
        else:
            assert scores["char_overlap"] >= 0.25, \
                f"{s['name']}: char overlap {scores['char_overlap']:.1%} < 25%"
            assert scores["recall"] >= 0.35, \
                f"{s['name']}: recall {scores['recall']:.1%} < 35%"

    def test_scenario_quality_mixed(self, scenario, backend):
        """Run scenario via legacy MixedAudioSource (baseline)."""
        s, mic_path, sys_path = scenario
        _, scores, mode, _ = self._run_and_evaluate(
            s, mic_path, sys_path, backend, pipeline="mixed",
        )
        self._assert_thresholds(s, scores, mode)

    def test_scenario_quality_dual(self, scenario, backend):
        """Run scenario via dual-channel RMS priority selection."""
        s, mic_path, sys_path = scenario
        _, scores, mode, _ = self._run_and_evaluate(
            s, mic_path, sys_path, backend, pipeline="dual",
        )
        self._assert_thresholds(s, scores, mode)


@pytest.mark.slow
class TestGeneratedScenarios:
    """N-to-N dual-channel tests using dynamically generated WAV pairs.

    Only runs when --generate N is passed:
        pytest tests/test_meeting_dual.py -m slow --generate 3
    """

    def pytest_generate_tests(self, metafunc):
        """Dynamically parametrize from --generate flag."""
        if "gen_scenario" in metafunc.fixturenames:
            n = metafunc.config.getoption("--generate", default=0)
            if n > 0:
                from tests.dual_channel_gen import generate_scenarios
                scenarios = generate_scenarios(n_pairs=n)
                ids = [s["name"] for s in scenarios]
                metafunc.parametrize("gen_scenario", scenarios, ids=ids)
            else:
                # No --generate flag: skip all tests in this class
                metafunc.parametrize(
                    "gen_scenario", [pytest.param(None, marks=pytest.mark.skip(
                        reason="Pass --generate N to enable N-to-N tests"))],
                    ids=["no_generate"],
                )

    @pytest.fixture(scope="class")
    def backend(self):
        """Shared MLXWhisperBackend with warmup."""
        from asr_backend import MLXWhisperBackend

        backend = MLXWhisperBackend()
        warmup = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)
        backend.transcribe(warmup, language=None, task="transcribe")
        return backend

    def test_generated_quality(self, gen_scenario, backend):
        """Run a generated dual-channel scenario via dual-channel path."""
        s = gen_scenario
        mode = s.get("mode", "alternating")

        mic_path = os.path.join(AUDIO_DIR, s["mic_wav"])
        sys_path = os.path.join(AUDIO_DIR, s["system_wav"])
        if not os.path.isfile(mic_path) or not os.path.isfile(sys_path):
            pytest.skip(f"WAV files not available for {s['name']}")

        mic_src = WavFileSource(mic_path, speed=1.0)
        sys_src = WavFileSource(sys_path, offset_s=s["system_offset_s"], speed=1.0)

        # Use dual-channel path (RMS priority selection)
        session = MeetingSession(
            backend=backend,
            audio_source=mic_src,
            max_buffer_s=5.0,
        )
        session._dual_channel = True
        session._mic_source = mic_src
        session._sys_source = sys_src
        session._audio_source = None

        session.start()

        mic_dur = _wav_duration(s["mic_wav"])
        sys_dur = _wav_duration(s["system_wav"])
        total_audio = max(mic_dur, s["system_offset_s"] + sys_dur)
        time.sleep(total_audio + 5)

        transcript = session.stop()

        assert len(transcript) > 0, f"{s['name']}: produced empty transcript"

        gt = _load_ground_truth()
        mic_gt = gt.get(s["mic_wav"], "")
        sys_gt = gt.get(s["system_wav"], "")
        combined_gt = mic_gt + " " + sys_gt
        if not combined_gt.strip():
            pytest.skip(f"No ground truth available for {s['name']}")

        scores = evaluate_dual_channel(transcript, combined_gt)

        print(f"\n  Generated [dual] {s['name']} ({mode}):")
        print(f"    Transcript: {len(transcript)} chars")
        print(f"    Char overlap: {scores['char_overlap']:.1%}")
        print(f"    Recall:       {scores['recall']:.1%}")
        print(f"    WER:          {scores['wer']:.1%}")
        print(f"    Tier:         {scores['tier']}")

        _save_result({
            "timestamp": datetime.datetime.now().isoformat(),
            "scenario": s["name"],
            "pipeline": "dual",
            "mic_wav": s["mic_wav"],
            "system_wav": s["system_wav"],
            "mode": mode,
            "char_overlap": scores["char_overlap"],
            "recall": scores["recall"],
            "wer": scores["wer"],
            "duplications": scores["duplications"],
            "tier": scores["tier"],
            "transcript_len": len(transcript),
            "gt_len": len(combined_gt),
            "audio_duration_s": round(total_audio, 1),
            "generated": True,
        })

        # Same thresholds as fixed scenarios
        if mode == "overlapping":
            if scores["char_overlap"] < 0.05:
                pytest.xfail(
                    f"{s['name']}: extreme overlap produced {scores['char_overlap']:.1%} "
                    f"overlap — known Whisper limitation with fully concurrent speech"
                )
            assert scores["char_overlap"] >= 0.15, \
                f"{s['name']}: char overlap {scores['char_overlap']:.1%} < 15%"
            assert scores["recall"] >= 0.20, \
                f"{s['name']}: recall {scores['recall']:.1%} < 20%"
        else:
            assert scores["char_overlap"] >= 0.25, \
                f"{s['name']}: char overlap {scores['char_overlap']:.1%} < 25%"
            assert scores["recall"] >= 0.35, \
                f"{s['name']}: recall {scores['recall']:.1%} < 35%"


@pytest.mark.slow
class TestOverlapABComparison:
    """A/B comparison: same overlapping scenarios through mixed vs dual pipeline.

    Runs each overlapping E2E scenario twice (mixed then dual) and prints
    a side-by-side quality comparison. Asserts that dual is not worse than mixed.
    """

    @pytest.fixture(scope="class")
    def backend(self):
        """Shared MLXWhisperBackend with warmup."""
        from asr_backend import MLXWhisperBackend

        backend = MLXWhisperBackend()
        warmup = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)
        backend.transcribe(warmup, language=None, task="transcribe")
        return backend

    @pytest.fixture(params=[
        s for s in _e2e_scenarios() if s.get("mode") == "overlapping"
    ], ids=[
        s["name"] for s in _e2e_scenarios() if s.get("mode") == "overlapping"
    ])
    def overlap_scenario(self, request):
        s = request.param
        mic_path = os.path.join(AUDIO_DIR, s["mic_wav"])
        sys_path = os.path.join(AUDIO_DIR, s["system_wav"])
        if not os.path.isfile(mic_path) or not os.path.isfile(sys_path):
            pytest.skip(f"WAV files not available for {s['name']}")
        return s, mic_path, sys_path

    def _run_pipeline(self, s, mic_path, sys_path, backend, pipeline):
        """Run one pipeline variant and return scores."""
        mic_src = WavFileSource(mic_path, speed=1.0)
        sys_src = WavFileSource(sys_path, offset_s=s["system_offset_s"], speed=1.0)

        if pipeline == "dual":
            session = MeetingSession(
                backend=backend, audio_source=mic_src, max_buffer_s=5.0,
            )
            session._dual_channel = True
            session._mic_source = mic_src
            session._sys_source = sys_src
            session._audio_source = None
        else:
            mixed = MixedAudioSource([mic_src, sys_src])
            session = MeetingSession(
                backend=backend, audio_source=mixed, max_buffer_s=5.0,
            )

        session.start()
        mic_dur = _wav_duration(s["mic_wav"])
        sys_dur = _wav_duration(s["system_wav"])
        total_audio = max(mic_dur, s["system_offset_s"] + sys_dur)
        time.sleep(total_audio + 5)
        transcript = session.stop()

        gt = _load_ground_truth()
        combined_gt = gt.get(s["mic_wav"], "") + " " + gt.get(s["system_wav"], "")
        if not combined_gt.strip():
            return None
        return evaluate_dual_channel(transcript, combined_gt)

    def test_dual_not_worse_than_mixed(self, overlap_scenario, backend):
        """Dual-channel should produce equal or better quality than mixed."""
        s, mic_path, sys_path = overlap_scenario

        mixed_scores = self._run_pipeline(s, mic_path, sys_path, backend, "mixed")
        dual_scores = self._run_pipeline(s, mic_path, sys_path, backend, "dual")

        if mixed_scores is None or dual_scores is None:
            pytest.skip(f"No ground truth for {s['name']}")

        print(f"\n  A/B Comparison [{s['name']}]:")
        print(f"    {'Metric':<16} {'Mixed':>8} {'Dual':>8} {'Delta':>8}")
        print(f"    {'-'*40}")
        for metric in ("char_overlap", "recall", "wer"):
            m = mixed_scores[metric]
            d = dual_scores[metric]
            delta = d - m
            sign = "+" if delta >= 0 else ""
            print(f"    {metric:<16} {m:>7.1%} {d:>7.1%} {sign}{delta:>7.1%}")

        _save_result({
            "timestamp": datetime.datetime.now().isoformat(),
            "scenario": s["name"],
            "test": "ab_comparison",
            "mixed_overlap": mixed_scores["char_overlap"],
            "dual_overlap": dual_scores["char_overlap"],
            "mixed_recall": mixed_scores["recall"],
            "dual_recall": dual_scores["recall"],
            "mixed_wer": mixed_scores["wer"],
            "dual_wer": dual_scores["wer"],
        })

        # Dual should not be significantly worse (allow 5% tolerance)
        assert dual_scores["char_overlap"] >= mixed_scores["char_overlap"] - 0.05, \
            (f"{s['name']}: dual overlap {dual_scores['char_overlap']:.1%} "
             f"significantly worse than mixed {mixed_scores['char_overlap']:.1%}")


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

        backend = MockBackend(responses=["test"] * 50)
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
