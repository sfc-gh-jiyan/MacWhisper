"""Audio capture abstraction for MacWhisper.

Provides pluggable audio sources for microphone, system audio,
and mixed (mic + system) capture. Used by Meeting Mode for
continuous recording from multiple sources.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)


class AudioSource(ABC):
    """Abstract audio source interface.

    Subclasses must implement start/stop and provide audio via callback.
    Audio format: int16, mono, 16kHz.
    """

    @abstractmethod
    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        """Start capturing audio. Callback receives int16 chunks."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing audio."""
        ...

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Return True if currently capturing."""
        ...


class MicrophoneSource(AudioSource):
    """Microphone audio capture via sounddevice/PortAudio."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = 1,
        blocksize: int = 1024,
        device: int | str | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.device = device
        self._stream: sd.InputStream | None = None
        self._callback: Callable[[np.ndarray], None] | None = None

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        if self._stream is not None:
            self.stop()
        self._callback = callback

        def _sd_callback(indata, frame_count, time_info, status):
            if self._callback is not None:
                self._callback(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.blocksize,
            device=self.device,
            callback=_sd_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._callback = None

    @property
    def is_active(self) -> bool:
        return self._stream is not None


def _find_system_audio_helper() -> str | None:
    """Locate the SystemAudioHelper binary.

    Search order:
    1. Next to this file: swift/SystemAudioHelper/.build/release/SystemAudioHelper
    2. Bundled in the .app: Contents/MacOS/SystemAudioHelper
    3. On PATH
    """
    here = Path(__file__).resolve().parent

    # Dev layout
    dev_path = here / "swift" / "SystemAudioHelper" / ".build" / "release" / "SystemAudioHelper"
    if dev_path.is_file():
        return str(dev_path)

    # Bundled .app layout
    app_path = here / "SystemAudioHelper"
    if app_path.is_file():
        return str(app_path)

    # PATH fallback
    import shutil
    found = shutil.which("SystemAudioHelper")
    return found


class SystemAudioSource(AudioSource):
    """Captures system audio via the SystemAudioHelper Swift binary.

    The helper uses ScreenCaptureKit (macOS 13+) to capture all system
    audio output and pipes raw PCM (int16, mono, 16kHz) to stdout.
    This class spawns the helper as a subprocess and reads its stdout
    in a background thread, forwarding chunks to the callback.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        blocksize: int = 1024,
        helper_path: str | None = None,
    ):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self._helper_path = helper_path or _find_system_audio_helper()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._callback: Callable[[np.ndarray], None] | None = None
        self._stop_event = threading.Event()

    @property
    def available(self) -> bool:
        """Return True if the helper binary can be found and exists."""
        return self._helper_path is not None and os.path.isfile(self._helper_path)

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        if self._proc is not None:
            self.stop()
        if not self.available:
            raise RuntimeError(
                "SystemAudioHelper binary not found. "
                "Build it with: cd swift/SystemAudioHelper && swift build -c release"
            )

        self._callback = callback
        self._stop_event.clear()

        self._proc = subprocess.Popen(
            [self._helper_path, "--sample-rate", str(self.sample_rate)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="system-audio-reader"
        )
        self._thread.start()

        # Log stderr in a separate thread
        self._stderr_thread = threading.Thread(
            target=self._log_stderr, daemon=True, name="system-audio-stderr"
        )
        self._stderr_thread.start()

        logger.info("SystemAudioSource started (pid=%d)", self._proc.pid)

    def stop(self) -> None:
        self._stop_event.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        self._callback = None
        logger.info("SystemAudioSource stopped")

    @property
    def is_active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _read_loop(self) -> None:
        """Read raw PCM from helper stdout and forward to callback."""
        bytes_per_read = self.blocksize * 2  # int16 = 2 bytes per sample
        try:
            while not self._stop_event.is_set():
                if self._proc is None or self._proc.stdout is None:
                    break
                data = self._proc.stdout.read(bytes_per_read)
                if not data:
                    break
                samples = np.frombuffer(data, dtype=np.int16).reshape(-1, 1).copy()
                if self._callback is not None:
                    self._callback(samples)
        except Exception as e:
            if not self._stop_event.is_set():
                logger.error("SystemAudioSource read error: %s", e)

    def _log_stderr(self) -> None:
        """Forward helper stderr to Python logger."""
        try:
            while not self._stop_event.is_set():
                if self._proc is None or self._proc.stderr is None:
                    break
                line = self._proc.stderr.readline()
                if not line:
                    break
                logger.debug("helper: %s", line.decode("utf-8", errors="replace").rstrip())
        except Exception:
            pass


class WavFileSource(AudioSource):
    """Audio source that plays back a WAV file at real-time pace.

    Reads a mono 16kHz int16 WAV file and delivers chunks via callback
    at the same rate as live audio, making it a drop-in replacement for
    MicrophoneSource or SystemAudioSource in tests.

    Args:
        wav_path: Path to a mono 16kHz int16 WAV file.
        blocksize: Samples per chunk delivered to callback.
        offset_s: Delay in seconds before playback starts (silence padding).
        speed: Playback speed multiplier (1.0 = real-time, 0 = no sleep).
    """

    def __init__(
        self,
        wav_path: str,
        blocksize: int = 1024,
        offset_s: float = 0.0,
        speed: float = 1.0,
    ):
        self._wav_path = wav_path
        self._blocksize = blocksize
        self._offset_s = offset_s
        self._speed = speed
        self._callback: Callable[[np.ndarray], None] | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        if self._thread is not None:
            self.stop()
        self._callback = callback
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._playback_loop, daemon=True, name="wav-file-source"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        self._callback = None

    @property
    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _playback_loop(self) -> None:
        """Read WAV and deliver chunks at real-time pace."""
        import time

        # Delay start if offset_s > 0
        if self._offset_s > 0 and self._speed > 0:
            delay = self._offset_s / self._speed
            # Sleep in small increments so stop_event can interrupt
            elapsed = 0.0
            while elapsed < delay and not self._stop_event.is_set():
                step = min(0.05, delay - elapsed)
                time.sleep(step)
                elapsed += step

        sleep_per_chunk = (self._blocksize / SAMPLE_RATE) / self._speed if self._speed > 0 else 0

        try:
            with wave.open(self._wav_path, "rb") as wf:
                assert wf.getsampwidth() == 2, f"Expected 16-bit WAV, got {wf.getsampwidth()*8}-bit"
                assert wf.getnchannels() == 1, f"Expected mono WAV, got {wf.getnchannels()} channels"
                sr = wf.getframerate()
                if sr != SAMPLE_RATE:
                    logger.warning("WAV sample rate %d != %d, resampling not supported", sr, SAMPLE_RATE)

                while not self._stop_event.is_set():
                    raw = wf.readframes(self._blocksize)
                    if not raw:
                        break
                    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 1).copy()
                    if self._callback is not None:
                        self._callback(samples)
                    if sleep_per_chunk > 0:
                        time.sleep(sleep_per_chunk)
        except Exception as e:
            if not self._stop_event.is_set():
                logger.error("WavFileSource playback error: %s", e)


class MixedAudioSource(AudioSource):
    """Combines multiple AudioSource instances into one stream.

    Audio from all sources is forwarded to the same callback.
    """

    def __init__(self, sources: list[AudioSource] | None = None):
        self._sources: list[AudioSource] = sources or []
        self._active = False

    def add_source(self, source: AudioSource) -> None:
        self._sources.append(source)

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        self._active = True
        for source in self._sources:
            source.start(callback)

    def stop(self) -> None:
        self._active = False
        for source in self._sources:
            source.stop()

    @property
    def is_active(self) -> bool:
        return self._active
