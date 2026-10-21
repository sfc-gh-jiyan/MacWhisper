"""open_input_stream: recover from stale PortAudio device lists (-9986)."""

from unittest import mock

import pytest
import sounddevice as sd

import audio_capture


def _err():
    return sd.PortAudioError("Error opening InputStream: Internal PortAudio error", -9986)


def test_success_first_try_no_reinit():
    with mock.patch.object(sd, "InputStream", return_value="S") as ctor, \
         mock.patch.object(sd, "_terminate") as term, mock.patch.object(sd, "_initialize") as init:
        assert audio_capture.open_input_stream(samplerate=16000) == "S"
    ctor.assert_called_once_with(samplerate=16000)
    term.assert_not_called()
    init.assert_not_called()


def test_reinit_and_retry_once_on_portaudio_error():
    with mock.patch.object(sd, "InputStream", side_effect=[_err(), "S"]) as ctor, \
         mock.patch.object(sd, "_terminate") as term, mock.patch.object(sd, "_initialize") as init:
        assert audio_capture.open_input_stream(channels=1) == "S"
    assert ctor.call_count == 2
    term.assert_called_once()
    init.assert_called_once()


def test_second_failure_raises():
    with mock.patch.object(sd, "InputStream", side_effect=[_err(), _err()]), \
         mock.patch.object(sd, "_terminate"), mock.patch.object(sd, "_initialize"):
        with pytest.raises(sd.PortAudioError):
            audio_capture.open_input_stream()


def test_reinit_failure_still_retries():
    with mock.patch.object(sd, "InputStream", side_effect=[_err(), "S"]), \
         mock.patch.object(sd, "_terminate", side_effect=RuntimeError("x")), \
         mock.patch.object(sd, "_initialize"):
        assert audio_capture.open_input_stream() == "S"
