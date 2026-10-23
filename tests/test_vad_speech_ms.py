import numpy as np
import vad


class _Model:
    def __init__(self, conf):
        self.conf, self.reset = conf, 0

    def __call__(self, tensor, sr):
        return type("C", (), {"item": lambda s: self.conf})()

    def reset_states(self):
        self.reset += 1


def _det(conf):
    d = vad.VoiceActivityDetector.__new__(vad.VoiceActivityDetector)
    d.threshold, d.sample_rate, d._use_silero, d._model = 0.5, 16000, True, _Model(conf)
    return d


def test_speech_ms_counts_speech_windows_and_resets_state():
    d = _det(0.9)
    assert 950 <= d.speech_ms(np.zeros(16000, dtype=np.int16)) <= 1000
    assert d._model.reset >= 1


def test_speech_ms_zero_on_silence():
    assert _det(0.1).speech_ms(np.zeros(16000, dtype=np.int16)) == 0


def test_speech_ms_none_without_silero():
    d = _det(0.9)
    d._use_silero = False
    assert d.speech_ms(np.zeros(16000, dtype=np.int16)) is None
