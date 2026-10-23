"""Push to Talk must stop when the key is up, even if the release event is lost or raced."""
import threading
import app as app_mod


class _App:
    _request_stop = app_mod.TranscriberApp._request_stop
    _ptt_watchdog = app_mod.TranscriberApp._ptt_watchdog

    def __init__(self):
        self.recording = True
        self._rec_gen = 1
        self._stop_lock = threading.Lock()
        self.stops = 0

    def _ptt_spec(self):
        return "alt_r"

    def _stop_and_transcribe(self):
        self.stops += 1
        self.recording = False


def test_request_stop_runs_once(monkeypatch):
    monkeypatch.setattr(app_mod.threading, "Thread",
                        lambda target, daemon: type("T", (), {"start": lambda s: target()})())
    a = _App()
    a._request_stop(1, "release")
    a._request_stop(1, "watchdog")
    assert a.stops == 1


def test_request_stop_ignores_old_recording():
    a = _App()
    a._rec_gen = 2
    a._request_stop(1, "watchdog")
    assert a.stops == 0


def test_watchdog_stops_when_key_physically_up(monkeypatch):
    monkeypatch.setattr(app_mod.shortcuts, "key_is_down", lambda tok: False)
    monkeypatch.setattr(app_mod.threading, "Thread",
                        lambda target, daemon: type("T", (), {"start": lambda s: target()})())
    a = _App()
    a._ptt_watchdog(1, sleep=lambda s: None)
    assert a.stops == 1 and not a.recording


def test_watchdog_keeps_recording_while_held(monkeypatch):
    state = iter([True, True, True])
    a = _App()

    def down(tok):
        try:
            return next(state)
        except StopIteration:
            a.recording = False      # user lets go normally; release path stops it
            return True
    monkeypatch.setattr(app_mod.shortcuts, "key_is_down", down)
    a._ptt_watchdog(1, sleep=lambda s: None)
    assert a.stops == 0


def test_watchdog_noop_for_unknown_key(monkeypatch):
    monkeypatch.setattr(app_mod.shortcuts, "key_is_down", lambda tok: None)
    a = _App()
    a._ptt_watchdog(1, sleep=lambda s: None)
    assert a.stops == 0 and a.recording
