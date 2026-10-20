"""Tests for login_item (fake SMAppService; never registers for real)."""

from types import SimpleNamespace

import login_item as li


class FakeService:
    def __init__(self, status=0, fail=None):
        self._status = status
        self.fail = fail
        self.calls = []

    def status(self):
        return self._status

    def registerAndReturnError_(self, _):
        self.calls.append("register")
        if self.fail:
            return False, SimpleNamespace(localizedDescription=lambda: self.fail)
        self._status = 1
        return True, None

    def unregisterAndReturnError_(self, _):
        self.calls.append("unregister")
        if self.fail:
            return False, SimpleNamespace(localizedDescription=lambda: self.fail)
        self._status = 0
        return True, None


def test_states():
    assert li.LoginItem(FakeService(0)).state() == li.OFF
    assert li.LoginItem(FakeService(1)).state() == li.ON
    assert li.LoginItem(FakeService(2)).state() == li.NEEDS_APPROVAL
    assert li.LoginItem(FakeService(3)).state() == li.UNAVAILABLE


def test_unavailable_when_not_running_as_the_app():
    item = li.LoginItem(FakeService(3))
    assert not item.available()
    ok, msg = item.set_enabled(True)
    assert not ok and "/Applications" in msg


def test_enable_and_disable():
    svc = FakeService(0)
    item = li.LoginItem(svc)
    assert item.set_enabled(True) == (True, "")
    assert item.state() == li.ON
    assert item.set_enabled(False) == (True, "")
    assert svc.calls == ["register", "unregister"]


def test_enable_when_already_on_is_a_noop():
    svc = FakeService(1)
    assert li.LoginItem(svc).set_enabled(True) == (True, "")
    assert svc.calls == []


def test_errors_are_reported_not_swallowed():
    item = li.LoginItem(FakeService(0, fail="Operation not permitted"))
    ok, msg = item.set_enabled(True)
    assert not ok and "Operation not permitted" in msg


def test_needs_approval_message():
    assert "Login Items" in li.LoginItem(FakeService(2)).hint()


def test_missing_framework_is_unavailable():
    item = li.LoginItem(None)
    assert item.state() == li.UNAVAILABLE
    assert item.set_enabled(True)[0] is False


def test_real_service_loads_without_registering():
    """Loads the real SMAppService class; only reads status (no side effects)."""
    item = li.LoginItem.for_this_app()
    assert item.state() in (li.OFF, li.ON, li.NEEDS_APPROVAL, li.UNAVAILABLE)


def test_real_register_returns_ok_and_error():
    """Regression: without metadata, register/unregister returned a bare bool
    (NSError** not an out-param), so `ok, err = ...` crashed. Checks the
    signature only; never calls register."""
    svc = li._main_app_service()
    for name in ("registerAndReturnError_", "unregisterAndReturnError_"):
        md = getattr(svc, name).__metadata__()
        assert md["arguments"][2]["type"] == b"o^@"
        assert md["retval"]["type"] == b"Z"
