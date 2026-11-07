"""Tests for meeting_labels.py (Task 1: Mic/Remote label and echo helpers)."""

from __future__ import annotations

from meeting import MeetingSegment
from meeting_labels import (
    ECHO_SIMILARITY,
    is_echo,
    ordered,
    source_label,
    text_similarity,
)


def test_labels():
    assert source_label("mic") == "Mic"
    assert source_label("remote") == "Remote"
    assert source_label("remote:2") == "Remote 2"
    assert source_label(None) is None


def test_echo_same_words_overlapping():
    m = MeetingSegment("OK, which warehouse is this?", 18.0, 21.0, source="mic")
    r = MeetingSegment("OK which warehouse is this", 17.6, 21.2, source="remote")
    assert is_echo(m, r)


def test_not_echo_when_no_time_overlap():
    m = MeetingSegment("which warehouse", 40.0, 42.0, source="mic")
    r = MeetingSegment("which warehouse", 10.0, 12.0, source="remote")
    assert not is_echo(m, r)


def test_short_real_reply_is_not_echo():
    m = MeetingSegment("yes", 5.0, 5.5, source="mic")
    r = MeetingSegment("yes, so the query ran for two hours", 4.0, 9.0, source="remote")
    assert not is_echo(m, r)


def test_ordered_by_start_time_stable():
    a = MeetingSegment("a", 5.0, 6.0, source="remote")
    b = MeetingSegment("b", 2.0, 3.0, source="mic")
    c = MeetingSegment("c", 5.0, 7.0, source="mic")
    assert [s.text for s in ordered([a, b, c])] == ["b", "a", "c"]
