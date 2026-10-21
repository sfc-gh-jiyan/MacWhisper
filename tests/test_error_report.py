"""error_report: group log errors into a de-duplicated summary."""

from datetime import datetime

import error_report

LOG = """\
2026-10-20 09:02:04 crash_guard ERROR Error in thread 'Thread-29 (_start_recording)': PortAudioError: Error opening InputStream [PaErrorCode -9986]
Traceback (most recent call last):
  File "x.py", line 1
2026-10-20 09:02:10 crash_guard ERROR Error in thread 'Thread-33 (_start_recording)': PortAudioError: Error opening InputStream [PaErrorCode -9986]
2026-10-20 10:00:00 __main__ WARNING Resynced modifiers [] -> ['cmd'] before shortcut match
2026-10-20 10:05:00 qwen_backend ERROR Qwen sidecar failure: Qwen3 request timed out after 21s
2026-10-18 08:00:00 qwen_backend ERROR old error from long ago
2026-10-20 11:00:00 meeting WARNING Live meeting engine failed (Qwen3 request timed out after 21s) — continuing with Whisper
"""

NOW = datetime(2026, 10, 20, 12, 0, 0)


def _groups(tmp_path, days=1):
    p = tmp_path / "m.log"
    p.write_text(LOG)
    return error_report.summarize(str(p), days=days, now=NOW)


def test_groups_same_error_with_different_thread_ids(tmp_path):
    g = _groups(tmp_path)
    pa = [x for x in g if "PortAudioError" in x.sample]
    assert len(pa) == 1
    assert pa[0].count == 2
    assert pa[0].first == "2026-10-20 09:02:04"
    assert pa[0].last == "2026-10-20 09:02:10"


def test_ignores_routine_warnings_and_old_entries(tmp_path):
    g = _groups(tmp_path)
    text = " ".join(x.sample for x in g)
    assert "Resynced" not in text
    assert "long ago" not in text


def test_includes_notable_warnings(tmp_path):
    g = _groups(tmp_path)
    assert any("Live meeting engine failed" in x.sample for x in g)


def test_sorted_by_count_desc(tmp_path):
    g = _groups(tmp_path)
    assert g[0].count >= g[-1].count


def test_days_window_includes_older(tmp_path):
    g = _groups(tmp_path, days=3)
    assert any("long ago" in x.sample for x in g)


def test_missing_log_returns_empty(tmp_path):
    assert error_report.summarize(str(tmp_path / "nope.log"), now=NOW) == []


def test_render_markdown(tmp_path):
    md = error_report.render(_groups(tmp_path), days=1, now=NOW)
    assert md.startswith("# MacWhisper error report")
    assert "| 2 |" in md
    assert "PortAudioError" in md


def test_render_empty():
    md = error_report.render([], days=1, now=NOW)
    assert "No errors" in md


def test_write_report(tmp_path):
    p = tmp_path / "m.log"
    p.write_text(LOG)
    out = tmp_path / "out" / "r.md"
    path = error_report.write_report(str(p), str(out), days=1, now=NOW)
    assert path == str(out)
    assert "PortAudioError" in out.read_text()
