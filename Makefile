.PHONY: smoke qa test test-full

# Use venv python if available, else system python
PYTHON := $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python)
REPORT_DIR := tests/TestReport

# ── Level 1: Smoke test (~14s) ────────────────────────────
# Pure logic, no ML model needed. Run after every code change.
smoke:
	$(PYTHON) -m pytest tests/test_app.py tests/test_meeting.py \
	  tests/test_online_processor.py tests/test_meeting_dual.py \
	  -m "not slow and not hardware" -q

# ── Level 2: Subtitle QA (~1-2 min) ──────────────────────
# Replay Google_sample.wav with full diagnostic report.
# Checks: latency, freezes, inference timing, stability.
qa:
	$(PYTHON) tests/test_replay.py --wav Google_sample.wav --diagnose \
	  --report-dir $(REPORT_DIR) --no-offline

# ── Level 3: Standard test (~5 min) ──────────────────────
# Requires mlx_whisper model. Run before every commit.
test: smoke qa
	$(PYTHON) -m pytest tests/test_meeting_dual.py -m "slow" -q

# ── Level 4: Full test (~15 min) ─────────────────────────
# Run before releases or after major changes.
test-full: smoke qa
	$(PYTHON) -m pytest tests/test_meeting_dual.py -m "slow" -q
	$(PYTHON) tests/test_replay.py --top 5 --diagnose --report-dir $(REPORT_DIR)
