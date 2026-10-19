.PHONY: smoke qa test test-full eval-meeting qwen-setup

# Use venv python if available, else system python
PYTHON := $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python)
REPORT_DIR := tests/TestReport

# ── Level 1: Smoke test (~14s) ────────────────────────────
# Pure logic, no ML model needed. Run after every code change.
smoke:
	$(PYTHON) -m pytest tests/test_app.py tests/test_meeting.py \
	  tests/test_online_processor.py tests/test_meeting_dual.py tests/test_overlay.py tests/test_crash_guard.py tests/test_error_alert.py tests/test_eval_meeting.py tests/test_compare_ptt.py tests/test_simulate_stream.py tests/test_qwen_sidecar.py tests/test_qwen_backend.py tests/test_qwen_text.py tests/test_qwen_engine.py \
	  -m "not slow and not hardware" -q

# ── Level 2: Subtitle QA (~1-2 min) ──────────────────────
# Replay 3 named sample WAVs of different lengths (10-70s) with diagnostic report.
# Checks: latency, freezes, inference timing, stability.
qa:
	$(PYTHON) tests/test_replay.py --named-only --min-duration 10 --max-duration 70 --spread 3 \
	  --diagnose --report-dir $(REPORT_DIR) --no-offline

# ── Qwen3-ASR sidecar env (isolated from the app venv) ──
qwen-setup:
	./scripts/setup_qwen.sh

# ── Meeting eval: score strategies vs references (local data only) ──
# References: ~/.macwhisper/eval/<id>.ref.txt; audio from ~/.macwhisper/meetings/
eval-meeting:
	$(PYTHON) tests/eval_meeting.py

# ── Level 3: Standard test (~5 min) ──────────────────────
# Requires mlx_whisper model. Run before every commit.
test: smoke qa
	$(PYTHON) -m pytest tests/test_meeting_dual.py -m "slow" -q

# ── Level 4: Full test (~15 min) ─────────────────────────
# Run before releases or after major changes.
test-full: smoke qa
	$(PYTHON) -m pytest tests/test_meeting_dual.py -m "slow" -q
	$(PYTHON) tests/test_replay.py --top 5 --diagnose --report-dir $(REPORT_DIR)
