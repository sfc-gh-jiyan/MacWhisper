.PHONY: smoke test test-full

# ── Level 1: Smoke test (~7s) ─────────────────────────────
# Pure logic, no ML model needed. Run after every code change.
smoke:
	python -m pytest tests/test_app.py tests/test_meeting.py \
	  tests/test_online_processor.py tests/test_meeting_dual.py \
	  -m "not slow and not hardware" -q

# ── Level 2: Standard test (~5 min) ──────────────────────
# Requires mlx_whisper model. Run before every commit.
test: smoke
	python -m pytest tests/test_meeting_dual.py -m "slow" -q
	python tests/test_replay.py --top 1

# ── Level 3: Full test (~15 min) ─────────────────────────
# Run before releases or after major changes.
test-full: smoke
	python -m pytest tests/test_meeting_dual.py -m "slow" -q
	python tests/test_replay.py --top 5 --diagnose
