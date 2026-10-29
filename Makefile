.PHONY: llm-setup smoke qa test test-full eval-meeting qwen-setup app helper reset-permissions

# Use venv python if available, else system python
PYTHON := $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python)

# ── Level 1: Smoke test (~14s) ────────────────────────────
# Pure logic, no ML model needed. Run after every code change.
smoke:
	$(PYTHON) -m pytest tests/test_app.py tests/test_meeting.py \
	  tests/test_meeting_dual.py tests/test_overlay.py tests/test_crash_guard.py tests/test_error_alert.py tests/test_eval_meeting.py tests/test_compare_ptt.py tests/test_simulate_stream.py tests/test_qwen_sidecar.py tests/test_qwen_backend.py tests/test_qwen_text.py tests/test_qwen_engine.py tests/test_shortcuts.py tests/test_settings_window.py tests/test_single_instance.py tests/test_login_item.py tests/test_quit_guard.py tests/test_qwen_meeting.py tests/test_open_input_stream.py tests/test_error_report.py tests/test_vocabulary.py tests/test_corrector.py tests/test_learn.py tests/test_correction_flow.py tests/test_correction_report.py tests/test_llm_backend.py tests/test_vocab_miner.py tests/test_vad_speech_ms.py tests/test_beautifier.py tests/test_ptt_watchdog.py tests/test_eval_meeting_live.py \
	  -m "not slow and not hardware" -q

# ── Level 2: Meeting QA (loads the Qwen3 model) ──────────
# Scores the live transcript the app saved and a Qwen3 sidecar pass over the
# whole meeting against hand-checked references (~/.macwhisper/eval/).
qa:
	$(PYTHON) tests/eval_meeting.py --strategies realtime sidecar

# ── /Applications/MacWhisper.app (native launcher; double-click to start) ──
app:
	./scripts/build_app.sh

# Clear MacWhisper.app's privacy permission records (then grant them again)
reset-permissions:
	./scripts/reset_permissions.sh

# ── Qwen3-ASR sidecar env (isolated from the app venv) ──
# ── System-audio helper for Meeting Mode (ScreenCaptureKit) ──
helper:
	./scripts/build_helper.sh

qwen-setup:
	./scripts/setup_qwen.sh

# ── Local text LLM for vocabulary learning (isolated .venv-llm) ──
llm-setup:
	uv venv --python 3.12 .venv-llm
	uv pip install --python .venv-llm/bin/python -r requirements-llm.txt

# ── Meeting eval: score strategies vs references (local data only) ──
# References: ~/.macwhisper/eval/<id>.ref.txt; audio from ~/.macwhisper/meetings/
eval-meeting:
	$(PYTHON) tests/eval_meeting.py

# ── Level 3: Standard test (~5 min) ──────────────────────
# Requires the Qwen3 sidecar (make qwen-setup). Run before every commit.
test: smoke qa
	$(PYTHON) -m pytest tests/test_meeting_dual.py -m "slow" -q

# ── Level 4: Full test (~15 min) ─────────────────────────
# Run before releases or after major changes.
test-full: smoke qa
	$(PYTHON) -m pytest tests/test_meeting_dual.py -m "slow" -q
	$(PYTHON) tests/eval_meeting_live.py --engine qwen3
