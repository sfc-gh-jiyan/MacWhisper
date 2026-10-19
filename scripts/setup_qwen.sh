#!/bin/bash
#
# setup_qwen.sh — create the isolated Qwen3-ASR sidecar env (.venv-qwen)
#
# The app's own venv is NOT touched. Safe to re-run.
#
#   ./scripts/setup_qwen.sh            # create/refresh env + check model load
#   ./scripts/setup_qwen.sh --no-check # skip the model load check
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_DIR="$ROOT/.venv-qwen"
MODEL="${MACWHISPER_QWEN_MODEL:-mlx-community/Qwen3-ASR-1.7B-bf16}"

if [ "$(uname -m)" != "arm64" ]; then
    echo "ERROR: Qwen3-ASR (MLX) requires Apple Silicon." >&2
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    echo "==> Creating $ENV_DIR with uv (Python 3.12)"
    uv venv "$ENV_DIR" --python 3.12 -q --allow-existing
    VIRTUAL_ENV="$ENV_DIR" uv pip install -q -r "$ROOT/requirements-qwen.txt"
else
    PY="$(command -v python3.12 || true)"
    if [ -z "$PY" ]; then
        echo "ERROR: need 'uv' (brew install uv) or python3.12 on PATH." >&2
        exit 1
    fi
    echo "==> Creating $ENV_DIR with $PY"
    "$PY" -m venv "$ENV_DIR"
    "$ENV_DIR/bin/pip" install -q -r "$ROOT/requirements-qwen.txt"
fi

if [ "${1:-}" != "--no-check" ]; then
    echo "==> Checking model load: $MODEL (first run downloads it)"
    "$ENV_DIR/bin/python" "$ROOT/qwen_sidecar.py" --model "$MODEL" --self-test
fi
echo "==> Qwen3-ASR sidecar ready: $ENV_DIR"
