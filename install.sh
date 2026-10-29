#!/bin/bash
#
# MacWhisper Installer
# Creates a macOS .app bundle and sets up the Python environment
#

set -e

APP_NAME="MacWhisper"
APP_PATH="/Applications/${APP_NAME}.app"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERSION=$(cat "$SCRIPT_DIR/VERSION" 2>/dev/null || echo "0.0.0")

echo "=========================================="
echo "  MacWhisper Installer  v${VERSION}"
echo "=========================================="
echo ""
echo "Project directory: $SCRIPT_DIR"
echo "App will be installed to: $APP_PATH"
echo ""

# ── Pre-flight checks ────────────────────────────────────

ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "ERROR: MacWhisper requires Apple Silicon (M1/M2/M3/M4)."
    echo "       Detected architecture: $ARCH"
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ from python.org or via Homebrew."
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10+ required. Found: python3 $PY_VERSION"
    exit 1
fi
echo "Python: $PY_VERSION (OK)"

# ── Step 1: Python virtual environment ────────────────────

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "[1/6] Creating Python virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
else
    echo "[1/6] Virtual environment already exists, skipping."
fi

echo "[2/6] Installing dependencies (this may take a few minutes)..."
"$SCRIPT_DIR/venv/bin/pip" install --progress-bar on -r "$SCRIPT_DIR/requirements.txt"

# ── Step 2: Create .app bundle ────────────────────────────

echo "[3/6] Creating ${APP_NAME}.app bundle..."

# Native launcher (embeds CPython): permissions belong to MacWhisper.app,
# not to Python or Terminal. See scripts/build_app.sh.
"$SCRIPT_DIR/scripts/build_app.sh"

# ── Step 3: Create data directory ─────────────────────────

mkdir -p "$HOME/.macwhisper/logs"
mkdir -p "$HOME/.macwhisper/audio"

# ── Step 4: Qwen3-ASR engine (isolated .venv-qwen + model download) ──

echo "[4/6] Setting up the Qwen3-ASR engine (first time downloads the model)..."
"$SCRIPT_DIR/scripts/setup_qwen.sh" \
    || echo "  ⚠  Qwen3-ASR setup failed. Run 'make qwen-setup' later (needs uv or python3.12)."

# ── Step 5: System-audio helper (Meeting Mode hears the other people) ──

echo "[5/6] Building the system-audio helper for Meeting Mode..."
"$SCRIPT_DIR/scripts/build_helper.sh" \
    || echo "  ⚠  Helper build failed: meetings will record your microphone only. Run 'make helper' later (needs: xcode-select --install)."

# ── Step 6: Verify ────────────────────────────────────────

echo "[6/6] Verifying installation..."
"$SCRIPT_DIR/venv/bin/python3" -c "import rumps, pynput, sounddevice, pyperclip, certifi, opencc; print('All dependencies OK')"

echo ""
echo "=========================================="
echo "  Installation complete!  v${VERSION}"
echo "=========================================="
echo ""
echo "  To launch:"
echo "    Option A: Open ${APP_NAME} from /Applications"
echo "    Option B: Run './run.sh' from this directory"
echo ""
echo "  First launch: double-click MacWhisper in /Applications, then grant"
echo "  (System Settings → Privacy & Security) to MacWhisper:"
echo ""
echo "    1. Microphone           — for recording audio (asked on first use)"
echo "    2. Accessibility        — for auto-paste"
echo "    3. Input Monitoring     — for the global hotkeys"
echo ""
echo "    ./run.sh runs from Terminal instead; then Terminal needs these permissions."
echo ""
echo "  Usage: hold Right ⌥ to talk; ⌃⌥M or ⌃⇧F8 start/stop a meeting."
echo "  All options and shortcuts: menu bar → Settings… (⌘,)"
echo ""
