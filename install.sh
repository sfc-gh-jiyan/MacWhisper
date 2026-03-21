#!/bin/bash
#
# MacWhisper Installer
# Creates a macOS .app bundle and sets up the Python environment
#

set -e

APP_NAME="MacWhisper"
APP_PATH="/Applications/${APP_NAME}.app"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  MacWhisper Installer"
echo "=========================================="
echo ""
echo "Project directory: $SCRIPT_DIR"
echo "App will be installed to: $APP_PATH"
echo ""

# ── Pre-flight: Apple Silicon check ───────────────────────

ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "ERROR: MacWhisper requires Apple Silicon (M1/M2/M3/M4)."
    echo "       Detected architecture: $ARCH"
    echo "       This Mac is not supported."
    exit 1
fi

# ── Step 1: Python virtual environment ────────────────────

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "[1/5] Creating Python virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
else
    echo "[1/5] Virtual environment already exists, skipping."
fi

echo "[2/5] Installing dependencies (this may take a few minutes)..."
"$SCRIPT_DIR/venv/bin/pip" install --progress-bar on -r "$SCRIPT_DIR/requirements.txt"

# ── Step 2: Create .app bundle ────────────────────────────

echo "[3/5] Creating ${APP_NAME}.app bundle..."

mkdir -p "${APP_PATH}/Contents/MacOS"
mkdir -p "${APP_PATH}/Contents/Resources"

# App icon
cp "${SCRIPT_DIR}/AppIcon.icns" "${APP_PATH}/Contents/Resources/AppIcon.icns"

# Launcher script — Python directly (no bash intermediary)
cat > "${APP_PATH}/Contents/MacOS/${APP_NAME}" << LAUNCHER
#!${SCRIPT_DIR}/venv/bin/python3
import os, sys, platform
if platform.machine() != "arm64":
    import subprocess
    r = subprocess.run(["/usr/bin/arch", "-arm64", sys.executable] + sys.argv)
    sys.exit(r.returncode)
_dir = "${SCRIPT_DIR}"
os.chdir(_dir)
sys.path.insert(0, _dir)
_log_dir = os.path.join(_dir, "logs")
os.makedirs(_log_dir, exist_ok=True)
_lf = open(os.path.join(_log_dir, "macwhisper.log"), "a")
sys.stdout = _lf
sys.stderr = _lf
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
import runpy
runpy.run_path(os.path.join(_dir, "app.py"), run_name="__main__")
LAUNCHER
chmod +x "${APP_PATH}/Contents/MacOS/${APP_NAME}"

# Info.plist
cat > "${APP_PATH}/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>MacWhisper</string>
  <key>CFBundleDisplayName</key>
  <string>MacWhisper</string>
  <key>CFBundleIdentifier</key>
  <string>com.macwhisper.app</string>
  <key>CFBundleVersion</key>
  <string>1.0</string>
  <key>CFBundleExecutable</key>
  <string>MacWhisper</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>NSMicrophoneUsageDescription</key>
  <string>MacWhisper needs microphone access for voice transcription.</string>
  <key>NSAccessibilityUsageDescription</key>
  <string>MacWhisper needs Accessibility access to paste transcribed text and detect the Right Option hotkey.</string>
  <key>LSArchitecturePriority</key>
  <array>
    <string>arm64</string>
  </array>
</dict>
</plist>
PLIST

# ── Step 3: Create logs directory ─────────────────────────

mkdir -p "$SCRIPT_DIR/logs"

# ── Step 4: Pre-download model ───────────────────────────

export SSL_CERT_FILE="$("$SCRIPT_DIR/venv/bin/python3" -c 'import certifi; print(certifi.where())')"
MODEL_REPO="$("$SCRIPT_DIR/venv/bin/python3" -c "
import json, os
cfg_path = os.path.expanduser('~/.macwhisper_config.json')
try:
    model = json.load(open(cfg_path))['current_model']
except Exception:
    model = 'mlx-community/whisper-small-mlx'
print(model)
")"
echo "[4/5] Pre-downloading model: ${MODEL_REPO} (first time only)..."
"$SCRIPT_DIR/venv/bin/python3" -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_REPO}')
print('Model downloaded OK')
" 2>&1 || echo "  ⚠  Model download failed (will retry on first launch). Check your internet connection."

# ── Step 5: Verify ────────────────────────────────────────

echo "[5/5] Verifying installation..."
"$SCRIPT_DIR/venv/bin/python3" -c "import rumps, mlx_whisper, pynput, sounddevice, pyperclip, certifi; print('All dependencies OK')"

echo ""
echo "=========================================="
echo "  Installation complete!"
echo "=========================================="
echo ""
echo "  To launch:"
echo "    Option A: Open ${APP_NAME} from /Applications"
echo "    Option B: Run './run.sh' from this directory"
echo ""
echo "  First time setup (System Settings → Privacy & Security):"
echo "    Grant these 3 permissions to MacWhisper (or your Terminal app if using ./run.sh):"
echo ""
echo "    1. Microphone           — for recording audio"
echo "    2. Accessibility        — for auto-paste (Cmd+V simulation)"
echo "    3. Input Monitoring     — for global hotkey (Right Option key)"
echo ""
echo "    ⚠  If you launch via Terminal/iTerm, add BOTH the terminal app AND MacWhisper."
echo ""
echo "  Usage:"
echo "    Hold Right Option key to record, release to transcribe"
echo "    Ctrl+Shift+M  — switch model (Small/Medium/Large)"
echo "    Ctrl+Shift+T  — toggle translate mode"
echo ""
