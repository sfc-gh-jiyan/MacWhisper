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

# ── Step 1: Python virtual environment ────────────────────

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "[1/4] Creating Python virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
else
    echo "[1/4] Virtual environment already exists, skipping."
fi

echo "[2/4] Installing dependencies..."
"$SCRIPT_DIR/venv/bin/pip" install -q -r "$SCRIPT_DIR/requirements.txt"

# ── Step 2: Create .app bundle ────────────────────────────

echo "[3/4] Creating ${APP_NAME}.app bundle..."

mkdir -p "${APP_PATH}/Contents/MacOS"
mkdir -p "${APP_PATH}/Contents/Resources"

# Launcher script
cat > "${APP_PATH}/Contents/MacOS/${APP_NAME}" << LAUNCHER
#!/bin/bash
cd "${SCRIPT_DIR}"
exec "${SCRIPT_DIR}/venv/bin/python3" app.py
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
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>NSMicrophoneUsageDescription</key>
  <string>MacWhisper needs microphone access for voice transcription.</string>
</dict>
</plist>
PLIST

# ── Step 3: Create logs directory ─────────────────────────

mkdir -p "$SCRIPT_DIR/logs"

# ── Step 4: Verify ────────────────────────────────────────

echo "[4/4] Verifying installation..."
"$SCRIPT_DIR/venv/bin/python3" -c "import rumps, mlx_whisper, pynput, sounddevice, pyperclip; print('All dependencies OK')"

echo ""
echo "=========================================="
echo "  Installation complete!"
echo "=========================================="
echo ""
echo "  To launch:"
echo "    Option A: Open ${APP_NAME} from /Applications"
echo "    Option B: Run './run.sh' from this directory"
echo ""
echo "  First time setup:"
echo "    1. macOS will ask for Microphone permission — allow it"
echo "    2. macOS will ask for Accessibility permission — allow it"
echo "       (System Settings → Privacy & Security → Accessibility)"
echo ""
echo "  Usage:"
echo "    Hold Right Option key to record, release to transcribe"
echo "    Ctrl+Shift+M  — switch model (Small/Medium/Large)"
echo "    Ctrl+Shift+T  — toggle translate mode"
echo ""
