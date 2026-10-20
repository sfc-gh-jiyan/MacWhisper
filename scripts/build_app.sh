#!/bin/bash
#
# build_app.sh — build /Applications/MacWhisper.app with a native launcher
#
# The launcher embeds CPython (see launcher/macwhisper_launcher.c) and runs
# this repo's app.py with the repo venv, so macOS grants privacy permissions
# to MacWhisper.app itself, not to Python or Terminal. Code changes in the
# repo need no rebuild; rebuild only after moving the repo or recreating venv.
#
#   ./scripts/build_app.sh                 # build + install to /Applications
#   ./scripts/build_app.sh --dest DIR      # build into DIR (tests; no install)
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="MacWhisper"
DEST="/Applications"
if [ "${1:-}" = "--dest" ]; then
    DEST="${2:?--dest needs a directory}"
fi
APP="$DEST/${APP_NAME}.app"
VERSION="$(cat "$ROOT/VERSION")"
VENV_PY="$ROOT/venv/bin/python3"

if [ "$(uname -m)" != "arm64" ]; then
    echo "ERROR: MacWhisper requires Apple Silicon." >&2
    exit 1
fi
if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: $VENV_PY not found. Run ./install.sh first." >&2
    exit 1
fi
if ! command -v clang >/dev/null 2>&1; then
    echo "ERROR: clang not found. Install the Command Line Tools: xcode-select --install" >&2
    exit 1
fi

# Build flags from the venv's base interpreter (whatever Python created the venv)
read -r PY_INCLUDE PY_LIBDIR PY_VER CACERT < <("$VENV_PY" - <<'EOF'
import sysconfig, certifi
print(sysconfig.get_paths()["include"],
      sysconfig.get_config_var("LIBDIR"),
      sysconfig.get_config_var("VERSION"),
      certifi.where())
EOF
)
PY_LIBNAME="python${PY_VER}"
PY_LDLIB="lib${PY_LIBNAME}.dylib"   # embedding needs the shared lib (LDLIBRARY may name a .a)
if [ ! -f "$PY_LIBDIR/$PY_LDLIB" ]; then
    echo "ERROR: $PY_LIBDIR/$PY_LDLIB not found — this Python cannot be embedded." >&2
    exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
BUNDLE="$STAGE/${APP_NAME}.app"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"

echo "==> Compiling native launcher (Python: $PY_LIBDIR/$PY_LDLIB)"
clang -O2 -arch arm64 -Wall -Wextra -Werror \
    -I"$PY_INCLUDE" \
    -DMW_REPO="\"$ROOT\"" \
    -DMW_VENV_PY="\"$VENV_PY\"" \
    -DMW_CACERT="\"$CACERT\"" \
    "$ROOT/launcher/macwhisper_launcher.c" \
    -o "$BUNDLE/Contents/MacOS/$APP_NAME" \
    -L"$PY_LIBDIR" -l"$PY_LIBNAME" -Wl,-rpath,"$PY_LIBDIR"

# The launcher must never exec: a re-exec changes the process's pid version
# and macOS can no longer resolve the app (no menu bar icon, permission
# mismatch). Fail the build if any exec* symbol is linked in.
if nm -u "$BUNDLE/Contents/MacOS/$APP_NAME" | grep -qE '^_exec[lv]'; then
    echo "ERROR: launcher links an exec* call — it must not re-exec itself." >&2
    exit 1
fi

cp "$ROOT/AppIcon.icns" "$BUNDLE/Contents/Resources/AppIcon.icns"

cat > "$BUNDLE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>${APP_NAME}</string>
  <key>CFBundleDisplayName</key><string>${APP_NAME}</string>
  <key>CFBundleIdentifier</key><string>com.macwhisper.app</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleExecutable</key><string>${APP_NAME}</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>LSUIElement</key><true/>
  <key>LSEnvironment</key>
  <dict>
    <key>MallocNanoZone</key><string>0</string>
  </dict>
  <key>NSMicrophoneUsageDescription</key>
  <string>MacWhisper needs microphone access to transcribe your speech.</string>
  <key>NSAppleEventsUsageDescription</key>
  <string>MacWhisper pastes the transcribed text into the app you are using.</string>
</dict>
</plist>
PLIST

echo "==> Signing (ad-hoc)"
codesign --force --sign - --identifier com.macwhisper.app "$BUNDLE"
codesign --verify --strict "$BUNDLE"

echo "==> Self-test"
# Run directly (not via LaunchServices), so pass the LSEnvironment value
if ! MallocNanoZone=0 "$BUNDLE/Contents/MacOS/$APP_NAME" --self-test; then
    echo "ERROR: the built app failed its self-test; nothing was installed." >&2
    exit 1
fi

mkdir -p "$DEST"
rm -rf "$APP"
cp -R "$BUNDLE" "$APP"
echo "==> Installed $APP (v$VERSION)"
