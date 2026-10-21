#!/usr/bin/env bash
# Build a self-contained MacWhisper.app that does not depend on the source
# checkout, the repo venvs or ~/.macwhisper.
#
#   scripts/build_standalone.sh [--dest DIR]      (default: ~/MacWhisperStandalone)
#
# Bundle: two relocatable CPythons (python-build-standalone), pinned wheels from
# packaging/*.lock, the app sources and the SystemAudioHelper. Models are not
# bundled — they download on first use into the Hugging Face cache.
# Ad-hoc signed only (phase 1). Nothing outside --dest and build/ is touched.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/MacWhisperStandalone"
while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PBS_TAG="20261013"
PY_APP="3.13.16"
PY_QWEN="3.12.15"
APP_NAME="MacWhisper"
BUNDLE_ID="com.macwhisper.standalone"
VERSION="$(cat "$REPO/VERSION")"

WORK="$REPO/build/standalone"
CACHE="$WORK/cache"
APP="$WORK/$APP_NAME.app"
C="$APP/Contents"
mkdir -p "$CACHE"

log() { printf '\n==> %s\n' "$*"; }

fetch_python() {  # version -> path to tarball (sha256-verified)
  local ver="$1"
  local name="cpython-${ver}+${PBS_TAG}-aarch64-apple-darwin-install_only.tar.gz"
  local url="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${name}"
  local tgz="$CACHE/$name"
  if [ ! -f "$tgz" ]; then
    curl -fL --retry 3 -o "$tgz.part" "$url"
    mv "$tgz.part" "$tgz"
  fi
  if [ ! -f "$CACHE/SHA256SUMS" ]; then
    curl -fsSL --retry 3 -o "$CACHE/SHA256SUMS" \
      "https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/SHA256SUMS"
  fi
  local want got
  want="$(awk -v n="$name" '$2==n {print $1}' "$CACHE/SHA256SUMS")"
  got="$(shasum -a 256 "$tgz" | awk '{print $1}')"
  if [ -z "$want" ] || [ "$want" != "$got" ]; then
    echo "checksum mismatch for $name (want '$want', got '$got')" >&2
    rm -f "$tgz"; exit 1
  fi
  echo "$tgz"
}

install_python() {  # version dest-dir lockfile
  local ver="$1" dir="$2" lock="$3"
  local tgz; tgz="$(fetch_python "$ver")"
  mkdir -p "$dir"
  tar -xzf "$tgz" -C "$dir" --strip-components 1
  rm -f "$dir"/lib/python3.*/EXTERNALLY-MANAGED
  uv pip install --quiet --python "$dir/bin/python3" --break-system-packages -r "$lock"
  uv pip check --python "$dir/bin/python3"
}

log "Clean $APP"
rm -rf "$APP"
mkdir -p "$C/MacOS" "$C/Resources/app"

log "App runtime: CPython $PY_APP"
install_python "$PY_APP" "$C/Resources/python" "$REPO/packaging/requirements-app.lock"

log "Qwen3 runtime: CPython $PY_QWEN"
install_python "$PY_QWEN" "$C/Resources/python-qwen" "$REPO/packaging/requirements-qwen.lock"

log "App sources"
cp "$REPO"/*.py "$C/Resources/app/"
rm -f "$C/Resources/app/setup.py"
cp "$REPO/VERSION" "$C/Resources/app/"
cp "$REPO/AppIcon.icns" "$C/Resources/AppIcon.icns"

log "SystemAudioHelper"
( cd "$REPO/swift/SystemAudioHelper" && swift build -c release --quiet )
cp "$REPO/swift/SystemAudioHelper/.build/release/SystemAudioHelper" "$C/Resources/app/"

log "Launcher"
PYH="$C/Resources/python"
clang -O2 -Werror -Wall -arch arm64 -mmacosx-version-min=13.0 \
  -I"$PYH/include/python3.13" \
  "$REPO/launcher/standalone_launcher.c" \
  -L"$PYH/lib" -lpython3.13 \
  -Wl,-rpath,@executable_path/../Resources/python/lib \
  -o "$C/MacOS/$APP_NAME"
if nm -u "$C/MacOS/$APP_NAME" | grep -qE '_exec[lv]'; then
  echo "launcher must not exec (pid version breaks menu bar / TCC)" >&2; exit 1
fi
if otool -L "$C/MacOS/$APP_NAME" | grep -E 'libpython' | grep -vq '@rpath'; then
  echo "launcher links libpython by absolute path:" >&2
  otool -L "$C/MacOS/$APP_NAME" >&2; exit 1
fi

log "Info.plist"
cat > "$C/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleExecutable</key><string>$APP_NAME</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSUIElement</key><true/>
  <key>LSEnvironment</key><dict><key>MallocNanoZone</key><string>0</string></dict>
  <key>NSMicrophoneUsageDescription</key><string>MacWhisper transcribes your speech.</string>
  <key>NSAudioCaptureUsageDescription</key><string>Meeting Mode transcribes system audio.</string>
</dict>
</plist>
PLIST

log "Strip caches"
find "$C/Resources" -name "__pycache__" -type d -prune -exec rm -rf {} +

log "Sign (ad-hoc, stable identifier requirement)"
codesign --force --sign - --timestamp=none \
  --requirements "=designated => identifier \"$BUNDLE_ID\"" "$C/Resources/app/SystemAudioHelper"
codesign --force --sign - --timestamp=none \
  --requirements "=designated => identifier \"$BUNDLE_ID\"" "$APP"
codesign --verify --verbose=1 "$APP"

log "Self-test (bundle runtime only)"
MACWHISPER_DATA_DIR="$WORK/selftest-data" MallocNanoZone=0 "$C/MacOS/$APP_NAME" --self-test | tee "$WORK/self_test.txt"
grep -q '^imports=ok$' "$WORK/self_test.txt"
grep -q '^qwen_imports=ok' "$WORK/self_test.txt"
grep -q "^sys_path_outside_bundle=\[\]$" "$WORK/self_test.txt" || {
  echo "sys.path reaches outside the bundle" >&2; exit 1; }

log "Install to $DEST"
mkdir -p "$DEST"
rm -rf "$DEST/$APP_NAME.app"
ditto "$APP" "$DEST/$APP_NAME.app"
du -sh "$DEST/$APP_NAME.app"
echo "Built $DEST/$APP_NAME.app ($BUNDLE_ID $VERSION)"
