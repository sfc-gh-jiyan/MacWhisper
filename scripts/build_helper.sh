#!/bin/bash
# Build SystemAudioHelper (ScreenCaptureKit), which Meeting Mode uses to record
# system audio. audio_capture.find_helper() looks for the release binary here.
set -euo pipefail

HELPER_DIR="$(cd "$(dirname "$0")/.." && pwd)/swift/SystemAudioHelper"

if ! command -v swift &>/dev/null; then
    echo "  swift not found — install the Command Line Tools: xcode-select --install" >&2
    exit 1
fi

cd "$HELPER_DIR"
swift build -c release 2>&1 | tail -3
test -x .build/release/SystemAudioHelper
echo "  Helper ready: $HELPER_DIR/.build/release/SystemAudioHelper"
