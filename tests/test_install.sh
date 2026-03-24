#!/bin/bash
#
# test_install.sh — Clean-install verification for MacWhisper
#
# Simulates a fresh user: wipes local venv/.app, clones fresh, runs install,
# verifies everything works.
#
# Usage:
#   ./test_install.sh          # Test from local source
#   ./test_install.sh --remote # Test from git clone (full end-to-end)
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

PASS=0
FAIL=0

check() {
    local desc="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} $desc"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗${NC} $desc"
        FAIL=$((FAIL + 1))
    fi
}

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

echo "=========================================="
echo "  MacWhisper Clean-Install Test"
echo "=========================================="
echo ""
echo "Work directory: $WORK_DIR"
echo ""

# ── Phase 1: Prepare clean source ────────────────────────

echo "Phase 1: Preparing clean source..."

if [ "$1" = "--remote" ]; then
    REMOTE_URL=$(cd "$SCRIPT_DIR" && git remote get-url origin 2>/dev/null)
    BRANCH=$(cd "$SCRIPT_DIR" && git branch --show-current)
    if [ -z "$REMOTE_URL" ]; then
        echo -e "${RED}ERROR: No git remote found${NC}"
        exit 1
    fi
    echo "  Cloning $REMOTE_URL ($BRANCH)..."
    git clone --branch "$BRANCH" --depth 1 "$REMOTE_URL" "$WORK_DIR/MacWhisper"
else
    echo "  Copying local source..."
    rsync -a --exclude=venv --exclude=dist --exclude=__pycache__ \
        --exclude=.git --exclude='*.pyc' --exclude='.DS_Store' \
        "$SCRIPT_DIR/" "$WORK_DIR/MacWhisper/"
fi

SRC="$WORK_DIR/MacWhisper"

# ── Phase 2: Pre-flight checks ───────────────────────────

echo ""
echo "Phase 2: Pre-flight checks..."

check "VERSION file exists" test -f "$SRC/VERSION"
check "app.py exists" test -f "$SRC/app.py"
check "install.sh exists" test -f "$SRC/install.sh"
check "requirements.txt exists" test -f "$SRC/requirements.txt"
check "AppIcon.icns exists" test -f "$SRC/AppIcon.icns"
check "run.sh exists" test -f "$SRC/run.sh"

VERSION=$(cat "$SRC/VERSION" 2>/dev/null || echo "MISSING")
APP_VERSION=$(grep -oE '__version__ = "[^"]+"' "$SRC/app.py" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
check "VERSION ($VERSION) matches app.py ($APP_VERSION)" test "$VERSION" = "$APP_VERSION"

# ── Phase 3: Dependency installation ─────────────────────

echo ""
echo "Phase 3: Installing dependencies (this takes a while)..."

python3 -m venv "$SRC/venv"
"$SRC/venv/bin/pip" install -q --upgrade pip
"$SRC/venv/bin/pip" install -q -r "$SRC/requirements.txt" 2>&1 | tail -1

echo ""
echo "Phase 4: Import verification..."

check "import rumps" "$SRC/venv/bin/python3" -c "import rumps"
check "import mlx_whisper" "$SRC/venv/bin/python3" -c "import mlx_whisper"
check "import sounddevice" "$SRC/venv/bin/python3" -c "import sounddevice"
check "import pynput" "$SRC/venv/bin/python3" -c "import pynput"
check "import pyperclip" "$SRC/venv/bin/python3" -c "import pyperclip"
check "import certifi" "$SRC/venv/bin/python3" -c "import certifi"
check "import opencc" "$SRC/venv/bin/python3" -c "import opencc"
check "import numpy" "$SRC/venv/bin/python3" -c "import numpy"

# ── Phase 5: Code quality ────────────────────────────────

echo ""
echo "Phase 5: Code quality checks..."

check "app.py compiles" "$SRC/venv/bin/python3" -m py_compile "$SRC/app.py"
check "no syntax errors" "$SRC/venv/bin/python3" -c "
import ast, sys
with open('$SRC/app.py') as f:
    ast.parse(f.read())
"
check "__version__ is set" grep -q '__version__' "$SRC/app.py"
check "crash handler exists" grep -q 'crash.log' "$SRC/app.py"
check "audio device check exists" grep -q '_check_audio_device' "$SRC/app.py"
check "data dir is ~/.macwhisper" grep -q '~/.macwhisper' "$SRC/app.py"

# ── Phase 6: Script checks ───────────────────────────────

echo ""
echo "Phase 6: Script syntax checks..."

check "install.sh syntax" bash -n "$SRC/install.sh"
check "bump_version.sh syntax" bash -n "$SRC/scripts/bump_version.sh"
check "build.sh syntax" bash -n "$SRC/scripts/build.sh"
check "release.sh syntax" bash -n "$SRC/scripts/release.sh"

# ── Summary ───────────────────────────────────────────────

echo ""
echo "=========================================="
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then
    echo -e "  ${GREEN}ALL $TOTAL CHECKS PASSED${NC}"
else
    echo -e "  ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC} (of $TOTAL)"
fi
echo "=========================================="

exit "$FAIL"
