#!/bin/bash
#
# check_install.sh — verify a finished `bash install.sh` (used by CI "Install test")
#
#   bash tests/check_install.sh                 # required + optional checks
#   bash tests/check_install.sh --allow-launch  # also run the app launch smoke
#                                               # outside CI (restarts your app!)
#   bash tests/check_install.sh --required-only # file/import checks only (no model, no launch)
#
# REQUIRED checks print ✓/✗ and make the script exit 1 on failure.
# OPTIONAL checks print ✓/⚠ and never fail (MLX / GUI inside a CI VM).
#
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="/Applications/MacWhisper.app"
EXE="$APP/Contents/MacOS/MacWhisper"            # scripts/build_app.sh EXE
PLIST="$APP/Contents/Info.plist"
HELPER="$ROOT/swift/SystemAudioHelper/.build/release/SystemAudioHelper"  # scripts/build_helper.sh
CRASH_LOG="$HOME/.macwhisper/logs/crash.log"    # app.py CRASH_LOG_PATH
MODEL="${MACWHISPER_QWEN_MODEL:-mlx-community/Qwen3-ASR-0.6B-bf16}"  # scripts/setup_qwen.sh
SELFTEST_CAP_S=300
LAUNCH_WAIT_S=30

ALLOW_LAUNCH=0
REQUIRED_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --allow-launch) ALLOW_LAUNCH=1 ;;
        --required-only) REQUIRED_ONLY=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

FAILED=0
ok()   { echo "✓ $*"; }
bad()  { echo "✗ $*"; FAILED=1; }
warn() { echo "⚠ $*"; }

# run_capped SECONDS CMD... — macOS has no `timeout`; returns 124 on timeout
run_capped() {
    local cap="$1"; shift
    "$@" & local pid=$! waited=0
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$waited" -ge "$cap" ]; then
            kill "$pid" 2>/dev/null; sleep 2; kill -9 "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            return 124
        fi
        sleep 1; waited=$((waited + 1))
    done
    wait "$pid"
}

echo "== REQUIRED =="

if [ -d "$APP" ]; then ok "$APP exists"; else bad "$APP missing"; fi

if [ -x "$EXE" ]; then
    ok "launcher executable: $EXE"
    archs="$(lipo -archs "$EXE" 2>/dev/null || true)"
    case " $archs " in
        *" arm64 "*) ok "launcher arch: $archs" ;;
        *) bad "launcher not arm64 (lipo: '${archs}'; file: $(file -b "$EXE"))" ;;
    esac
else
    bad "launcher missing or not executable: $EXE"
fi

want_ver="$(cat "$ROOT/VERSION" 2>/dev/null || echo '?')"
got_ver="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$PLIST" 2>/dev/null || echo '?')"
if [ "$got_ver" = "$want_ver" ]; then
    ok "CFBundleShortVersionString $got_ver == VERSION"
else
    bad "CFBundleShortVersionString '$got_ver' != VERSION '$want_ver'"
fi

if "$ROOT/venv/bin/python" -c "import rumps, pynput, sounddevice, pyperclip, certifi, opencc, numpy" 2>/dev/null; then
    ok "venv imports rumps pynput sounddevice pyperclip certifi opencc numpy"
else
    bad "venv import failed: $("$ROOT/venv/bin/python" -c 'import rumps, pynput, sounddevice, pyperclip, certifi, opencc, numpy' 2>&1 | tail -1)"
fi

if [ -x "$ROOT/.venv-qwen/bin/python" ]; then
    ok ".venv-qwen exists"
    if "$ROOT/.venv-qwen/bin/python" -c "import mlx_audio" 2>/dev/null; then
        ok ".venv-qwen imports mlx_audio"
    else
        bad ".venv-qwen cannot import mlx_audio"
    fi
else
    bad ".venv-qwen missing ($ROOT/.venv-qwen/bin/python)"
fi

if [ -x "$HELPER" ]; then ok "SystemAudioHelper: $HELPER"; else bad "SystemAudioHelper missing: $HELPER"; fi

if [ "$REQUIRED_ONLY" -eq 1 ]; then
    echo "== OPTIONAL == skipped (--required-only)"
else
echo "== OPTIONAL =="

if [ -x "$ROOT/.venv-qwen/bin/python" ]; then
    run_capped "$SELFTEST_CAP_S" "$ROOT/.venv-qwen/bin/python" "$ROOT/qwen_sidecar.py" --model "$MODEL" --self-test
    rc=$?
    if [ "$rc" -eq 0 ]; then ok "qwen_sidecar --self-test ($MODEL)"
    elif [ "$rc" -eq 124 ]; then warn "qwen_sidecar --self-test timed out after ${SELFTEST_CAP_S}s"
    else warn "qwen_sidecar --self-test exited $rc"; fi
else
    warn "qwen_sidecar --self-test skipped (no .venv-qwen)"
fi

if [ -z "${CI:-}" ] && [ "$ALLOW_LAUNCH" -ne 1 ]; then
    warn "launch smoke skipped: not in CI (\$CI unset). It opens, then quits /Applications/MacWhisper.app,"
    warn "  which would restart your running app. Pass --allow-launch to run it anyway."
elif [ ! -d "$APP" ]; then
    warn "launch smoke skipped: $APP missing"
else
    before=0
    [ -f "$CRASH_LOG" ] && before="$(grep -c 'MacWhisper session started' "$CRASH_LOG")"
    open "$APP"
    pid=""
    for _ in $(seq 1 "$LAUNCH_WAIT_S"); do
        pid="$(pgrep -f 'MacWhisper.app/Contents/MacOS' | head -1)"
        [ -n "$pid" ] && break
        sleep 1
    done
    if [ -z "$pid" ]; then
        warn "launch smoke: no MacWhisper process within ${LAUNCH_WAIT_S}s"
    else
        sleep 5   # let it reach crash_guard.enable_faulthandler
        stat="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')"
        case "$stat" in
            "") warn "launch smoke: pid $pid exited shortly after start" ;;
            T*) warn "launch smoke: pid $pid is stopped (stat $stat)" ;;
            *)  ok "launch smoke: pid $pid alive (stat $stat)" ;;
        esac
        after=0
        [ -f "$CRASH_LOG" ] && after="$(grep -c 'MacWhisper session started' "$CRASH_LOG")"
        if [ "$after" -gt "$before" ]; then ok "crash.log has a new 'MacWhisper session started' line"
        else warn "crash.log: no new 'MacWhisper session started' line ($CRASH_LOG)"; fi
    fi
    osascript -e 'tell application "MacWhisper" to quit' >/dev/null 2>&1 || true
    sleep 3
    pkill -f 'MacWhisper.app/Contents/MacOS' 2>/dev/null || true
    pkill -f 'qwen_sidecar.py' 2>/dev/null || true
fi
fi  # REQUIRED_ONLY

echo
if [ "$FAILED" -ne 0 ]; then echo "✗ install check FAILED"; exit 1; fi
echo "✓ install check passed (required)"
