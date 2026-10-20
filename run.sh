#!/bin/bash
#
# MacWhisper — Quick launcher (no .app needed)
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Kill any running MacWhisper processes (both ./run.sh and /Applications/MacWhisper.app)
PIDS=$(pgrep -f "(python.*app\.py|python.*qwen_sidecar\.py|MacWhisper\.app/Contents/MacOS)" 2>/dev/null | grep -v $$)
if [ -n "$PIDS" ]; then
    echo "Stopping previous MacWhisper (PID: $PIDS)..."
    echo "$PIDS" | xargs kill 2>/dev/null
    # Wait for a normal quit (up to 10 s). Starting while the old one still
    # runs would be refused by the single-instance lock.
    DEADLINE=$((SECONDS + 10))
    while :; do
        ALIVE=""
        for p in $PIDS; do kill -0 "$p" 2>/dev/null && ALIVE="$ALIVE $p"; done
        [ -z "$ALIVE" ] || [ "$SECONDS" -ge "$DEADLINE" ] && break
        sleep 0.25
    done
    if [ -n "$ALIVE" ]; then
        echo "Previous MacWhisper did not quit in 10 s — forcing it (PID:$ALIVE)"
        kill -9 $ALIVE 2>/dev/null
        sleep 0.5
        rm -f "$HOME/.macwhisper/.running"   # our forced stop, not a crash
    fi
fi

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Virtual environment not found. Run ./install.sh first."
    exit 1
fi

cd "$SCRIPT_DIR"
export SSL_CERT_FILE="$(venv/bin/python3 -c 'import certifi; print(certifi.where())')"
# Workaround: macOS nano malloc zone can corrupt heap when mlx + AppKit coexist
export MallocNanoZone=0
exec venv/bin/python3 app.py
