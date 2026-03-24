#!/bin/bash
#
# MacWhisper — Quick launcher (no .app needed)
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Kill any running MacWhisper processes (both ./run.sh and /Applications/MacWhisper.app)
PIDS=$(pgrep -f "(python.*app\.py|MacWhisper\.app/Contents/MacOS)" 2>/dev/null | grep -v $$)
if [ -n "$PIDS" ]; then
    echo "Stopping previous MacWhisper (PID: $PIDS)..."
    echo "$PIDS" | xargs kill 2>/dev/null
    sleep 0.5
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
