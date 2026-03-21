#!/bin/bash
#
# MacWhisper — Quick launcher (no .app needed)
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Virtual environment not found. Run ./install.sh first."
    exit 1
fi

cd "$SCRIPT_DIR"
exec venv/bin/python3 app.py
