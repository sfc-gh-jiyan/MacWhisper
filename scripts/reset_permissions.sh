#!/bin/bash
#
# reset_permissions.sh — clear MacWhisper.app's privacy permission records
#
# Rebuilding the app (or older versions with a different signature) can leave
# permission entries that no longer match the installed app: the switch looks
# on in System Settings but macOS does not grant it. This removes ONLY
# MacWhisper's entries so you can grant them once, cleanly.
#
#   ./scripts/reset_permissions.sh          # (make reset-permissions)
#
set -euo pipefail

BUNDLE_ID="com.macwhisper.app"

# Never reset without a bundle id: `tccutil reset SERVICE` alone resets the
# service for EVERY app.
[ -n "$BUNDLE_ID" ] || { echo "ERROR: empty bundle id" >&2; exit 1; }

if pgrep -f "(MacWhisper\.app/Contents/MacOS/MacWhisper|python.*app\.py)" >/dev/null; then
    echo "MacWhisper is running. Quit it first (menu bar → Quit), then run this again." >&2
    exit 1
fi

for service in Accessibility ListenEvent PostEvent Microphone; do
    if tccutil reset "$service" "$BUNDLE_ID" >/dev/null 2>&1; then
        echo "  reset $service for $BUNDLE_ID"
    else
        echo "  $service: nothing to reset"
    fi
done

cat <<EOF

Done. Now, in this order:
  1. System Settings → Privacy & Security → Accessibility:
       + → Applications → MacWhisper, switch it on
  2. Same for Input Monitoring
  3. Double-click MacWhisper in Applications (Microphone is asked on first use)

Check: ~/.macwhisper/logs/macwhisper.log should show
  "Permissions (...): accessibility=True input_monitoring=True"
EOF
