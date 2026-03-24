#!/bin/bash
#
# bump_version.sh — Bump MacWhisper version across all files
#
# Usage:
#   ./bump_version.sh 0.4.0
#   ./bump_version.sh patch   # 0.3.1 -> 0.3.2
#   ./bump_version.sh minor   # 0.3.1 -> 0.4.0
#   ./bump_version.sh major   # 0.3.1 -> 1.0.0
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION_FILE="$SCRIPT_DIR/VERSION"
APP_FILE="$SCRIPT_DIR/app.py"

OLD_VERSION=$(cat "$VERSION_FILE" 2>/dev/null || echo "0.0.0")
IFS='.' read -r MAJOR MINOR PATCH <<< "$OLD_VERSION"

if [ -z "$1" ]; then
    echo "Usage: $0 <version|patch|minor|major>"
    echo "Current version: $OLD_VERSION"
    exit 1
fi

case "$1" in
    patch)  NEW_VERSION="$MAJOR.$MINOR.$((PATCH + 1))" ;;
    minor)  NEW_VERSION="$MAJOR.$((MINOR + 1)).0" ;;
    major)  NEW_VERSION="$((MAJOR + 1)).0.0" ;;
    *)      NEW_VERSION="$1" ;;
esac

# Validate version format
if ! echo "$NEW_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "ERROR: Invalid version format: $NEW_VERSION (expected X.Y.Z)"
    exit 1
fi

echo "Bumping version: $OLD_VERSION -> $NEW_VERSION"

# 1) VERSION file
echo "$NEW_VERSION" > "$VERSION_FILE"
echo "  Updated VERSION"

# 2) app.py __version__
sed -i '' "s/^__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" "$APP_FILE"
echo "  Updated app.py __version__"

# 3) app.py docstring
sed -i '' "s/MacWhisper - macOS Menu Bar App  v.*/MacWhisper - macOS Menu Bar App  v$NEW_VERSION/" "$APP_FILE"
echo "  Updated app.py docstring"

echo ""
echo "Done. Version is now $NEW_VERSION"
echo ""
echo "Next steps:"
echo "  git add VERSION app.py"
echo "  git commit -m \"chore: bump version to $NEW_VERSION\""
echo "  git tag v$NEW_VERSION"
