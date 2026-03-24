#!/bin/bash
#
# build.sh — Create a source tarball for MacWhisper release
#
# Produces: dist/MacWhisper-<version>.tar.gz + SHA256 hash
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERSION=$(cat "$SCRIPT_DIR/VERSION" 2>/dev/null || echo "0.0.0")
DIST_DIR="$SCRIPT_DIR/dist"
ARCHIVE_NAME="MacWhisper-${VERSION}.tar.gz"

echo "Building MacWhisper v${VERSION}..."

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# Create tarball from tracked files only (respects .gitignore)
cd "$SCRIPT_DIR"
git archive --format=tar.gz --prefix="MacWhisper-${VERSION}/" -o "$DIST_DIR/$ARCHIVE_NAME" HEAD

# Generate SHA256
cd "$DIST_DIR"
shasum -a 256 "$ARCHIVE_NAME" > "${ARCHIVE_NAME}.sha256"
SHA=$(awk '{print $1}' "${ARCHIVE_NAME}.sha256")

echo ""
echo "Build complete:"
echo "  Archive: $DIST_DIR/$ARCHIVE_NAME"
echo "  SHA256:  $SHA"
echo ""
echo "Next: ./release.sh to upload to GitHub"
