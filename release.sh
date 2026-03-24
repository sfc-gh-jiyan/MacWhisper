#!/bin/bash
#
# release.sh — Create a GitHub Release with the source tarball
#
# Prerequisites:
#   - gh CLI installed and authenticated
#   - build.sh already run (dist/ directory exists)
#   - Git tag for the version exists
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERSION=$(cat "$SCRIPT_DIR/VERSION" 2>/dev/null || echo "0.0.0")
TAG="v${VERSION}"
DIST_DIR="$SCRIPT_DIR/dist"
ARCHIVE="$DIST_DIR/MacWhisper-${VERSION}.tar.gz"

# Verify prerequisites
if ! command -v gh &>/dev/null; then
    echo "ERROR: gh CLI not found. Install via: brew install gh"
    exit 1
fi

if [ ! -f "$ARCHIVE" ]; then
    echo "ERROR: $ARCHIVE not found. Run ./build.sh first."
    exit 1
fi

if ! git tag -l "$TAG" | grep -q "$TAG"; then
    echo "Tag $TAG does not exist. Creating..."
    git tag "$TAG"
    git push origin "$TAG"
fi

echo "Creating GitHub Release $TAG..."

gh release create "$TAG" \
    "$ARCHIVE" \
    --title "MacWhisper $TAG" \
    --notes "## MacWhisper $TAG

### Installation
\`\`\`bash
# Download and extract
curl -L https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/download/$TAG/MacWhisper-${VERSION}.tar.gz | tar xz
cd MacWhisper-${VERSION}
bash install.sh
\`\`\`

### SHA256
\`\`\`
$(cat "$DIST_DIR/MacWhisper-${VERSION}.tar.gz.sha256")
\`\`\`
"

echo ""
echo "Release created: $TAG"
echo "URL: $(gh release view "$TAG" --json url -q .url)"
