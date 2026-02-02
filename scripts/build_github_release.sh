#!/bin/bash
# Build script for llamatelemetry GitHub release packages
# Creates both source and wheel distributions for GitHub releases

set -e

echo "========================================"
echo "llamatelemetry GitHub Release Build Script"
echo "========================================"
echo ""

# Get version from pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
echo "Building llamatelemetry v$VERSION for GitHub releases"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info
echo "✓ Cleaned"
echo ""

# Build source distribution
echo "Building source distribution..."
python3 -m build --sdist
echo "✓ Source distribution created"
echo ""

# Build wheel distribution
echo "Building wheel distribution..."
python3 -m build --wheel
echo "✓ Wheel distribution created"
echo ""

# List created files
echo "========================================"
echo "Build complete! Created files:"
echo "========================================"
ls -lh dist/
echo ""

# Calculate checksums
echo "Calculating SHA256 checksums..."
cd dist/
for file in *; do
    sha256sum "$file" > "$file.sha256"
    echo "✓ $file.sha256"
done
cd ..
echo ""

echo "========================================"
echo "GitHub Release Packages Ready!"
echo "========================================"
echo ""
echo "Files ready for upload to GitHub releases:"
echo "  - dist/llamatelemetry-$VERSION.tar.gz (source)"
echo "  - dist/llamatelemetry-$VERSION-py3-none-any.whl (wheel)"
echo "  - dist/*.sha256 (checksums)"
echo ""
echo "Next steps:"
echo "  1. Create GitHub release: gh release create v$VERSION"
echo "  2. Upload files: gh release upload v$VERSION dist/*"
echo "  3. Also upload: llamatelemetry-binaries-cuda12-t4-v$VERSION.tar.gz"
echo ""
