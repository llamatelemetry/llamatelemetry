#!/bin/bash
set -e

BUILD_DIR="/media/waqasm86/External1/Project-Nvidia/llama.cpp/build_cuda12_940m"
TEMP_PKG_DIR="/tmp/test_pkg_debug"

echo "Cleaning up old test directory..."
rm -rf "$TEMP_PKG_DIR"
mkdir -p "$TEMP_PKG_DIR/bin"
mkdir -p "$TEMP_PKG_DIR/lib"

echo ""
echo "=== Copying binaries ==="
BINARY_COUNT=0
for binary in llama-server llama-cli llama-quantize llama-embedding llama-bench; do
    if [ -f "$BUILD_DIR/bin/${binary}" ]; then
        echo "  Copying: $binary"
        cp "$BUILD_DIR/bin/${binary}" "$TEMP_PKG_DIR/bin/"
        chmod +x "$TEMP_PKG_DIR/bin/${binary}"
        ((BINARY_COUNT++))
    else
        echo "  Skipping: $binary (not found)"
    fi
done
echo "Total binaries copied: $BINARY_COUNT"

echo ""
echo "=== Copying libraries ==="
LIB_COUNT=0

# Method 1: Try with if/ls check
if ls "$BUILD_DIR"/bin/*.so* 1> /dev/null 2>&1; then
    echo "  Libraries found, copying..."
    cp -a "$BUILD_DIR"/bin/*.so* "$TEMP_PKG_DIR/lib/" 2>&1
    LIB_COUNT=$(find "$TEMP_PKG_DIR/lib/" -type f -o -type l | wc -l)
    echo "  Total libraries copied: $LIB_COUNT"
else
    echo "  No .so files found!"
fi

echo ""
echo "=== Results ==="
echo "Binaries in temp_pkg/bin:"
ls -lh "$TEMP_PKG_DIR/bin/" | tail -5

echo ""
echo "Libraries in temp_pkg/lib:"
ls -lh "$TEMP_PKG_DIR/lib/" | head -10

echo ""
echo "Total sizes:"
du -sh "$TEMP_PKG_DIR/bin"
du -sh "$TEMP_PKG_DIR/lib"
du -sh "$TEMP_PKG_DIR"

echo ""
echo "✓ Test completed successfully!"
