#!/bin/bash
################################################################################
# Create GitHub Release Package for llamatelemetry
#
# Legacy packaging script. llamatelemetry v2.2.0 targets Kaggle dual Tesla T4 only.
# This script may include older multi-target packaging logic.
#
# The package will be uploaded to GitHub Releases page (NOT main repo)
# to keep the main repository small and avoid large binary commits.
#
# Package contents:
#   - llama-server and tools (compiled for CUDA 12)
#   - Shared libraries (.so files)
#   - For v2.2.0, binaries are optimized for Tesla T4 (SM 7.5)
#
# Usage:
#   1. Build llama.cpp with cmake (manually)
#   2. Run this script to package binaries
#   3. Upload generated .tar.gz to GitHub Releases
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="/media/waqasm86/External1/Project-Nvidia"
LLAMA_CPP_DIR="${PROJECT_ROOT}/llama.cpp"
OUTPUT_DIR="${PROJECT_ROOT}/release-packages"
VERSION="1.2.0"  # Update this for each release

################################################################################
# Functions
################################################################################

show_header() {
    echo -e "${BLUE}========================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================================================${NC}"
    echo ""
}

show_step() {
    echo -e "${YELLOW}▶ $1${NC}"
}

show_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

show_error() {
    echo -e "${RED}✗ $1${NC}"
}

show_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

################################################################################
# Main Script
################################################################################

show_header "Create llamatelemetry GitHub Release Package"

echo -e "${CYAN}Package Configuration:${NC}"
echo "  Version:     v${VERSION}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  CUDA:        12.x"
echo "  Targets:     Legacy 940M (CC 5.0) & Tesla T4 (CC 7.5)"
echo ""

# Ask which build to package
echo -e "${YELLOW}Which build do you want to package?${NC}"
echo "  1) GeForce 940M (CC 5.0) - legacy/local"
echo "  2) Tesla T4 (CC 7.5) - Kaggle T4 build"
echo "  3) Both (create separate packages)"
echo ""
read -p "Enter choice (1/2/3): " choice

case $choice in
    1)
        BUILDS=("940m")
        BUILD_NAMES=("GeForce 940M")
        BUILD_DIRS=("build_cuda12_940m")
        ;;
    2)
        BUILDS=("t4")
        BUILD_NAMES=("Tesla T4")
        BUILD_DIRS=("build_cuda12_t4")
        ;;
    3)
        BUILDS=("940m" "t4")
        BUILD_NAMES=("GeForce 940M" "Tesla T4")
        BUILD_DIRS=("build_cuda12_940m" "build_cuda12_t4")
        ;;
    *)
        show_error "Invalid choice"
        exit 1
        ;;
esac
echo ""

################################################################################
# Process each build
################################################################################

for i in "${!BUILDS[@]}"; do
    TARGET="${BUILDS[$i]}"
    TARGET_NAME="${BUILD_NAMES[$i]}"
    BUILD_DIR="${LLAMA_CPP_DIR}/${BUILD_DIRS[$i]}"

    show_header "Processing: ${TARGET_NAME} (${TARGET})"

    # Verify build exists
    show_step "Checking build directory..."
    if [ ! -d "$BUILD_DIR" ]; then
        show_error "Build directory not found: $BUILD_DIR"
        echo "  Please build llama.cpp first with CMake"
        exit 1
    fi
    show_success "Build directory found"
    echo ""

    # Verify llama-server exists
    show_step "Checking llama-server binary..."
    if [ ! -f "$BUILD_DIR/bin/llama-server" ]; then
        show_error "llama-server not found in build"
        echo "  Expected: $BUILD_DIR/bin/llama-server"
        exit 1
    fi

    FILE_SIZE=$(stat -c%s "$BUILD_DIR/bin/llama-server")
    SIZE_MB=$((FILE_SIZE / 1024 / 1024))
    show_success "Found llama-server (${SIZE_MB} MB)"
    echo ""

    # Create temporary package structure
    show_step "Creating package structure..."
    TEMP_PKG_DIR="${OUTPUT_DIR}/temp_${TARGET}"
    rm -rf "$TEMP_PKG_DIR"
    mkdir -p "$TEMP_PKG_DIR/bin"
    mkdir -p "$TEMP_PKG_DIR/lib"
    show_success "Package structure created"
    echo ""

    # Copy binaries
    show_step "Copying binaries..."
    BINARY_COUNT=0

    # Essential binaries
    for binary in llama-server llama-cli llama-quantize llama-embedding llama-bench; do
        if [ -f "$BUILD_DIR/bin/${binary}" ]; then
            cp "$BUILD_DIR/bin/${binary}" "$TEMP_PKG_DIR/bin/"
            chmod +x "$TEMP_PKG_DIR/bin/${binary}"
            BINARY_COUNT=$((BINARY_COUNT + 1))
            show_success "  ${binary}"
        fi
    done

    echo "  Total binaries: $BINARY_COUNT"
    echo ""

    # Copy shared libraries
    show_step "Copying shared libraries..."
    LIB_COUNT=0

    # Copy from bin/ directory (where CMake usually puts them)
    # Use cp -a to preserve symlinks
    if ls "$BUILD_DIR"/bin/*.so* 1> /dev/null 2>&1; then
        cp -a "$BUILD_DIR"/bin/*.so* "$TEMP_PKG_DIR/lib/" 2>/dev/null || true
        LIB_COUNT=$(find "$TEMP_PKG_DIR/lib/" -type f -o -type l | wc -l)
    fi

    # Also check lib/ directory (some builds put libraries here)
    if [ -d "$BUILD_DIR/lib" ]; then
        if ls "$BUILD_DIR"/lib/*.so* 1> /dev/null 2>&1; then
            for lib in "$BUILD_DIR"/lib/*.so*; do
                if [ -f "$lib" ]; then
                    LIBNAME=$(basename "$lib")
                    if [ ! -e "$TEMP_PKG_DIR/lib/$LIBNAME" ]; then
                        cp -a "$lib" "$TEMP_PKG_DIR/lib/"
                    fi
                fi
            done
            LIB_COUNT=$(find "$TEMP_PKG_DIR/lib/" -type f -o -type l | wc -l)
        fi
    fi

    # Also check ggml/src directory (if libraries exist there)
    if [ -d "$BUILD_DIR/ggml/src" ]; then
        if ls "$BUILD_DIR"/ggml/src/*.so* 1> /dev/null 2>&1; then
            for lib in "$BUILD_DIR"/ggml/src/*.so*; do
                if [ -f "$lib" ]; then
                    LIBNAME=$(basename "$lib")
                    if [ ! -e "$TEMP_PKG_DIR/lib/$LIBNAME" ]; then
                        cp -a "$lib" "$TEMP_PKG_DIR/lib/"
                    fi
                fi
            done
            LIB_COUNT=$(find "$TEMP_PKG_DIR/lib/" -type f -o -type l | wc -l)
        fi
    fi

    show_success "Copied $LIB_COUNT library files"
    echo ""

    # Show library list
    show_info "Libraries included:"
    ls -1 "$TEMP_PKG_DIR/lib/" | head -10 | while read lib; do
        echo "    - $lib"
    done
    if [ $LIB_COUNT -gt 10 ]; then
        echo "    ... and $((LIB_COUNT - 10)) more"
    fi
    echo ""

    # Create README
    show_step "Creating README..."
    cat > "$TEMP_PKG_DIR/README.md" << EOF
# llamatelemetry CUDA 12 Binaries - ${TARGET_NAME}

Version: v${VERSION}
Built: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Target: ${TARGET_NAME}

## Contents

- \`bin/\` - Executable binaries
  - llama-server (main inference server)
  - llama-cli (command-line interface)
  - llama-quantize (model quantization tool)
  - llama-embedding (embedding generation)
  - llama-bench (performance benchmarking)

- \`lib/\` - Shared libraries (.so files)
  - libllama.so
  - libggml-*.so (GGML backends)
  - CUDA libraries

## Installation

### For llamatelemetry Python package:

\`\`\`bash
# Extract to llamatelemetry package directory
tar -xzf llamatelemetry-binaries-cuda12-${TARGET}.tar.gz
cp -r bin/* /path/to/llamatelemetry/llamatelemetry/binaries/cuda12/
cp -r lib/* /path/to/llamatelemetry/llamatelemetry/lib/
chmod +x /path/to/llamatelemetry/llamatelemetry/binaries/cuda12/*
\`\`\`

### Standalone usage:

\`\`\`bash
# Extract and run
tar -xzf llamatelemetry-binaries-cuda12-${TARGET}.tar.gz
export LD_LIBRARY_PATH=\$(pwd)/lib:\$LD_LIBRARY_PATH
./bin/llama-server --help
\`\`\`

## Requirements

- CUDA 12.x runtime
- NVIDIA GPU with Compute Capability 5.0 or higher
- Linux x86_64

## Build Information

- CUDA Version: 12.x
- Compute Capability: Built for ${TARGET_NAME}
- Build Type: Release
- CUDA Graphs: Enabled
- Shared Libraries: Yes

## License

Same as llama.cpp project (MIT License)

## Support

- GitHub: https://github.com/waqasm86/llamatelemetry
- Issues: https://github.com/waqasm86/llamatelemetry/issues
EOF
    show_success "README created"
    echo ""

    # Calculate package size
    show_step "Calculating package size..."
    TOTAL_SIZE=$(du -sh "$TEMP_PKG_DIR" | cut -f1)
    show_info "Package size (uncompressed): $TOTAL_SIZE"
    echo ""

    # Create tar.gz archive
    ARCHIVE_NAME="llamatelemetry-binaries-cuda12-${TARGET}.tar.gz"
    ARCHIVE_PATH="${OUTPUT_DIR}/${ARCHIVE_NAME}"

    show_step "Creating archive: $ARCHIVE_NAME"

    # Create parent directory if needed
    mkdir -p "$OUTPUT_DIR"

    # Create archive (compress from temp directory)
    cd "$OUTPUT_DIR"
    tar -czf "$ARCHIVE_NAME" -C "temp_${TARGET}" .

    if [ -f "$ARCHIVE_PATH" ]; then
        ARCHIVE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)
        show_success "Archive created: $ARCHIVE_NAME ($ARCHIVE_SIZE)"
    else
        show_error "Failed to create archive"
        exit 1
    fi
    echo ""

    # Verify archive
    show_step "Verifying archive..."
    if tar -tzf "$ARCHIVE_PATH" > /dev/null 2>&1; then
        FILE_COUNT=$(tar -tzf "$ARCHIVE_PATH" | wc -l)
        show_success "Archive is valid ($FILE_COUNT files)"
    else
        show_error "Archive verification failed"
        exit 1
    fi
    echo ""

    # Cleanup temp directory
    show_step "Cleaning up temporary files..."
    rm -rf "$TEMP_PKG_DIR"
    show_success "Cleanup complete"
    echo ""

    # Show archive details
    show_header "Package Summary: ${TARGET_NAME}"
    echo -e "${CYAN}Archive:${NC}      $ARCHIVE_NAME"
    echo -e "${CYAN}Size:${NC}         $ARCHIVE_SIZE"
    echo -e "${CYAN}Location:${NC}     $ARCHIVE_PATH"
    echo -e "${CYAN}Binaries:${NC}     $BINARY_COUNT"
    echo -e "${CYAN}Libraries:${NC}    $LIB_COUNT"
    echo ""
done

################################################################################
# Final Summary
################################################################################

show_header "Release Package Creation Complete!"

echo -e "${GREEN}Created packages:${NC}"
for TARGET in "${BUILDS[@]}"; do
    ARCHIVE_NAME="llamatelemetry-binaries-cuda12-${TARGET}.tar.gz"
    if [ -f "${OUTPUT_DIR}/${ARCHIVE_NAME}" ]; then
        SIZE=$(du -h "${OUTPUT_DIR}/${ARCHIVE_NAME}" | cut -f1)
        echo "  ✓ $ARCHIVE_NAME ($SIZE)"
    fi
done
echo ""

echo -e "${CYAN}Location:${NC}"
echo "  ${OUTPUT_DIR}/"
ls -lh "$OUTPUT_DIR"/*.tar.gz
echo ""

echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Test the package locally:"
echo "   cd ${OUTPUT_DIR}"
echo "   tar -xzf llamatelemetry-binaries-cuda12-940m.tar.gz"
echo "   export LD_LIBRARY_PATH=\$(pwd)/lib:\$LD_LIBRARY_PATH"
echo "   ./bin/llama-server --help"
echo ""

echo "2. Upload to GitHub Releases:"
echo "   a. Go to: https://github.com/waqasm86/llamatelemetry/releases"
echo "   b. Click 'Draft a new release'"
echo "   c. Tag: v${VERSION}"
echo "   d. Title: llamatelemetry v${VERSION} - CUDA 12 Binaries"
echo "   e. Upload these files:"
for TARGET in "${BUILDS[@]}"; do
    echo "      - llamatelemetry-binaries-cuda12-${TARGET}.tar.gz"
done
echo ""

echo "3. Update bootstrap.py URL (if version changed):"
echo "   Edit: llamatelemetry/llamatelemetry/_internal/bootstrap.py"
echo "   Update GITHUB_RELEASE_URL to: v${VERSION}"
echo ""

echo "4. Update package version:"
echo "   Edit: llamatelemetry/llamatelemetry/__init__.py"
echo "   Update __version__ to: '${VERSION}'"
echo ""

echo -e "${RED}IMPORTANT:${NC}"
echo "  - These binaries go to GitHub RELEASES page (not main repo)"
echo "  - Main repo stays small; binaries live in GitHub Releases"
echo "  - NEVER upload .gguf model files anywhere"
echo ""

show_success "All done! Archives ready for upload to GitHub Releases."
