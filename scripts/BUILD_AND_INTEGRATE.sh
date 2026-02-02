#!/bin/bash
################################################################################
# Complete Build and Integration Script for llamatelemetry CUDA 12 Binaries
#
# This script:
# 1. Builds llama.cpp with CUDA 12 for your target GPU
# 2. Integrates binaries into llamatelemetry package structure
# 3. Verifies all paths and detection logic
# 4. Tests the complete workflow
#
# Usage:
#   ./BUILD_AND_INTEGRATE.sh 940m    # For GeForce 940M
#   ./BUILD_AND_INTEGRATE.sh t4      # For Tesla T4
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
LLCUDA_DIR="${PROJECT_ROOT}/llamatelemetry"
LLCUDA_PKG_DIR="${LLCUDA_DIR}/llamatelemetry"

# CUDA Configuration
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"

# Target GPU selection
TARGET="${1:-auto}"

################################################################################
# GPU Profiles
################################################################################
declare -A GPU_940M
GPU_940M[name]="GeForce 940M"
GPU_940M[cc]="50"
GPU_940M[arch]="Maxwell"
GPU_940M[fa]="OFF"
GPU_940M[cublas]="ON"

declare -A GPU_T4
GPU_T4[name]="Tesla T4"
GPU_T4[cc]="75"
GPU_T4[arch]="Turing"
GPU_T4[fa]="ON"
GPU_T4[cublas]="OFF"

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

detect_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo ""
        return 1
    fi

    local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

    if [[ "$gpu_name" == *"940M"* ]]; then
        echo "940m"
    elif [[ "$gpu_name" == *"T4"* ]]; then
        echo "t4"
    else
        echo "940m"  # Default to 940M for local system
    fi
}

################################################################################
# Main Script
################################################################################

show_header "llamatelemetry CUDA 12 Build and Integration"

# Auto-detect if needed
if [ "$TARGET" == "auto" ]; then
    show_step "Auto-detecting GPU..."
    TARGET=$(detect_gpu)
    if [ -z "$TARGET" ]; then
        show_error "Could not detect GPU, defaulting to 940m"
        TARGET="940m"
    fi
    show_success "Detected target: $TARGET"
    echo ""
fi

# Set GPU profile
if [ "$TARGET" == "940m" ]; then
    GPU_NAME="${GPU_940M[name]}"
    GPU_CC="${GPU_940M[cc]}"
    GPU_ARCH="${GPU_940M[arch]}"
    GPU_FA="${GPU_940M[fa]}"
    GPU_CUBLAS="${GPU_940M[cublas]}"
elif [ "$TARGET" == "t4" ]; then
    GPU_NAME="${GPU_T4[name]}"
    GPU_CC="${GPU_T4[cc]}"
    GPU_ARCH="${GPU_T4[arch]}"
    GPU_FA="${GPU_T4[fa]}"
    GPU_CUBLAS="${GPU_T4[cublas]}"
else
    show_error "Unknown target: $TARGET"
    echo "Usage: $0 [940m|t4|auto]"
    exit 1
fi

echo -e "${CYAN}Target Configuration:${NC}"
echo "  GPU:          $GPU_NAME"
echo "  Architecture: $GPU_ARCH"
echo "  Compute Cap:  $GPU_CC"
echo "  FlashAttention: $GPU_FA"
echo "  Force cuBLAS:   $GPU_CUBLAS"
echo ""

# Directory Configuration
BUILD_DIR="${LLAMA_CPP_DIR}/build_cuda12_${TARGET}"
INSTALL_BIN_DIR="${LLCUDA_PKG_DIR}/binaries/cuda12"
INSTALL_LIB_DIR="${LLCUDA_PKG_DIR}/lib"

echo -e "${CYAN}Directory Configuration:${NC}"
echo "  llama.cpp:   $LLAMA_CPP_DIR"
echo "  Build dir:   $BUILD_DIR"
echo "  llamatelemetry pkg:  $LLCUDA_PKG_DIR"
echo "  Binaries ->: $INSTALL_BIN_DIR"
echo "  Libraries->: $INSTALL_LIB_DIR"
echo ""

################################################################################
# STEP 1: Verify Prerequisites
################################################################################

show_header "Step 1: Verify Prerequisites"

show_step "Checking CUDA installation..."
if [ ! -f "$CUDA_COMPILER" ]; then
    show_error "CUDA compiler not found at $CUDA_COMPILER"
    exit 1
fi
$CUDA_COMPILER --version | head -1
show_success "CUDA found"
echo ""

show_step "Checking llama.cpp source..."
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    show_error "llama.cpp not found at $LLAMA_CPP_DIR"
    exit 1
fi
show_success "llama.cpp source found"
echo ""

show_step "Checking llamatelemetry package..."
if [ ! -d "$LLCUDA_PKG_DIR" ]; then
    show_error "llamatelemetry package not found at $LLCUDA_PKG_DIR"
    exit 1
fi
show_success "llamatelemetry package found"
echo ""

################################################################################
# STEP 2: Create Directory Structure
################################################################################

show_header "Step 2: Create llamatelemetry Directory Structure"

show_step "Creating binaries directory..."
mkdir -p "$INSTALL_BIN_DIR"
show_success "Created: $INSTALL_BIN_DIR"

show_step "Creating lib directory..."
mkdir -p "$INSTALL_LIB_DIR"
show_success "Created: $INSTALL_LIB_DIR"

show_step "Creating models cache..."
mkdir -p "${LLCUDA_PKG_DIR}/models"
show_success "Created: ${LLCUDA_PKG_DIR}/models"

echo ""

################################################################################
# STEP 3: CMake Configuration
################################################################################

show_header "Step 3: CMake Configuration"

echo -e "${CYAN}You need to run the following CMake command:${NC}"
echo ""
echo "cd $LLAMA_CPP_DIR"
echo ""
cat << EOF
cmake -B build_cuda12_${TARGET} \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DGGML_CUDA=ON \\
    -DCMAKE_CUDA_ARCHITECTURES="${GPU_CC}" \\
    -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}" \\
    -DGGML_NATIVE=OFF \\
    -DGGML_CUDA_FORCE_CUBLAS=${GPU_CUBLAS} \\
    -DGGML_CUDA_FA=${GPU_FA} \\
    -DGGML_CUDA_GRAPHS=ON \\
    -DLLAMA_BUILD_SERVER=ON \\
    -DLLAMA_BUILD_TOOLS=ON \\
    -DBUILD_SHARED_LIBS=ON \\
    -DCMAKE_INSTALL_RPATH='\$ORIGIN/../lib' \\
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
EOF
echo ""

read -p "Press Enter when you have run the CMake configuration command..."

# Verify build directory was created
if [ ! -d "$BUILD_DIR" ]; then
    show_error "Build directory not found. Did you run the CMake command?"
    exit 1
fi
show_success "CMake configuration verified"
echo ""

################################################################################
# STEP 4: Build
################################################################################

show_header "Step 4: Build llama.cpp"

echo -e "${CYAN}You need to run the following build command:${NC}"
echo ""
echo "cd $LLAMA_CPP_DIR"
echo "cmake --build build_cuda12_${TARGET} --config Release -j\$(nproc)"
echo ""

read -p "Press Enter when the build has completed..."

# Verify build artifacts exist
if [ ! -f "${BUILD_DIR}/bin/llama-server" ]; then
    show_error "llama-server binary not found. Did the build succeed?"
    exit 1
fi
show_success "Build artifacts verified"
echo ""

################################################################################
# STEP 5: Integration
################################################################################

show_header "Step 5: Integrate Binaries into llamatelemetry"

show_step "Copying binaries..."
cp "${BUILD_DIR}/bin/llama-server" "$INSTALL_BIN_DIR/" && \
chmod +x "${INSTALL_BIN_DIR}/llama-server"
show_success "llama-server copied"

for binary in llama-cli llama-quantize llama-embedding llama-bench; do
    if [ -f "${BUILD_DIR}/bin/${binary}" ]; then
        cp "${BUILD_DIR}/bin/${binary}" "$INSTALL_BIN_DIR/" && \
        chmod +x "${INSTALL_BIN_DIR}/${binary}"
        show_success "${binary} copied"
    fi
done
echo ""

show_step "Copying shared libraries..."
# Copy all .so files from build/bin
for lib in "${BUILD_DIR}"/bin/*.so*; do
    if [ -f "$lib" ]; then
        cp "$lib" "$INSTALL_LIB_DIR/"
        show_success "Copied $(basename $lib)"
    fi
done

# Also check ggml/src directory
if [ -d "${BUILD_DIR}/ggml/src" ]; then
    for lib in "${BUILD_DIR}"/ggml/src/*.so*; do
        if [ -f "$lib" ]; then
            cp "$lib" "$INSTALL_LIB_DIR/"
            show_success "Copied $(basename $lib)"
        fi
    done
fi
echo ""

################################################################################
# STEP 6: Verification
################################################################################

show_header "Step 6: Verification"

show_step "Checking installed binaries..."
ls -lh "$INSTALL_BIN_DIR"/
echo ""

show_step "Checking installed libraries..."
ls -lh "$INSTALL_LIB_DIR"/ | head -10
echo ""

show_step "Verifying llama-server CUDA linking..."
ldd "${INSTALL_BIN_DIR}/llama-server" | grep -E "(cuda|cublas)" || echo "  Note: Might be statically linked"
echo ""

show_step "Testing llama-server execution..."
export LD_LIBRARY_PATH="$INSTALL_LIB_DIR:$LD_LIBRARY_PATH"
if "${INSTALL_BIN_DIR}/llama-server" --help | head -5; then
    show_success "llama-server executes correctly"
else
    show_error "llama-server failed to execute"
    exit 1
fi
echo ""

################################################################################
# STEP 7: Update llamatelemetry Package
################################################################################

show_header "Step 7: Update Environment Variables"

echo -e "${CYAN}Add to your ~/.bashrc or ~/.zshrc:${NC}"
echo ""
cat << EOF
# llamatelemetry CUDA 12 Configuration for ${GPU_NAME}
export LLAMA_SERVER_PATH="${INSTALL_BIN_DIR}/llama-server"
export LD_LIBRARY_PATH="${INSTALL_LIB_DIR}:\$LD_LIBRARY_PATH"
EOF
echo ""

################################################################################
# STEP 8: Create Verification Script
################################################################################

show_header "Step 8: Creating Test Scripts"

# Create Python test script
TEST_SCRIPT="${PROJECT_ROOT}/test_llamatelemetry_integration.py"
cat > "$TEST_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""
llamatelemetry Integration Test Script
Tests the complete workflow from import to inference
"""

import os
import sys
from pathlib import Path

# Set up environment (adjust paths as needed)
PROJECT_ROOT = Path(__file__).parent
LLCUDA_DIR = PROJECT_ROOT / "llamatelemetry" / "llamatelemetry"
BIN_DIR = LLCUDA_DIR / "binaries" / "cuda12"
LIB_DIR = LLCUDA_DIR / "lib"

# Configure environment
os.environ['LLAMA_SERVER_PATH'] = str(BIN_DIR / "llama-server")
current_ld = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = f"{LIB_DIR}:{current_ld}"

print("=" * 70)
print("llamatelemetry Integration Test")
print("=" * 70)
print()

print("Environment:")
print(f"  LLAMA_SERVER_PATH: {os.environ.get('LLAMA_SERVER_PATH')}")
print(f"  LD_LIBRARY_PATH:   {LIB_DIR}")
print()

# Test 1: Import llamatelemetry
print("[1/5] Testing import...")
try:
    import llamatelemetry
    print(f"  ✓ llamatelemetry v{llamatelemetry.__version__} imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import llamatelemetry: {e}")
    sys.exit(1)
print()

# Test 2: Check GPU compatibility
print("[2/5] Testing GPU detection...")
try:
    compat = llamatelemetry.check_gpu_compatibility()
    print(f"  Platform:     {compat.get('platform', 'unknown')}")
    print(f"  GPU:          {compat.get('gpu_name', 'unknown')}")
    print(f"  Compute Cap:  {compat.get('compute_capability', 'unknown')}")
    print(f"  Compatible:   {compat.get('compatible', False)}")

    if not compat.get('compatible'):
        print(f"  Warning:      {compat.get('reason', 'Unknown reason')}")
except Exception as e:
    print(f"  ✗ GPU check failed: {e}")
print()

# Test 3: Find llama-server
print("[3/5] Testing llama-server detection...")
try:
    from llamatelemetry.server import ServerManager
    manager = ServerManager()
    server_path = manager.find_llama_server()

    if server_path:
        print(f"  ✓ Found: {server_path}")
        print(f"    Size: {server_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"    Executable: {os.access(server_path, os.X_OK)}")
    else:
        print(f"  ✗ llama-server not found")
        print(f"  Expected at: {BIN_DIR / 'llama-server'}")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Server detection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 4: Check libraries
print("[4/5] Testing library detection...")
try:
    if LIB_DIR.exists():
        libs = list(LIB_DIR.glob("*.so*"))
        print(f"  ✓ Found {len(libs)} shared libraries")
        for lib in sorted(libs)[:5]:
            print(f"    - {lib.name}")
        if len(libs) > 5:
            print(f"    ... and {len(libs) - 5} more")
    else:
        print(f"  ✗ Library directory not found: {LIB_DIR}")
except Exception as e:
    print(f"  ✗ Library check failed: {e}")
print()

# Test 5: Server execution test
print("[5/5] Testing llama-server execution...")
try:
    import subprocess
    result = subprocess.run(
        [str(server_path), "--help"],
        capture_output=True,
        text=True,
        timeout=5
    )

    if result.returncode == 0:
        print(f"  ✓ llama-server executes successfully")
        # Show first few lines of help
        help_lines = result.stdout.split('\n')[:3]
        for line in help_lines:
            if line.strip():
                print(f"    {line}")
    else:
        print(f"  ✗ llama-server failed with code {result.returncode}")
        print(f"    stderr: {result.stderr[:200]}")
except Exception as e:
    print(f"  ✗ Execution test failed: {e}")
print()

print("=" * 70)
print("Integration Test Complete!")
print("=" * 70)
print()
print("Next steps:")
print("  1. Install llamatelemetry in development mode:")
print(f"     cd {PROJECT_ROOT / 'llamatelemetry'}")
print("     pip install -e .")
print()
print("  2. Test with a model (requires model download):")
print("     python3 -c \"import llamatelemetry; engine = llamatelemetry.InferenceEngine(); print(engine)\"")
print()
PYEOF

chmod +x "$TEST_SCRIPT"
show_success "Created test script: $TEST_SCRIPT"
echo ""

################################################################################
# SUMMARY
################################################################################

show_header "Integration Complete!"

echo -e "${GREEN}✓ Binaries integrated into llamatelemetry package${NC}"
echo -e "${GREEN}✓ Libraries installed${NC}"
echo -e "${GREEN}✓ Directory structure created${NC}"
echo ""

echo -e "${YELLOW}What the integration did:${NC}"
echo "  1. Created ${INSTALL_BIN_DIR}/ with llama-server and tools"
echo "  2. Created ${INSTALL_LIB_DIR}/ with shared libraries (.so files)"
echo "  3. Set up proper permissions (chmod +x on binaries)"
echo "  4. Verified execution path"
echo ""

echo -e "${YELLOW}How llamatelemetry will find llama-server:${NC}"
echo "  The server.py find_llama_server() function searches in this order:"
echo "  1. LLAMA_SERVER_PATH environment variable"
echo "  2. ✓ Package binaries: ${INSTALL_BIN_DIR}/llama-server"
echo "  3. LLAMA_CPP_DIR environment variable"
echo "  4. Cache directory: ~/.cache/llamatelemetry/"
echo "  5. System paths: /usr/local/bin, /usr/bin, etc."
echo ""

echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Run integration test:"
echo "     python3 $TEST_SCRIPT"
echo ""
echo "  2. Install llamatelemetry package:"
echo "     cd ${LLCUDA_DIR}"
echo "     pip install -e ."
echo ""
echo "  3. Test inference:"
echo "     python3 -c \"import llamatelemetry; print(llamatelemetry.__version__)\""
echo ""

echo -e "${CYAN}For Kaggle:${NC}"
echo "  - Build with: ./build_cuda12_tesla_t4_colab.sh"
echo "  - Create tar.gz: tar -czf llamatelemetry-binaries-t4.tar.gz -C ${LLCUDA_PKG_DIR} binaries lib"
echo "  - Upload to GitHub releases"
echo "  - Bootstrap will auto-download in Kaggle"
echo ""

show_success "All done! Run the test script to verify integration."
