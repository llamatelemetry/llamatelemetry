#!/bin/bash
################################################################################
# Manual CMake Build Script for NVIDIA Tesla T4 (CC 7.5)
# System: Kaggle, CUDA 12.4/12.6, Python 3.12
#
# This script contains the exact CMake commands you need to run manually.
# It does NOT execute them automatically - you run each command yourself.
################################################################################

cat << 'EOF'
================================================================================
CMake Build Commands for NVIDIA Tesla T4 (Compute Capability 7.5)
================================================================================

Target GPU:    Tesla T4
Architecture:  Turing
Compute Cap:   7.5
CUDA Version:  12.4/12.6 (Kaggle)
System:        Kaggle (Python 3.12)

================================================================================
STEP 1: Setup (in Kaggle)
================================================================================

# Clone llama.cpp if not already present
!git clone https://github.com/ggml-org/llama.cpp
%cd llama.cpp

# Verify CUDA
!nvcc --version
!nvidia-smi

================================================================================
STEP 2: Configure with CMake (run this command)
================================================================================

!cmake -B build_cuda12_t4 \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_FORCE_CUBLAS=OFF \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_CURL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

================================================================================
CMake Options Explained:
================================================================================

-DCMAKE_BUILD_TYPE=Release
  → Build optimized release version

-DGGML_CUDA=ON
  → Enable CUDA support

-DCMAKE_CUDA_ARCHITECTURES="75"
  → Target Compute Capability 7.5 (Turing architecture)
  → CRITICAL for Tesla T4

-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
  → Use Kaggle's CUDA compiler

-DGGML_NATIVE=OFF
  → Build for CC 7.5, not the build machine's GPU
  → Ensures the binary works on all T4 GPUs

-DGGML_CUDA_FORCE_CUBLAS=OFF
  → Use optimized custom CUDA kernels (faster than cuBLAS on T4)

-DGGML_CUDA_FA=ON
  → Enable FlashAttention (T4 supports this, ~2x faster)
  → Requires CC >= 7.0

-DGGML_CUDA_FA_ALL_QUANTS=ON
  → Enable FlashAttention for all quantization formats

-DGGML_CUDA_GRAPHS=ON
  → Enable CUDA graphs for optimized kernel execution

-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128
  → Optimize for multi-GPU peer-to-peer transfers

-DLLAMA_BUILD_SERVER=ON
  → Build llama-server executable (required for llamatelemetry)

-DBUILD_SHARED_LIBS=ON
  → Build as shared libraries (.so files)

================================================================================
STEP 3: Build (run this command)
================================================================================

This will take 5-15 minutes in Kaggle:

!cmake --build build_cuda12_t4 --config Release -j$(nproc)

Explanation:
  --config Release  : Build in release mode (optimized)
  -j$(nproc)        : Use all CPU cores (Kaggle has 2 cores usually)

================================================================================
STEP 4: Verify Build Success
================================================================================

!ls -lh build_cuda12_t4/bin/llama-server
!ls -lh build_cuda12_t4/bin/*.so*

Expected outputs:
  llama-server     : ~150-200 MB
  libllama.so      : ~50-100 MB
  libggml-*.so     : Multiple library files

================================================================================
STEP 5: Test Binary (in Kaggle)
================================================================================

import os
os.environ['LD_LIBRARY_PATH'] = '/content/llama.cpp/build_cuda12_t4/bin'

!./build_cuda12_t4/bin/llama-server --help

Should display help text without errors.

================================================================================
STEP 6: Create Package (in Kaggle)
================================================================================

# Create directory structure
!mkdir -p /content/package_t4/bin
!mkdir -p /content/package_t4/lib

# Copy binaries
!cp build_cuda12_t4/bin/llama-server /content/package_t4/bin/
!cp build_cuda12_t4/bin/llama-cli /content/package_t4/bin/
!cp build_cuda12_t4/bin/llama-quantize /content/package_t4/bin/
!cp build_cuda12_t4/bin/llama-embedding /content/package_t4/bin/

# Copy libraries
!cp build_cuda12_t4/bin/*.so* /content/package_t4/lib/

# Create tar.gz
!cd /content && tar -czf llamatelemetry-binaries-cuda12-t4.tar.gz package_t4/

# Download
from google.colab import files
files.download('/content/llamatelemetry-binaries-cuda12-t4.tar.gz')

================================================================================
ALTERNATIVE: Use Local Script (after downloading from Kaggle)
================================================================================

If you build in Kaggle and download the build directory, you can use:

cd /media/waqasm86/External1/Project-Nvidia
./CREATE_RELEASE_PACKAGE.sh

Select option "2" for Tesla T4

================================================================================
Notes for Tesla T4
================================================================================

Hardware Capabilities:
  - 15GB VRAM (large context support)
  - FlashAttention supported (2x faster inference)
  - Tensor cores for INT8/FP16
  - Excellent for 1-13B parameter models

Recommended Settings:
  - gpu_layers: 26-35
  - ctx_size: 2048-8192
  - Use Q4_K_M or Q5_K_M quantization
  - Expected speed: 25-60 tokens/sec

Compatible Models:
  - Llama 2 7B
  - Llama 3.1 8B
  - Mistral 7B
  - Gemma 7B
  - Phi-3 Medium 14B
  - Small 13B models

FlashAttention Benefits:
  - ~2x faster inference
  - Lower memory usage
  - Better long-context performance
  - Enabled by default for T4

================================================================================
Troubleshooting
================================================================================

If CMake fails in Kaggle:
  1. Check CUDA: !nvcc --version
  2. Verify T4: !nvidia-smi
  3. Update cmake: !pip install cmake --upgrade

If build fails:
  1. Check Kaggle GPU is allocated (Runtime → Change runtime type → T4)
  2. Clear output and restart: Runtime → Factory reset runtime
  3. Check disk space: !df -h

If llama-server crashes:
  1. Library path: Set LD_LIBRARY_PATH correctly
  2. CUDA version mismatch: Kaggle CUDA != Build CUDA
  3. Wrong CC: Built for wrong GPU (should be 75 for T4)

Kaggle-specific issues:
  1. Session timeout: Save build artifacts to Google Drive
  2. Runtime disconnect: Build in multiple steps
  3. GPU not allocated: Select T4 GPU in runtime settings

================================================================================
Performance Comparison: T4 vs 940M
================================================================================

| Feature              | GeForce 940M | Tesla T4    |
|----------------------|--------------|-------------|
| VRAM                 | 1GB          | 15GB        |
| Compute Capability   | 5.0          | 7.5         |
| FlashAttention       | ❌ No        | ✅ Yes      |
| Recommended Layers   | 10-15        | 26-35       |
| Max Model Size       | 3B params    | 13B params  |
| Expected Speed       | 10-20 tok/s  | 25-60 tok/s |
| Tensor Cores         | ❌ No        | ✅ Yes      |

================================================================================
After Build: Upload to GitHub Releases
================================================================================

1. Download the .tar.gz file from Kaggle
2. Go to: https://github.com/waqasm86/llamatelemetry/releases
3. Create new release: v2.2.0
4. Upload: llamatelemetry-binaries-cuda12-t4.tar.gz
5. Add release notes describing T4 optimizations

================================================================================
Local Build (if you have T4 GPU locally)
================================================================================

If building on a local system with T4:

cd /media/waqasm86/External1/Project-Nvidia/llama.cpp

cmake -B build_cuda12_t4 \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DGGML_CUDA_FA=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DBUILD_SHARED_LIBS=ON

cmake --build build_cuda12_t4 --config Release -j$(nproc)

================================================================================
EOF
