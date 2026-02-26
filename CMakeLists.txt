cmake_minimum_required(VERSION 3.24)
project(llamatelemetry LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Find or install pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ============================================================================
# llamatelemetry v0.1.0 - Tesla T4 Only (SM 7.5) - CUDA 12 Optimized
# SDK version: 0.1.0 | Binary artifact: llama.cpp v0.1.0
# Runtime target: Kaggle dual T4 environment (builds can be produced elsewhere)
# ============================================================================

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "llamatelemetry v0.1.0 - Targeting Tesla T4 (SM 7.5)")
    set(CMAKE_CUDA_ARCHITECTURES "75")
else()
    message(STATUS "Using user-specified CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
    # Verify SM 7.5+ for optimal performance
    if(CMAKE_CUDA_ARCHITECTURES LESS 75)
        message(WARNING "llamatelemetry v0.1.0 is optimized for SM 7.5+ (Tensor Cores + FlashAttention)")
        message(WARNING "Your architecture: SM ${CMAKE_CUDA_ARCHITECTURES}")
        message(WARNING "Recommended: Tesla T4, RTX 20xx+, A100, H100")
    endif()
endif()

message(STATUS "Building for Tesla T4 (SM 7.5)")
message(STATUS "Features: FlashAttention + Tensor Cores + CUDA Graphs")

# Source files
set(CUDA_SOURCES
    csrc/core/device.cu
    csrc/core/tensor.cu
    csrc/ops/matmul.cu
)

set(CPP_SOURCES
    csrc/bindings.cpp
)

# Create Python module
pybind11_add_module(llamatelemetry_cpp
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

# Include directories
target_include_directories(llamatelemetry_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(llamatelemetry_cpp PRIVATE
    CUDA::cudart_static
    CUDA::cublas_static
    CUDA::cublasLt_static
    CUDA::culibos
    ${CMAKE_DL_LIBS}
    pthread
    rt
)

# CUDA compilation flags
set_target_properties(llamatelemetry_cpp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Optimization flags for Tesla T4 (SM 7.5)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(llamatelemetry_cpp PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math              # Faster math operations
            -lineinfo                    # Profiling support
            --ptxas-options=-v          # Verbose PTX assembly
            -Xptxas=-warn-spills        # Warn about register spills
            -gencode=arch=compute_75,code=sm_75  # Explicit T4 target
        >
        $<$<COMPILE_LANGUAGE:CXX>:
            -O3
            -march=native               # CPU optimizations
        >
    )
    message(STATUS "Release build: Tensor Core and FlashAttention optimizations enabled")
else()
    target_compile_options(llamatelemetry_cpp PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -g                          # Debug symbols
            -G                          # Device debug
            -lineinfo                   # Line info for profiling
        >
        $<$<COMPILE_LANGUAGE:CXX>:
            -g                          # Debug symbols
        >
    )
    message(STATUS "Debug build: Profiling and debug symbols enabled")
endif()

# Install target
install(TARGETS llamatelemetry_cpp
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
