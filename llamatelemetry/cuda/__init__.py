"""
llamatelemetry CUDA Optimization APIs

Advanced CUDA optimizations for Tesla T4 inference including:
- CUDA Graphs for reduced kernel launch overhead
- Triton custom kernels for specialized operations
- Tensor Core utilities for mixed-precision acceleration

These APIs provide low-level control for maximum performance.
"""

from .graphs import (
    CUDAGraph,
    GraphPool,
    capture_graph,
    replay_graph,
    enable_cuda_graphs,
)

from .triton_kernels import (
    TritonKernel,
    register_kernel,
    get_kernel,
    list_kernels,
)

from .tensor_core import (
    TensorCoreConfig,
    enable_tensor_cores,
    matmul_tensor_core,
    check_tensor_core_support,
)

__all__ = [
    # CUDA Graphs
    'CUDAGraph',
    'GraphPool',
    'capture_graph',
    'replay_graph',
    'enable_cuda_graphs',

    # Triton kernels
    'TritonKernel',
    'register_kernel',
    'get_kernel',
    'list_kernels',

    # Tensor Core
    'TensorCoreConfig',
    'enable_tensor_cores',
    'matmul_tensor_core',
    'check_tensor_core_support',
]
