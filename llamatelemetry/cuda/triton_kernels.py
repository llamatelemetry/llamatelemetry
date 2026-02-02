"""
Triton Kernel Integration

Custom Triton kernels for specialized operations on Tesla T4.
Provides a registry system for managing and deploying Triton kernels.

Triton allows writing GPU kernels in Python with performance comparable to CUDA.

References:
    - Triton Language: https://triton-lang.org/
    - OpenAI Triton: https://github.com/openai/triton
"""

import torch
from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass
import warnings


# Check if Triton is available
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    warnings.warn(
        "Triton not available. Install with: pip install triton\n"
        "Some optimizations will be disabled."
    )


@dataclass
class KernelConfig:
    """Configuration for Triton kernel."""
    name: str
    block_size: int = 128
    num_warps: int = 4
    num_stages: int = 2


class TritonKernel:
    """
    Wrapper for Triton kernels with automatic configuration.

    Example:
        >>> @triton.jit
        >>> def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        >>>     pid = tl.program_id(0)
        >>>     block_start = pid * BLOCK_SIZE
        >>>     offsets = block_start + tl.arange(0, BLOCK_SIZE)
        >>>     mask = offsets < n_elements
        >>>     x = tl.load(x_ptr + offsets, mask=mask)
        >>>     y = tl.load(y_ptr + offsets, mask=mask)
        >>>     output = x + y
        >>>     tl.store(output_ptr + offsets, output, mask=mask)
        >>>
        >>> kernel = TritonKernel("add", add_kernel)
        >>> kernel.launch(x, y, output, n_elements)
    """

    def __init__(
        self,
        name: str,
        kernel_func: Optional[Callable] = None,
        config: Optional[KernelConfig] = None,
    ):
        """
        Initialize Triton kernel wrapper.

        Args:
            name: Kernel name
            kernel_func: Triton JIT-compiled function
            config: Kernel configuration
        """
        self.name = name
        self.kernel_func = kernel_func
        self.config = config or KernelConfig(name=name)

        if not TRITON_AVAILABLE and kernel_func is not None:
            warnings.warn(f"Triton not available, kernel '{name}' will not work")

    def launch(
        self,
        *args,
        grid: Optional[Tuple[int, ...]] = None,
        **kwargs
    ):
        """
        Launch kernel with arguments.

        Args:
            *args: Kernel arguments (tensors, scalars)
            grid: Grid dimensions (auto-compute if None)
            **kwargs: Additional kernel arguments

        Example:
            >>> kernel.launch(x, y, output, n_elements, grid=(n_blocks,))
        """
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")

        if self.kernel_func is None:
            raise RuntimeError(f"Kernel '{self.name}' not initialized")

        # Auto-compute grid if not provided
        if grid is None:
            grid = self._compute_grid(*args)

        # Launch with configuration
        self.kernel_func[grid](
            *args,
            BLOCK_SIZE=self.config.block_size,
            num_warps=self.config.num_warps,
            num_stages=self.config.num_stages,
            **kwargs
        )

    def _compute_grid(self, *args) -> Tuple[int, ...]:
        """Auto-compute grid dimensions based on input size."""
        # Simple heuristic: use first tensor's size
        for arg in args:
            if isinstance(arg, torch.Tensor):
                n_elements = arg.numel()
                n_blocks = (n_elements + self.config.block_size - 1) // self.config.block_size
                return (n_blocks,)

        # Default to 1 block
        return (1,)


# Global kernel registry
_KERNEL_REGISTRY: Dict[str, TritonKernel] = {}


def register_kernel(
    name: str,
    kernel_func: Callable,
    config: Optional[KernelConfig] = None,
):
    """
    Register a Triton kernel.

    Args:
        name: Kernel name
        kernel_func: Triton JIT function
        config: Optional configuration

    Example:
        >>> @triton.jit
        >>> def my_kernel(...):
        >>>     ...
        >>>
        >>> register_kernel("my_kernel", my_kernel)
    """
    kernel = TritonKernel(name, kernel_func, config)
    _KERNEL_REGISTRY[name] = kernel
    return kernel


def get_kernel(name: str) -> Optional[TritonKernel]:
    """
    Get registered kernel by name.

    Args:
        name: Kernel name

    Returns:
        TritonKernel or None if not found

    Example:
        >>> kernel = get_kernel("add_kernel")
        >>> kernel.launch(x, y, output, n)
    """
    return _KERNEL_REGISTRY.get(name)


def list_kernels() -> List[str]:
    """
    List all registered kernel names.

    Returns:
        List of kernel names
    """
    return list(_KERNEL_REGISTRY.keys())


# ============================================================================
# Built-in Optimized Kernels for Tesla T4
# ============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Optimized element-wise addition."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    @triton.jit
    def fused_layernorm_kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        mean_ptr,
        rstd_ptr,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused LayerNorm for RMS normalization."""
        row_idx = tl.program_id(0)
        row_start = row_idx * n_cols

        # Load row
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0)

        # Compute mean
        mean = tl.sum(x, axis=0) / n_cols
        tl.store(mean_ptr + row_idx, mean)

        # Compute variance
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(rstd_ptr + row_idx, rstd)

        # Normalize
        x_normed = x_centered * rstd

        # Apply affine transformation
        weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        output = x_normed * weight + bias

        # Store
        tl.store(output_ptr + row_start + cols, output, mask=mask)

    @triton.jit
    def softmax_kernel(
        input_ptr,
        output_ptr,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Numerically stable softmax."""
        row_idx = tl.program_id(0)
        row_start = row_idx * n_cols

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        # Load
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf'))

        # Max for numerical stability
        x_max = tl.max(x, axis=0)
        x_shifted = x - x_max

        # Exp
        numerator = tl.exp(x_shifted)
        denominator = tl.sum(numerator, axis=0)

        # Normalize
        output = numerator / denominator

        # Store
        tl.store(output_ptr + row_start + cols, output, mask=mask)

    # Register built-in kernels
    register_kernel("add", add_kernel)
    register_kernel("layernorm", fused_layernorm_kernel)
    register_kernel("softmax", softmax_kernel)


# ============================================================================
# High-level API functions
# ============================================================================

def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition using Triton kernel.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        x + y

    Example:
        >>> a = torch.randn(1024, device='cuda')
        >>> b = torch.randn(1024, device='cuda')
        >>> c = triton_add(a, b)
    """
    if not TRITON_AVAILABLE:
        return x + y  # Fallback to PyTorch

    assert x.shape == y.shape
    output = torch.empty_like(x)

    kernel = get_kernel("add")
    n_elements = x.numel()
    grid = ((n_elements + 127) // 128,)

    kernel.kernel_func[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=128,
    )

    return output


def triton_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    LayerNorm using Triton kernel.

    Args:
        x: Input tensor [batch, features]
        weight: Scale parameters
        bias: Shift parameters
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor
    """
    if not TRITON_AVAILABLE:
        return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)

    batch_size, n_cols = x.shape
    output = torch.empty_like(x)
    mean = torch.empty(batch_size, device=x.device)
    rstd = torch.empty(batch_size, device=x.device)

    kernel = get_kernel("layernorm")
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (batch_size,)

    kernel.kernel_func[grid](
        x, weight, bias, output, mean, rstd,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def triton_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax using Triton kernel.

    Args:
        x: Input tensor
        dim: Dimension to normalize

    Returns:
        Softmax output
    """
    if not TRITON_AVAILABLE:
        return torch.softmax(x, dim=dim)

    if dim != -1:
        # Transpose to make last dim the target
        raise NotImplementedError("Only dim=-1 supported for now")

    batch_size, n_cols = x.shape
    output = torch.empty_like(x)

    kernel = get_kernel("softmax")
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (batch_size,)

    kernel.kernel_func[grid](
        x, output, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
