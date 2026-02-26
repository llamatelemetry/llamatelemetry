"""
llamatelemetry.kaggle.gpu_context - GPU context manager for clean RAPIDS/CUDA isolation.

Replaces manual environment variable management:

Before:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import cudf, cugraph
    # ... operations ...
    # Forgot to restore!

After:
    with GPUContext(gpu_ids=[1]):
        import cudf, cugraph
        # RAPIDS operations on GPU 1 only
    # Automatically restored
"""

import os
from typing import List, Optional
from contextlib import contextmanager


class GPUContext:
    """
    Context manager for GPU isolation.

    Ensures CUDA_VISIBLE_DEVICES is set correctly and restored on exit.
    Useful for split-GPU workflows where LLM runs on GPU 0 and
    RAPIDS/analytics run on GPU 1.

    Example:
        >>> # Run RAPIDS on GPU 1 while llama-server uses GPU 0
        >>> with GPUContext(gpu_ids=[1]) as ctx:
        ...     import cudf
        ...     df = cudf.DataFrame({"a": [1, 2, 3]})
        ...     print(f"Using GPU: {ctx.visible_devices}")

        >>> # Multi-GPU context
        >>> with GPUContext(gpu_ids=[0, 1]):
        ...     # Operations on both GPUs
        ...     pass
    """

    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        restore_on_exit: bool = True
    ):
        """
        Initialize GPU context.

        Args:
            gpu_ids: GPU IDs to make visible (None = all GPUs)
            restore_on_exit: Restore original CUDA_VISIBLE_DEVICES on exit
        """
        self.gpu_ids = gpu_ids
        self.restore_on_exit = restore_on_exit
        self._original_value: Optional[str] = None
        self._entered = False

    @property
    def visible_devices(self) -> str:
        """Current CUDA_VISIBLE_DEVICES value."""
        return os.environ.get("CUDA_VISIBLE_DEVICES", "")

    @property
    def is_active(self) -> bool:
        """Check if context is currently active."""
        return self._entered

    def __enter__(self) -> "GPUContext":
        """Enter context and set CUDA_VISIBLE_DEVICES."""
        # Save original value
        self._original_value = os.environ.get("CUDA_VISIBLE_DEVICES")
        self._entered = True

        # Set new value
        if self.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in self.gpu_ids)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore CUDA_VISIBLE_DEVICES."""
        self._entered = False

        # Restore original value
        if self.restore_on_exit:
            if self._original_value is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = self._original_value

        return False  # Don't suppress exceptions

    def set_devices(self, gpu_ids: List[int]) -> None:
        """
        Change visible devices while in context.

        Args:
            gpu_ids: New GPU IDs to make visible
        """
        self.gpu_ids = gpu_ids
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)

    def __repr__(self) -> str:
        status = "active" if self._entered else "inactive"
        return f"GPUContext(gpu_ids={self.gpu_ids}, status={status})"


@contextmanager
def rapids_gpu(gpu_id: int = 1):
    """
    Convenience context manager for RAPIDS on specific GPU.

    This is optimized for the split-GPU workflow where GPU 0 is used
    for LLM inference and GPU 1 for RAPIDS analytics.

    Args:
        gpu_id: GPU ID for RAPIDS operations (default: 1)

    Example:
        >>> with rapids_gpu(1):
        ...     import cudf
        ...     df = cudf.DataFrame({"x": [1, 2, 3]})
        ...     # All RAPIDS ops on GPU 1
    """
    with GPUContext(gpu_ids=[gpu_id]) as ctx:
        yield ctx


@contextmanager
def llm_gpu(gpu_ids: Optional[List[int]] = None):
    """
    Convenience context manager for LLM inference GPUs.

    Args:
        gpu_ids: GPU IDs for LLM (default: [0, 1] for dual GPU)

    Example:
        >>> with llm_gpu([0, 1]):
        ...     engine = InferenceEngine()
        ...     # LLM uses GPUs 0 and 1
    """
    if gpu_ids is None:
        gpu_ids = [0, 1]

    with GPUContext(gpu_ids=gpu_ids) as ctx:
        yield ctx


@contextmanager
def single_gpu(gpu_id: int = 0):
    """
    Convenience context manager for single GPU operations.

    Args:
        gpu_id: GPU ID to use (default: 0)

    Example:
        >>> with single_gpu(0):
        ...     # All ops on GPU 0 only
        ...     pass
    """
    with GPUContext(gpu_ids=[gpu_id]) as ctx:
        yield ctx


def get_current_gpu_context() -> Optional[List[int]]:
    """
    Get current CUDA_VISIBLE_DEVICES as list of GPU IDs.

    Returns:
        List of GPU IDs, or None if not set
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not value:
        return None

    try:
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError:
        return None


def set_gpu_for_rapids(gpu_id: int = 1) -> str:
    """
    Set GPU for RAPIDS without context manager (persistent).

    Use this when you don't need automatic restoration.

    Args:
        gpu_id: GPU ID for RAPIDS

    Returns:
        Previous CUDA_VISIBLE_DEVICES value
    """
    previous = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return previous


def reset_gpu_context(value: Optional[str] = None) -> None:
    """
    Reset CUDA_VISIBLE_DEVICES to a specific value.

    Args:
        value: Value to set (None to unset)
    """
    if value is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = value


__all__ = [
    "GPUContext",
    "rapids_gpu",
    "llm_gpu",
    "single_gpu",
    "get_current_gpu_context",
    "set_gpu_for_rapids",
    "reset_gpu_context",
]
