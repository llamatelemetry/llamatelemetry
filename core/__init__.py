"""
llamatelemetry.core - Core tensor and device management

This module provides PyTorch-style tensor operations with direct CUDA control.
"""

import numpy as np
from typing import List, Union, Optional, Tuple
import llamatelemetry_cpp
from llamatelemetry_cpp import get_device_properties, get_device_count


# Re-export C++ types
DType = llamatelemetry_cpp.DType
Device = llamatelemetry_cpp.Device
DeviceProperties = llamatelemetry_cpp.DeviceProperties


from .server import InferenceEngine
from . import core

__all__ = [
    "InferenceEngine",
    "core",
]

class Tensor:
    """
    PyTorch-style tensor with CUDA acceleration

    Example:
        >>> import llamatelemetry
        >>> x = llamatelemetry.Tensor([2, 3], dtype=llamatelemetry.DType.Float32, device=0)
        >>> print(x.shape)
        [2, 3]
    """

    def __init__(self, shape: List[int], dtype: DType = DType.Float32, device: int = 0):
        """
        Create a new tensor

        Args:
            shape: Tensor dimensions
            dtype: Data type (default: Float32)
            device: CUDA device ID (default: 0)
        """
        self._tensor = llamatelemetry_cpp.Tensor(shape, dtype, device)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, device: int = 0) -> 'Tensor':
        """
        Create tensor from NumPy array

        Args:
            arr: NumPy array
            device: Target CUDA device

        Returns:
            New Tensor object
        """
        # Map NumPy dtype to llamatelemetry DType
        dtype_map = {
            np.float32: DType.Float32,
            np.float16: DType.Float16,
            np.int32: DType.Int32,
            np.int64: DType.Int64,
            np.uint8: DType.UInt8,
        }

        if arr.dtype.type not in dtype_map:
            raise ValueError(f"Unsupported NumPy dtype: {arr.dtype}")

        tensor = cls(list(arr.shape), dtype_map[arr.dtype.type], device)

        # TODO: Copy data from NumPy to GPU
        # For now, this is a placeholder

        return tensor

    @classmethod
    def zeros(cls, shape: List[int], dtype: DType = DType.Float32, device: int = 0) -> 'Tensor':
        """Create zero-filled tensor"""
        t = cls.__new__(cls)
        t._tensor = llamatelemetry_cpp.Tensor.zeros(shape, dtype, device)
        return t

    @classmethod
    def ones(cls, shape: List[int], dtype: DType = DType.Float32, device: int = 0) -> 'Tensor':
        """Create ones-filled tensor"""
        t = cls.__new__(cls)
        t._tensor = llamatelemetry_cpp.Tensor.ones(shape, dtype, device)
        return t

    @property
    def shape(self) -> List[int]:
        """Tensor shape"""
        return self._tensor.shape

    @property
    def dtype(self) -> DType:
        """Data type"""
        return self._tensor.dtype

    @property
    def device(self) -> int:
        """Device ID"""
        return self._tensor.device

    @property
    def ndim(self) -> int:
        """Number of dimensions"""
        return self._tensor.ndim()

    def numel(self) -> int:
        """Number of elements"""
        return self._tensor.numel()

    def to(self, device: int) -> 'Tensor':
        """
        Move tensor to specified device

        Args:
            device: Target device ID

        Returns:
            New tensor on target device
        """
        t = Tensor.__new__(Tensor)
        t._tensor = self._tensor.to(device)
        return t

    def is_contiguous(self) -> bool:
        """Check if tensor is contiguous in memory"""
        return self._tensor.is_contiguous()

    def contiguous(self) -> 'Tensor':
        """Return contiguous version of tensor"""
        t = Tensor.__new__(Tensor)
        t._tensor = self._tensor.contiguous()
        return t

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication: self @ other"""
        result = Tensor.__new__(Tensor)
        result._tensor = llamatelemetry_cpp.ops.matmul(self._tensor, other._tensor)
        return result

    def __repr__(self) -> str:
        return repr(self._tensor)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Matrix multiplication

    Args:
        a: Left tensor [M, K]
        b: Right tensor [K, N]

    Returns:
        Result tensor [M, N]
    """
    return a @ b


def get_device_count() -> int:
    """Get number of available CUDA devices"""
    return Device.get_device_count()


def get_device_properties(device_id: int = 0) -> DeviceProperties:
    """Get properties of CUDA device"""
    return Device.get_device_properties(device_id)


def synchronize(device_id: int = -1):
    """Synchronize CUDA device"""
    Device.synchronize(device_id)


__all__ = [
    'Tensor',
    'DType',
    'Device',
    'DeviceProperties',
    'matmul',
    'get_device_count',
    'get_device_properties',
    'synchronize',
]
