"""
Tensor Core Utilities

Utilities for leveraging Tesla T4's Tensor Cores for accelerated
mixed-precision operations.

Tesla T4 (Turing architecture, SM 7.5) supports:
- FP16 matrix operations with Tensor Cores
- INT8 operations for quantized inference
- Automatic mixed precision (AMP) support

Tensor Cores provide up to 8x performance boost for suitable operations.

References:
    - NVIDIA Tensor Cores: https://developer.nvidia.com/tensor-cores
    - PyTorch AMP: https://pytorch.org/docs/stable/amp.html
"""

import torch
from typing import Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class TensorCoreConfig:
    """
    Configuration for Tensor Core operations.

    Attributes:
        enabled: Enable Tensor Core acceleration
        dtype: Data type for Tensor Core ops (float16 or bfloat16)
        allow_tf32: Allow TF32 mode for FP32 operations
        allow_fp16: Allow FP16 accumulation
    """
    enabled: bool = True
    dtype: torch.dtype = torch.float16
    allow_tf32: bool = True
    allow_fp16: bool = True


def check_tensor_core_support(device: int = 0) -> bool:
    """
    Check if device supports Tensor Cores.

    Tesla T4 has SM 7.5, which supports Tensor Cores.

    Args:
        device: CUDA device ID

    Returns:
        True if Tensor Cores are supported

    Example:
        >>> if check_tensor_core_support():
        >>>     print("Tensor Cores available!")
    """
    if not torch.cuda.is_available():
        return False

    # Get compute capability
    major, minor = torch.cuda.get_device_capability(device)
    compute_cap = major * 10 + minor

    # Tensor Cores available from SM 7.0 (Volta)
    # Tesla T4 is SM 7.5 (Turing)
    if compute_cap >= 70:
        device_name = torch.cuda.get_device_name(device)
        print(f"✓ Tensor Cores supported on {device_name} (SM {major}.{minor})")
        return True
    else:
        print(f"✗ Tensor Cores not supported (SM {major}.{minor} < 7.0)")
        return False


def enable_tensor_cores(
    dtype: torch.dtype = torch.float16,
    allow_tf32: bool = True,
) -> TensorCoreConfig:
    """
    Enable Tensor Core optimizations globally.

    Args:
        dtype: Preferred dtype (float16 or bfloat16)
        allow_tf32: Allow TF32 for FP32 operations

    Returns:
        TensorCoreConfig with applied settings

    Example:
        >>> config = enable_tensor_cores(dtype=torch.float16, allow_tf32=True)
        >>> # All subsequent matmuls will use Tensor Cores when possible
    """
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, Tensor Core settings ignored")
        return TensorCoreConfig(enabled=False)

    # Check support
    if not check_tensor_core_support():
        warnings.warn("Tensor Cores not supported on this device")
        return TensorCoreConfig(enabled=False)

    # Enable TF32 for FP32 operations (Ampere and newer, but set anyway)
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for FP32 operations")

    # Enable FP16 accumulation
    # Note: PyTorch automatically uses Tensor Cores for FP16 matmul
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    print(f"✓ Tensor Cores enabled with {dtype}")

    return TensorCoreConfig(
        enabled=True,
        dtype=dtype,
        allow_tf32=allow_tf32,
        allow_fp16=True,
    )


def matmul_tensor_core(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Matrix multiplication using Tensor Cores.

    Automatically converts inputs to FP16, performs matmul with Tensor Cores,
    and optionally converts back.

    Args:
        A: First matrix [M, K]
        B: Second matrix [K, N]
        out: Optional output tensor
        dtype: Computation dtype (float16 for Tensor Cores)

    Returns:
        Result matrix [M, N]

    Example:
        >>> A = torch.randn(1024, 2048, device='cuda')
        >>> B = torch.randn(2048, 4096, device='cuda')
        >>> C = matmul_tensor_core(A, B)  # Fast FP16 Tensor Core matmul
    """
    if not torch.cuda.is_available():
        return torch.matmul(A, B, out=out)

    original_dtype = A.dtype

    # Convert to FP16 for Tensor Core acceleration
    if A.dtype != dtype:
        A = A.to(dtype)
    if B.dtype != dtype:
        B = B.to(dtype)

    # Perform matmul (PyTorch automatically uses Tensor Cores)
    if out is not None:
        result = torch.matmul(A, B, out=out)
    else:
        result = torch.matmul(A, B)

    # Convert back to original dtype if needed
    if original_dtype != dtype:
        result = result.to(original_dtype)

    return result


def enable_amp(
    dtype: torch.dtype = torch.float16,
    cache_enabled: bool = True,
) -> Tuple[torch.cuda.amp.GradScaler, torch.cuda.amp.autocast]:
    """
    Enable Automatic Mixed Precision (AMP) training.

    AMP automatically uses Tensor Cores where beneficial while
    maintaining numerical stability.

    Args:
        dtype: AMP dtype (float16 or bfloat16)
        cache_enabled: Enable AMP cache

    Returns:
        Tuple of (GradScaler, autocast context)

    Example:
        >>> scaler, autocast = enable_amp()
        >>>
        >>> for batch in dataloader:
        >>>     with autocast:
        >>>         output = model(batch)
        >>>         loss = criterion(output, target)
        >>>
        >>>     scaler.scale(loss).backward()
        >>>     scaler.step(optimizer)
        >>>     scaler.update()
    """
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, AMP disabled")
        return None, None

    # Create gradient scaler for loss scaling
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Create autocast context
    autocast_ctx = torch.cuda.amp.autocast(
        enabled=True,
        dtype=dtype,
        cache_enabled=cache_enabled,
    )

    print(f"✓ AMP enabled with {dtype}")

    return scaler, autocast_ctx


class TensorCoreMatMul(torch.nn.Module):
    """
    Matrix multiplication module with Tensor Core optimization.

    Drop-in replacement for torch.matmul with automatic FP16 conversion.

    Example:
        >>> matmul = TensorCoreMatMul(dtype=torch.float16)
        >>> C = matmul(A, B)
    """

    def __init__(self, dtype: torch.dtype = torch.float16):
        """
        Initialize Tensor Core matmul module.

        Args:
            dtype: Computation dtype
        """
        super().__init__()
        self.dtype = dtype

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Perform matrix multiplication.

        Args:
            A: First matrix
            B: Second matrix

        Returns:
            A @ B using Tensor Cores
        """
        return matmul_tensor_core(A, B, dtype=self.dtype)


def optimize_for_tensor_cores(
    model: torch.nn.Module,
    dtype: torch.dtype = torch.float16,
) -> torch.nn.Module:
    """
    Optimize model for Tensor Core inference.

    Converts model to FP16 and enables optimizations.

    Args:
        model: PyTorch model
        dtype: Target dtype (float16 or bfloat16)

    Returns:
        Optimized model

    Example:
        >>> model = MyModel()
        >>> model = optimize_for_tensor_cores(model)
        >>> # Inference now uses Tensor Cores
        >>> output = model(input)
    """
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, optimization skipped")
        return model

    # Check Tensor Core support
    if not check_tensor_core_support():
        warnings.warn("Tensor Cores not supported, optimization limited")

    # Move to CUDA and convert dtype
    model = model.cuda()
    model = model.to(dtype)

    # Enable optimizations
    model.eval()  # Inference mode
    torch.backends.cudnn.benchmark = True

    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"✓ Model optimized for Tensor Cores ({dtype})")

    return model


def get_tensor_core_info(device: int = 0) -> dict:
    """
    Get Tensor Core capabilities for device.

    Args:
        device: CUDA device ID

    Returns:
        Dictionary with Tensor Core information

    Example:
        >>> info = get_tensor_core_info()
        >>> print(f"Tensor Cores: {info['supported']}")
        >>> print(f"Performance: {info['estimated_speedup']}x")
    """
    if not torch.cuda.is_available():
        return {'supported': False, 'reason': 'CUDA not available'}

    major, minor = torch.cuda.get_device_capability(device)
    compute_cap = major * 10 + minor
    device_name = torch.cuda.get_device_name(device)

    info = {
        'device': device_name,
        'compute_capability': f"{major}.{minor}",
        'supported': compute_cap >= 70,
    }

    if info['supported']:
        # Tesla T4 is SM 7.5 (Turing)
        if compute_cap == 75:  # Tesla T4
            info['architecture'] = 'Turing'
            info['fp16_tflops'] = 65  # Theoretical peak
            info['int8_tops'] = 130
            info['estimated_speedup'] = '2-4x'
        elif compute_cap >= 80:  # Ampere or newer
            info['architecture'] = 'Ampere+'
            info['estimated_speedup'] = '4-8x'
        else:  # Volta
            info['architecture'] = 'Volta'
            info['estimated_speedup'] = '2-3x'
    else:
        info['reason'] = f"Compute capability {major}.{minor} < 7.0"

    return info
