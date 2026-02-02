"""
llamatelemetry.api.multigpu - Multi-GPU Configuration Module

Configuration and utilities for running llama.cpp server with multiple GPUs.
Specifically optimized for Kaggle's 2× NVIDIA Tesla T4 (32GB total VRAM).
"""

import subprocess
import os
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import shutil


class SplitMode(Enum):
    """
    GPU split mode for multi-GPU inference.
    
    Attributes:
        NONE: Single GPU mode
        LAYER: Split model layers across GPUs (default)
        ROW: Split tensor rows across GPUs (requires tensor parallelism)
    """
    NONE = "none"
    LAYER = "layer"
    ROW = "row"


@dataclass
class GPUInfo:
    """
    GPU device information.
    
    Attributes:
        id: GPU device ID (0, 1, 2, ...)
        name: GPU name/model (e.g., "Tesla T4")
        memory_total: Total VRAM in bytes
        memory_free: Available VRAM in bytes
        memory_used: Used VRAM in bytes
        compute_capability: CUDA compute capability (e.g., "7.5")
        driver_version: NVIDIA driver version
        cuda_version: CUDA runtime version
        utilization: GPU utilization percentage
        temperature: GPU temperature in Celsius
        power_draw: Current power draw in Watts
        power_limit: Power limit in Watts
    """
    id: int
    name: str
    memory_total: int
    memory_free: int = 0
    memory_used: int = 0
    compute_capability: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    utilization: int = 0
    temperature: int = 0
    power_draw: float = 0.0
    power_limit: float = 0.0
    
    @property
    def memory_total_gb(self) -> float:
        """Total memory in GB."""
        return self.memory_total / (1024**3)
    
    @property
    def memory_free_gb(self) -> float:
        """Free memory in GB."""
        return self.memory_free / (1024**3)
    
    @property
    def memory_used_gb(self) -> float:
        """Used memory in GB."""
        return self.memory_used / (1024**3)


@dataclass
class MultiGPUConfig:
    """
    Multi-GPU configuration for llama.cpp server.
    
    This class encapsulates all multi-GPU settings and provides
    methods to generate command-line arguments for llama-server.
    
    Attributes:
        n_gpu_layers: Number of layers to offload to GPU(s) (-1 = all)
        main_gpu: Primary GPU device ID for computation
        split_mode: How to split model across GPUs
        tensor_split: VRAM split ratios per GPU (e.g., [0.5, 0.5])
        use_mmap: Use memory-mapped files for model loading
        use_mlock: Lock model in memory (no swap)
        flash_attention: Enable flash attention
        no_kv_offload: Don't offload KV cache to GPU
        
    Example:
        >>> # Kaggle 2× T4 configuration
        >>> config = MultiGPUConfig(
        ...     n_gpu_layers=-1,  # All layers on GPU
        ...     split_mode=SplitMode.LAYER,
        ...     tensor_split=[0.5, 0.5]  # Equal split
        ... )
        >>> 
        >>> # Generate CLI arguments
        >>> args = config.to_cli_args()
        >>> print(args)
        ['-ngl', '-1', '--split-mode', 'layer', '--tensor-split', '0.5,0.5']
    """
    n_gpu_layers: int = -1
    main_gpu: int = 0
    split_mode: SplitMode = SplitMode.LAYER
    tensor_split: Optional[List[float]] = None
    use_mmap: bool = True
    use_mlock: bool = False
    flash_attention: bool = True
    no_kv_offload: bool = False
    
    # Additional memory settings
    ctx_size: int = 0  # 0 = use model default
    batch_size: int = 2048
    ubatch_size: int = 512
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.tensor_split:
            # Normalize tensor_split to sum to 1.0
            total = sum(self.tensor_split)
            if total > 0 and abs(total - 1.0) > 0.01:
                self.tensor_split = [x / total for x in self.tensor_split]
    
    def to_cli_args(self) -> List[str]:
        """
        Convert configuration to llama-server CLI arguments.
        
        Returns:
            List of command-line argument strings
        """
        args = []
        
        # GPU layers
        if self.n_gpu_layers != 0:
            args.extend(["-ngl", str(self.n_gpu_layers)])
        
        # Main GPU
        if self.main_gpu > 0:
            args.extend(["--main-gpu", str(self.main_gpu)])
        
        # Split mode (only if we have tensor_split)
        if self.tensor_split and len(self.tensor_split) > 1:
            args.extend(["--split-mode", self.split_mode.value])
            
            # Tensor split ratios
            split_str = ",".join(str(x) for x in self.tensor_split)
            args.extend(["--tensor-split", split_str])
        
        # Memory mapping
        if not self.use_mmap:
            args.append("--no-mmap")
        
        # Memory locking
        if self.use_mlock:
            args.append("--mlock")
        
        # Flash attention
        if self.flash_attention:
            args.append("-fa")
        
        # KV cache offloading
        if self.no_kv_offload:
            args.append("--no-kv-offload")
        
        # Context size
        if self.ctx_size > 0:
            args.extend(["-c", str(self.ctx_size)])
        
        # Batch sizes
        if self.batch_size != 2048:
            args.extend(["-b", str(self.batch_size)])
        if self.ubatch_size != 512:
            args.extend(["-ub", str(self.ubatch_size)])
        
        return args
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_gpu_layers": self.n_gpu_layers,
            "main_gpu": self.main_gpu,
            "split_mode": self.split_mode.value,
            "tensor_split": self.tensor_split,
            "use_mmap": self.use_mmap,
            "use_mlock": self.use_mlock,
            "flash_attention": self.flash_attention,
            "no_kv_offload": self.no_kv_offload,
            "ctx_size": self.ctx_size,
            "batch_size": self.batch_size,
            "ubatch_size": self.ubatch_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiGPUConfig":
        """Create from dictionary."""
        split_mode = data.get("split_mode", "layer")
        if isinstance(split_mode, str):
            split_mode = SplitMode(split_mode)
        
        return cls(
            n_gpu_layers=data.get("n_gpu_layers", -1),
            main_gpu=data.get("main_gpu", 0),
            split_mode=split_mode,
            tensor_split=data.get("tensor_split"),
            use_mmap=data.get("use_mmap", True),
            use_mlock=data.get("use_mlock", False),
            flash_attention=data.get("flash_attention", True),
            no_kv_offload=data.get("no_kv_offload", False),
            ctx_size=data.get("ctx_size", 0),
            batch_size=data.get("batch_size", 2048),
            ubatch_size=data.get("ubatch_size", 512)
        )


# =============================================================================
# GPU Detection Functions
# =============================================================================

def run_nvidia_smi(args: List[str] = None) -> Optional[str]:
    """
    Run nvidia-smi command and return output.
    
    Args:
        args: Additional command-line arguments
        
    Returns:
        Command output or None on error
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    
    cmd = [nvidia_smi] + (args or [])
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def detect_gpus() -> List[GPUInfo]:
    """
    Detect available NVIDIA GPUs.
    
    Returns:
        List of GPUInfo objects for each GPU
        
    Example:
        >>> gpus = detect_gpus()
        >>> for gpu in gpus:
        ...     print(f"GPU {gpu.id}: {gpu.name} ({gpu.memory_total_gb:.1f} GB)")
        GPU 0: Tesla T4 (15.0 GB)
        GPU 1: Tesla T4 (15.0 GB)
    """
    # Query format: index, name, memory.total, memory.free, memory.used, 
    #               compute_capability, driver_version, utilization, temp, power
    query = [
        "--query-gpu=index,name,memory.total,memory.free,memory.used,"
        "compute_cap,driver_version,utilization.gpu,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits"
    ]
    
    output = run_nvidia_smi(query)
    if not output:
        return []
    
    gpus = []
    for line in output.strip().split("\n"):
        if not line:
            continue
        
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 11:
            continue
        
        try:
            gpu = GPUInfo(
                id=int(parts[0]),
                name=parts[1],
                memory_total=int(float(parts[2])) * 1024 * 1024,  # MiB to bytes
                memory_free=int(float(parts[3])) * 1024 * 1024,
                memory_used=int(float(parts[4])) * 1024 * 1024,
                compute_capability=parts[5],
                driver_version=parts[6],
                utilization=int(parts[7]) if parts[7] != "[N/A]" else 0,
                temperature=int(parts[8]) if parts[8] != "[N/A]" else 0,
                power_draw=float(parts[9]) if parts[9] != "[N/A]" else 0.0,
                power_limit=float(parts[10]) if parts[10] != "[N/A]" else 0.0
            )
            gpus.append(gpu)
        except (ValueError, IndexError):
            continue
    
    return gpus


def get_cuda_version() -> Optional[str]:
    """
    Get CUDA runtime version.
    
    Returns:
        CUDA version string (e.g., "12.4") or None
    """
    # Try nvidia-smi first
    output = run_nvidia_smi()
    if output:
        match = re.search(r"CUDA Version:\s*([\d.]+)", output)
        if match:
            return match.group(1)
    
    # Try nvcc
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            result = subprocess.run(
                [nvcc, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            match = re.search(r"release\s+([\d.]+)", result.stdout)
            if match:
                return match.group(1)
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    return None


def get_total_vram() -> int:
    """
    Get total VRAM across all GPUs in bytes.
    
    Returns:
        Total VRAM in bytes
    """
    gpus = detect_gpus()
    return sum(gpu.memory_total for gpu in gpus)


def get_free_vram() -> int:
    """
    Get total free VRAM across all GPUs in bytes.
    
    Returns:
        Free VRAM in bytes
    """
    gpus = detect_gpus()
    return sum(gpu.memory_free for gpu in gpus)


def is_multi_gpu() -> bool:
    """Check if multiple GPUs are available."""
    return len(detect_gpus()) > 1


def gpu_count() -> int:
    """Get number of available GPUs."""
    return len(detect_gpus())


# =============================================================================
# Configuration Presets
# =============================================================================

def kaggle_t4_dual_config(model_size_gb: float = 7.0) -> MultiGPUConfig:
    """
    Get optimized configuration for Kaggle's 2× Tesla T4 GPUs.
    
    Kaggle provides:
    - 2× NVIDIA Tesla T4 GPUs (15 GB VRAM each, 30 GB total)
    - SM 7.5 (Turing architecture)
    - Flash Attention support
    
    Args:
        model_size_gb: Approximate model size in GB for VRAM allocation
        
    Returns:
        Optimized MultiGPUConfig for Kaggle T4s
        
    Example:
        >>> config = kaggle_t4_dual_config(model_size_gb=7)
        >>> print(config.tensor_split)  # [0.5, 0.5]
    """
    # Calculate optimal tensor split based on model size
    # T4 has 15GB each, leave some headroom for KV cache
    available_per_gpu = 14.0  # GB, with safety margin
    total_available = available_per_gpu * 2
    
    if model_size_gb <= available_per_gpu:
        # Model fits on single GPU, but split for parallelism
        tensor_split = [0.5, 0.5]
    else:
        # Need both GPUs
        tensor_split = [0.5, 0.5]
    
    return MultiGPUConfig(
        n_gpu_layers=-1,  # All layers on GPU
        main_gpu=0,
        split_mode=SplitMode.LAYER,
        tensor_split=tensor_split,
        use_mmap=True,
        use_mlock=False,  # Kaggle has limited RAM
        flash_attention=True,  # T4 supports flash attention
        no_kv_offload=False,
        ctx_size=8192,  # Good balance for T4
        batch_size=2048,
        ubatch_size=512
    )


def colab_t4_single_config() -> MultiGPUConfig:
    """
    Legacy single-T4 configuration (pre-2.2.0).

    llamatelemetry v2.2.0 targets Kaggle dual T4 only. This helper remains for
    legacy compatibility with single-T4 environments.

    Returns:
        MultiGPUConfig for a single Tesla T4
    """
    return MultiGPUConfig(
        n_gpu_layers=-1,
        main_gpu=0,
        split_mode=SplitMode.NONE,
        tensor_split=None,
        use_mmap=True,
        use_mlock=False,
        flash_attention=True,
        no_kv_offload=False,
        ctx_size=4096,  # Conservative for single GPU
        batch_size=1024,
        ubatch_size=256
    )


def auto_config() -> MultiGPUConfig:
    """
    Automatically detect GPUs and create optimal configuration.
    
    Returns:
        Automatically configured MultiGPUConfig
    """
    gpus = detect_gpus()
    
    if not gpus:
        # No GPUs detected, CPU only
        return MultiGPUConfig(
            n_gpu_layers=0,
            flash_attention=False
        )
    
    if len(gpus) == 1:
        # Single GPU
        gpu = gpus[0]
        return MultiGPUConfig(
            n_gpu_layers=-1,
            main_gpu=0,
            split_mode=SplitMode.NONE,
            tensor_split=None,
            flash_attention="7.5" <= gpu.compute_capability,  # Turing+
            ctx_size=4096 if gpu.memory_total_gb < 16 else 8192
        )
    
    # Multi-GPU: calculate split based on VRAM
    total_vram = sum(gpu.memory_total for gpu in gpus)
    tensor_split = [gpu.memory_total / total_vram for gpu in gpus]
    
    # Use most powerful GPU as main
    main_gpu = max(range(len(gpus)), key=lambda i: gpus[i].memory_total)
    
    # Check flash attention support (all GPUs must support it)
    flash_support = all(
        gpu.compute_capability >= "7.5" for gpu in gpus
    )
    
    return MultiGPUConfig(
        n_gpu_layers=-1,
        main_gpu=main_gpu,
        split_mode=SplitMode.LAYER,
        tensor_split=tensor_split,
        flash_attention=flash_support,
        ctx_size=8192
    )


# =============================================================================
# Model VRAM Estimation
# =============================================================================

def estimate_model_vram(
    param_count: float,
    quantization: str = "Q4_K_M",
    ctx_size: int = 4096
) -> float:
    """
    Estimate VRAM usage for a model.
    
    Args:
        param_count: Number of parameters in billions (e.g., 7 for 7B)
        quantization: Quantization type (Q4_K_M, Q8_0, etc.)
        ctx_size: Context size
        
    Returns:
        Estimated VRAM usage in GB
        
    Example:
        >>> estimate_model_vram(7, "Q4_K_M", 4096)
        5.2  # ~5.2 GB for Llama 7B Q4_K_M
    """
    # Bits per weight for different quantization types
    bits_per_weight = {
        "F32": 32,
        "F16": 16,
        "BF16": 16,
        "Q8_0": 8.5,
        "Q6_K": 6.6,
        "Q5_K_M": 5.7,
        "Q5_K_S": 5.5,
        "Q5_0": 5.5,
        "Q4_K_M": 4.8,
        "Q4_K_S": 4.5,
        "Q4_0": 4.5,
        "Q3_K_M": 3.9,
        "Q3_K_S": 3.4,
        "Q2_K": 2.6,
        "IQ4_XS": 4.3,
        "IQ3_XS": 3.3,
        "IQ2_XS": 2.3,
        "IQ1_S": 1.6,
    }
    
    bpw = bits_per_weight.get(quantization.upper(), 4.8)  # Default to Q4_K_M
    
    # Model weights in GB
    weights_gb = (param_count * 1e9 * bpw) / (8 * 1024**3)
    
    # KV cache estimate (rough)
    # ~2 bytes per token per layer for FP16 KV cache
    # Assume 32 layers for 7B model, scale linearly
    layers = int(param_count * 4.5)  # Rough estimate
    kv_cache_gb = (ctx_size * layers * 2 * 2) / (1024**3)  # 2 bytes * 2 (K and V)
    
    # Overhead for CUDA kernels, temporary buffers
    overhead = 0.5  # GB
    
    return weights_gb + kv_cache_gb + overhead


def can_fit_model(
    param_count: float,
    quantization: str = "Q4_K_M",
    ctx_size: int = 4096
) -> Tuple[bool, float]:
    """
    Check if a model can fit in available VRAM.
    
    Args:
        param_count: Number of parameters in billions
        quantization: Quantization type
        ctx_size: Context size
        
    Returns:
        Tuple of (can_fit, estimated_vram_gb)
    """
    estimated = estimate_model_vram(param_count, quantization, ctx_size)
    available = get_free_vram() / (1024**3)
    return (estimated <= available * 0.9, estimated)  # 90% safety margin


def recommend_quantization(
    param_count: float,
    ctx_size: int = 4096
) -> str:
    """
    Recommend best quantization for available VRAM.
    
    Args:
        param_count: Number of parameters in billions
        ctx_size: Context size
        
    Returns:
        Recommended quantization type
    """
    available = get_free_vram() / (1024**3)
    
    # Try quantization levels from highest quality to lowest
    quants = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q2_K", "IQ2_XS"]
    
    for quant in quants:
        estimated = estimate_model_vram(param_count, quant, ctx_size)
        if estimated <= available * 0.85:  # 85% safety margin
            return quant
    
    return "IQ2_XS"  # Fallback to smallest


# =============================================================================
# Environment Helpers
# =============================================================================

def set_cuda_visible_devices(*device_ids: int) -> None:
    """
    Set CUDA_VISIBLE_DEVICES environment variable.
    
    Args:
        *device_ids: GPU device IDs to make visible
        
    Example:
        >>> set_cuda_visible_devices(0, 1)  # Use GPUs 0 and 1
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)


def get_cuda_visible_devices() -> List[int]:
    """
    Get current CUDA_VISIBLE_DEVICES.
    
    Returns:
        List of visible GPU IDs, or empty list if not set
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not value:
        return []
    return [int(d) for d in value.split(",") if d.strip()]


def print_gpu_info() -> None:
    """Print GPU information to stdout."""
    gpus = detect_gpus()
    cuda_version = get_cuda_version()
    
    print(f"CUDA Version: {cuda_version or 'Not detected'}")
    print(f"Number of GPUs: {len(gpus)}")
    print()
    
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.name}")
        print(f"  Memory: {gpu.memory_free_gb:.1f} / {gpu.memory_total_gb:.1f} GB "
              f"({gpu.memory_used_gb:.1f} GB used)")
        print(f"  Compute Capability: {gpu.compute_capability}")
        print(f"  Driver: {gpu.driver_version}")
        if gpu.temperature > 0:
            print(f"  Temperature: {gpu.temperature}°C")
        if gpu.power_draw > 0:
            print(f"  Power: {gpu.power_draw:.1f} / {gpu.power_limit:.1f} W")
        print()
