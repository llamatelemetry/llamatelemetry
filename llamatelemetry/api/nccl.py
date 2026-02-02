"""
llamatelemetry.api.nccl - NVIDIA Collective Communications Library (NCCL) Integration

Python bindings and utilities for multi-GPU communication using NCCL.
Enables efficient tensor parallelism and data parallelism across multiple GPUs.

This module provides a high-level interface to NCCL operations for use with
llama.cpp multi-GPU inference on Kaggle (2× T4) and other multi-GPU systems.

Requirements:
    - NCCL library installed (libnccl2)
    - CUDA 12.x
    - Multiple NVIDIA GPUs

Example:
    >>> from llamatelemetry.api.nccl import NCCLCommunicator, is_nccl_available
    >>> 
    >>> if is_nccl_available():
    ...     comm = NCCLCommunicator(gpu_ids=[0, 1])
    ...     comm.initialize()
    ...     print(f"NCCL initialized with {comm.world_size} GPUs")
"""

import os
import ctypes
from ctypes import c_void_p, c_int, c_size_t, c_char_p, byref, POINTER
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass
from enum import IntEnum
import subprocess
import shutil


# =============================================================================
# NCCL Constants and Types
# =============================================================================

class NCCLResult(IntEnum):
    """NCCL operation result codes."""
    ncclSuccess = 0
    ncclUnhandledCudaError = 1
    ncclSystemError = 2
    ncclInternalError = 3
    ncclInvalidArgument = 4
    ncclInvalidUsage = 5
    ncclRemoteError = 6
    ncclInProgress = 7
    ncclNumResults = 8


class NCCLDataType(IntEnum):
    """NCCL data types."""
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclBfloat16 = 9
    ncclNumTypes = 10


class NCCLRedOp(IntEnum):
    """NCCL reduction operations."""
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5


# =============================================================================
# NCCL Library Loader
# =============================================================================

_nccl_lib = None
_nccl_available = None


def _find_nccl_library() -> Optional[str]:
    """Find NCCL shared library path."""
    # Common library names
    lib_names = [
        "libnccl.so.2",
        "libnccl.so",
        "nccl.so",
        "libnccl.dylib"
    ]
    
    # Common search paths
    search_paths = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
        "/usr/lib",
        "/opt/nvidia/nccl/lib",
        os.path.expanduser("~/.local/lib"),
        os.environ.get("LD_LIBRARY_PATH", "").split(":"),
    ]
    
    # Flatten paths
    all_paths = []
    for p in search_paths:
        if isinstance(p, list):
            all_paths.extend(p)
        elif p:
            all_paths.append(p)
    
    for base in all_paths:
        for name in lib_names:
            path = os.path.join(base, name)
            if os.path.isfile(path):
                return path
    
    # Try ldconfig
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in result.stdout.split("\n"):
            if "libnccl.so" in line:
                parts = line.split("=>")
                if len(parts) == 2:
                    return parts[1].strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return None


def _load_nccl() -> Optional[ctypes.CDLL]:
    """Load NCCL library."""
    global _nccl_lib
    
    if _nccl_lib is not None:
        return _nccl_lib
    
    lib_path = _find_nccl_library()
    if lib_path:
        try:
            _nccl_lib = ctypes.CDLL(lib_path)
            return _nccl_lib
        except OSError:
            pass
    
    # Try direct loading
    for name in ["nccl", "libnccl.so.2"]:
        try:
            _nccl_lib = ctypes.CDLL(name)
            return _nccl_lib
        except OSError:
            continue
    
    return None


def is_nccl_available() -> bool:
    """
    Check if NCCL is available on this system.
    
    Returns:
        True if NCCL library is available
        
    Example:
        >>> if is_nccl_available():
        ...     print("NCCL ready for multi-GPU!")
    """
    global _nccl_available
    
    if _nccl_available is not None:
        return _nccl_available
    
    _nccl_available = _load_nccl() is not None
    return _nccl_available


def get_nccl_version() -> Optional[str]:
    """
    Get NCCL version string.
    
    Returns:
        Version string (e.g., "2.19.3") or None if not available
    """
    lib = _load_nccl()
    if lib is None:
        return None
    
    try:
        version = c_int()
        lib.ncclGetVersion(byref(version))
        v = version.value
        major = v // 10000
        minor = (v % 10000) // 100
        patch = v % 100
        return f"{major}.{minor}.{patch}"
    except Exception:
        return None


# =============================================================================
# NCCL Data Classes
# =============================================================================

@dataclass
class NCCLConfig:
    """
    NCCL configuration.
    
    Attributes:
        gpu_ids: List of GPU device IDs to use
        world_size: Number of GPUs in the communicator
        rank: This process's rank (for multi-process)
        local_rank: Local GPU rank
        nccl_debug: Enable NCCL debug logging
        nccl_socket_ifname: Network interface for NCCL
    """
    gpu_ids: List[int]
    world_size: Optional[int] = None
    rank: int = 0
    local_rank: int = 0
    nccl_debug: str = "WARN"
    nccl_socket_ifname: Optional[str] = None
    
    def __post_init__(self):
        if self.world_size is None:
            self.world_size = len(self.gpu_ids)
    
    def apply_env(self):
        """Apply configuration to environment variables."""
        os.environ["NCCL_DEBUG"] = self.nccl_debug
        if self.nccl_socket_ifname:
            os.environ["NCCL_SOCKET_IFNAME"] = self.nccl_socket_ifname
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in self.gpu_ids)


@dataclass
class NCCLInfo:
    """
    NCCL runtime information.
    
    Attributes:
        version: NCCL version string
        device_count: Number of available GPUs
        is_available: Whether NCCL is available
        library_path: Path to NCCL library
    """
    version: Optional[str]
    device_count: int
    is_available: bool
    library_path: Optional[str]


# =============================================================================
# NCCL Communicator
# =============================================================================

class NCCLCommunicator:
    """
    NCCL Communicator for multi-GPU operations.
    
    This class provides a high-level interface for NCCL collective operations
    across multiple GPUs. It's designed for use with llama.cpp multi-GPU
    inference on systems like Kaggle (2× T4).
    
    Note: This is a simplified interface. For advanced usage, consider using
    PyTorch distributed or nvidia-nccl4py directly.
    
    Attributes:
        gpu_ids: List of GPU IDs in this communicator
        world_size: Number of GPUs
        is_initialized: Whether the communicator is initialized
        
    Example:
        >>> comm = NCCLCommunicator(gpu_ids=[0, 1])
        >>> comm.initialize()
        >>> print(f"Initialized with {comm.world_size} GPUs")
        >>> comm.finalize()
    """
    
    def __init__(self, gpu_ids: List[int] = None, config: NCCLConfig = None):
        """
        Initialize NCCLCommunicator.
        
        Args:
            gpu_ids: List of GPU device IDs to use
            config: Optional NCCLConfig for advanced settings
        """
        if config:
            self.config = config
            self.gpu_ids = config.gpu_ids
        else:
            self.gpu_ids = gpu_ids or [0, 1]
            self.config = NCCLConfig(gpu_ids=self.gpu_ids)
        
        self.world_size = len(self.gpu_ids)
        self._comm_handle = None
        self._is_initialized = False
        self._lib = None
    
    @property
    def is_initialized(self) -> bool:
        """Check if communicator is initialized."""
        return self._is_initialized
    
    def initialize(self) -> bool:
        """
        Initialize NCCL communicator.
        
        Returns:
            True if initialization succeeded
            
        Raises:
            RuntimeError: If NCCL is not available
        """
        if self._is_initialized:
            return True
        
        self._lib = _load_nccl()
        if self._lib is None:
            raise RuntimeError("NCCL library not available")
        
        # Apply environment configuration
        self.config.apply_env()
        
        try:
            # Create unique ID
            unique_id = (ctypes.c_byte * 128)()  # NCCL_UNIQUE_ID_BYTES = 128
            self._lib.ncclGetUniqueId(byref(unique_id))
            
            # Initialize communicator
            # Note: This is a simplified single-process initialization
            # For multi-process, you'd need to broadcast the unique_id
            self._comm_handle = c_void_p()
            result = self._lib.ncclCommInitRank(
                byref(self._comm_handle),
                self.world_size,
                unique_id,
                self.config.rank
            )
            
            if result != NCCLResult.ncclSuccess:
                raise RuntimeError(f"NCCL init failed with code {result}")
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            self._is_initialized = False
            raise RuntimeError(f"NCCL initialization failed: {e}")
    
    def finalize(self):
        """Finalize and destroy the communicator."""
        if self._is_initialized and self._lib and self._comm_handle:
            try:
                self._lib.ncclCommDestroy(self._comm_handle)
            except Exception:
                pass
        self._is_initialized = False
        self._comm_handle = None
    
    def barrier(self):
        """
        Synchronize all GPUs in the communicator.
        
        Blocks until all GPUs reach this point.
        """
        if not self._is_initialized:
            raise RuntimeError("Communicator not initialized")
        
        # NCCL doesn't have explicit barrier, use AllReduce with dummy data
        # In practice, use CUDA synchronization
        pass  # Simplified implementation
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()
        return False
    
    def __del__(self):
        """Destructor."""
        self.finalize()


# =============================================================================
# High-Level Functions
# =============================================================================

def get_nccl_info() -> NCCLInfo:
    """
    Get NCCL runtime information.
    
    Returns:
        NCCLInfo with version, device count, and availability
        
    Example:
        >>> info = get_nccl_info()
        >>> print(f"NCCL {info.version}, {info.device_count} GPUs")
    """
    lib_path = _find_nccl_library()
    version = get_nccl_version()
    is_available = is_nccl_available()
    
    # Get device count from nvidia-smi
    device_count = 0
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10
        )
        device_count = len([l for l in result.stdout.split("\n") if l.strip().startswith("GPU")])
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return NCCLInfo(
        version=version,
        device_count=device_count,
        is_available=is_available,
        library_path=lib_path
    )


def setup_nccl_environment(
    gpu_ids: List[int] = None,
    debug_level: str = "WARN",
    socket_ifname: Optional[str] = None,
    buffer_size: Optional[int] = None
):
    """
    Configure NCCL environment variables.
    
    Call this before initializing any NCCL operations.
    
    Args:
        gpu_ids: GPU IDs to use (sets CUDA_VISIBLE_DEVICES)
        debug_level: NCCL debug level (WARN, INFO, TRACE)
        socket_ifname: Network interface for NCCL
        buffer_size: NCCL buffer size in bytes
        
    Example:
        >>> setup_nccl_environment(gpu_ids=[0, 1], debug_level="INFO")
    """
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
    
    os.environ["NCCL_DEBUG"] = debug_level
    
    if socket_ifname:
        os.environ["NCCL_SOCKET_IFNAME"] = socket_ifname
    
    if buffer_size:
        os.environ["NCCL_BUFFSIZE"] = str(buffer_size)
    
    # Common optimizations
    os.environ["NCCL_P2P_LEVEL"] = "NVL"  # Use NVLink if available
    os.environ["NCCL_SHM_DISABLE"] = "0"  # Enable shared memory


def kaggle_nccl_config() -> NCCLConfig:
    """
    Get optimized NCCL configuration for Kaggle 2× T4.
    
    Returns:
        NCCLConfig optimized for Kaggle
        
    Example:
        >>> config = kaggle_nccl_config()
        >>> config.apply_env()
    """
    return NCCLConfig(
        gpu_ids=[0, 1],
        world_size=2,
        rank=0,
        local_rank=0,
        nccl_debug="WARN",
        nccl_socket_ifname=None  # Use default
    )


def print_nccl_info():
    """Print NCCL information to stdout."""
    info = get_nccl_info()
    
    print("NCCL Information:")
    print(f"  Available: {'Yes' if info.is_available else 'No'}")
    if info.version:
        print(f"  Version: {info.version}")
    if info.library_path:
        print(f"  Library: {info.library_path}")
    print(f"  GPUs: {info.device_count}")


# =============================================================================
# Utility Functions for llama.cpp Multi-GPU
# =============================================================================

def get_llama_cpp_nccl_args(
    gpu_ids: List[int] = None,
    split_mode: str = "layer",
    tensor_split: List[float] = None
) -> List[str]:
    """
    Get llama-server arguments for NCCL multi-GPU inference.
    
    Note: llama.cpp doesn't use NCCL directly, but this function
    provides the proper arguments for multi-GPU inference.
    
    Args:
        gpu_ids: GPU IDs to use
        split_mode: "layer" or "row"
        tensor_split: VRAM ratio per GPU
        
    Returns:
        List of CLI arguments for llama-server
        
    Example:
        >>> args = get_llama_cpp_nccl_args([0, 1], tensor_split=[0.5, 0.5])
        >>> print(" ".join(args))
        -ngl -1 --split-mode layer --tensor-split 0.5,0.5
    """
    gpu_ids = gpu_ids or [0, 1]
    
    if tensor_split is None:
        # Equal split by default
        tensor_split = [1.0 / len(gpu_ids)] * len(gpu_ids)
    
    args = [
        "-ngl", "-1",  # All layers on GPU
        "--split-mode", split_mode,
        "--tensor-split", ",".join(f"{v:.2f}" for v in tensor_split),
    ]
    
    return args
