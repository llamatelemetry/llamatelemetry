"""
llamatelemetry.telemetry.resource - GPU-aware OpenTelemetry Resource

Builds an OTel Resource enriched with NVIDIA GPU attributes:
compute capability, device name, VRAM, driver version, CUDA version,
and NCCL availability.
"""

import subprocess
from typing import Optional


def _nvidia_smi_query() -> dict:
    """Run nvidia-smi and return parsed GPU attributes."""
    attrs = {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "name": parts[0],
                        "memory_total": parts[1],
                        "driver_version": parts[2],
                        "compute_capability": parts[3],
                    })
            if gpus:
                attrs["gpu.count"] = len(gpus)
                attrs["gpu.name"] = gpus[0]["name"]
                attrs["gpu.memory_total"] = gpus[0]["memory_total"]
                attrs["gpu.driver_version"] = gpus[0]["driver_version"]
                attrs["gpu.compute_capability"] = gpus[0]["compute_capability"]
                # Multi-GPU: list all names
                if len(gpus) > 1:
                    attrs["gpu.names"] = ",".join(g["name"] for g in gpus)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return attrs


def _cuda_version() -> Optional[str]:
    """Detect CUDA toolkit version via nvcc."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    parts = line.split("release")
                    if len(parts) > 1:
                        return parts[1].strip().split(",")[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _nccl_available() -> bool:
    """Check if NCCL shared library is loadable."""
    import ctypes
    try:
        ctypes.CDLL("libnccl.so")
        return True
    except OSError:
        try:
            ctypes.CDLL("libnccl.so.2")
            return True
        except OSError:
            return False


def build_gpu_resource(service_name: str = "llamatelemetry", service_version: str = "2.2.0"):
    """
    Build an OpenTelemetry Resource with GPU and NCCL attributes.

    Args:
        service_name: Service name for the resource
        service_version: Service version

    Returns:
        opentelemetry.sdk.resources.Resource with GPU attributes,
        or a plain dict if OTel SDK is not installed.
    """
    attributes = {
        "service.name": service_name,
        "service.version": service_version,
        "llamatelemetry.version": "2.2.0",
    }

    # Add GPU info
    gpu_attrs = _nvidia_smi_query()
    attributes.update(gpu_attrs)

    # Add CUDA version
    cuda_ver = _cuda_version()
    if cuda_ver:
        attributes["cuda.version"] = cuda_ver

    # Add NCCL availability
    attributes["nccl.available"] = _nccl_available()

    # Detect platform
    import os
    if os.path.exists("/kaggle"):
        attributes["platform"] = "kaggle"
    elif "COLAB_GPU" in os.environ:
        attributes["platform"] = "colab"
    else:
        attributes["platform"] = "local"

    try:
        from opentelemetry.sdk.resources import Resource
        return Resource.create(attributes)
    except ImportError:
        return attributes
