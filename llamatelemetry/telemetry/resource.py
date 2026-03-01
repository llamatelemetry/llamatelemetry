"""
llamatelemetry.telemetry.resource - GPU-aware OpenTelemetry Resource

Builds an OTel Resource enriched with NVIDIA GPU attributes:
compute capability, device name, VRAM, driver version, CUDA version,
and NCCL availability. Optionally enriches with llama.cpp build and
server metadata for balanced inference + observability.
"""

import os
import subprocess
from typing import Optional, Dict, Any


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


def _parse_llama_server_version(output: str) -> Dict[str, str]:
    """
    Parse llama-server --version output into a dict.

    Expected output lines (llama.cpp common/arg.cpp):
      version: <build_number> (<commit>)
      built with <compiler> for <target>
    """
    info: Dict[str, str] = {}
    if not output:
        return info

    for line in output.splitlines():
        line = line.strip()
        if line.lower().startswith("version:"):
            # version: 1234 (abcdef)
            parts = line.split("version:", 1)[-1].strip()
            if "(" in parts and ")" in parts:
                build_number = parts.split("(", 1)[0].strip()
                commit = parts.split("(", 1)[-1].split(")")[0].strip()
                if build_number:
                    info["llama_cpp.build_number"] = build_number
                if commit:
                    info["llama_cpp.commit"] = commit
                if build_number and commit:
                    info["llama_cpp.build_info"] = f"b{build_number}-{commit}"
        elif line.lower().startswith("built with"):
            # built with <compiler> for <target>
            parts = line.split("built with", 1)[-1].strip()
            if " for " in parts:
                compiler, target = parts.split(" for ", 1)
                compiler = compiler.strip()
                target = target.strip()
                if compiler:
                    info["llama_cpp.compiler"] = compiler
                if target:
                    info["llama_cpp.build_target"] = target
    return info


def _llama_server_version_info(server_path: Optional[str]) -> Dict[str, str]:
    """Run llama-server --version and parse build info."""
    if not server_path:
        return {}
    try:
        result = subprocess.run(
            [server_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return _parse_llama_server_version(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {}


def _llama_server_props(server_url: Optional[str], timeout: float = 2.0) -> Dict[str, Any]:
    """Fetch /props from llama-server when available."""
    if not server_url:
        return {}
    try:
        import requests
        response = requests.get(f"{server_url}/props", timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return {}
    return {}


def build_gpu_resource(
    service_name: str = "llamatelemetry",
    service_version: str = "0.1.0",
    llama_server_url: Optional[str] = None,
    llama_server_path: Optional[str] = None,
):
    """
    Build an OpenTelemetry Resource with GPU and NCCL attributes.

    Args:
        service_name: Service name for the resource
        service_version: Service version
        llama_server_url: Optional llama-server URL for /props metadata
        llama_server_path: Optional llama-server path for --version metadata

    Returns:
        opentelemetry.sdk.resources.Resource with GPU attributes,
        or a plain dict if OTel SDK is not installed.
    """
    if llama_server_path is None:
        llama_server_path = os.environ.get("LLAMA_SERVER_PATH")

    attributes: Dict[str, Any] = {
        "service.name": service_name,
        "service.version": service_version,
        "llamatelemetry.version": "0.1.0",
        "llamatelemetry.binary_version": "0.1.0",  # llama.cpp artifact version
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

    # Add NCCL version and library path if available
    try:
        from ..api.nccl import get_nccl_info
        nccl_info = get_nccl_info()
        if nccl_info.version:
            attributes["nccl.version"] = nccl_info.version
        if nccl_info.library_path:
            attributes["nccl.library_path"] = nccl_info.library_path
    except Exception:
        pass

    # Add NCCL environment flags for observability
    env_flags = [
        "NCCL_DEBUG",
        "NCCL_SOCKET_IFNAME",
        "NCCL_P2P_LEVEL",
        "NCCL_BUFFSIZE",
        "NCCL_IB_HCA",
        "NCCL_NET_GDR_LEVEL",
        "NCCL_SHM_DISABLE",
    ]
    for flag in env_flags:
        if flag in os.environ:
            attributes[f"nccl.env.{flag.lower()}"] = os.environ.get(flag)

    # Detect platform
    if os.path.exists("/kaggle"):
        attributes["platform"] = "kaggle"
    elif "COLAB_GPU" in os.environ:
        attributes["platform"] = "colab"
    else:
        attributes["platform"] = "local"

    # Enrich with llama.cpp build info (from llama-server --version)
    if llama_server_path:
        attributes["llama_cpp.server_path"] = str(llama_server_path)
    version_info = _llama_server_version_info(llama_server_path)
    if version_info:
        attributes.update(version_info)

    # Enrich with llama.cpp server properties (from /props)
    props = _llama_server_props(llama_server_url)
    if isinstance(props, dict) and props:
        model_path = props.get("model_path")
        if model_path:
            attributes["llama_cpp.model_path"] = model_path
        total_slots = props.get("total_slots")
        if total_slots is not None:
            attributes["llama_cpp.total_slots"] = total_slots
        build_info = props.get("build_info")
        if build_info:
            attributes["llama_cpp.build_info"] = build_info
        modalities = props.get("modalities")
        if isinstance(modalities, dict):
            for key, value in modalities.items():
                attributes[f"llama_cpp.modalities.{key}"] = value

    try:
        from opentelemetry.sdk.resources import Resource
        return Resource.create(attributes)
    except ImportError:
        return attributes
