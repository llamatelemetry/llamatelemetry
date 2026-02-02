"""
llamatelemetry.utils - Utility Functions

Helper functions for llamatelemetry package including installation helpers,
environment detection, and configuration management.
"""

from typing import Optional, Dict, Any, List
import os
import subprocess
import platform
from pathlib import Path


def detect_cuda() -> Dict[str, Any]:
    """
    Detect CUDA installation and GPU information.

    Returns:
        Dictionary with CUDA information:
        - available: bool - Whether CUDA is available
        - version: str - CUDA version
        - gpus: list - List of GPU information dictionaries
    """
    info = {
        'available': False,
        'version': None,
        'gpus': []
    }

    try:
        # Check nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            info['available'] = True

            # Parse GPU information
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        info['gpus'].append({
                            'name': parts[0],
                            'memory': parts[1],
                            'driver_version': parts[2],
                            'compute_capability': parts[3]
                        })

            # Try to get CUDA version
            try:
                nvcc_result = subprocess.run(
                    ['nvcc', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if nvcc_result.returncode == 0:
                    # Parse version from output like "Cuda compilation tools, release 12.0, V12.0.140"
                    for line in nvcc_result.stdout.split('\n'):
                        if 'release' in line.lower():
                            parts = line.split('release')
                            if len(parts) > 1:
                                version_str = parts[1].strip().split(',')[0].strip()
                                info['version'] = version_str
                                break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return info


def check_gpu_compatibility(min_compute_cap: float = 5.0) -> Dict[str, Any]:
    """
    Check if GPU is compatible with llamatelemetry binaries.

    llamatelemetry binaries are compiled for compute capability 5.0+ (Maxwell and newer):
    - 5.0: Maxwell (GTX 900 series, Tesla M40)
    - 6.x: Pascal (GTX 10xx, Tesla P100)
    - 7.0: Volta (Tesla V100)
    - 7.5: Turing (RTX 20xx, Tesla T4, GTX 16xx)
    - 8.0: Ampere (RTX 30xx, A100)
    - 8.6: Ampere (RTX 30xx)
    - 8.9: Ada Lovelace (RTX 40xx)

    Args:
        min_compute_cap: Minimum compute capability required (default: 5.0)

    Returns:
        Dictionary with compatibility information:
        - compatible: bool - Whether GPU is compatible
        - compute_capability: float - GPU compute capability
        - gpu_name: str - GPU name
        - reason: str - Explanation if not compatible
        - platform: str - Detected platform (local/colab/kaggle)
    """
    result = {
        'compatible': False,
        'compute_capability': None,
        'gpu_name': None,
        'reason': None,
        'platform': 'local'
    }

    # Detect platform
    if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
        result['platform'] = 'colab'
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        result['platform'] = 'kaggle'

    cuda_info = detect_cuda()

    if not cuda_info['available']:
        result['reason'] = 'No CUDA GPU detected. Please ensure NVIDIA drivers are installed.'
        return result

    if not cuda_info['gpus']:
        result['reason'] = 'CUDA available but no GPUs found.'
        return result

    # Check first GPU (most common case)
    gpu = cuda_info['gpus'][0]
    result['gpu_name'] = gpu['name']

    try:
        compute_cap = float(gpu['compute_capability'])
        result['compute_capability'] = compute_cap

        if compute_cap < min_compute_cap:
            result['reason'] = (
                f"GPU compute capability {compute_cap} is below minimum required {min_compute_cap}. "
                f"llamatelemetry requires Maxwell architecture or newer (compute capability 5.0+)."
            )
            return result

        # All checks passed
        result['compatible'] = True
        result['reason'] = f"GPU {gpu['name']} (compute capability {compute_cap}) is compatible."

    except (ValueError, TypeError):
        result['reason'] = f"Could not parse compute capability: {gpu.get('compute_capability')}"
        return result

    return result


def get_llama_cpp_cuda_path() -> Optional[Path]:
    """
    Get the path to Ubuntu-Cuda-Llama.cpp-Executable installation if it exists.

    Returns:
        Path to Ubuntu-Cuda-Llama.cpp-Executable directory, or None if not found
    """
    # Check environment variable first
    env_path = os.getenv('LLAMA_CPP_DIR')
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # Check common locations
    possible_paths = [
        Path.home() / 'Ubuntu-Cuda-Llama.cpp-Executable',
        Path.cwd() / 'Ubuntu-Cuda-Llama.cpp-Executable',
        Path('/opt/Ubuntu-Cuda-Llama.cpp-Executable'),
        Path.home() / 'llama.cpp' / 'build',  # Common build directory
    ]

    for path in possible_paths:
        if path.exists() and (path / 'bin' / 'llama-server').exists():
            return path

    return None


def setup_environment() -> Dict[str, str]:
    """
    Setup environment variables for optimal llamatelemetry operation.

    Returns:
        Dictionary of environment variables that were set
    """
    env_vars = {}

    # Set Ubuntu-Cuda-Llama.cpp-Executable path if found
    llama_cpp_path = get_llama_cpp_cuda_path()
    if llama_cpp_path and 'LLAMA_CPP_DIR' not in os.environ:
        os.environ['LLAMA_CPP_DIR'] = str(llama_cpp_path)
        env_vars['LLAMA_CPP_DIR'] = str(llama_cpp_path)

        # Also set library path for Linux
        if platform.system() == 'Linux':
            lib_path = llama_cpp_path / 'lib'
            if lib_path.exists():
                ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
                new_path = f"{lib_path}:{ld_library_path}" if ld_library_path else str(lib_path)
                os.environ['LD_LIBRARY_PATH'] = new_path
                env_vars['LD_LIBRARY_PATH'] = new_path

    # Set CUDA-related environment variables for optimal performance
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        # Use all available GPUs by default
        cuda_info = detect_cuda()
        if cuda_info['available'] and cuda_info['gpus']:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(len(cuda_info['gpus'])))
            env_vars['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']

    return env_vars


def find_gguf_models(directory: str = None) -> List[Path]:
    """
    Find GGUF model files in a directory.

    Args:
        directory: Directory to search (default: Ubuntu-Cuda-Llama.cpp-Executable/bin and current directory)

    Returns:
        List of paths to GGUF model files
    """
    models = []

    # Directories to search
    search_dirs = []

    if directory:
        search_dirs.append(Path(directory))
    else:
        # Default search locations
        llama_cpp_path = get_llama_cpp_cuda_path()
        if llama_cpp_path:
            search_dirs.append(llama_cpp_path / 'bin')
        search_dirs.append(Path.cwd())
        search_dirs.append(Path.home() / 'models')

    # Search for .gguf files
    for search_dir in search_dirs:
        if search_dir.exists():
            for gguf_file in search_dir.glob('*.gguf'):
                models.append(gguf_file)

    return sorted(models)


def print_system_info():
    """
    Print comprehensive system information for debugging.

    Displays:
    - Python version and executable
    - Operating system
    - CUDA availability and GPU info
    - Ubuntu-Cuda-Llama.cpp-Executable installation status
    - Available GGUF models
    """
    import sys

    print("=" * 60)
    print("llamatelemetry System Information")
    print("=" * 60)

    # Python info
    print(f"\nPython:")
    print(f"  Version: {sys.version}")
    print(f"  Executable: {sys.executable}")

    # OS info
    print(f"\nOperating System:")
    print(f"  System: {platform.system()}")
    print(f"  Release: {platform.release()}")
    print(f"  Machine: {platform.machine()}")

    # CUDA info
    cuda_info = detect_cuda()
    print(f"\nCUDA:")
    print(f"  Available: {cuda_info['available']}")
    if cuda_info['available']:
        print(f"  Version: {cuda_info['version'] or 'Unknown'}")
        print(f"  GPUs: {len(cuda_info['gpus'])}")
        for i, gpu in enumerate(cuda_info['gpus']):
            print(f"    GPU {i}: {gpu['name']}")
            print(f"      Memory: {gpu['memory']}")
            print(f"      Driver: {gpu['driver_version']}")
            print(f"      Compute: {gpu['compute_capability']}")

    # Ubuntu-Cuda-Llama.cpp-Executable info
    llama_cpp_path = get_llama_cpp_cuda_path()
    print(f"\nUbuntu-Cuda-Llama.cpp-Executable:")
    if llama_cpp_path:
        print(f"  Found: Yes")
        print(f"  Location: {llama_cpp_path}")
        server_path = llama_cpp_path / 'bin' / 'llama-server'
        print(f"  Server: {'Found' if server_path.exists() else 'Not found'}")
    else:
        print(f"  Found: No")
        print(f"  Set LLAMA_CPP_DIR environment variable to specify location")

    # GGUF models
    models = find_gguf_models()
    print(f"\nGGUF Models Found: {len(models)}")
    for model in models[:5]:  # Show first 5
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"  - {model.name} ({size_mb:.1f} MB)")
    if len(models) > 5:
        print(f"  ... and {len(models) - 5} more")

    print("\n" + "=" * 60)


def create_config_file(config_path: Optional[Path] = None) -> Path:
    """
    Create a default configuration file for llamatelemetry.

    Args:
        config_path: Path for config file (default: ~/.llamatelemetry/config.json)

    Returns:
        Path to created config file
    """
    import json

    if config_path is None:
        config_path = Path.home() / '.llamatelemetry' / 'config.json'

    config_path.parent.mkdir(parents=True, exist_ok=True)

    default_config = {
        'server': {
            'url': 'http://127.0.0.1:8090',
            'port': 8090,
            'host': '127.0.0.1',
            'auto_start': True,
        },
        'model': {
            'gpu_layers': 99,
            'ctx_size': 2048,
            'n_parallel': 1,
        },
        'inference': {
            'max_tokens': 128,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
        },
        'paths': {
            'llama_cpp_dir': str(get_llama_cpp_cuda_path()) if get_llama_cpp_cuda_path() else None,
            'models_dir': str(Path.home() / 'models'),
        }
    }

    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)

    return config_path


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from file.

    Args:
        config_path: Path to config file (default: ~/.llamatelemetry/config.json)

    Returns:
        Configuration dictionary
    """
    import json

    if config_path is None:
        config_path = Path.home() / '.llamatelemetry' / 'config.json'

    if not config_path.exists():
        # Create default config
        create_config_file(config_path)

    with open(config_path, 'r') as f:
        return json.load(f)


def get_recommended_gpu_layers(model_size_gb: float, vram_gb: float) -> int:
    """
    Get recommended number of GPU layers based on model size and available VRAM.

    Args:
        model_size_gb: Model size in GB
        vram_gb: Available VRAM in GB

    Returns:
        Recommended number of GPU layers
    """
    # Rule of thumb: model needs ~1.2x its size in VRAM for full GPU offload
    # due to KV cache and other overhead

    if vram_gb >= model_size_gb * 1.2:
        # Full GPU offload
        return 99

    # Partial offload based on available VRAM
    ratio = vram_gb / (model_size_gb * 1.2)

    if ratio >= 0.8:
        return 40  # Most layers
    elif ratio >= 0.6:
        return 30  # Many layers
    elif ratio >= 0.4:
        return 20  # Some layers
    elif ratio >= 0.2:
        return 10  # Few layers
    else:
        return 0   # CPU only


def validate_model_path(model_path: str) -> bool:
    """
    Validate that a model path exists and is a GGUF file.

    Args:
        model_path: Path to model file

    Returns:
        True if valid, False otherwise
    """
    path = Path(model_path)
    return path.exists() and path.suffix.lower() == '.gguf'


def auto_configure_for_model(model_path: Path, vram_gb: Optional[float] = None) -> Dict[str, Any]:
    """
    Auto-configure optimal inference settings based on hardware and model.

    Automatically detects GPU VRAM and recommends optimal settings for:
    - gpu_layers: Number of layers to offload to GPU
    - ctx_size: Context window size
    - batch_size: Batch size for prompt processing
    - ubatch_size: Micro-batch size (critical for low VRAM)

    Args:
        model_path: Path to GGUF model file
        vram_gb: Optional VRAM in GB (auto-detected if not provided)

    Returns:
        Dictionary with recommended settings

    Example:
        >>> from llamatelemetry.utils import auto_configure_for_model
        >>> from pathlib import Path
        >>> settings = auto_configure_for_model(Path("model.gguf"))
        >>> print(settings)
        {'gpu_layers': 20, 'ctx_size': 1024, 'batch_size': 512, 'ubatch_size': 128}
    """
    # Detect VRAM if not provided
    if vram_gb is None:
        cuda_info = detect_cuda()
        if cuda_info['available'] and cuda_info['gpus']:
            # Parse VRAM from first GPU
            gpu = cuda_info['gpus'][0]
            mem_str = gpu['memory'].split()[0]  # e.g., "1024 MiB" or "8.0 GiB"

            # Convert to GB
            if 'GiB' in gpu['memory']:
                vram_gb = float(mem_str)
            elif 'MiB' in gpu['memory']:
                vram_gb = float(mem_str) / 1024
            else:
                # Fallback: assume MiB
                vram_gb = float(mem_str) / 1024
        else:
            # No CUDA detected, use CPU-only defaults
            print("⚠ CUDA not detected, using CPU-only configuration")
            return {
                'gpu_layers': 0,
                'ctx_size': 512,
                'batch_size': 256,
                'ubatch_size': 64,
                'n_parallel': 1
            }

    # Get model information
    try:
        from .models import ModelInfo
        model_info = ModelInfo.from_file(str(model_path))
        model_size_gb = model_info.file_size_mb / 1024

        # Use ModelInfo's built-in recommendations
        settings = model_info.get_recommended_settings(vram_gb=vram_gb)

        print(f"✓ Auto-configured for {vram_gb:.1f} GB VRAM")
        print(f"  GPU Layers: {settings['gpu_layers']}")
        print(f"  Context Size: {settings['ctx_size']}")
        print(f"  Batch Size: {settings['batch_size']}")
        print(f"  Micro-batch Size: {settings['ubatch_size']}")

        return settings

    except Exception as e:
        # Fallback to basic estimation
        print(f"⚠ Could not read model metadata, using conservative defaults: {e}")

        # Conservative defaults based on VRAM
        if vram_gb >= 8:
            return {
                'gpu_layers': 99,
                'ctx_size': 4096,
                'batch_size': 2048,
                'ubatch_size': 512,
                'n_parallel': 2
            }
        elif vram_gb >= 4:
            return {
                'gpu_layers': 40,
                'ctx_size': 2048,
                'batch_size': 1024,
                'ubatch_size': 256,
                'n_parallel': 1
            }
        elif vram_gb >= 2:
            return {
                'gpu_layers': 20,
                'ctx_size': 1024,
                'batch_size': 512,
                'ubatch_size': 128,
                'n_parallel': 1
            }
        elif vram_gb >= 1:
            return {
                'gpu_layers': 8,
                'ctx_size': 512,
                'batch_size': 512,
                'ubatch_size': 128,
                'n_parallel': 1
            }
        else:
            return {
                'gpu_layers': 0,
                'ctx_size': 512,
                'batch_size': 256,
                'ubatch_size': 64,
                'n_parallel': 1
            }
