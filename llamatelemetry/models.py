"""
llamatelemetry.models - Model Management and Discovery

This module provides utilities for:
- Discovering local GGUF models
- Downloading models from HuggingFace with confirmation
- Getting model metadata and information
- Managing model collections
- Recommending optimal settings per model

Examples:
    Smart model loading (auto-download from registry):
    >>> from llamatelemetry.models import load_model_smart
    >>> model_path = load_model_smart("gemma-3-1b-Q4_K_M")  # Asks for confirmation

    List registry models:
    >>> from llamatelemetry.models import list_registry_models
    >>> models = list_registry_models()

    Download from HuggingFace:
    >>> from llamatelemetry.models import download_model
    >>> path = download_model("TheBloke/Llama-2-7B-GGUF", "llama-2-7b.Q4_K_M.gguf")

    Get model info:
    >>> from llamatelemetry.models import ModelInfo
    >>> info = ModelInfo.from_file("model.gguf")
    >>> print(info.get_recommended_settings())
"""

from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import struct
import shutil
import json
import os
import subprocess


class ModelInfo:
    """
    GGUF model information extractor.

    Reads metadata from GGUF files to provide model information
    and recommend optimal inference settings.

    Examples:
        >>> info = ModelInfo.from_file("gemma-3-1b-it-Q4_K_M.gguf")
        >>> print(f"Architecture: {info.architecture}")
        >>> print(f"Parameter count: {info.parameter_count}")
        >>> settings = info.get_recommended_settings(vram_gb=1)
        >>> print(settings)
    """

    def __init__(self, filepath: str):
        """
        Initialize model info from GGUF file.

        Args:
            filepath: Path to GGUF model file
        """
        self.filepath = Path(filepath)
        self.metadata: Dict[str, Any] = {}
        self.architecture: Optional[str] = None
        self.parameter_count: Optional[int] = None
        self.context_length: Optional[int] = None
        self.embedding_length: Optional[int] = None
        self.quantization: Optional[str] = None
        self.file_size_mb: float = 0.0

        if self.filepath.exists():
            self.file_size_mb = self.filepath.stat().st_size / (1024 * 1024)
            self._parse_metadata()

    def _parse_metadata(self):
        """Parse GGUF metadata from file."""
        try:
            with open(self.filepath, 'rb') as f:
                # Read GGUF magic and version
                magic = f.read(4)
                if magic != b'GGUF':
                    return

                version = struct.unpack('<I', f.read(4))[0]

                # Read tensor and metadata counts
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

                # For simplicity, we'll use gguf library if available
                try:
                    from gguf import GGUFReader
                    reader = GGUFReader(self.filepath)

                    # Extract common metadata
                    for field in reader.fields.values():
                        self.metadata[field.name] = field.parts[field.data[0]] if field.data else None

                    # Extract key information
                    self.architecture = self.metadata.get('general.architecture', 'unknown')
                    self.context_length = self.metadata.get(f'{self.architecture}.context_length')
                    self.embedding_length = self.metadata.get(f'{self.architecture}.embedding_length')
                    self.quantization = self.metadata.get('general.file_type', 'unknown')

                    # Estimate parameter count from embedding length and block count
                    block_count = self.metadata.get(f'{self.architecture}.block_count', 0)
                    if self.embedding_length and block_count:
                        # Rough estimate: params ≈ embedding_dim^2 * layers * 12 (for transformer)
                        self.parameter_count = int((self.embedding_length ** 2) * block_count * 12 / 1e9)

                except ImportError:
                    # Fallback to basic parsing
                    self.architecture = "unknown"

        except Exception:
            pass

    def get_recommended_settings(self, vram_gb: float = 8.0) -> Dict[str, Any]:
        """
        Get recommended inference settings based on model and hardware.

        Args:
            vram_gb: Available VRAM in GB

        Returns:
            Dictionary with recommended settings
        """
        model_size_gb = self.file_size_mb / 1024

        # Estimate GPU layers based on VRAM
        if vram_gb >= model_size_gb * 1.5:
            gpu_layers = 99  # Full GPU offload
            ctx_size = self.context_length or 4096
            batch_size = 2048
            ubatch_size = 512
        elif vram_gb >= model_size_gb:
            gpu_layers = 40  # Most layers
            ctx_size = min(self.context_length or 2048, 2048)
            batch_size = 1024
            ubatch_size = 256
        elif vram_gb >= model_size_gb * 0.5:
            gpu_layers = 20  # Some layers
            ctx_size = 1024
            batch_size = 512
            ubatch_size = 128
        elif vram_gb >= model_size_gb * 0.3:
            gpu_layers = 10  # Few layers
            ctx_size = 512
            batch_size = 512
            ubatch_size = 128
        else:
            gpu_layers = 0  # CPU only
            ctx_size = 512
            batch_size = 256
            ubatch_size = 64

        return {
            "gpu_layers": gpu_layers,
            "ctx_size": ctx_size,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            "recommended_vram_gb": model_size_gb * 1.5,
            "min_vram_gb": model_size_gb * 0.3
        }

    @classmethod
    def from_file(cls, filepath: str) -> 'ModelInfo':
        """Create ModelInfo from file path."""
        return cls(filepath)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filepath": str(self.filepath),
            "filename": self.filepath.name,
            "file_size_mb": self.file_size_mb,
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "context_length": self.context_length,
            "embedding_length": self.embedding_length,
            "quantization": self.quantization,
            "metadata": self.metadata
        }

    def __repr__(self) -> str:
        return f"ModelInfo(file='{self.filepath.name}', arch='{self.architecture}', size={self.file_size_mb:.1f}MB)"


def list_models(directories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    List all GGUF models in specified directories.

    Args:
        directories: List of directories to search (default: auto-detect)

    Returns:
        List of model information dictionaries
    """
    from .utils import find_gguf_models

    if directories is None:
        # Use default search locations
        model_paths = find_gguf_models()
    else:
        model_paths = []
        for directory in directories:
            dir_path = Path(directory)
            if dir_path.exists():
                model_paths.extend(dir_path.glob("*.gguf"))

    models = []
    for path in model_paths:
        try:
            info = ModelInfo.from_file(str(path))
            models.append(info.to_dict())
        except Exception:
            # Add basic info if parsing fails
            models.append({
                "filepath": str(path),
                "filename": path.name,
                "file_size_mb": path.stat().st_size / (1024 * 1024),
                "architecture": "unknown",
                "parameter_count": None,
                "context_length": None,
                "embedding_length": None,
                "quantization": "unknown",
                "metadata": {}
            })

    return sorted(models, key=lambda x: x['file_size_mb'], reverse=True)


def download_model(
    repo_id: str,
    filename: str,
    output_dir: Optional[str] = None,
    show_progress: bool = True
) -> str:
    """
    Download GGUF model from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID (e.g., "TheBloke/Llama-2-7B-GGUF")
        filename: Model filename to download
        output_dir: Output directory (default: current directory)
        show_progress: Show download progress

    Returns:
        Path to downloaded model file

    Examples:
        >>> path = download_model(
        ...     "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        ...     "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        ... )
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub required for model downloads. "
            "Install with: pip install huggingface_hub"
        )

    if output_dir is None:
        output_dir = os.getcwd()

    print(f"Downloading {filename} from {repo_id}...")

    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=output_dir,
            resume_download=True
        )

        print(f"✓ Downloaded to: {model_path}")
        return model_path

    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")


def get_model_recommendations(vram_gb: float = 8.0) -> List[Dict[str, str]]:
    """
    Get recommended models based on available VRAM.

    Args:
        vram_gb: Available VRAM in GB

    Returns:
        List of recommended model configurations
    """
    recommendations = []

    if vram_gb >= 24:
        recommendations.extend([
            {
                "name": "Llama 3.1 70B Q4_K_M",
                "repo": "lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF",
                "file": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
                "size_gb": 42,
                "description": "Very large, highly capable model"
            },
            {
                "name": "Mixtral 8x7B Q4_K_M",
                "repo": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
                "file": "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
                "size_gb": 26,
                "description": "Mixture of experts, excellent performance"
            }
        ])
    elif vram_gb >= 12:
        recommendations.extend([
            {
                "name": "Llama 3.1 8B Q5_K_M",
                "repo": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                "file": "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
                "size_gb": 6.0,
                "description": "High quality 8B model"
            },
            {
                "name": "Mistral 7B Instruct Q5_K_M",
                "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                "file": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
                "size_gb": 5.1,
                "description": "Excellent instruction-following"
            }
        ])
    elif vram_gb >= 6:
        recommendations.extend([
            {
                "name": "Llama 3.1 8B Q4_K_M",
                "repo": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                "size_gb": 4.9,
                "description": "Good quality with lower VRAM usage"
            },
            {
                "name": "Phi-3 Medium Q5_K_M",
                "repo": "microsoft/Phi-3-medium-4k-instruct-gguf",
                "file": "Phi-3-medium-4k-instruct-q5_k_m.gguf",
                "size_gb": 5.4,
                "description": "Microsoft's efficient model"
            }
        ])
    elif vram_gb >= 3:
        recommendations.extend([
            {
                "name": "Phi-3 Mini Q5_K_M",
                "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
                "file": "Phi-3-mini-4k-instruct-q5_k_m.gguf",
                "size_gb": 2.5,
                "description": "Small but capable model"
            },
            {
                "name": "Gemma 2 2B Q5_K_M",
                "repo": "google/gemma-2-2b-it-GGUF",
                "file": "2b_it_v2.gguf",
                "size_gb": 1.8,
                "description": "Google's efficient small model"
            }
        ])
    else:  # < 3 GB VRAM
        recommendations.extend([
            {
                "name": "Gemma 2B Q4_K_M",
                "repo": "google/gemma-2b-it-GGUF",
                "file": "gemma-2b-it.Q4_K_M.gguf",
                "size_gb": 1.5,
                "description": "Tiny but functional"
            },
            {
                "name": "Phi-3 Mini Q4_K_M",
                "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
                "file": "Phi-3-mini-4k-instruct-q4_k_m.gguf",
                "size_gb": 2.2,
                "description": "Small efficient model"
            },
            {
                "name": "TinyLlama 1.1B Q5_K_M",
                "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "file": "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
                "size_gb": 0.8,
                "description": "Very small, for testing"
            }
        ])

    return recommendations


def print_model_catalog(vram_gb: Optional[float] = None):
    """
    Print catalog of recommended models.

    Args:
        vram_gb: Optional VRAM to filter recommendations
    """
    if vram_gb is None:
        # Try to detect VRAM
        try:
            from .utils import detect_cuda
            cuda_info = detect_cuda()
            if cuda_info['available'] and cuda_info['gpus']:
                gpu = cuda_info['gpus'][0]
                mem_str = gpu['memory'].split()[0]  # e.g., "1024 MiB"
                if 'GiB' in gpu['memory']:
                    vram_gb = float(mem_str)
                else:  # MiB
                    vram_gb = float(mem_str) / 1024
        except Exception:
            vram_gb = 8.0  # Default assumption

    recommendations = get_model_recommendations(vram_gb)

    print("=" * 80)
    print(f"Recommended Models for {vram_gb:.1f} GB VRAM")
    print("=" * 80)

    for i, model in enumerate(recommendations, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   Repository: {model['repo']}")
        print(f"   Filename: {model['file']}")
        print(f"   Size: {model['size_gb']:.1f} GB")
        print(f"   Description: {model['description']}")
        print(f"   Download command:")
        print(f"   >>> from llamatelemetry.models import download_model")
        print(f"   >>> download_model('{model['repo']}', '{model['file']}')")

    print("\n" + "=" * 80)


class ModelManager:
    """
    Manages a collection of models with metadata.

    Provides utilities to organize, search, and select models
    from a local collection.

    Examples:
        >>> manager = ModelManager()
        >>> manager.scan_directories(["/path/to/models"])
        >>> models = manager.find_by_size(max_gb=2.0)
        >>> model = manager.get_best_for_vram(vram_gb=4.0)
    """

    def __init__(self, directories: Optional[List[str]] = None):
        """
        Initialize model manager.

        Args:
            directories: Directories to scan for models
        """
        self.models: List[ModelInfo] = []
        if directories:
            self.scan_directories(directories)

    def scan_directories(self, directories: List[str]):
        """
        Scan directories for GGUF models.

        Args:
            directories: List of directory paths
        """
        self.models = []

        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                continue

            for gguf_file in dir_path.glob("**/*.gguf"):
                try:
                    info = ModelInfo.from_file(str(gguf_file))
                    self.models.append(info)
                except Exception:
                    pass

    def find_by_size(self, min_mb: float = 0, max_mb: float = float('inf')) -> List[ModelInfo]:
        """Find models within size range."""
        return [
            m for m in self.models
            if min_mb <= m.file_size_mb <= max_mb
        ]

    def find_by_architecture(self, architecture: str) -> List[ModelInfo]:
        """Find models by architecture."""
        return [
            m for m in self.models
            if m.architecture and architecture.lower() in m.architecture.lower()
        ]

    def get_best_for_vram(self, vram_gb: float) -> Optional[ModelInfo]:
        """
        Get best model that fits in given VRAM.

        Args:
            vram_gb: Available VRAM in GB

        Returns:
            Best fitting ModelInfo or None
        """
        max_size_mb = vram_gb * 1024 * 0.7  # Use 70% of VRAM for model

        fitting_models = self.find_by_size(max_mb=max_size_mb)

        if not fitting_models:
            return None

        # Return largest model that fits
        return max(fitting_models, key=lambda m: m.file_size_mb)

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self):
        return iter(self.models)


# ============================================================================
# Smart Model Loading with Registry and Auto-Download
# ============================================================================

def load_model_smart(
    model_name_or_path: str,
    cache_dir: Optional[Path] = None,
    interactive: bool = True,
    force_download: bool = False
) -> Path:
    """
    Smart model loader with auto-download and confirmation.

    Handles three cases:
    1. Local path exists → returns path
    2. Model name in registry → downloads from HuggingFace (with confirmation)
    3. HuggingFace syntax "repo:file" → downloads directly

    Args:
        model_name_or_path: Model name from registry, local path, or "repo:file"
        cache_dir: Cache directory (default: llamatelemetry/models/)
        interactive: Ask for confirmation before downloading
        force_download: Re-download even if cached

    Returns:
        Path to model file

    Raises:
        ValueError: If model not found or download cancelled
        FileNotFoundError: If local path doesn't exist

    Examples:
        >>> # From registry (auto-downloads)
        >>> path = load_model_smart("gemma-3-1b-Q4_K_M")

        >>> # Local path
        >>> path = load_model_smart("/path/to/model.gguf")

        >>> # HuggingFace syntax
        >>> path = load_model_smart("unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf")
    """
    from ._internal.registry import MODEL_REGISTRY, get_model_info

    # Get default cache directory
    if cache_dir is None:
        # Import at runtime to avoid circular dependency
        from . import _MODEL_CACHE
        cache_dir = _MODEL_CACHE

    # Case 1: Local path exists
    path_obj = Path(model_name_or_path).expanduser()
    if path_obj.exists() and path_obj.is_file():
        print(f"✓ Using local model: {path_obj.name}")
        return path_obj

    # Case 2: Model in registry
    if model_name_or_path in MODEL_REGISTRY:
        model_info = get_model_info(model_name_or_path)
        cached_path = cache_dir / model_info['file']

        # Check if already cached
        if cached_path.exists() and not force_download:
            print(f"✓ Using cached model: {cached_path.name}")
            return cached_path

        # Ask for confirmation
        if interactive:
            print("\n" + "=" * 70)
            print(f"Model: {model_name_or_path}")
            print(f"Description: {model_info['description']}")
            print(f"Size: {model_info['size_mb']} MB (~{model_info['size_mb']/1024:.1f} GB)")
            print(f"Minimum VRAM: {model_info['min_vram_gb']} GB")
            print(f"Source: https://huggingface.co/{model_info['repo']}")
            print(f"Cache location: {cached_path}")
            print("=" * 70)

            response = input("\nDownload this model? [Y/n]: ").strip().lower()
            if response and response not in ['y', 'yes']:
                print("\n❌ Model download cancelled by user")
                print("   To proceed, re-run with 'Y' or pre-download the model manually")
                return None

        # Download
        print(f"\nDownloading {model_info['file']}...")
        print(f"This may take a while ({model_info['size_mb']} MB)...")

        try:
            from huggingface_hub import hf_hub_download
            from tqdm import tqdm

            downloaded_path = hf_hub_download(
                repo_id=model_info['repo'],
                filename=model_info['file'],
                cache_dir=str(cache_dir),
                resume_download=True,
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False
            )

            # Ensure it's in our cache directory
            downloaded_path_obj = Path(downloaded_path)
            if downloaded_path_obj != cached_path:
                # Copy or move to cache
                if not cached_path.exists():
                    shutil.copy(downloaded_path_obj, cached_path)

            print(f"✓ Model downloaded: {cached_path.name}")
            return cached_path

        except ImportError:
            raise ImportError(
                "huggingface_hub required for model downloads. "
                "Install with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")

    # Case 3: HuggingFace syntax "repo:file"
    if '/' in model_name_or_path and ':' in model_name_or_path:
        try:
            repo_id, filename = model_name_or_path.split(':', 1)
        except ValueError:
            raise ValueError(
                f"Invalid HuggingFace format. Use 'repo/id:filename.gguf', "
                f"got: {model_name_or_path}"
            )

        cached_path = cache_dir / filename

        # Check if already cached
        if cached_path.exists() and not force_download:
            print(f"✓ Using cached model: {cached_path.name}")
            return cached_path

        # Ask for confirmation
        if interactive:
            print("\n" + "=" * 70)
            print(f"Repository: {repo_id}")
            print(f"File: {filename}")
            print(f"Cache location: {cached_path}")
            print("=" * 70)

            response = input("\nDownload this model? [Y/n]: ").strip().lower()
            if response and response not in ['y', 'yes']:
                print("\n❌ Model download cancelled by user")
                print("   To proceed, re-run with 'Y' or pre-download the model manually")
                return None

        # Download
        print(f"\nDownloading {filename} from {repo_id}...")

        try:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(cache_dir),
                resume_download=True,
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False
            )

            downloaded_path_obj = Path(downloaded_path)
            if downloaded_path_obj != cached_path:
                if not cached_path.exists():
                    shutil.copy(downloaded_path_obj, cached_path)

            print(f"✓ Model downloaded: {cached_path.name}")
            return cached_path

        except ImportError:
            raise ImportError(
                "huggingface_hub required for model downloads. "
                "Install with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")

    # Case 4: Not found
    raise ValueError(
        f"Model not found: '{model_name_or_path}'\n\n"
        f"Available options:\n"
        f"  1. Use a model name from registry: {', '.join(list(MODEL_REGISTRY.keys())[:5])}...\n"
        f"  2. Provide a local file path: /path/to/model.gguf\n"
        f"  3. Use HuggingFace syntax: repo/id:filename.gguf"
    )


def list_registry_models() -> Dict[str, Dict[str, Any]]:
    """
    List all models available in the registry.

    Returns:
        Dictionary of model name -> model info

    Example:
        >>> from llamatelemetry.models import list_registry_models
        >>> models = list_registry_models()
        >>> for name, info in models.items():
        ...     print(f"{name}: {info['description']}")
    """
    from ._internal.registry import MODEL_REGISTRY
    return MODEL_REGISTRY.copy()


def print_registry_models(vram_gb: Optional[float] = None):
    """
    Print formatted list of registry models.

    Args:
        vram_gb: Optional VRAM filter - only show compatible models

    Example:
        >>> from llamatelemetry.models import print_registry_models
        >>> print_registry_models(vram_gb=1.0)  # Show models for 1GB VRAM
    """
    from ._internal.registry import MODEL_REGISTRY, find_models_by_vram

    if vram_gb is not None:
        models = find_models_by_vram(vram_gb)
        print(f"Models compatible with {vram_gb} GB VRAM:")
        print("=" * 80)
    else:
        models = MODEL_REGISTRY
        print("All available models in registry:")
        print("=" * 80)

    if not models:
        print(f"No models found for {vram_gb} GB VRAM")
        print(f"Try increasing VRAM or check models with print_registry_models()")
        return

    for i, (name, info) in enumerate(models.items(), 1):
        print(f"\n{i}. {name}")
        print(f"   Description: {info['description']}")
        print(f"   Size: {info['size_mb']} MB (~{info['size_mb']/1024:.1f} GB)")
        print(f"   Min VRAM: {info['min_vram_gb']} GB")
        print(f"   Repository: {info['repo']}")
        print(f"   \n   Usage: engine.load_model('{name}')")

    print("\n" + "=" * 80)
