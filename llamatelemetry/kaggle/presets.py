"""
llamatelemetry.kaggle.presets - Server presets for common Kaggle configurations.

Replaces verbose manual configuration with type-safe enums and presets.

Before:
    server.start_server(
        model_path=model_path,
        host="127.0.0.1", port=8080,
        gpu_layers=99, ctx_size=4096, batch_size=512,
        tensor_split="0.5,0.5"  # Error-prone string!
    )

After:
    engine = env.create_engine(model, preset=ServerPreset.KAGGLE_DUAL_T4)
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


class TensorSplitMode(Enum):
    """
    GPU tensor split modes for multi-GPU inference.

    Replaces error-prone string values like "0.5,0.5" with type-safe enums.

    Example:
        >>> from llamatelemetry.kaggle import TensorSplitMode
        >>> mode = TensorSplitMode.DUAL_50_50
        >>> print(mode.to_string())  # "0.5,0.5"
    """
    NONE = "none"              # Single GPU
    EQUAL = "equal"            # Split equally across all GPUs
    BALANCED = "balanced"      # Auto-balance based on VRAM
    CUSTOM = "custom"          # User-defined split ratios

    # Preset ratios for dual GPU
    DUAL_50_50 = "0.5,0.5"
    DUAL_60_40 = "0.6,0.4"
    DUAL_70_30 = "0.7,0.3"
    DUAL_40_60 = "0.4,0.6"
    DUAL_30_70 = "0.3,0.7"

    def to_string(self) -> Optional[str]:
        """Convert to tensor_split string for llama-server."""
        if self == TensorSplitMode.NONE:
            return None
        elif self == TensorSplitMode.EQUAL:
            return "0.5,0.5"
        elif self == TensorSplitMode.BALANCED:
            return "0.5,0.5"  # Default to equal, auto_config will adjust
        elif self.value.startswith("0."):
            return self.value
        return None

    def to_list(self) -> Optional[List[float]]:
        """Convert to list of floats."""
        string = self.to_string()
        if string is None:
            return None
        return [float(x) for x in string.split(",")]


class ServerPreset(Enum):
    """
    Pre-configured server settings for common environments.

    Example:
        >>> from llamatelemetry.kaggle import ServerPreset, get_preset_config
        >>> config = get_preset_config(ServerPreset.KAGGLE_DUAL_T4)
        >>> print(config.ctx_size)  # 8192
    """
    AUTO = auto()              # Auto-detect optimal settings
    KAGGLE_DUAL_T4 = auto()    # 2x T4, 30GB total, split-GPU
    KAGGLE_SINGLE_T4 = auto()  # 1x T4, 15GB
    COLAB_T4 = auto()          # Colab T4 (legacy)
    COLAB_A100 = auto()        # Colab A100
    LOCAL_3090 = auto()        # RTX 3090 24GB
    LOCAL_4090 = auto()        # RTX 4090 24GB
    CPU_ONLY = auto()          # No GPU


@dataclass
class PresetConfig:
    """
    Configuration for a server preset.

    Attributes:
        name: Human-readable preset name
        server_url: Server URL with port
        port: Server port
        host: Server host
        gpu_layers: Number of layers to offload to GPU(s)
        tensor_split: VRAM split ratios per GPU
        split_mode: How to split model across GPUs
        main_gpu: Primary GPU device ID
        ctx_size: Context window size
        batch_size: Batch size for prompt processing
        ubatch_size: Micro-batch size
        flash_attention: Enable flash attention
        use_mmap: Use memory-mapped files
        n_parallel: Number of parallel sequences
    """
    name: str
    server_url: str = "http://127.0.0.1:8080"
    port: int = 8080
    host: str = "127.0.0.1"

    # GPU settings
    gpu_layers: int = 99
    tensor_split: Optional[List[float]] = None
    split_mode: TensorSplitMode = TensorSplitMode.NONE
    main_gpu: int = 0

    # Memory settings
    ctx_size: int = 4096
    batch_size: int = 512
    ubatch_size: int = 128

    # Performance
    flash_attention: bool = True
    use_mmap: bool = True

    # Parallel processing
    n_parallel: int = 1

    def to_load_kwargs(self) -> Dict[str, Any]:
        """
        Convert to kwargs for InferenceEngine.load_model().

        Returns:
            Dictionary of keyword arguments
        """
        kwargs = {
            "gpu_layers": self.gpu_layers,
            "ctx_size": self.ctx_size,
            "batch_size": self.batch_size,
            "ubatch_size": self.ubatch_size,
            "n_parallel": self.n_parallel,
        }

        if self.tensor_split:
            kwargs["tensor_split"] = ",".join(str(v) for v in self.tensor_split)
            kwargs["split_mode"] = "layer"

        if self.flash_attention:
            kwargs["flash_attn"] = True

        return kwargs

    def to_server_kwargs(self) -> Dict[str, Any]:
        """
        Convert to kwargs for ServerManager.start_server().

        Returns:
            Dictionary of keyword arguments
        """
        kwargs = self.to_load_kwargs()
        kwargs["port"] = self.port
        kwargs["host"] = self.host
        return kwargs


# Preset configurations
PRESET_CONFIGS: Dict[ServerPreset, PresetConfig] = {
    ServerPreset.KAGGLE_DUAL_T4: PresetConfig(
        name="Kaggle 2x T4",
        port=8080,
        gpu_layers=99,
        tensor_split=[0.5, 0.5],
        split_mode=TensorSplitMode.DUAL_50_50,
        ctx_size=8192,
        batch_size=2048,
        ubatch_size=512,
        flash_attention=True,
        n_parallel=4,
    ),
    ServerPreset.KAGGLE_SINGLE_T4: PresetConfig(
        name="Kaggle 1x T4",
        port=8080,
        gpu_layers=99,
        tensor_split=None,
        split_mode=TensorSplitMode.NONE,
        ctx_size=4096,
        batch_size=1024,
        ubatch_size=256,
        flash_attention=True,
        n_parallel=2,
    ),
    ServerPreset.COLAB_T4: PresetConfig(
        name="Colab T4",
        port=8080,
        gpu_layers=99,
        tensor_split=None,
        split_mode=TensorSplitMode.NONE,
        ctx_size=4096,
        batch_size=1024,
        ubatch_size=256,
        flash_attention=True,
        n_parallel=2,
    ),
    ServerPreset.COLAB_A100: PresetConfig(
        name="Colab A100",
        port=8080,
        gpu_layers=99,
        tensor_split=None,
        split_mode=TensorSplitMode.NONE,
        ctx_size=16384,
        batch_size=4096,
        ubatch_size=1024,
        flash_attention=True,
        n_parallel=8,
    ),
    ServerPreset.LOCAL_3090: PresetConfig(
        name="Local RTX 3090",
        port=8080,
        gpu_layers=99,
        tensor_split=None,
        split_mode=TensorSplitMode.NONE,
        ctx_size=8192,
        batch_size=2048,
        ubatch_size=512,
        flash_attention=True,
        n_parallel=4,
    ),
    ServerPreset.LOCAL_4090: PresetConfig(
        name="Local RTX 4090",
        port=8080,
        gpu_layers=99,
        tensor_split=None,
        split_mode=TensorSplitMode.NONE,
        ctx_size=16384,
        batch_size=4096,
        ubatch_size=1024,
        flash_attention=True,
        n_parallel=8,
    ),
    ServerPreset.CPU_ONLY: PresetConfig(
        name="CPU Only",
        port=8080,
        gpu_layers=0,
        tensor_split=None,
        split_mode=TensorSplitMode.NONE,
        ctx_size=2048,
        batch_size=512,
        ubatch_size=128,
        flash_attention=False,
        n_parallel=1,
    ),
}


def get_preset_config(preset: ServerPreset) -> PresetConfig:
    """
    Get configuration for a preset.

    Args:
        preset: ServerPreset enum value

    Returns:
        PresetConfig for the specified preset

    Example:
        >>> config = get_preset_config(ServerPreset.KAGGLE_DUAL_T4)
        >>> print(config.tensor_split)  # [0.5, 0.5]
    """
    if preset == ServerPreset.AUTO:
        return _auto_detect_preset()

    return PRESET_CONFIGS.get(preset, _auto_detect_preset())


def _auto_detect_preset() -> PresetConfig:
    """Auto-detect best preset based on available hardware."""
    try:
        from ..api.multigpu import detect_gpus
        gpus = detect_gpus()

        if not gpus:
            return PRESET_CONFIGS[ServerPreset.CPU_ONLY]

        # Check for dual T4 (Kaggle)
        if len(gpus) >= 2 and all("T4" in g.name for g in gpus):
            return PRESET_CONFIGS[ServerPreset.KAGGLE_DUAL_T4]

        # Check for single T4
        if len(gpus) == 1 and "T4" in gpus[0].name:
            return PRESET_CONFIGS[ServerPreset.KAGGLE_SINGLE_T4]

        # Check for A100
        if any("A100" in g.name for g in gpus):
            return PRESET_CONFIGS[ServerPreset.COLAB_A100]

        # Check for 4090
        if any("4090" in g.name for g in gpus):
            return PRESET_CONFIGS[ServerPreset.LOCAL_4090]

        # Check for 3090
        if any("3090" in g.name for g in gpus):
            return PRESET_CONFIGS[ServerPreset.LOCAL_3090]

        # Default to single T4 settings for other GPUs
        return PRESET_CONFIGS[ServerPreset.KAGGLE_SINGLE_T4]

    except Exception:
        return PRESET_CONFIGS[ServerPreset.CPU_ONLY]


__all__ = [
    "TensorSplitMode",
    "ServerPreset",
    "PresetConfig",
    "get_preset_config",
    "PRESET_CONFIGS",
]
