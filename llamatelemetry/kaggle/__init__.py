"""
llamatelemetry.kaggle - Kaggle-specific utilities for zero-boilerplate setup.

This module provides utilities optimized for Kaggle notebooks with
dual Tesla T4 GPUs (30GB total VRAM).

Quick Start:
    >>> from llamatelemetry.kaggle import KaggleEnvironment
    >>>
    >>> # One-liner setup (replaces 50+ lines of boilerplate)
    >>> env = KaggleEnvironment.setup()
    >>>
    >>> # Create engine with optimal settings
    >>> engine = env.create_engine("gemma-3-4b-Q4_K_M")
    >>> result = engine.infer("Hello!")
    >>> print(result.text)
    >>>
    >>> # RAPIDS on GPU 1
    >>> with env.rapids_context():
    ...     import cudf
    ...     df = cudf.DataFrame({"x": [1, 2, 3]})

Available Classes:
    - KaggleEnvironment: Main entry point for Kaggle setup
    - ServerPreset: Pre-configured server settings
    - TensorSplitMode: GPU tensor split configurations
    - KaggleSecrets: Auto-load Kaggle secrets
    - GPUContext: GPU isolation context manager

Available Functions:
    - quick_setup(): Alias for KaggleEnvironment.setup()
    - auto_load_secrets(): Load all known secrets
    - rapids_gpu(): Context manager for RAPIDS on specific GPU
    - llm_gpu(): Context manager for LLM inference GPUs
"""

from .environment import KaggleEnvironment, quick_setup
from .presets import (
    ServerPreset,
    TensorSplitMode,
    PresetConfig,
    get_preset_config,
    PRESET_CONFIGS,
)
from .secrets import (
    KaggleSecrets,
    auto_load_secrets,
    setup_huggingface_auth,
    setup_graphistry_auth,
)
from .gpu_context import (
    GPUContext,
    rapids_gpu,
    llm_gpu,
    single_gpu,
    get_current_gpu_context,
    set_gpu_for_rapids,
    reset_gpu_context,
)

__all__ = [
    # Main entry point
    "KaggleEnvironment",
    "quick_setup",

    # Presets
    "ServerPreset",
    "TensorSplitMode",
    "PresetConfig",
    "get_preset_config",
    "PRESET_CONFIGS",

    # Secrets
    "KaggleSecrets",
    "auto_load_secrets",
    "setup_huggingface_auth",
    "setup_graphistry_auth",

    # GPU Context
    "GPUContext",
    "rapids_gpu",
    "llm_gpu",
    "single_gpu",
    "get_current_gpu_context",
    "set_gpu_for_rapids",
    "reset_gpu_context",
]
