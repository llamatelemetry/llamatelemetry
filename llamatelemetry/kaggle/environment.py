"""
llamatelemetry.kaggle.environment - Zero-boilerplate Kaggle setup.

This module provides KaggleEnvironment, the main entry point for Kaggle notebooks.
It replaces 50+ lines of boilerplate with a single line:

    env = KaggleEnvironment.setup()

Features:
    - Auto-detect GPUs (count, names, VRAM, compute capability)
    - Auto-load secrets (HF_TOKEN, Graphistry keys)
    - Auto-select preset (KAGGLE_DUAL_T4, KAGGLE_SINGLE_T4)
    - Create optimally-configured InferenceEngine
    - GPU isolation for RAPIDS workloads
"""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
import os

if TYPE_CHECKING:
    from .. import InferenceEngine
    from .presets import ServerPreset
    from .gpu_context import GPUContext


@dataclass
class KaggleEnvironment:
    """
    Unified Kaggle environment configuration.

    Automatically detects:
    - GPU count and type (T4, etc.)
    - Available VRAM per GPU
    - Kaggle secrets (HF_TOKEN, Graphistry keys)
    - Optimal server presets

    Example:
        >>> # One-liner setup
        >>> env = KaggleEnvironment.setup()
        >>> print(f"GPUs: {env.gpu_count}x {env.gpu_names[0]}")
        >>> print(f"Total VRAM: {env.total_vram_gb} GB")

        >>> # Create engine with optimal settings
        >>> engine = env.create_engine("gemma-3-4b-Q4_K_M")
        >>> result = engine.infer("Hello!")

        >>> # RAPIDS on GPU 1
        >>> with env.rapids_context():
        ...     import cudf
        ...     df = cudf.DataFrame({"x": [1, 2, 3]})
    """

    # Auto-detected properties
    gpu_count: int = 0
    gpu_names: List[str] = field(default_factory=list)
    total_vram_gb: float = 0.0
    vram_per_gpu_gb: List[float] = field(default_factory=list)
    compute_capability: str = ""
    cuda_version: str = ""

    # Secrets (auto-loaded from Kaggle)
    hf_token: Optional[str] = None
    graphistry_key_id: Optional[str] = None
    graphistry_key_secret: Optional[str] = None

    # Configuration
    preset: Optional["ServerPreset"] = None
    telemetry_enabled: bool = True
    graphistry_enabled: bool = False
    rapids_gpu_id: int = 1  # GPU for RAPIDS (default: GPU 1)
    llm_gpu_ids: List[int] = field(default_factory=lambda: [0, 1])

    # Internal state
    _engine: Optional["InferenceEngine"] = None
    _graphistry_registered: bool = False
    _setup_complete: bool = False

    @classmethod
    def setup(
        cls,
        enable_telemetry: bool = True,
        enable_graphistry: bool = False,
        auto_load_secrets: bool = True,
        split_gpu_mode: bool = True,
        verbose: bool = True,
    ) -> "KaggleEnvironment":
        """
        One-liner Kaggle setup.

        This is the main entry point for KaggleEnvironment. It automatically
        detects hardware, loads secrets, and configures optimal settings.

        Args:
            enable_telemetry: Enable OpenTelemetry tracing (default: True)
            enable_graphistry: Register Graphistry from secrets (default: False)
            auto_load_secrets: Load HF_TOKEN, Graphistry keys from Kaggle secrets
            split_gpu_mode: Use GPU 0 for LLM, GPU 1 for RAPIDS (default: True)
            verbose: Print setup summary

        Returns:
            Configured KaggleEnvironment instance

        Example:
            >>> env = KaggleEnvironment.setup()
            >>> print(f"Ready with {env.gpu_count} GPUs ({env.total_vram_gb:.1f} GB)")
        """
        from .presets import ServerPreset

        env = cls()

        # Detect GPUs
        env._detect_gpus()

        # Auto-select preset based on detected hardware
        env._select_preset(split_gpu_mode)

        # Load secrets
        if auto_load_secrets:
            env._load_secrets()

        # Register Graphistry if requested
        if enable_graphistry:
            env._register_graphistry()

        env.telemetry_enabled = enable_telemetry
        env.graphistry_enabled = enable_graphistry
        env._setup_complete = True

        if verbose:
            env._print_summary()

        return env

    def _detect_gpus(self) -> None:
        """Detect available GPUs and their properties."""
        try:
            from ..api.multigpu import detect_gpus, get_cuda_version

            gpus = detect_gpus()
            self.gpu_count = len(gpus)
            self.gpu_names = [g.name for g in gpus]
            self.vram_per_gpu_gb = [g.memory_total_gb for g in gpus]
            self.total_vram_gb = sum(self.vram_per_gpu_gb)

            if gpus:
                self.compute_capability = gpus[0].compute_capability

            self.cuda_version = get_cuda_version() or ""

        except Exception:
            # Fallback if detection fails
            self.gpu_count = 0
            self.gpu_names = []
            self.vram_per_gpu_gb = []
            self.total_vram_gb = 0.0

    def _select_preset(self, split_gpu_mode: bool) -> None:
        """Select optimal preset based on detected hardware."""
        from .presets import ServerPreset

        if self.gpu_count >= 2 and all("T4" in name for name in self.gpu_names):
            self.preset = ServerPreset.KAGGLE_DUAL_T4
            if split_gpu_mode:
                self.llm_gpu_ids = [0, 1]  # Both for LLM (tensor split)
                self.rapids_gpu_id = 1  # RAPIDS on GPU 1 when not running LLM
        elif self.gpu_count == 1:
            self.preset = ServerPreset.KAGGLE_SINGLE_T4
            self.llm_gpu_ids = [0]
            self.rapids_gpu_id = 0
        else:
            self.preset = ServerPreset.AUTO
            self.llm_gpu_ids = []
            self.rapids_gpu_id = 0

    def _load_secrets(self) -> None:
        """Load secrets from Kaggle or environment."""
        from .secrets import auto_load_secrets as _auto_load_secrets

        secrets = _auto_load_secrets(set_env=True)

        self.hf_token = secrets.get("HF_TOKEN") or secrets.get("HUGGING_FACE_HUB_TOKEN")
        self.graphistry_key_id = secrets.get("Graphistry_Personal_Key_ID")
        self.graphistry_key_secret = secrets.get("Graphistry_Personal_Secret_Key")

    def _register_graphistry(self) -> None:
        """Register Graphistry with auto-loaded secrets."""
        if not self.graphistry_key_id or not self.graphistry_key_secret:
            # Try to load secrets if not already loaded
            from .secrets import setup_graphistry_auth
            self._graphistry_registered = setup_graphistry_auth()
            return

        try:
            import graphistry
            graphistry.register(
                api=3,
                protocol="https",
                server="hub.graphistry.com",
                personal_key_id=self.graphistry_key_id,
                personal_key_secret=self.graphistry_key_secret
            )
            self._graphistry_registered = True
        except Exception as e:
            print(f"Warning: Graphistry registration failed: {e}")
            self._graphistry_registered = False

    def _print_summary(self) -> None:
        """Print environment summary."""
        print("=" * 60)
        print("LLAMATELEMETRY KAGGLE ENVIRONMENT")
        print("=" * 60)

        # GPU info
        if self.gpu_count > 0:
            gpu_name = self.gpu_names[0] if self.gpu_names else "Unknown"
            print(f"GPUs:              {self.gpu_count}x {gpu_name}")
            print(f"Total VRAM:        {self.total_vram_gb:.1f} GB")
            print(f"Compute Cap:       {self.compute_capability}")
        else:
            print("GPUs:              None detected (CPU mode)")

        print(f"CUDA Version:      {self.cuda_version or 'Not detected'}")

        # Preset
        preset_name = self.preset.name if self.preset else "AUTO"
        print(f"Preset:            {preset_name}")

        # Features
        print(f"Telemetry:         {'Enabled' if self.telemetry_enabled else 'Disabled'}")
        print(f"HF Token:          {'Set' if self.hf_token else 'Not set'}")
        print(f"Graphistry:        {'Registered' if self._graphistry_registered else 'Not registered'}")

        print("=" * 60)

    def create_engine(
        self,
        model_name_or_path: str,
        preset: Optional["ServerPreset"] = None,
        auto_start: bool = True,
        verbose: bool = True,
        **kwargs
    ) -> "InferenceEngine":
        """
        Create InferenceEngine with optimal Kaggle settings.

        This method creates an InferenceEngine configured for the detected
        hardware with telemetry and optimal batch/context sizes.

        Args:
            model_name_or_path: Model name from registry or path to GGUF file
            preset: Override preset (default: auto-detected)
            auto_start: Start server automatically (default: True)
            verbose: Print status messages (default: True)
            **kwargs: Additional InferenceEngine parameters

        Returns:
            Configured InferenceEngine ready for inference

        Example:
            >>> engine = env.create_engine("gemma-3-4b-Q4_K_M")
            >>> result = engine.infer("Hello, world!")
            >>> print(result.text)
        """
        from .. import InferenceEngine
        from .presets import get_preset_config

        # Get preset configuration
        preset = preset or self.preset
        config = get_preset_config(preset)

        # Create engine with telemetry
        engine = InferenceEngine(
            server_url=config.server_url,
            enable_telemetry=self.telemetry_enabled,
            telemetry_config={
                "service_name": "llamatelemetry-kaggle",
                "enable_graphistry": self.graphistry_enabled,
            } if self.telemetry_enabled else None,
        )

        # Merge preset config with kwargs
        load_kwargs = {**config.to_load_kwargs(), **kwargs}

        # Load model
        if auto_start:
            engine.load_model(
                model_name_or_path,
                auto_start=True,
                auto_configure=True,
                interactive_download=not bool(self.hf_token),  # Skip prompt if token set
                verbose=verbose,
                **load_kwargs
            )

        self._engine = engine
        return engine

    def rapids_context(self) -> "GPUContext":
        """
        Get GPU context for RAPIDS operations (isolated to rapids_gpu_id).

        Use this context manager to run RAPIDS/cuDF/cuGraph operations
        on a dedicated GPU while the LLM uses other GPUs.

        Returns:
            GPUContext configured for RAPIDS GPU

        Example:
            >>> with env.rapids_context():
            ...     import cudf, cugraph
            ...     df = cudf.DataFrame({"x": [1, 2, 3]})
            ...     # All RAPIDS ops on GPU 1
        """
        from .gpu_context import GPUContext
        return GPUContext(gpu_ids=[self.rapids_gpu_id])

    def llm_context(self) -> "GPUContext":
        """
        Get GPU context for LLM operations.

        Use this context manager to ensure LLM operations use
        the designated LLM GPUs.

        Returns:
            GPUContext configured for LLM GPUs

        Example:
            >>> with env.llm_context():
            ...     result = engine.infer("Hello!")
        """
        from .gpu_context import GPUContext
        return GPUContext(gpu_ids=self.llm_gpu_ids)

    def get_model_download_path(self) -> Path:
        """
        Get the default model download directory for Kaggle.

        Returns:
            Path to model download directory
        """
        if os.path.exists("/kaggle/working"):
            return Path("/kaggle/working/models")
        return Path.home() / ".cache" / "llamatelemetry" / "models"

    def download_model(
        self,
        repo_id: str,
        filename: str,
        local_dir: Optional[Path] = None
    ) -> Path:
        """
        Download a model from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID (e.g., "unsloth/gemma-3-4b-it-GGUF")
            filename: Model filename (e.g., "gemma-3-4b-it-Q4_K_M.gguf")
            local_dir: Download directory (default: auto-detect)

        Returns:
            Path to downloaded model file

        Example:
            >>> model_path = env.download_model(
            ...     "unsloth/gemma-3-4b-it-GGUF",
            ...     "gemma-3-4b-it-Q4_K_M.gguf"
            ... )
        """
        from huggingface_hub import hf_hub_download

        if local_dir is None:
            local_dir = self.get_model_download_path()

        local_dir.mkdir(parents=True, exist_ok=True)

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            token=self.hf_token,
        )

        return Path(model_path)

    def __repr__(self) -> str:
        return (
            f"KaggleEnvironment("
            f"gpus={self.gpu_count}x{self.gpu_names[0] if self.gpu_names else 'None'}, "
            f"vram={self.total_vram_gb:.1f}GB, "
            f"preset={self.preset.name if self.preset else 'AUTO'})"
        )


def quick_setup(**kwargs) -> KaggleEnvironment:
    """
    Quick setup alias for KaggleEnvironment.setup().

    Example:
        >>> from llamatelemetry.kaggle import quick_setup
        >>> env = quick_setup()
    """
    return KaggleEnvironment.setup(**kwargs)


__all__ = [
    "KaggleEnvironment",
    "quick_setup",
]
