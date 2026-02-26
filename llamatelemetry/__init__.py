"""
llamatelemetry - Clean, Ultra-Lightweight CUDA 12 LLM Inference for Python 3.11+

Streamlined PyTorch-style package with hybrid bootstrap architecture.
Lightweight Python package with auto-download of CUDA binaries and libraries.
No manual setup required - just pip install and use!

Version 0.1.0 - Initial release (renamed from llcuda, built on llama.cpp binaries v0.1.0).

Examples:
    Basic usage (auto-download model from registry):
    >>> import llamatelemetry
    >>> engine = llamatelemetry.InferenceEngine()
    >>> engine.load_model("gemma-3-1b-Q4_K_M")  # Auto-downloads and configures
    >>> result = engine.infer("What is AI?", max_tokens=100)
    >>> print(result.text)

    Using local model:
    >>> engine = llamatelemetry.InferenceEngine()
    >>> engine.load_model("/path/to/model.gguf", auto_start=True)
    >>> result = engine.infer("What is AI?")

Key Features:
    - Lightweight Python package with auto-download of CUDA binaries
    - Python 3.11+ optimized
    - Kaggle-only runtime target (2x Tesla T4, SM 7.5)
    - Split-GPU workflow (GPU 0: LLM, GPU 1: Graphistry/RAPIDS)
    - Optimized for small GGUF models (1B-5B)
    - Clean, maintainable codebase
"""

from typing import Optional, List, Dict, Any
from contextlib import nullcontext
import os
import sys
import subprocess
import requests
import time
from pathlib import Path
import logging

# ============================================================================
# AUTO-CONFIGURATION (PyTorch-style)
# Automatically configure paths to bundled CUDA binaries and libraries
# ============================================================================

_LLCUDA_DIR = Path(__file__).parent
_BIN_DIR = _LLCUDA_DIR / "binaries" / "cuda12"
_LIB_DIR = _LLCUDA_DIR / "lib"
_MODEL_CACHE = _LLCUDA_DIR / "models"

# Ensure model cache directory exists
_MODEL_CACHE.mkdir(parents=True, exist_ok=True)


# Auto-configure LD_LIBRARY_PATH for bundled shared libraries
if _LIB_DIR.exists():
    _lib_path_str = str(_LIB_DIR.absolute())
    _current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    if _lib_path_str not in _current_ld_path:
        if _current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{_lib_path_str}:{_current_ld_path}"
        else:
            os.environ["LD_LIBRARY_PATH"] = _lib_path_str

    # Log the setup (optional: helps with debugging)
    logging.info(f"llamatelemetry: Set LD_LIBRARY_PATH to include {_lib_path_str}")
else:
    logging.warning(
        "llamatelemetry: Library directory not found - shared libraries may not load correctly"
    )


# Auto-configure LLAMA_SERVER_PATH to bundled executable
_LLAMA_SERVER = _BIN_DIR / "llama-server"
if _LLAMA_SERVER.exists():
    os.environ["LLAMA_SERVER_PATH"] = str(_LLAMA_SERVER.absolute())
    # Make executable if not already
    if not os.access(_LLAMA_SERVER, os.X_OK):
        try:
            os.chmod(_LLAMA_SERVER, 0o755)
        except Exception:
            pass  # Ignore permission errors
else:
    # ========================================================================
    # HYBRID BOOTSTRAP: Download binaries and models on first import
    # ========================================================================
    try:
        from ._internal.bootstrap import bootstrap

        bootstrap()

        # Re-apply env vars after bootstrap (in case paths were created during download)
        if _LIB_DIR.exists():
            _lib_path_str = str(_LIB_DIR.absolute())
            _current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if _lib_path_str not in _current_ld_path:
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{_lib_path_str}:{_current_ld_path}"
                    if _current_ld_path
                    else _lib_path_str
                )

        if _LLAMA_SERVER.exists():
            os.environ["LLAMA_SERVER_PATH"] = str(_LLAMA_SERVER.absolute())
            if not os.access(_LLAMA_SERVER, os.X_OK):
                os.chmod(_LLAMA_SERVER, 0o755)

    except Exception as e:
        import warnings

        warnings.warn(
            f"llamatelemetry bootstrap failed: {e}\n"
            "Some features may not work. Please check your installation.",
            RuntimeWarning,
        )


# Enhanced path debugging
if not _LIB_DIR.exists():
    # Try to find lib directory in package
    import llamatelemetry as lc

    package_dir = Path(lc.__file__).parent
    possible_lib_dirs = [
        package_dir / "lib",
        package_dir.parent / "lib",
        Path("/usr/local/lib/python3.12/dist-packages/llamatelemetry/lib"),
        Path("/usr/lib/python3/dist-packages/llamatelemetry/lib"),
    ]

    for lib_dir in possible_lib_dirs:
        if lib_dir.exists():
            _LIB_DIR = lib_dir
            _lib_path_str = str(_LIB_DIR.absolute())
            _current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if _lib_path_str not in _current_ld_path:
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{_lib_path_str}:{_current_ld_path}"
                    if _current_ld_path
                    else _lib_path_str
                )
            break

if not _LLAMA_SERVER.exists():
    # Try to find server in alternative locations
    possible_server_paths = [
        _BIN_DIR / "llama-server",
        Path("/usr/local/bin/llama-server"),
        Path("/usr/bin/llama-server"),
        Path.home() / ".cache/llamatelemetry/llama-server",
    ]

    for server_path in possible_server_paths:
        if server_path.exists():
            os.environ["LLAMA_SERVER_PATH"] = str(server_path.absolute())
            if not os.access(server_path, os.X_OK):
                try:
                    os.chmod(server_path, 0o755)
                except:
                    pass
            break


from .server import ServerManager
from .utils import (
    detect_cuda,
    check_gpu_compatibility,
    get_llama_cpp_cuda_path,
    setup_environment,
    find_gguf_models,
    print_system_info,
    load_config,
    create_config_file,
    get_recommended_gpu_layers,
    validate_model_path,
)

__version__ = "0.1.0"  # SDK version (binary artifact is llama.cpp v0.1.0)
__all__ = [
    # Core classes
    "InferenceEngine",
    "InferResult",
    "ServerManager",
    "bootstrap",
    # Utility functions
    "check_cuda_available",
    "get_cuda_device_info",
    "check_gpu_compatibility",
    "detect_cuda",
    "setup_environment",
    "find_gguf_models",
    "print_system_info",
    "get_llama_cpp_cuda_path",
    "quick_infer",
    # Existing modules (lazily imported)
    "jupyter",
    "chat",
    "embeddings",
    "models",
    # New API modules (v2.1+)
    "quantization",
    "unsloth",
    "cuda",
    "inference",
    # OpenTelemetry observability (v2.2+)
    "telemetry",
    # Kaggle zero-boilerplate setup (v0.2.0+)
    "kaggle",
]


class InferenceEngine:
    """
    High-level Python interface for LLM inference with CUDA acceleration.

    This class provides an easy-to-use API for running LLM inference with
    automatic server management. It can automatically find and start
    llama-server, or connect to an existing server instance.

    Examples:
        Auto-start mode (easiest):
        >>> engine = InferenceEngine()
        >>> engine.load_model("model.gguf", auto_start=True, gpu_layers=99)
        >>> result = engine.infer("What is AI?", max_tokens=100)
        >>> print(result.text)

        Connect to existing server:
        >>> engine = InferenceEngine(server_url="http://127.0.0.1:8090")
        >>> result = engine.infer("What is AI?", max_tokens=100)
        >>> print(result.text)

        With context manager (auto-cleanup):
        >>> with InferenceEngine() as engine:
        ...     engine.load_model("model.gguf", auto_start=True)
        ...     result = engine.infer("Hello!")
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8090",
        enable_telemetry: bool = False,
        telemetry_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the inference engine.

        Args:
            server_url: URL of llama-server backend (default: http://127.0.0.1:8090)
        """
        self.server_url = server_url
        self._model_loaded = False
        self._server_manager: Optional[ServerManager] = None
        self._model_path: Optional[Path] = None
        self._model_name: Optional[str] = None
        self._metrics = {
            "requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0.0,
            "latencies": [],
        }
        self._telemetry_enabled = enable_telemetry
        self._telemetry_config = telemetry_config or {}
        self._tracer = None
        self._meter = None
        self._metrics_collector = None

        if self._telemetry_enabled:
            self._init_telemetry()

    def _init_telemetry(self) -> None:
        """Initialize OpenTelemetry tracing and metrics if available."""
        try:
            from .telemetry import setup_telemetry, get_metrics_collector

            tracer, meter = setup_telemetry(
                service_name=self._telemetry_config.get(
                    "service_name", "llamatelemetry"
                ),
                service_version=self._telemetry_config.get(
                    "service_version", __version__
                ),
                otlp_endpoint=self._telemetry_config.get("otlp_endpoint"),
                enable_graphistry=self._telemetry_config.get(
                    "enable_graphistry", False
                ),
                graphistry_server=self._telemetry_config.get("graphistry_server"),
            )
            self._tracer = tracer
            self._meter = meter
            self._metrics_collector = get_metrics_collector()
        except Exception:
            # Telemetry is optional; fail silently if not available
            self._tracer = None
            self._meter = None
            self._metrics_collector = None

    @staticmethod
    def check_for_updates():
        try:
            response = requests.get("https://api.github.com/repos/llamatelemetry/llamatelemetry/releases/latest", timeout=2)
            latest = response.json()["tag_name"].lstrip("v")
            if latest != __version__:
                print(
                    f"llamatelemetry: New version available ({latest}) - pip install --upgrade git+https://github.com/llamatelemetry/llamatelemetry.git"
                )
        except Exception:
            pass  # Silent fail

    # Update checks are opt-in; call InferenceEngine.check_for_updates() manually.

    def check_server(self) -> bool:
        """
        Check if llama-server is running and accessible.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def load_model(
        self,
        model_name_or_path: str,
        gpu_layers: Optional[int] = None,
        ctx_size: Optional[int] = None,
        auto_start: bool = True,
        auto_configure: bool = True,
        n_parallel: int = 1,
        verbose: bool = True,
        interactive_download: bool = True,
        silent: bool = False,
        **kwargs,
    ) -> Optional[bool]:
        """
        Load a GGUF model for inference with smart loading and auto-configuration.

        This method supports three loading modes:
        1. Registry name: "gemma-3-1b-Q4_K_M" (auto-downloads from HuggingFace)
        2. Local path: "/path/to/model.gguf"
        3. HuggingFace syntax: "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf"

        Args:
            model_name_or_path: Model name from registry, local path, or HF syntax
            gpu_layers: Number of layers to offload to GPU (None = auto-configure)
            ctx_size: Context size (None = auto-configure)
            auto_start: Automatically start server if not running (default: True)
            auto_configure: Automatically configure optimal settings (default: True)
            n_parallel: Number of parallel sequences (default: 1)
            verbose: Print status messages (default: True)
            interactive_download: Ask for confirmation before downloading (default: True)
            silent: Suppress all llama-server output/warnings (default: False)
            **kwargs: Additional server parameters (batch_size, ubatch_size, etc.)

        Returns:
            True if model loaded successfully, None if user cancelled download

        Raises:
            FileNotFoundError: If model file not found
            ConnectionError: If server not running and auto_start=False
            RuntimeError: If server fails to start

        Note:
            If interactive_download=True and user selects 'No' when prompted,
            the method returns None gracefully without raising an exception.

        Examples:
            >>> # Auto-download from registry
            >>> engine.load_model("gemma-3-1b-Q4_K_M")

            >>> # Local path with manual settings
            >>> engine.load_model("/path/to/model.gguf", gpu_layers=20, ctx_size=2048)

            >>> # HuggingFace download
            >>> engine.load_model("unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf")
        """
        from .models import load_model_smart
        from .utils import auto_configure_for_model

        # Step 1: Smart model loading (handles registry, local, HF)
        if verbose:
            print(f"Loading model: {model_name_or_path}")

        model_path = load_model_smart(
            model_name_or_path, interactive=interactive_download
        )

        # Check if user cancelled download (returns None)
        if model_path is None:
            if not silent:
                print("\nℹ️  Model loading stopped. No model loaded.")
            return  # Exit gracefully without raising exception

        # Step 2: Auto-configure if requested and no manual settings provided
        auto_settings = {}
        if auto_configure and (gpu_layers is None or ctx_size is None):
            if verbose:
                print("\nAuto-configuring optimal settings...")

            auto_settings = auto_configure_for_model(model_path)

            # Use auto-configured values if not manually specified
            if gpu_layers is None:
                gpu_layers = auto_settings["gpu_layers"]
            if ctx_size is None:
                ctx_size = auto_settings["ctx_size"]

            # Merge auto-configured settings with kwargs
            if "batch_size" not in kwargs:
                kwargs["batch_size"] = auto_settings.get("batch_size", 512)
            if "ubatch_size" not in kwargs:
                kwargs["ubatch_size"] = auto_settings.get("ubatch_size", 128)
        else:
            # Use defaults if not auto-configuring
            if gpu_layers is None:
                gpu_layers = 99
            if ctx_size is None:
                ctx_size = 2048
            # Set default auto_settings for later use
            auto_settings = {"batch_size": 512, "ubatch_size": 128}

        # Step 3: Validate model path exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Step 4: Start server if needed
        if not self.check_server():
            if auto_start:
                if verbose:
                    print(f"\nStarting llama-server...")

                # Create server manager
                self._server_manager = ServerManager(server_url=self.server_url)

                # Extract port from server URL
                port = int(self.server_url.split(":")[-1].split("/")[0])

                # Extract batch parameters from kwargs or use defaults
                batch_size = kwargs.pop(
                    "batch_size",
                    auto_settings.get("batch_size", 512) if auto_configure else 512,
                )
                ubatch_size = kwargs.pop(
                    "ubatch_size",
                    auto_settings.get("ubatch_size", 128) if auto_configure else 128,
                )

                success = self._server_manager.start_server(
                    model_path=str(model_path),
                    port=port,
                    gpu_layers=gpu_layers if gpu_layers is not None else 99,
                    ctx_size=ctx_size if ctx_size is not None else 2048,
                    n_parallel=n_parallel,
                    batch_size=batch_size,
                    ubatch_size=ubatch_size,
                    verbose=verbose,
                    silent=silent,
                    **kwargs,
                )

                if not success:
                    raise RuntimeError("Failed to start llama-server")

            else:
                raise ConnectionError(
                    f"llama-server not running at {self.server_url}. "
                    "Set auto_start=True to start automatically, or start the server manually."
                )

        self._model_loaded = True
        self._model_path = model_path
        self._model_name = model_path.name

        if verbose:
            print(f"\n✓ Model loaded and ready for inference")
            print(f"  Server: {self.server_url}")
            print(f"  GPU Layers: {gpu_layers}")
            print(f"  Context Size: {ctx_size}")

        return True

    def infer(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: int = 0,
        stop_sequences: Optional[List[str]] = None,
    ) -> "InferResult":
        """
        Run inference on a single prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (default: 128)
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling threshold (default: 0.9)
            top_k: Top-k sampling limit (default: 40)
            seed: Random seed (0 = random, default: 0)
            stop_sequences: List of stop sequences (default: None)

        Returns:
            InferResult object with generated text and metrics
        """
        start_time = time.time()
        prompt_tokens = len(prompt.split()) if prompt else 0

        span_cm = nullcontext()
        if self._tracer:
            span_cm = self._tracer.start_as_current_span("llm.inference")

        with span_cm as span:
            payload = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "stream": False,
            }

            if seed > 0:
                payload["seed"] = seed

            if stop_sequences:
                payload["stop"] = stop_sequences

            try:
                response = requests.post(
                    f"{self.server_url}/completion", json=payload, timeout=120
                )

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()

                    text = data.get("content", "")
                    tokens_generated = data.get("tokens_predicted", len(text.split()))

                    # Update metrics
                    self._metrics["requests"] += 1
                    self._metrics["total_tokens"] += tokens_generated
                    self._metrics["total_latency_ms"] += latency_ms
                    self._metrics["latencies"].append(latency_ms)

                    if self._metrics_collector:
                        try:
                            self._metrics_collector.record_inference(
                                latency_ms=latency_ms,
                                tokens=tokens_generated,
                                model=self._model_name or "",
                            )
                        except Exception:
                            pass

                    if span:
                        try:
                            from .telemetry.tracer import annotate_inference_span

                            annotate_inference_span(
                                span=span,
                                model=self._model_name or "",
                                prompt_tokens=prompt_tokens,
                                output_tokens=tokens_generated,
                                latency_ms=latency_ms,
                                gpu_id=0,
                                split_mode="none",
                            )
                        except Exception:
                            pass

                    result = InferResult()
                    result.success = True
                    result.text = text
                    result.tokens_generated = tokens_generated
                    result.latency_ms = latency_ms
                    result.tokens_per_sec = (
                        tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0
                    )

                    return result
                else:
                    result = InferResult()
                    result.success = False
                    result.error_message = (
                        f"Server error: {response.status_code} - {response.text}"
                    )
                    if span:
                        try:
                            span.set_attribute("llm.error", result.error_message)
                        except Exception:
                            pass
                    return result

            except requests.exceptions.Timeout as e:
                result = InferResult()
                result.success = False
                result.error_message = "Request timeout - server took too long to respond"
                if span:
                    try:
                        span.record_exception(e)
                        span.set_attribute("llm.error", result.error_message)
                    except Exception:
                        pass
                return result
            except requests.exceptions.RequestException as e:
                result = InferResult()
                result.success = False
                result.error_message = f"Connection error: {str(e)}"
                if span:
                    try:
                        span.record_exception(e)
                        span.set_attribute("llm.error", result.error_message)
                    except Exception:
                        pass
                return result
            except Exception as e:
                result = InferResult()
                result.success = False
                result.error_message = f"Unexpected error: {str(e)}"
                if span:
                    try:
                        span.record_exception(e)
                        span.set_attribute("llm.error", result.error_message)
                    except Exception:
                        pass
                return result

    def infer_stream(
        self,
        prompt: str,
        callback: Any,
        max_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs,
    ) -> "InferResult":
        """
        Run streaming inference with callback for each chunk.

        Args:
            prompt: Input prompt text
            callback: Function called for each generated chunk
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters (top_p, top_k, seed)

        Returns:
            InferResult object with complete response and metrics
        """
        # For simplicity, just call regular infer and invoke callback once
        result = self.infer(prompt, max_tokens, temperature, **kwargs)
        if result.success and callback:
            callback(result.text)
        return result

    def batch_infer(
        self, prompts: List[str], max_tokens: int = 128, **kwargs
    ) -> List["InferResult"]:
        """
        Run batch inference on multiple prompts.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per prompt
            **kwargs: Additional parameters (temperature, top_p, top_k)

        Returns:
            List of InferResult objects
        """
        results = []
        for prompt in prompts:
            result = self.infer(prompt, max_tokens, **kwargs)
            results.append(result)
        return results

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.

        Returns:
            Dictionary with latency, throughput, and GPU metrics
        """
        latencies = self._metrics["latencies"]

        if latencies:
            sorted_latencies = sorted(latencies)
            mean_latency = self._metrics["total_latency_ms"] / len(latencies)
            p50_idx = len(sorted_latencies) // 2
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)

            p50 = sorted_latencies[p50_idx] if p50_idx < len(sorted_latencies) else 0
            p95 = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else 0
            p99 = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else 0
        else:
            mean_latency = p50 = p95 = p99 = 0

        return {
            "latency": {
                "mean_ms": mean_latency,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "min_ms": min(latencies) if latencies else 0,
                "max_ms": max(latencies) if latencies else 0,
                "sample_count": len(latencies),
            },
            "throughput": {
                "total_tokens": self._metrics["total_tokens"],
                "total_requests": self._metrics["requests"],
                "tokens_per_sec": self._metrics["total_tokens"]
                / (self._metrics["total_latency_ms"] / 1000)
                if self._metrics["total_latency_ms"] > 0
                else 0,
                "requests_per_sec": self._metrics["requests"]
                / (self._metrics["total_latency_ms"] / 1000)
                if self._metrics["total_latency_ms"] > 0
                else 0,
            },
        }

    def reset_metrics(self):
        """Reset performance metrics counters."""
        self._metrics = {
            "requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0.0,
            "latencies": [],
        }

    def unload_model(self):
        """Unload the current model and stop server if managed by this instance."""
        if self._server_manager is not None:
            self._server_manager.stop_server()
            self._server_manager = None
        self._model_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model_loaded

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: cleanup server."""
        self.unload_model()
        return False

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.unload_model()


class InferResult:
    """Wrapper for inference results with convenient access."""

    def __init__(self):
        self._success = False
        self._text = ""
        self._tokens_generated = 0
        self._latency_ms = 0.0
        self._tokens_per_sec = 0.0
        self._error_message = ""

    @property
    def success(self) -> bool:
        """Whether inference succeeded."""
        return self._success

    @success.setter
    def success(self, value: bool):
        self._success = value

    @property
    def text(self) -> str:
        """Generated text."""
        return self._text

    @text.setter
    def text(self, value: str):
        self._text = value

    @property
    def tokens_generated(self) -> int:
        """Number of tokens generated."""
        return self._tokens_generated

    @tokens_generated.setter
    def tokens_generated(self, value: int):
        self._tokens_generated = value

    @property
    def latency_ms(self) -> float:
        """Inference latency in milliseconds."""
        return self._latency_ms

    @latency_ms.setter
    def latency_ms(self, value: float):
        self._latency_ms = value

    @property
    def tokens_per_sec(self) -> float:
        """Generation throughput in tokens/second."""
        return self._tokens_per_sec

    @tokens_per_sec.setter
    def tokens_per_sec(self, value: float):
        self._tokens_per_sec = value

    @property
    def error_message(self) -> str:
        """Error message if inference failed."""
        return self._error_message

    @error_message.setter
    def error_message(self, value: str):
        self._error_message = value

    def __repr__(self) -> str:
        if self.success:
            return (
                f"InferResult(tokens={self.tokens_generated}, "
                f"latency={self.latency_ms:.2f}ms, "
                f"throughput={self.tokens_per_sec:.2f} tok/s)"
            )
        else:
            return f"InferResult(Error: {self.error_message})"

    def __str__(self) -> str:
        return self.text


def check_cuda_available() -> bool:
    """
    Check if CUDA is available on the system.

    Returns:
        True if CUDA is available, False otherwise
    """
    cuda_info = detect_cuda()
    return cuda_info["available"]


def get_cuda_device_info() -> Optional[Dict[str, Any]]:
    """
    Get CUDA device information.

    Returns:
        Dictionary with GPU info or None if CUDA unavailable
    """
    cuda_info = detect_cuda()
    if not cuda_info["available"]:
        return None

    return {"cuda_version": cuda_info["version"], "gpus": cuda_info["gpus"]}


# Convenience function
def quick_infer(
    prompt: str,
    model_path: Optional[str] = None,
    max_tokens: int = 128,
    server_url: str = "http://127.0.0.1:8090",
    auto_start: bool = True,
) -> str:
    """
    Quick inference with minimal setup.

    Args:
        prompt: Input prompt
        model_path: Path to GGUF model (required if auto_start=True)
        max_tokens: Maximum tokens to generate
        server_url: llama-server URL
        auto_start: Automatically start server if needed

    Returns:
        Generated text
    """
    engine = InferenceEngine(server_url=server_url)

    if auto_start and model_path:
        engine.load_model(model_path, auto_start=True, verbose=False)
    elif not engine.check_server():
        return "Error: Server not running and no model path provided"

    result = engine.infer(prompt, max_tokens=max_tokens)
    return result.text if result.success else f"Error: {result.error_message}"
