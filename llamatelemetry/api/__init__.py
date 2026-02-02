"""
llamatelemetry.api - Comprehensive llama.cpp Server API Client Module

This module provides a complete Python SDK for interacting with llama.cpp server,
with full coverage of all API endpoints and features.

Based on: https://github.com/ggml-org/llama.cpp/tree/master/tools/server

Features:
    - Full OpenAI-compatible chat/completions API
    - Anthropic-compatible Messages API
    - Native llama.cpp completion API with all sampling parameters
    - Embeddings and reranking endpoints
    - Tokenization and detokenization
    - LoRA adapter management
    - Slot management and KV cache control
    - Model loading/unloading
    - Health monitoring and metrics
    - Multi-GPU support (Kaggle 2× T4)
    - GGUF model utilities (parsing, quantization, conversion)

Example:
    >>> from llamatelemetry.api import LlamaCppClient
    >>> 
    >>> client = LlamaCppClient("http://localhost:8080")
    >>> 
    >>> # OpenAI-compatible chat
    >>> response = client.chat.completions.create(
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     max_tokens=100
    ... )
    >>> print(response.choices[0].message.content)
    
    >>> # Native completion with advanced sampling
    >>> response = client.complete(
    ...     prompt="The future of AI is",
    ...     n_predict=50,
    ...     temperature=0.7,
    ...     mirostat=2
    ... )
    
    >>> # Multi-GPU configuration
    >>> from llamatelemetry.api import MultiGPUConfig, kaggle_t4_dual_config
    >>> config = kaggle_t4_dual_config()
    >>> print(config.to_cli_args())
"""

# Client imports
from .client import (
    LlamaCppClient,
    ChatCompletionsAPI,
    EmbeddingsClientAPI,
    ModelsClientAPI,
    SlotsClientAPI,
    LoraClientAPI,
    # Data classes
    Message,
    Choice,
    Usage,
    Timings,
    CompletionResponse,
    EmbeddingData,
    EmbeddingsResponse,
    RerankResult,
    RerankResponse,
    TokenizeResponse,
    ModelInfo,
    SlotInfo,
    HealthStatus,
    LoraAdapter,
)

# Multi-GPU imports
from .multigpu import (
    MultiGPUConfig,
    SplitMode,
    GPUInfo,
    detect_gpus,
    get_cuda_version,
    get_total_vram,
    get_free_vram,
    is_multi_gpu,
    gpu_count,
    kaggle_t4_dual_config,
    colab_t4_single_config,
    auto_config,
    estimate_model_vram,
    can_fit_model,
    recommend_quantization,
    set_cuda_visible_devices,
    get_cuda_visible_devices,
    print_gpu_info,
)

# GGUF imports
from .gguf import (
    GGUFMetadata,
    GGUFTensorInfo,
    GGUFModelInfo,
    GGMLType,
    GGUFValueType,
    QUANT_TYPE_NAMES,
    parse_gguf_header,
    quantize,
    convert_hf_to_gguf,
    merge_lora,
    generate_imatrix,
    find_gguf_models,
    get_model_summary,
    compare_models,
    validate_gguf,
    get_recommended_quant,
)

__all__ = [
    # Client
    "LlamaCppClient",
    "ChatCompletionsAPI",
    "EmbeddingsClientAPI", 
    "ModelsClientAPI",
    "SlotsClientAPI",
    "LoraClientAPI",
    # Data classes
    "Message",
    "Choice",
    "Usage",
    "Timings",
    "CompletionResponse",
    "EmbeddingData",
    "EmbeddingsResponse",
    "RerankResult",
    "RerankResponse",
    "TokenizeResponse",
    "ModelInfo",
    "SlotInfo",
    "HealthStatus",
    "LoraAdapter",
    # Multi-GPU
    "MultiGPUConfig",
    "SplitMode",
    "GPUInfo",
    "detect_gpus",
    "get_cuda_version",
    "get_total_vram",
    "get_free_vram",
    "is_multi_gpu",
    "gpu_count",
    "kaggle_t4_dual_config",
    "colab_t4_single_config",
    "auto_config",
    "estimate_model_vram",
    "can_fit_model",
    "recommend_quantization",
    "set_cuda_visible_devices",
    "get_cuda_visible_devices",
    "print_gpu_info",
    # GGUF
    "GGUFMetadata",
    "GGUFTensorInfo",
    "GGUFModelInfo",
    "GGMLType",
    "GGUFValueType",
    "QUANT_TYPE_NAMES",
    "parse_gguf_header",
    "quantize",
    "convert_hf_to_gguf",
    "merge_lora",
    "generate_imatrix",
    "find_gguf_models",
    "get_model_summary",
    "compare_models",
    "validate_gguf",
    "get_recommended_quant",
    # NCCL
    "NCCLCommunicator",
    "NCCLConfig",
    "NCCLInfo",
    "is_nccl_available",
    "get_nccl_version",
    "get_nccl_info",
    "setup_nccl_environment",
    "kaggle_nccl_config",
    "print_nccl_info",
    "get_llama_cpp_nccl_args",
]

# NCCL imports (optional, may not be available on all systems)
try:
    from .nccl import (
        NCCLCommunicator,
        NCCLConfig,
        NCCLInfo,
        is_nccl_available,
        get_nccl_version,
        get_nccl_info,
        setup_nccl_environment,
        kaggle_nccl_config,
        print_nccl_info,
        get_llama_cpp_nccl_args,
    )
except ImportError:
    # NCCL not available - provide stubs
    def is_nccl_available():
        return False
    
    def get_nccl_version():
        return None
    
    NCCLCommunicator = None
    NCCLConfig = None
    NCCLInfo = None
    get_nccl_info = None
    setup_nccl_environment = None
    kaggle_nccl_config = None
    print_nccl_info = None
    get_llama_cpp_nccl_args = None
