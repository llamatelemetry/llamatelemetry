"""
llamatelemetry Advanced Inference APIs

Enhanced inference capabilities including:
- FlashAttention v2/v3 integration
- KV-cache optimization for long contexts
- Batch inference optimization
- Speculative decoding

These APIs complement the main InferenceEngine for specialized use cases.
"""

from .flash_attn import (
    FlashAttentionConfig,
    enable_flash_attention,
    flash_attention_forward,
)

from .kv_cache import (
    KVCache,
    KVCacheConfig,
    PagedKVCache,
    optimize_kv_cache,
)

from .batch import (
    BatchInferenceOptimizer,
    ContinuousBatching,
    batch_inference_optimized,
)

__all__ = [
    # FlashAttention
    'FlashAttentionConfig',
    'enable_flash_attention',
    'flash_attention_forward',

    # KV-cache
    'KVCache',
    'KVCacheConfig',
    'PagedKVCache',
    'optimize_kv_cache',

    # Batch optimization
    'BatchInferenceOptimizer',
    'ContinuousBatching',
    'batch_inference_optimized',
]
