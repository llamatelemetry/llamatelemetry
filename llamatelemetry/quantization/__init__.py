"""
llamatelemetry Quantization API

Provides quantization utilities for converting models to GGUF format with various
quantization schemes (NF4, Q4_K_M, Q5_K_M, etc.) optimized for Tesla T4 GPU.

This module enables tight integration with Unsloth fine-tuned models, allowing
seamless export from training to optimized inference.
"""

from .nf4 import (
    quantize_nf4,
    dequantize_nf4,
    NF4Quantizer,
    NF4Config,
)

from .gguf import (
    convert_to_gguf,
    GGUFConverter,
    GGUFQuantType,
    save_gguf,
    load_gguf_metadata,
)

from .dynamic import (
    DynamicQuantizer,
    quantize_dynamic,
    AutoQuantConfig,
)

__all__ = [
    # NF4 quantization
    'quantize_nf4',
    'dequantize_nf4',
    'NF4Quantizer',
    'NF4Config',

    # GGUF conversion
    'convert_to_gguf',
    'GGUFConverter',
    'GGUFQuantType',
    'save_gguf',
    'load_gguf_metadata',

    # Dynamic quantization
    'DynamicQuantizer',
    'quantize_dynamic',
    'AutoQuantConfig',
]
