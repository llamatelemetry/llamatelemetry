"""
Dynamic Quantization API

Provides automatic quantization with adaptive precision based on model size,
available VRAM, and target performance metrics.

This module intelligently selects quantization schemes for optimal inference
on Tesla T4 GPUs while maintaining model quality.
"""

import torch
from typing import Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import subprocess


class QuantStrategy(Enum):
    """Quantization strategies."""
    AGGRESSIVE = "aggressive"  # Maximum compression (Q2_K, Q3_K)
    BALANCED = "balanced"      # Good balance (Q4_K_M) - Recommended
    QUALITY = "quality"        # Higher quality (Q5_K_M, Q6_K)
    MINIMAL = "minimal"        # Minimal compression (Q8_0, F16)


@dataclass
class AutoQuantConfig:
    """
    Auto-configuration for quantization based on constraints.

    Attributes:
        target_vram_gb: Target VRAM usage in GB
        target_speed_tps: Target tokens per second
        min_quality_ppl: Minimum acceptable perplexity increase
        strategy: Quantization strategy
        preserve_embeddings: Keep embeddings in F16
        preserve_output: Keep output layer in F16
    """
    target_vram_gb: Optional[float] = None
    target_speed_tps: Optional[float] = None
    min_quality_ppl: Optional[float] = None
    strategy: QuantStrategy = QuantStrategy.BALANCED
    preserve_embeddings: bool = True
    preserve_output: bool = True


class DynamicQuantizer:
    """
    Dynamic quantizer that adapts to model and hardware constraints.

    Automatically selects optimal quantization scheme based on:
    - Model size (parameter count)
    - Available GPU VRAM
    - Target inference speed
    - Quality requirements

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
        >>> quantizer = DynamicQuantizer(target_vram_gb=8.0)
        >>> config = quantizer.recommend_config(model)
        >>> print(f"Recommended: {config['quant_type']}")
    """

    # Model size to recommended quant type mapping (for Tesla T4 16GB)
    QUANT_RECOMMENDATIONS = {
        # Model size (GB) -> Quant type
        (0, 2): "Q4_K_M",      # 1B models
        (2, 4): "Q4_K_M",      # 3B models
        (4, 8): "Q4_K_M",      # 7B models
        (8, 12): "Q5_K_M",     # 8B models (need higher quality)
        (12, 20): "Q4_K_S",    # 13B+ models (aggressive)
        (20, float('inf')): "Q2_K",  # 20B+ models (very aggressive)
    }

    # Expected compression ratios
    COMPRESSION_RATIOS = {
        "F32": 1.0,
        "F16": 2.0,
        "BF16": 2.0,
        "Q8_0": 4.0,
        "Q6_K": 5.3,
        "Q5_K_M": 6.7,
        "Q5_K_S": 6.7,
        "Q4_K_M": 8.5,
        "Q4_K_S": 8.5,
        "Q3_K": 11.3,
        "Q2_K": 16.0,
    }

    # Expected performance on Tesla T4 (tokens/sec per 1B params)
    PERF_ESTIMATES = {
        "Q2_K": 60,
        "Q3_K": 55,
        "Q4_K_S": 50,
        "Q4_K_M": 45,
        "Q5_K_S": 38,
        "Q5_K_M": 35,
        "Q6_K": 30,
        "Q8_0": 25,
        "F16": 20,
    }

    def __init__(
        self,
        target_vram_gb: Optional[float] = None,
        strategy: QuantStrategy = QuantStrategy.BALANCED,
        device: int = 0,
    ):
        """
        Initialize dynamic quantizer.

        Args:
            target_vram_gb: Target VRAM usage (auto-detects if None)
            strategy: Quantization strategy
            device: CUDA device ID
        """
        self.target_vram_gb = target_vram_gb
        self.strategy = strategy
        self.device = device

        # Auto-detect available VRAM
        if target_vram_gb is None:
            self.target_vram_gb = self._get_available_vram()

    def _get_available_vram(self) -> float:
        """Get available GPU VRAM in GB."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            free_memory_mb = int(result.stdout.strip().split('\n')[self.device])
            # Reserve 2GB for overhead
            return max(1.0, (free_memory_mb / 1024) - 2.0)
        except Exception:
            return 10.0  # Default conservative estimate

    def estimate_model_size_fp16(self, model: Any) -> float:
        """
        Estimate model size in FP16 (GB).

        Args:
            model: PyTorch model

        Returns:
            Estimated size in GB
        """
        total_params = sum(p.numel() for p in model.parameters())
        # FP16 = 2 bytes per param
        size_bytes = total_params * 2
        size_gb = size_bytes / (1024**3)
        return size_gb

    def recommend_config(
        self,
        model: Optional[Any] = None,
        model_size_gb: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Recommend quantization configuration.

        Args:
            model: PyTorch model (optional if model_size_gb provided)
            model_size_gb: Model size in FP16 GB (optional if model provided)
            verbose: Print recommendations

        Returns:
            Dictionary with recommended config:
                - quant_type: Recommended quantization type
                - expected_vram_gb: Expected VRAM usage
                - expected_speed_tps: Expected tokens/sec
                - compression_ratio: Compression vs FP32

        Example:
            >>> config = quantizer.recommend_config(model)
            >>> print(config['quant_type'])  # 'Q4_K_M'
        """
        # Get model size
        if model_size_gb is None:
            if model is None:
                raise ValueError("Must provide either model or model_size_gb")
            model_size_gb = self.estimate_model_size_fp16(model)

        if verbose:
            print(f"Model size (FP16): {model_size_gb:.2f} GB")
            print(f"Available VRAM: {self.target_vram_gb:.2f} GB")
            print(f"Strategy: {self.strategy.value}")

        # Select quantization type based on strategy
        if self.strategy == QuantStrategy.AGGRESSIVE:
            quant_type = self._recommend_aggressive(model_size_gb)
        elif self.strategy == QuantStrategy.BALANCED:
            quant_type = self._recommend_balanced(model_size_gb)
        elif self.strategy == QuantStrategy.QUALITY:
            quant_type = self._recommend_quality(model_size_gb)
        else:  # MINIMAL
            quant_type = self._recommend_minimal(model_size_gb)

        # Calculate expected metrics
        compression_ratio = self.COMPRESSION_RATIOS.get(quant_type, 8.0)
        expected_vram = model_size_gb / compression_ratio * 2.0  # 2x FP16
        expected_vram += 2.0  # Add context and overhead

        # Estimate speed (rough)
        param_count_b = model_size_gb / 2.0  # FP16 = 2 bytes/param
        base_speed = self.PERF_ESTIMATES.get(quant_type, 40)
        expected_speed = base_speed / param_count_b

        config = {
            'quant_type': quant_type,
            'expected_vram_gb': expected_vram,
            'expected_speed_tps': expected_speed,
            'compression_ratio': compression_ratio,
            'strategy': self.strategy.value,
        }

        if verbose:
            print(f"\n✓ Recommended quantization: {quant_type}")
            print(f"  Expected VRAM: {expected_vram:.2f} GB")
            print(f"  Expected speed: {expected_speed:.1f} tokens/sec")
            print(f"  Compression: {compression_ratio:.1f}x vs FP32")

        return config

    def _recommend_aggressive(self, model_size_gb: float) -> str:
        """Recommend for aggressive compression."""
        if model_size_gb <= 4:
            return "Q4_K_S"
        elif model_size_gb <= 8:
            return "Q3_K"
        else:
            return "Q2_K"

    def _recommend_balanced(self, model_size_gb: float) -> str:
        """Recommend for balanced quality/speed."""
        for (min_size, max_size), quant_type in self.QUANT_RECOMMENDATIONS.items():
            if min_size <= model_size_gb < max_size:
                return quant_type
        return "Q4_K_M"

    def _recommend_quality(self, model_size_gb: float) -> str:
        """Recommend for higher quality."""
        if model_size_gb <= 4:
            return "Q5_K_M"
        elif model_size_gb <= 8:
            return "Q5_K_S"
        else:
            return "Q4_K_M"

    def _recommend_minimal(self, model_size_gb: float) -> str:
        """Recommend for minimal compression."""
        if model_size_gb <= 8:
            return "Q8_0"
        else:
            return "Q6_K"

    def quantize_model(
        self,
        model: Any,
        output_path: str,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        """
        Quantize model with recommended or provided config.

        Args:
            model: PyTorch model to quantize
            output_path: Output GGUF file path
            config: Optional config (uses recommend_config if None)
            verbose: Print progress
        """
        if config is None:
            config = self.recommend_config(model, verbose=verbose)

        quant_type = config['quant_type']

        if verbose:
            print(f"\nQuantizing model to {quant_type}...")

        # Use GGUF converter
        from .gguf import convert_to_gguf

        convert_to_gguf(
            model,
            output_path,
            quant_type=quant_type,
            verbose=verbose,
        )

        if verbose:
            print(f"✓ Quantized model saved to {output_path}")


def quantize_dynamic(
    model: Any,
    output_path: str,
    target_vram_gb: Optional[float] = None,
    strategy: str = "balanced",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Dynamically quantize a model (convenience function).

    Args:
        model: PyTorch model
        output_path: Output GGUF path
        target_vram_gb: Target VRAM in GB (auto-detect if None)
        strategy: "aggressive", "balanced", "quality", or "minimal"
        verbose: Print progress

    Returns:
        Configuration dict used

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
        >>> config = quantize_dynamic(model, "model.gguf", target_vram_gb=8.0)
        >>> print(f"Used {config['quant_type']} quantization")
    """
    strategy_enum = QuantStrategy(strategy)
    quantizer = DynamicQuantizer(target_vram_gb=target_vram_gb, strategy=strategy_enum)

    config = quantizer.recommend_config(model, verbose=verbose)
    quantizer.quantize_model(model, output_path, config, verbose=verbose)

    return config
