"""
Unsloth to GGUF Exporter

Export Unsloth fine-tuned models to GGUF format for llamatelemetry inference.
Handles LoRA merging, quantization, and metadata preservation.
"""

import torch
from typing import Optional, Union, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import json


@dataclass
class ExportConfig:
    """
    Configuration for Unsloth model export.

    Attributes:
        quant_type: Quantization type (Q4_K_M, Q5_K_M, etc.)
        merge_lora: Merge LoRA adapters before export
        preserve_tokenizer: Save tokenizer alongside GGUF
        metadata: Additional metadata to include
        verbose: Print export progress
    """
    quant_type: str = "Q4_K_M"
    merge_lora: bool = True
    preserve_tokenizer: bool = True
    metadata: Optional[Dict[str, Any]] = None
    verbose: bool = True


class UnslothExporter:
    """
    Export Unsloth models to GGUF format.

    Provides seamless conversion from Unsloth fine-tuned models to
    llamatelemetry-compatible GGUF files with quantization.

    Example:
        >>> from unsloth import FastLanguageModel
        >>> model, tokenizer = FastLanguageModel.from_pretrained("my_model")
        >>>
        >>> exporter = UnslothExporter()
        >>> exporter.export(
        >>>     model,
        >>>     tokenizer,
        >>>     "model-q4.gguf",
        >>>     quant_type="Q4_K_M"
        >>> )
    """

    def __init__(self):
        """Initialize Unsloth exporter."""
        pass

    def export(
        self,
        model: Any,
        tokenizer: Any,
        output_path: Union[str, Path],
        config: Optional[ExportConfig] = None,
    ) -> Path:
        """
        Export Unsloth model to GGUF format.

        Args:
            model: Unsloth model (with or without adapters)
            tokenizer: Associated tokenizer
            output_path: Output GGUF file path
            config: Export configuration (uses defaults if None)

        Returns:
            Path to exported GGUF file

        Example:
            >>> exporter = UnslothExporter()
            >>> exporter.export(model, tokenizer, "model.gguf")
        """
        if config is None:
            config = ExportConfig()

        output_path = Path(output_path)

        if config.verbose:
            print(f"Exporting Unsloth model to GGUF")
            print(f"  Output: {output_path}")
            print(f"  Quantization: {config.quant_type}")

        # Check if model has LoRA adapters
        has_adapters = self._check_has_adapters(model)

        if has_adapters and config.merge_lora:
            if config.verbose:
                print("  Merging LoRA adapters...")
            model = self._merge_adapters(model)

        # Extract base model if wrapped
        base_model = self._extract_base_model(model)

        # Export to GGUF using quantization module
        from ..quantization import convert_to_gguf

        if config.verbose:
            print(f"  Converting to GGUF with {config.quant_type}...")

        convert_to_gguf(
            base_model,
            output_path,
            tokenizer=tokenizer,
            quant_type=config.quant_type,
            verbose=config.verbose,
        )

        # Save tokenizer if requested
        if config.preserve_tokenizer:
            self._save_tokenizer(tokenizer, output_path.parent)

        # Save metadata
        if config.metadata:
            self._save_metadata(config.metadata, output_path)

        if config.verbose:
            print(f"✓ Export complete: {output_path}")

        return output_path

    def _check_has_adapters(self, model: Any) -> bool:
        """Check if model has LoRA adapters."""
        try:
            from peft import PeftModel
            return isinstance(model, PeftModel)
        except ImportError:
            return False

    def _merge_adapters(self, model: Any) -> Any:
        """Merge LoRA adapters into base model."""
        try:
            # PEFT method
            return model.merge_and_unload()
        except Exception as e:
            print(f"Warning: Could not merge adapters: {e}")
            return model

    def _extract_base_model(self, model: Any) -> Any:
        """Extract base model from wrappers."""
        # Try to unwrap PEFT model
        if hasattr(model, 'base_model'):
            model = model.base_model

        # Try to unwrap other wrappers
        if hasattr(model, 'model'):
            model = model.model

        return model

    def _save_tokenizer(self, tokenizer: Any, output_dir: Path):
        """Save tokenizer alongside GGUF."""
        try:
            tokenizer_path = output_dir / "tokenizer"
            tokenizer_path.mkdir(exist_ok=True, parents=True)
            tokenizer.save_pretrained(str(tokenizer_path))
            print(f"  ✓ Tokenizer saved to {tokenizer_path}")
        except Exception as e:
            print(f"  Warning: Could not save tokenizer: {e}")

    def _save_metadata(self, metadata: Dict[str, Any], gguf_path: Path):
        """Save additional metadata as JSON."""
        try:
            metadata_path = gguf_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  ✓ Metadata saved to {metadata_path}")
        except Exception as e:
            print(f"  Warning: Could not save metadata: {e}")

    def export_with_unsloth_native(
        self,
        model: Any,
        tokenizer: Any,
        output_dir: Union[str, Path],
        quant_method: str = "q4_k_m",
    ):
        """
        Export using Unsloth's native save_pretrained_gguf method.

        This uses Unsloth's built-in export functionality when available.

        Args:
            model: Unsloth model
            tokenizer: Tokenizer
            output_dir: Output directory
            quant_method: Quantization method

        Example:
            >>> exporter.export_with_unsloth_native(
            >>>     model, tokenizer, "output", quant_method="q4_k_m"
            >>> )
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"Exporting with Unsloth native method")
        print(f"  Output directory: {output_dir}")
        print(f"  Quantization: {quant_method}")

        # Check if Unsloth export is available
        if not hasattr(model, 'save_pretrained_gguf'):
            raise AttributeError(
                "Model does not have save_pretrained_gguf method. "
                "Use export() instead."
            )

        try:
            model.save_pretrained_gguf(
                str(output_dir),
                tokenizer,
                quantization_method=quant_method,
            )
            print(f"✓ Export complete using Unsloth native method")
        except Exception as e:
            print(f"Error during Unsloth export: {e}")
            print("Falling back to llamatelemetry export method...")

            # Fallback to llamatelemetry method
            gguf_path = output_dir / f"model-{quant_method}.gguf"
            config = ExportConfig(quant_type=quant_method.upper())
            self.export(model, tokenizer, gguf_path, config)


def export_to_llamatelemetry(
    model: Any,
    tokenizer: Any,
    output_path: Union[str, Path],
    quant_type: str = "Q4_K_M",
    merge_lora: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Export Unsloth model for llamatelemetry inference (convenience function).

    Args:
        model: Unsloth model
        tokenizer: Tokenizer
        output_path: Output GGUF path
        quant_type: Quantization type
        merge_lora: Merge LoRA adapters
        verbose: Print progress

    Returns:
        Path to exported file

    Example:
        >>> from unsloth import FastLanguageModel
        >>> model, tokenizer = FastLanguageModel.from_pretrained("my_model")
        >>> export_to_llamatelemetry(model, tokenizer, "model.gguf")
    """
    config = ExportConfig(
        quant_type=quant_type,
        merge_lora=merge_lora,
        verbose=verbose,
    )

    exporter = UnslothExporter()
    return exporter.export(model, tokenizer, output_path, config)


def export_to_gguf(
    model: Any,
    tokenizer: Any,
    output_path: Union[str, Path],
    quant_type: str = "Q4_K_M",
    **kwargs,
) -> Path:
    """
    Alias for export_to_llamatelemetry (for compatibility).

    Args:
        model: Model to export
        tokenizer: Tokenizer
        output_path: Output path
        quant_type: Quantization type
        **kwargs: Additional arguments for ExportConfig

    Returns:
        Path to exported file
    """
    return export_to_llamatelemetry(model, tokenizer, output_path, quant_type, **kwargs)
