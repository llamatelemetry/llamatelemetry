"""
LoRA Adapter Management

Handle LoRA adapters from Unsloth fine-tuning, including merging,
extraction, and management for llamatelemetry deployment.
"""

import torch
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AdapterConfig:
    """
    Configuration for LoRA adapter handling.

    Attributes:
        adapter_name: Name of the adapter
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        target_modules: List of modules with adapters
        lora_dropout: Dropout rate
    """
    adapter_name: str = "default"
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.0

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


class LoRAAdapter:
    """
    LoRA adapter manager for Unsloth models.

    Handles loading, merging, and extracting LoRA adapters for deployment.

    Example:
        >>> adapter = LoRAAdapter(model)
        >>> if adapter.has_adapters():
        >>>     merged_model = adapter.merge()
        >>>     adapter.save_merged(merged_model, "merged_model")
    """

    def __init__(self, model: Any):
        """
        Initialize adapter manager.

        Args:
            model: Model (with or without adapters)
        """
        self.model = model
        self.adapter_config = None

        # Detect adapter configuration
        self._detect_adapters()

    def _detect_adapters(self):
        """Detect if model has LoRA adapters and extract config."""
        try:
            from peft import PeftModel

            if isinstance(self.model, PeftModel):
                peft_config = self.model.peft_config.get('default', None)
                if peft_config is not None:
                    self.adapter_config = AdapterConfig(
                        adapter_name='default',
                        r=peft_config.r,
                        lora_alpha=peft_config.lora_alpha,
                        target_modules=list(peft_config.target_modules),
                        lora_dropout=peft_config.lora_dropout,
                    )
        except ImportError:
            pass

    def has_adapters(self) -> bool:
        """
        Check if model has LoRA adapters.

        Returns:
            True if adapters are present
        """
        return self.adapter_config is not None

    def merge(self) -> Any:
        """
        Merge LoRA adapters into base model.

        Returns:
            Model with merged adapters

        Example:
            >>> adapter = LoRAAdapter(model)
            >>> merged = adapter.merge()
        """
        if not self.has_adapters():
            print("No adapters to merge")
            return self.model

        print("Merging LoRA adapters...")
        print(f"  Rank: {self.adapter_config.r}")
        print(f"  Alpha: {self.adapter_config.lora_alpha}")
        print(f"  Target modules: {len(self.adapter_config.target_modules)}")

        try:
            merged_model = self.model.merge_and_unload()
            print("✓ Adapters merged successfully")
            return merged_model
        except Exception as e:
            print(f"Error merging adapters: {e}")
            return self.model

    def extract_adapter_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract adapter weights as dictionary.

        Returns:
            Dictionary of adapter weights

        Example:
            >>> adapter = LoRAAdapter(model)
            >>> weights = adapter.extract_adapter_weights()
            >>> print(f"Found {len(weights)} adapter tensors")
        """
        if not self.has_adapters():
            return {}

        adapter_weights = {}

        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                adapter_weights[name] = param.detach().cpu()

        return adapter_weights

    def save_merged(
        self,
        merged_model: Any,
        output_path: str,
        save_tokenizer: bool = True,
        tokenizer: Optional[Any] = None,
    ):
        """
        Save merged model to disk.

        Args:
            merged_model: Model with merged adapters
            output_path: Output directory
            save_tokenizer: Also save tokenizer
            tokenizer: Tokenizer to save (if save_tokenizer=True)

        Example:
            >>> merged = adapter.merge()
            >>> adapter.save_merged(merged, "output", tokenizer=tokenizer)
        """
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        print(f"Saving merged model to {output_path}")

        # Save model
        try:
            merged_model.save_pretrained(str(output_path))
            print("✓ Model saved")
        except Exception as e:
            print(f"Error saving model: {e}")

        # Save tokenizer
        if save_tokenizer and tokenizer is not None:
            try:
                tokenizer.save_pretrained(str(output_path))
                print("✓ Tokenizer saved")
            except Exception as e:
                print(f"Error saving tokenizer: {e}")

    def get_adapter_info(self) -> Dict[str, Any]:
        """
        Get adapter information.

        Returns:
            Dictionary with adapter details

        Example:
            >>> info = adapter.get_adapter_info()
            >>> print(f"LoRA rank: {info['rank']}")
        """
        if not self.has_adapters():
            return {'has_adapters': False}

        return {
            'has_adapters': True,
            'adapter_name': self.adapter_config.adapter_name,
            'rank': self.adapter_config.r,
            'alpha': self.adapter_config.lora_alpha,
            'target_modules': self.adapter_config.target_modules,
            'dropout': self.adapter_config.lora_dropout,
        }


def merge_lora_adapters(model: Any) -> Any:
    """
    Merge LoRA adapters into model (convenience function).

    Args:
        model: Model with LoRA adapters

    Returns:
        Model with merged adapters

    Example:
        >>> merged_model = merge_lora_adapters(model)
    """
    adapter = LoRAAdapter(model)
    return adapter.merge()


def extract_base_model(model: Any) -> Any:
    """
    Extract base model from PEFT wrapper (convenience function).

    Args:
        model: Wrapped model

    Returns:
        Base model without wrapper

    Example:
        >>> base_model = extract_base_model(peft_model)
    """
    # Try PEFT unwrapping
    if hasattr(model, 'base_model'):
        model = model.base_model

    # Try model attribute
    if hasattr(model, 'model'):
        model = model.model

    return model
