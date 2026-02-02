"""
Unsloth Model Loader

Load Unsloth fine-tuned models directly into llamatelemetry-compatible format.
Supports both local and HuggingFace Hub models.
"""

import torch
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import sys


def check_unsloth_available() -> bool:
    """
    Check if Unsloth is installed.

    Returns:
        True if Unsloth is available

    Example:
        >>> if check_unsloth_available():
        >>>     print("Unsloth is installed")
    """
    try:
        import unsloth
        return True
    except ImportError:
        return False


class UnslothModelLoader:
    """
    Loader for Unsloth fine-tuned models.

    Handles loading Unsloth models with LoRA adapters and prepares them
    for GGUF export or direct inference with llamatelemetry.

    Example:
        >>> loader = UnslothModelLoader()
        >>> model, tokenizer = loader.load("path/to/unsloth_model")
        >>> # Or from HuggingFace
        >>> model, tokenizer = loader.load("username/model-name")
    """

    def __init__(
        self,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize Unsloth model loader.

        Args:
            max_seq_length: Maximum sequence length
            load_in_4bit: Load model in 4-bit (recommended)
            dtype: Data type (auto-detect if None)
        """
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit

        if dtype is None:
            # Auto-detect best dtype
            import torch
            major, minor = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.dtype = dtype

    def load(
        self,
        model_name: str,
        adapter_path: Optional[str] = None,
        merge_adapters: bool = False,
    ) -> Tuple[Any, Any]:
        """
        Load Unsloth model and tokenizer.

        Args:
            model_name: Model name or path (local or HuggingFace)
            adapter_path: Optional path to LoRA adapters
            merge_adapters: Merge adapters into base model

        Returns:
            Tuple of (model, tokenizer)

        Example:
            >>> loader = UnslothModelLoader()
            >>> model, tokenizer = loader.load(
            >>>     "unsloth/llama-3-8b-Instruct",
            >>>     adapter_path="./my_adapters",
            >>>     merge_adapters=True
            >>> )
        """
        if not check_unsloth_available():
            raise ImportError(
                "Unsloth is not installed. Install with:\n"
                "  pip install unsloth"
            )

        from unsloth import FastLanguageModel

        print(f"Loading Unsloth model: {model_name}")
        print(f"  Max sequence length: {self.max_seq_length}")
        print(f"  Load in 4-bit: {self.load_in_4bit}")
        print(f"  Dtype: {self.dtype}")

        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )

        # Load adapters if specified
        if adapter_path is not None:
            print(f"Loading adapters from: {adapter_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)

            if merge_adapters:
                print("Merging adapters into base model...")
                model = model.merge_and_unload()

        print("✓ Model loaded successfully")

        return model, tokenizer

    def load_for_inference(
        self,
        model_name: str,
        adapter_path: Optional[str] = None,
    ) -> Tuple[Any, Any]:
        """
        Load model optimized for inference.

        Automatically enables inference mode and sets optimal configs.

        Args:
            model_name: Model name or path
            adapter_path: Optional adapter path

        Returns:
            Tuple of (model, tokenizer)
        """
        from unsloth import FastLanguageModel

        model, tokenizer = self.load(
            model_name,
            adapter_path,
            merge_adapters=True  # Always merge for inference
        )

        # Enable inference mode
        FastLanguageModel.for_inference(model)

        return model, tokenizer

    def load_with_peft_config(
        self,
        model_name: str,
        peft_config: Dict[str, Any],
    ) -> Tuple[Any, Any]:
        """
        Load model and apply PEFT configuration.

        Args:
            model_name: Model name
            peft_config: PEFT configuration dict

        Returns:
            Tuple of (model, tokenizer)
        """
        from unsloth import FastLanguageModel
        from peft import get_peft_model, LoraConfig

        # Load base model
        model, tokenizer = self.load(model_name)

        # Apply PEFT config
        lora_config = LoraConfig(**peft_config)
        model = get_peft_model(model, lora_config)

        return model, tokenizer


def load_unsloth_model(
    model_name: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    adapter_path: Optional[str] = None,
    merge_adapters: bool = False,
) -> Tuple[Any, Any]:
    """
    Load Unsloth model (convenience function).

    Args:
        model_name: Model name or path
        max_seq_length: Max sequence length
        load_in_4bit: Load in 4-bit
        adapter_path: Optional adapter path
        merge_adapters: Merge adapters

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load_unsloth_model(
        >>>     "unsloth/llama-3-8b-Instruct",
        >>>     max_seq_length=2048,
        >>> )
    """
    loader = UnslothModelLoader(
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    return loader.load(
        model_name,
        adapter_path=adapter_path,
        merge_adapters=merge_adapters,
    )
