"""
llamatelemetry.api.gguf - GGUF Model Utilities

Comprehensive tools for working with GGUF (GPT-Generated Unified Format) models.
Includes model info extraction, quantization, conversion, and metadata handling.
"""

import os
import subprocess
import struct
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, BinaryIO, Tuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum


# =============================================================================
# GGUF Constants
# =============================================================================

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3  # Current version
GGUF_DEFAULT_ALIGNMENT = 32


class GGUFValueType(IntEnum):
    """GGUF metadata value types."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(IntEnum):
    """GGML tensor quantization types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    BF16 = 29
    Q4_0_4_4 = 30
    Q4_0_4_8 = 31
    Q4_0_8_8 = 32
    TQ1_0 = 33
    TQ2_0 = 34


# Common quantization type mappings for CLI tools
QUANT_TYPE_NAMES = {
    "F32": GGMLType.F32,
    "F16": GGMLType.F16,
    "BF16": GGMLType.BF16,
    "Q8_0": GGMLType.Q8_0,
    "Q6_K": GGMLType.Q6_K,
    "Q5_K": GGMLType.Q5_K,
    "Q5_K_M": GGMLType.Q5_K,
    "Q5_K_S": GGMLType.Q5_K,
    "Q5_0": GGMLType.Q5_0,
    "Q4_K": GGMLType.Q4_K,
    "Q4_K_M": GGMLType.Q4_K,
    "Q4_K_S": GGMLType.Q4_K,
    "Q4_0": GGMLType.Q4_0,
    "Q3_K": GGMLType.Q3_K,
    "Q3_K_M": GGMLType.Q3_K,
    "Q3_K_S": GGMLType.Q3_K,
    "Q3_K_L": GGMLType.Q3_K,
    "Q2_K": GGMLType.Q2_K,
    "IQ4_XS": GGMLType.IQ4_XS,
    "IQ4_NL": GGMLType.IQ4_NL,
    "IQ3_S": GGMLType.IQ3_S,
    "IQ3_XXS": GGMLType.IQ3_XXS,
    "IQ2_S": GGMLType.IQ2_S,
    "IQ2_XS": GGMLType.IQ2_XS,
    "IQ2_XXS": GGMLType.IQ2_XXS,
    "IQ1_S": GGMLType.IQ1_S,
}


# =============================================================================
# Quantization Type Information (from gguf-docs by Julia Turc)
# =============================================================================

@dataclass
class QuantTypeInfo:
    """
    Quantization type information.
    
    Based on gguf-docs (https://github.com/iuliaturc/gguf-docs)
    """
    name: str
    generation: str  # "legacy", "k-quant", "i-quant"
    bits_per_weight: float  # Approximate bits per weight
    quality_score: int  # 1-10 relative quality
    requires_imatrix: bool  # Whether importance matrix is recommended
    description: str


# Quantization type details for user guidance
QUANT_TYPE_INFO = {
    # Legacy quants (1st generation) - affine scalar quantization
    "Q4_0": QuantTypeInfo("Q4_0", "legacy", 4.0, 5, False, "4-bit symmetric quantization"),
    "Q4_1": QuantTypeInfo("Q4_1", "legacy", 4.5, 6, False, "4-bit asymmetric quantization with offset"),
    "Q5_0": QuantTypeInfo("Q5_0", "legacy", 5.0, 6, False, "5-bit symmetric quantization"),
    "Q5_1": QuantTypeInfo("Q5_1", "legacy", 5.5, 7, False, "5-bit asymmetric quantization with offset"),
    "Q8_0": QuantTypeInfo("Q8_0", "legacy", 8.0, 9, False, "8-bit symmetric quantization (near-lossless)"),
    "Q8_1": QuantTypeInfo("Q8_1", "legacy", 8.5, 9, False, "8-bit asymmetric quantization"),
    
    # K-quants (2nd generation) - double quantization + mixed precision
    "Q2_K": QuantTypeInfo("Q2_K", "k-quant", 2.6, 3, True, "2-bit K-quant with super-blocks"),
    "Q3_K_S": QuantTypeInfo("Q3_K_S", "k-quant", 3.4, 5, True, "3-bit K-quant, small variant"),
    "Q3_K_M": QuantTypeInfo("Q3_K_M", "k-quant", 3.9, 6, True, "3-bit K-quant, medium variant"),
    "Q3_K_L": QuantTypeInfo("Q3_K_L", "k-quant", 4.3, 6, False, "3-bit K-quant, large variant"),
    "Q4_K_S": QuantTypeInfo("Q4_K_S", "k-quant", 4.5, 7, False, "4-bit K-quant, small variant"),
    "Q4_K_M": QuantTypeInfo("Q4_K_M", "k-quant", 4.8, 8, False, "4-bit K-quant, medium (RECOMMENDED)"),
    "Q5_K_S": QuantTypeInfo("Q5_K_S", "k-quant", 5.5, 8, False, "5-bit K-quant, small variant"),
    "Q5_K_M": QuantTypeInfo("Q5_K_M", "k-quant", 5.7, 9, False, "5-bit K-quant, medium variant"),
    "Q6_K": QuantTypeInfo("Q6_K", "k-quant", 6.6, 9, False, "6-bit K-quant (best K-quant quality)"),
    
    # I-quants (3rd generation) - vector quantization
    "IQ1_S": QuantTypeInfo("IQ1_S", "i-quant", 1.5, 1, True, "~1.5 bpw vector quant (experimental)"),
    "IQ2_XXS": QuantTypeInfo("IQ2_XXS", "i-quant", 2.0, 2, True, "~2.0 bpw, extra extra small"),
    "IQ2_XS": QuantTypeInfo("IQ2_XS", "i-quant", 2.3, 3, True, "~2.3 bpw, extra small"),
    "IQ2_S": QuantTypeInfo("IQ2_S", "i-quant", 2.5, 4, True, "~2.5 bpw, small"),
    "IQ3_XXS": QuantTypeInfo("IQ3_XXS", "i-quant", 3.0, 5, True, "~3.0 bpw, extra extra small"),
    "IQ3_XS": QuantTypeInfo("IQ3_XS", "i-quant", 3.3, 6, True, "~3.3 bpw, extra small (good for 70B)"),
    "IQ3_S": QuantTypeInfo("IQ3_S", "i-quant", 3.5, 6, True, "~3.5 bpw, small"),
    "IQ4_XS": QuantTypeInfo("IQ4_XS", "i-quant", 4.0, 7, False, "~4.0 bpw, near Q4_K quality"),
    "IQ4_NL": QuantTypeInfo("IQ4_NL", "i-quant", 4.5, 8, False, "~4.5 bpw, non-linear"),
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GGUFMetadata:
    """
    GGUF model metadata.
    
    Contains all metadata extracted from a GGUF file header.
    """
    # General
    general_architecture: str = ""
    general_name: str = ""
    general_author: str = ""
    general_url: str = ""
    general_description: str = ""
    general_license: str = ""
    general_file_type: int = 0
    general_quantization_version: int = 0
    
    # Model architecture
    context_length: int = 0
    embedding_length: int = 0
    block_count: int = 0
    head_count: int = 0
    head_count_kv: int = 0
    rope_freq_base: float = 0.0
    rope_scaling_type: str = ""
    
    # Tokenizer
    tokenizer_model: str = ""
    vocab_size: int = 0
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: int = 0
    chat_template: str = ""
    
    # Raw metadata
    raw: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def param_count(self) -> int:
        """Estimate parameter count from architecture."""
        if not self.block_count or not self.embedding_length:
            return 0
        
        # Rough estimate based on transformer architecture
        d_model = self.embedding_length
        n_layers = self.block_count
        n_heads = self.head_count or 32
        
        # Attention weights
        attn_params = n_layers * 4 * d_model * d_model
        
        # FFN weights (assume 4x expansion)
        ffn_params = n_layers * 3 * d_model * d_model * 4
        
        # Embeddings
        embed_params = self.vocab_size * d_model * 2
        
        return attn_params + ffn_params + embed_params
    
    @property
    def param_count_b(self) -> float:
        """Parameter count in billions."""
        return self.param_count / 1e9


@dataclass
class GGUFTensorInfo:
    """Information about a tensor in the GGUF file."""
    name: str
    n_dims: int
    shape: Tuple[int, ...]
    dtype: GGMLType
    offset: int
    
    @property
    def n_elements(self) -> int:
        """Total number of elements in the tensor."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result


@dataclass
class GGUFModelInfo:
    """
    Complete GGUF model information.
    
    Contains metadata, tensor information, and file statistics.
    """
    path: str
    file_size: int
    version: int
    tensor_count: int
    metadata_kv_count: int
    metadata: GGUFMetadata
    tensors: List[GGUFTensorInfo] = field(default_factory=list)
    
    @property
    def size_gb(self) -> float:
        """File size in GB."""
        return self.file_size / (1024**3)
    
    @property
    def quantization_type(self) -> str:
        """Infer quantization type from tensors."""
        if not self.tensors:
            return "unknown"
        
        # Look at the first few weight tensors
        weight_types = {}
        for tensor in self.tensors:
            if "weight" in tensor.name.lower():
                dtype_name = tensor.dtype.name if hasattr(tensor.dtype, 'name') else str(tensor.dtype)
                weight_types[dtype_name] = weight_types.get(dtype_name, 0) + 1
        
        if not weight_types:
            return "unknown"
        
        # Return most common type
        return max(weight_types, key=weight_types.get)


# =============================================================================
# GGUF Reader (Simplified)
# =============================================================================

def read_string(f: BinaryIO) -> str:
    """Read a GGUF string (length-prefixed)."""
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def read_value(f: BinaryIO, value_type: int) -> Any:
    """Read a typed value from GGUF file."""
    if value_type == GGUFValueType.UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif value_type == GGUFValueType.INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif value_type == GGUFValueType.UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif value_type == GGUFValueType.INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif value_type == GGUFValueType.UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif value_type == GGUFValueType.INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif value_type == GGUFValueType.FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif value_type == GGUFValueType.UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif value_type == GGUFValueType.INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif value_type == GGUFValueType.FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    elif value_type == GGUFValueType.BOOL:
        return struct.unpack("<B", f.read(1))[0] != 0
    elif value_type == GGUFValueType.STRING:
        return read_string(f)
    elif value_type == GGUFValueType.ARRAY:
        array_type = struct.unpack("<I", f.read(4))[0]
        array_len = struct.unpack("<Q", f.read(8))[0]
        return [read_value(f, array_type) for _ in range(array_len)]
    else:
        raise ValueError(f"Unknown value type: {value_type}")


def parse_gguf_header(path: str, read_tensors: bool = False) -> GGUFModelInfo:
    """
    Parse GGUF file header and extract model information.
    
    Args:
        path: Path to GGUF file
        read_tensors: Whether to read tensor information (slower)
        
    Returns:
        GGUFModelInfo with extracted data
        
    Example:
        >>> info = parse_gguf_header("model.gguf")
        >>> print(f"Model: {info.metadata.general_name}")
        >>> print(f"Architecture: {info.metadata.general_architecture}")
        >>> print(f"Context length: {info.metadata.context_length}")
    """
    with open(path, "rb") as f:
        # Read magic and version
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF file: wrong magic number {hex(magic)}")
        
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", f.read(8))[0]
        
        # Read metadata key-value pairs
        raw_metadata: Dict[str, Any] = {}
        for _ in range(metadata_kv_count):
            key = read_string(f)
            value_type = struct.unpack("<I", f.read(4))[0]
            value = read_value(f, value_type)
            raw_metadata[key] = value
        
        # Parse into structured metadata
        metadata = GGUFMetadata(raw=raw_metadata)
        
        # General fields
        metadata.general_architecture = raw_metadata.get("general.architecture", "")
        metadata.general_name = raw_metadata.get("general.name", "")
        metadata.general_author = raw_metadata.get("general.author", "")
        metadata.general_url = raw_metadata.get("general.url", "")
        metadata.general_description = raw_metadata.get("general.description", "")
        metadata.general_license = raw_metadata.get("general.license", "")
        metadata.general_file_type = raw_metadata.get("general.file_type", 0)
        metadata.general_quantization_version = raw_metadata.get("general.quantization_version", 0)
        
        # Architecture-specific fields
        arch = metadata.general_architecture
        if arch:
            metadata.context_length = raw_metadata.get(f"{arch}.context_length", 0)
            metadata.embedding_length = raw_metadata.get(f"{arch}.embedding_length", 0)
            metadata.block_count = raw_metadata.get(f"{arch}.block_count", 0)
            metadata.head_count = raw_metadata.get(f"{arch}.attention.head_count", 0)
            metadata.head_count_kv = raw_metadata.get(f"{arch}.attention.head_count_kv", 0)
            metadata.rope_freq_base = raw_metadata.get(f"{arch}.rope.freq_base", 0.0)
            metadata.rope_scaling_type = raw_metadata.get(f"{arch}.rope.scaling.type", "")
        
        # Tokenizer fields
        metadata.tokenizer_model = raw_metadata.get("tokenizer.ggml.model", "")
        metadata.vocab_size = len(raw_metadata.get("tokenizer.ggml.tokens", []))
        metadata.bos_token_id = raw_metadata.get("tokenizer.ggml.bos_token_id", 0)
        metadata.eos_token_id = raw_metadata.get("tokenizer.ggml.eos_token_id", 0)
        metadata.pad_token_id = raw_metadata.get("tokenizer.ggml.padding_token_id", 0)
        metadata.chat_template = raw_metadata.get("tokenizer.chat_template", "")
        
        # Read tensor info if requested
        tensors: List[GGUFTensorInfo] = []
        if read_tensors:
            for _ in range(tensor_count):
                name = read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
                dtype = GGMLType(struct.unpack("<I", f.read(4))[0])
                offset = struct.unpack("<Q", f.read(8))[0]
                
                tensors.append(GGUFTensorInfo(
                    name=name,
                    n_dims=n_dims,
                    shape=shape,
                    dtype=dtype,
                    offset=offset
                ))
        
        file_size = os.path.getsize(path)
        
        return GGUFModelInfo(
            path=path,
            file_size=file_size,
            version=version,
            tensor_count=tensor_count,
            metadata_kv_count=metadata_kv_count,
            metadata=metadata,
            tensors=tensors
        )


# =============================================================================
# GGUF CLI Tool Wrappers
# =============================================================================

def find_llama_tool(name: str) -> Optional[str]:
    """
    Find a llama.cpp tool binary.
    
    Args:
        name: Tool name (e.g., "llama-quantize", "llama-gguf")
        
    Returns:
        Path to tool or None if not found
    """
    # Check common locations
    search_paths = [
        # llamatelemetry bundle location
        os.path.expanduser("~/.llamatelemetry/bin"),
        # Built from source
        "./build/bin",
        "../llama.cpp/build/bin",
        # System PATH
        ""
    ]
    
    for base in search_paths:
        if base:
            tool_path = os.path.join(base, name)
        else:
            tool_path = shutil.which(name)
            if tool_path:
                return tool_path
            continue
        
        if os.path.isfile(tool_path) and os.access(tool_path, os.X_OK):
            return tool_path
    
    return None


def quantize(
    input_path: str,
    output_path: str,
    quant_type: str = "Q4_K_M",
    n_threads: Optional[int] = None,
    allow_requantize: bool = False,
    leave_output_tensor: bool = False,
    pure: bool = False,
    imatrix_path: Optional[str] = None,
    include_weights: Optional[List[str]] = None,
    exclude_weights: Optional[List[str]] = None,
    output_tensor_type: Optional[str] = None,
    token_embedding_type: Optional[str] = None,
    llama_quantize_path: Optional[str] = None
) -> bool:
    """
    Quantize a GGUF model to a different precision.
    
    Args:
        input_path: Input GGUF file
        output_path: Output GGUF file
        quant_type: Target quantization (Q4_K_M, Q8_0, etc.)
        n_threads: Number of threads
        allow_requantize: Allow requantizing already-quantized model
        leave_output_tensor: Leave output.weight as is
        pure: Disable k-quant mixtures
        imatrix_path: Path to importance matrix file
        include_weights: Tensor name patterns to include
        exclude_weights: Tensor name patterns to exclude
        output_tensor_type: Output tensor quantization
        token_embedding_type: Token embedding quantization
        llama_quantize_path: Path to llama-quantize binary
        
    Returns:
        True if successful
        
    Example:
        >>> quantize("model-f16.gguf", "model-q4km.gguf", "Q4_K_M")
    """
    tool = llama_quantize_path or find_llama_tool("llama-quantize")
    if not tool:
        raise FileNotFoundError("llama-quantize not found. Install llama.cpp first.")
    
    cmd = [tool, input_path, output_path, quant_type]
    
    if n_threads:
        cmd.extend(["--threads", str(n_threads)])
    if allow_requantize:
        cmd.append("--allow-requantize")
    if leave_output_tensor:
        cmd.append("--leave-output-tensor")
    if pure:
        cmd.append("--pure")
    if imatrix_path:
        cmd.extend(["--imatrix", imatrix_path])
    if include_weights:
        for pattern in include_weights:
            cmd.extend(["--include-weights", pattern])
    if exclude_weights:
        for pattern in exclude_weights:
            cmd.extend(["--exclude-weights", pattern])
    if output_tensor_type:
        cmd.extend(["--output-tensor-type", output_tensor_type])
    if token_embedding_type:
        cmd.extend(["--token-embedding-type", token_embedding_type])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def convert_hf_to_gguf(
    model_path: str,
    output_path: Optional[str] = None,
    outtype: str = "f16",
    vocab_only: bool = False,
    pad_vocab: bool = False,
    skip_unknown: bool = False,
    metadata: Optional[Dict[str, str]] = None,
    python_path: str = "python3",
    convert_script: Optional[str] = None
) -> bool:
    """
    Convert Hugging Face model to GGUF format.
    
    Args:
        model_path: Path to HuggingFace model directory
        output_path: Output GGUF file path
        outtype: Output type (f32, f16, bf16, q8_0)
        vocab_only: Only convert vocabulary
        pad_vocab: Add padding token if missing
        skip_unknown: Skip unknown tensors
        metadata: Additional metadata key-value pairs
        python_path: Python interpreter path
        convert_script: Path to convert_hf_to_gguf.py
        
    Returns:
        True if successful
        
    Example:
        >>> convert_hf_to_gguf("./my_model", "./my_model.gguf", outtype="f16")
    """
    # Find convert script
    if not convert_script:
        script_locations = [
            os.path.expanduser("~/.llamatelemetry/scripts/convert_hf_to_gguf.py"),
            "../llama.cpp/convert_hf_to_gguf.py",
            "./convert_hf_to_gguf.py"
        ]
        for loc in script_locations:
            if os.path.isfile(loc):
                convert_script = loc
                break
        else:
            raise FileNotFoundError("convert_hf_to_gguf.py not found")
    
    cmd = [python_path, convert_script, model_path]
    
    if output_path:
        cmd.extend(["--outfile", output_path])
    cmd.extend(["--outtype", outtype])
    
    if vocab_only:
        cmd.append("--vocab-only")
    if pad_vocab:
        cmd.append("--pad-vocab")
    if skip_unknown:
        cmd.append("--skip-unknown")
    if metadata:
        for key, value in metadata.items():
            cmd.extend(["--metadata", f"{key}={value}"])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def merge_lora(
    base_model: str,
    lora_path: str,
    output_path: str,
    lora_scale: float = 1.0,
    n_threads: Optional[int] = None,
    llama_export_path: Optional[str] = None
) -> bool:
    """
    Merge LoRA adapter into base model.
    
    Args:
        base_model: Base GGUF model path
        lora_path: LoRA adapter path (GGUF format)
        output_path: Output merged GGUF path
        lora_scale: LoRA scaling factor
        n_threads: Number of threads
        llama_export_path: Path to llama-export-lora binary
        
    Returns:
        True if successful
        
    Example:
        >>> merge_lora("base.gguf", "lora.gguf", "merged.gguf", lora_scale=0.5)
    """
    tool = llama_export_path or find_llama_tool("llama-export-lora")
    if not tool:
        raise FileNotFoundError("llama-export-lora not found")
    
    cmd = [
        tool,
        "--model", base_model,
        "--lora", lora_path,
        "--output", output_path,
        "--lora-scale", str(lora_scale)
    ]
    
    if n_threads:
        cmd.extend(["--threads", str(n_threads)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def generate_imatrix(
    model_path: str,
    output_path: str,
    data_file: str,
    ctx_size: int = 512,
    n_batch: int = 512,
    n_threads: Optional[int] = None,
    n_gpu_layers: int = -1,
    llama_imatrix_path: Optional[str] = None
) -> bool:
    """
    Generate importance matrix for quantization.
    
    The importance matrix helps produce higher quality quantizations
    by measuring which weights are most important during inference.
    
    Args:
        model_path: Model GGUF file
        output_path: Output imatrix file
        data_file: Calibration data file (text or GGUF tokens)
        ctx_size: Context size
        n_batch: Batch size
        n_threads: Number of threads
        n_gpu_layers: GPU layers to offload
        llama_imatrix_path: Path to llama-imatrix binary
        
    Returns:
        True if successful
        
    Example:
        >>> generate_imatrix("model.gguf", "model.imatrix", "calibration.txt")
        >>> quantize("model.gguf", "model-iq.gguf", "IQ4_XS", imatrix_path="model.imatrix")
    """
    tool = llama_imatrix_path or find_llama_tool("llama-imatrix")
    if not tool:
        raise FileNotFoundError("llama-imatrix not found")
    
    cmd = [
        tool,
        "--model", model_path,
        "--output", output_path,
        "--file", data_file,
        "--ctx-size", str(ctx_size),
        "--batch-size", str(n_batch),
        "-ngl", str(n_gpu_layers)
    ]
    
    if n_threads:
        cmd.extend(["--threads", str(n_threads)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# =============================================================================
# Model Discovery
# =============================================================================

def find_gguf_models(directory: str, recursive: bool = True) -> List[str]:
    """
    Find all GGUF files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Search subdirectories
        
    Returns:
        List of GGUF file paths
    """
    pattern = "**/*.gguf" if recursive else "*.gguf"
    return [str(p) for p in Path(directory).glob(pattern)]


def get_model_summary(path: str) -> str:
    """
    Get a human-readable model summary.
    
    Args:
        path: Path to GGUF file
        
    Returns:
        Formatted summary string
    """
    try:
        info = parse_gguf_header(path)
        meta = info.metadata
        
        lines = [
            f"Model: {meta.general_name or Path(path).stem}",
            f"Architecture: {meta.general_architecture}",
            f"Size: {info.size_gb:.2f} GB",
            f"Parameters: ~{meta.param_count_b:.1f}B" if meta.param_count_b > 0 else "",
            f"Context: {meta.context_length:,}" if meta.context_length else "",
            f"Vocab: {meta.vocab_size:,}" if meta.vocab_size else "",
            f"Tensors: {info.tensor_count:,}",
        ]
        
        if meta.general_author:
            lines.append(f"Author: {meta.general_author}")
        if meta.general_license:
            lines.append(f"License: {meta.general_license}")
        if meta.chat_template:
            lines.append(f"Chat template: Yes")
        
        return "\n".join(line for line in lines if line)
    except Exception as e:
        return f"Error reading {path}: {e}"


def compare_models(path1: str, path2: str) -> Dict[str, Any]:
    """
    Compare two GGUF models.
    
    Args:
        path1: First model path
        path2: Second model path
        
    Returns:
        Dictionary with comparison results
    """
    info1 = parse_gguf_header(path1)
    info2 = parse_gguf_header(path2)
    
    return {
        "size_diff_gb": info2.size_gb - info1.size_gb,
        "size_ratio": info2.file_size / info1.file_size,
        "same_architecture": info1.metadata.general_architecture == info2.metadata.general_architecture,
        "same_vocab": info1.metadata.vocab_size == info2.metadata.vocab_size,
        "same_context": info1.metadata.context_length == info2.metadata.context_length,
        "model1": {
            "name": info1.metadata.general_name,
            "size_gb": info1.size_gb,
            "quantization": info1.quantization_type
        },
        "model2": {
            "name": info2.metadata.general_name,
            "size_gb": info2.size_gb,
            "quantization": info2.quantization_type
        }
    }


# =============================================================================
# Helper Functions
# =============================================================================

def validate_gguf(path: str) -> Tuple[bool, str]:
    """
    Validate a GGUF file.
    
    Args:
        path: Path to GGUF file
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        with open(path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != GGUF_MAGIC:
                return False, f"Invalid magic number: {hex(magic)}"
            
            version = struct.unpack("<I", f.read(4))[0]
            if version > GGUF_VERSION:
                return False, f"Unsupported GGUF version: {version}"
            
            # Try to read tensor count
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            if tensor_count == 0:
                return False, "No tensors in file"
        
        return True, f"Valid GGUF v{version} with {tensor_count} tensors"
    except Exception as e:
        return False, f"Error reading file: {e}"


def get_recommended_quant(
    original_size_gb: float,
    target_size_gb: float
) -> str:
    """
    Recommend quantization type to achieve target size.
    
    Args:
        original_size_gb: Original model size in GB (F16)
        target_size_gb: Target size in GB
        
    Returns:
        Recommended quantization type
    """
    ratio = target_size_gb / original_size_gb
    
    # Approximate size ratios for different quantization types
    quant_ratios = [
        (0.12, "IQ1_S"),
        (0.15, "IQ2_XS"),
        (0.18, "IQ2_S"),
        (0.22, "IQ3_S"),
        (0.25, "Q2_K"),
        (0.28, "IQ4_XS"),
        (0.32, "Q3_K_S"),
        (0.35, "Q3_K_M"),
        (0.38, "Q4_K_S"),
        (0.42, "Q4_K_M"),
        (0.48, "Q5_K_S"),
        (0.52, "Q5_K_M"),
        (0.58, "Q6_K"),
        (0.65, "Q8_0"),
        (1.0, "F16"),
    ]
    
    for threshold, quant in quant_ratios:
        if ratio <= threshold:
            return quant
    
    return "F16"


# =============================================================================
# Kaggle T4 Quantization Helpers
# =============================================================================

def estimate_gguf_size(
    param_count: int,
    quant_type: str
) -> float:
    """
    Estimate GGUF file size for a given model and quantization.
    
    Args:
        param_count: Number of model parameters
        quant_type: Quantization type (Q4_K_M, IQ3_XS, etc.)
        
    Returns:
        Estimated file size in GB
        
    Example:
        >>> size = estimate_gguf_size(7_000_000_000, "Q4_K_M")  # 7B model
        >>> print(f"{size:.1f} GB")  # ~4.0 GB
    """
    # Get bits per weight from QUANT_TYPE_INFO
    if quant_type in QUANT_TYPE_INFO:
        bpw = QUANT_TYPE_INFO[quant_type].bits_per_weight
    else:
        # Default estimates
        bpw_map = {
            "F32": 32.0, "F16": 16.0, "BF16": 16.0,
            "Q8_0": 8.0, "Q6_K": 6.6, "Q5_K_M": 5.7,
            "Q4_K_M": 4.8, "Q4_0": 4.0, "Q3_K_M": 3.9,
        }
        bpw = bpw_map.get(quant_type, 4.8)
    
    # Size = params * bits_per_weight / 8 (bytes) / 1e9 (GB)
    # Add ~10% overhead for metadata and quantization constants
    size_gb = (param_count * bpw / 8 / 1e9) * 1.1
    return size_gb


def recommend_quant_for_kaggle(
    param_count: int,
    dual_t4: bool = True,
    context_size: int = 4096,
    prefer_quality: bool = True
) -> Dict[str, Any]:
    """
    Recommend quantization for Kaggle T4 environment.
    
    Based on gguf-docs by Julia Turc.
    
    Args:
        param_count: Number of model parameters (e.g., 7_000_000_000 for 7B)
        dual_t4: Whether using dual T4 (30GB total) or single (15GB)
        context_size: Target context window size
        prefer_quality: Prefer quality over fitting larger models
        
    Returns:
        Dictionary with recommendation details
        
    Example:
        >>> rec = recommend_quant_for_kaggle(70_000_000_000, dual_t4=True)
        >>> print(rec["quant_type"])  # "IQ3_XS"
        >>> print(rec["fits"])  # True
    """
    available_vram = 30.0 if dual_t4 else 15.0
    
    # Reserve VRAM for KV cache: ~1GB per 4096 context for 7B model
    # Scale proportionally with model size
    kv_cache_gb = (context_size / 4096) * (param_count / 7e9) * 1.0
    usable_vram = available_vram - kv_cache_gb - 1.0  # 1GB safety margin
    
    # Try quantizations in order of quality (descending)
    quality_order = [
        "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S",
        "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S",
        "IQ4_XS", "IQ3_XS", "IQ3_S", "IQ2_S", "IQ2_XS"
    ]
    
    # If not preferring quality, try smaller quants first for large models
    if not prefer_quality and param_count > 30e9:
        quality_order = [
            "IQ3_XS", "IQ3_S", "IQ4_XS", "Q3_K_M",
            "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q6_K"
        ]
    
    for quant in quality_order:
        estimated_size = estimate_gguf_size(param_count, quant)
        if estimated_size <= usable_vram:
            info = QUANT_TYPE_INFO.get(quant)
            return {
                "quant_type": quant,
                "fits": True,
                "estimated_size_gb": round(estimated_size, 1),
                "usable_vram_gb": round(usable_vram, 1),
                "headroom_gb": round(usable_vram - estimated_size, 1),
                "quality_score": info.quality_score if info else 7,
                "requires_imatrix": info.requires_imatrix if info else False,
                "description": info.description if info else "",
                "generation": info.generation if info else "k-quant",
            }
    
    # Nothing fits
    smallest_quant = "IQ2_XS"
    return {
        "quant_type": smallest_quant,
        "fits": False,
        "estimated_size_gb": round(estimate_gguf_size(param_count, smallest_quant), 1),
        "usable_vram_gb": round(usable_vram, 1),
        "headroom_gb": -1,
        "quality_score": 3,
        "requires_imatrix": True,
        "description": "Model too large for Kaggle T4",
        "generation": "i-quant",
    }


def print_quant_guide():
    """
    Print a guide to GGUF quantization types.
    
    Useful for interactive sessions in Kaggle notebooks.
    """
    print("=" * 70)
    print("GGUF Quantization Guide for Kaggle T4")
    print("=" * 70)
    print()
    print("LEGEND:")
    print("  BPW = Bits Per Weight (lower = smaller file)")
    print("  Quality = Relative quality score (1-10, higher = better)")
    print("  Imatrix = Whether importance matrix is recommended")
    print()
    print("-" * 70)
    print(f"{'Type':<12} {'Gen':<8} {'BPW':<6} {'Quality':<8} {'Imatrix':<8} Description")
    print("-" * 70)
    
    # Group by generation
    for gen in ["legacy", "k-quant", "i-quant"]:
        for name, info in QUANT_TYPE_INFO.items():
            if info.generation == gen:
                imat = "Yes" if info.requires_imatrix else "No"
                print(f"{name:<12} {info.generation:<8} {info.bits_per_weight:<6.1f} "
                      f"{info.quality_score:<8} {imat:<8} {info.description}")
    
    print("-" * 70)
    print()
    print("KAGGLE RECOMMENDATIONS:")
    print("  Single T4 (15GB): Q4_K_M for 7B, Q3_K_M for 13B")
    print("  Dual T4 (30GB):   Q6_K for 7B, Q4_K_M for 13B, IQ3_XS for 70B")
    print()
    print("See: https://github.com/iuliaturc/gguf-docs")
    print("=" * 70)
