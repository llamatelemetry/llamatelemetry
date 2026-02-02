"""
GGUF Parser for llamatelemetry v2.0

Parses GGUF (GGML Universal Format) model files with zero-copy memory mapping.

GGUF Format Specification:
- Magic: 0x47474655 ("GGUF")
- Version: 3 (big-endian support)
- Alignment: Default 32 bytes
- Structure: Header -> Metadata KV pairs -> Tensor info -> Tensor data

Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

import struct
import mmap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import IntEnum
from dataclasses import dataclass


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
    """GGML quantization types."""
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
    IQ1_M = 29
    BF16 = 30
    Q4_0_4_4 = 31
    Q4_0_4_8 = 32
    Q4_0_8_8 = 33


# Type sizes in bytes (for quantized types, this is the block size)
GGML_TYPE_SIZE = {
    GGMLType.F32: 4,
    GGMLType.F16: 2,
    GGMLType.Q4_0: 18,  # 18 bytes per 32 elements
    GGMLType.Q4_1: 20,  # 20 bytes per 32 elements
    GGMLType.Q5_0: 22,
    GGMLType.Q5_1: 24,
    GGMLType.Q8_0: 34,
    GGMLType.Q8_1: 36,
    GGMLType.I8: 1,
    GGMLType.I16: 2,
    GGMLType.I32: 4,
    GGMLType.I64: 8,
    GGMLType.F64: 8,
    GGMLType.BF16: 2,
}

# Block sizes (number of elements per block for quantized types)
GGML_BLOCK_SIZE = {
    GGMLType.F32: 1,
    GGMLType.F16: 1,
    GGMLType.Q4_0: 32,
    GGMLType.Q4_1: 32,
    GGMLType.Q5_0: 32,
    GGMLType.Q5_1: 32,
    GGMLType.Q8_0: 32,
    GGMLType.Q8_1: 32,
    GGMLType.I8: 1,
    GGMLType.I16: 1,
    GGMLType.I32: 1,
    GGMLType.I64: 1,
    GGMLType.F64: 1,
    GGMLType.BF16: 1,
}


@dataclass
class GGUFTensorInfo:
    """Information about a tensor in the GGUF file."""
    name: str
    n_dims: int
    shape: List[int]
    ggml_type: GGMLType
    offset: int  # Offset in the file where tensor data starts

    @property
    def numel(self) -> int:
        """Total number of elements in the tensor."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def nbytes(self) -> int:
        """Total size in bytes."""
        block_size = GGML_BLOCK_SIZE.get(self.ggml_type, 1)
        type_size = GGML_TYPE_SIZE.get(self.ggml_type, 4)
        n_blocks = (self.numel + block_size - 1) // block_size
        return n_blocks * type_size

    def __repr__(self) -> str:
        shape_str = "x".join(map(str, self.shape))
        return f"GGUFTensorInfo(name='{self.name}', shape=[{shape_str}], type={self.ggml_type.name})"


class GGUFReader:
    """
    GGUF file reader with memory-mapped tensor access.

    Usage:
        with GGUFReader("model.gguf") as reader:
            print(f"Model: {reader.metadata.get('general.name', 'unknown')}")
            print(f"Tensors: {len(reader.tensors)}")

            for name, info in reader.tensors.items():
                print(f"{name}: {info.shape} ({info.ggml_type.name})")

            # Get tensor data (memory-mapped, zero-copy)
            tensor_data = reader.get_tensor_data("model.embed_tokens.weight")
    """

    GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
    GGUF_VERSION = 3
    DEFAULT_ALIGNMENT = 32

    def __init__(self, file_path: Union[str, Path]):
        """
        Open GGUF file for reading.

        Args:
            file_path: Path to GGUF file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {file_path}")

        self.file = None
        self.mmap = None
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, GGUFTensorInfo] = {}
        self.alignment = self.DEFAULT_ALIGNMENT
        self.tensor_data_offset = 0

        self._open()
        self._parse_header()

    def _open(self):
        """Open file and create memory map."""
        self.file = open(self.file_path, 'rb')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def _read_uint8(self, offset: int) -> Tuple[int, int]:
        """Read uint8 at offset, return (value, new_offset)."""
        return struct.unpack_from('<B', self.mmap, offset)[0], offset + 1

    def _read_uint32(self, offset: int) -> Tuple[int, int]:
        """Read uint32 at offset, return (value, new_offset)."""
        return struct.unpack_from('<I', self.mmap, offset)[0], offset + 4

    def _read_uint64(self, offset: int) -> Tuple[int, int]:
        """Read uint64 at offset, return (value, new_offset)."""
        return struct.unpack_from('<Q', self.mmap, offset)[0], offset + 8

    def _read_int32(self, offset: int) -> Tuple[int, int]:
        """Read int32 at offset, return (value, new_offset)."""
        return struct.unpack_from('<i', self.mmap, offset)[0], offset + 4

    def _read_int64(self, offset: int) -> Tuple[int, int]:
        """Read int64 at offset, return (value, new_offset)."""
        return struct.unpack_from('<q', self.mmap, offset)[0], offset + 8

    def _read_float32(self, offset: int) -> Tuple[float, int]:
        """Read float32 at offset, return (value, new_offset)."""
        return struct.unpack_from('<f', self.mmap, offset)[0], offset + 4

    def _read_float64(self, offset: int) -> Tuple[float, int]:
        """Read float64 at offset, return (value, new_offset)."""
        return struct.unpack_from('<d', self.mmap, offset)[0], offset + 8

    def _read_string(self, offset: int) -> Tuple[str, int]:
        """Read string at offset, return (string, new_offset)."""
        length, offset = self._read_uint64(offset)
        string = self.mmap[offset:offset + length].decode('utf-8')
        return string, offset + length

    def _read_value(self, offset: int, value_type: GGUFValueType) -> Tuple[Any, int]:
        """Read a value of given type at offset."""
        if value_type == GGUFValueType.UINT8:
            return self._read_uint8(offset)
        elif value_type == GGUFValueType.INT8:
            val, offset = self._read_uint8(offset)
            return struct.unpack('b', struct.pack('B', val))[0], offset
        elif value_type == GGUFValueType.UINT16:
            val = struct.unpack_from('<H', self.mmap, offset)[0]
            return val, offset + 2
        elif value_type == GGUFValueType.INT16:
            val = struct.unpack_from('<h', self.mmap, offset)[0]
            return val, offset + 2
        elif value_type == GGUFValueType.UINT32:
            return self._read_uint32(offset)
        elif value_type == GGUFValueType.INT32:
            return self._read_int32(offset)
        elif value_type == GGUFValueType.UINT64:
            return self._read_uint64(offset)
        elif value_type == GGUFValueType.INT64:
            return self._read_int64(offset)
        elif value_type == GGUFValueType.FLOAT32:
            return self._read_float32(offset)
        elif value_type == GGUFValueType.FLOAT64:
            return self._read_float64(offset)
        elif value_type == GGUFValueType.BOOL:
            val, offset = self._read_uint8(offset)
            return bool(val), offset
        elif value_type == GGUFValueType.STRING:
            return self._read_string(offset)
        elif value_type == GGUFValueType.ARRAY:
            # Read array type
            array_type, offset = self._read_uint32(offset)
            array_type = GGUFValueType(array_type)
            # Read array length
            array_len, offset = self._read_uint64(offset)
            # Read array elements
            array = []
            for _ in range(array_len):
                val, offset = self._read_value(offset, array_type)
                array.append(val)
            return array, offset
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def _parse_header(self):
        """Parse GGUF header, metadata, and tensor info."""
        offset = 0

        # Read magic
        magic, offset = self._read_uint32(offset)
        if magic != self.GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: 0x{magic:08x} (expected 0x{self.GGUF_MAGIC:08x})")

        # Read version
        version, offset = self._read_uint32(offset)
        if version != self.GGUF_VERSION:
            raise ValueError(f"Unsupported GGUF version: {version} (expected {self.GGUF_VERSION})")

        # Read tensor count
        tensor_count, offset = self._read_uint64(offset)

        # Read metadata count
        metadata_count, offset = self._read_uint64(offset)

        # Read metadata key-value pairs
        for _ in range(metadata_count):
            key, offset = self._read_string(offset)
            value_type, offset = self._read_uint32(offset)
            value_type = GGUFValueType(value_type)
            value, offset = self._read_value(offset, value_type)
            self.metadata[key] = value

            # Check for alignment override
            if key == "general.alignment":
                self.alignment = value

        # Read tensor info
        for _ in range(tensor_count):
            name, offset = self._read_string(offset)
            n_dims, offset = self._read_uint32(offset)

            # Read dimensions (in reverse order)
            shape = []
            for _ in range(n_dims):
                dim, offset = self._read_uint64(offset)
                shape.append(dim)
            shape = list(reversed(shape))  # GGUF stores dims in reverse

            ggml_type, offset = self._read_uint32(offset)
            ggml_type = GGMLType(ggml_type)

            tensor_offset, offset = self._read_uint64(offset)

            tensor_info = GGUFTensorInfo(
                name=name,
                n_dims=n_dims,
                shape=shape,
                ggml_type=ggml_type,
                offset=tensor_offset
            )
            self.tensors[name] = tensor_info

        # Align to alignment boundary
        self.tensor_data_offset = (offset + self.alignment - 1) // self.alignment * self.alignment

    def get_tensor_data(self, tensor_name: str) -> memoryview:
        """
        Get memory-mapped view of tensor data (zero-copy).

        Args:
            tensor_name: Name of tensor

        Returns:
            memoryview of tensor data

        Raises:
            KeyError: If tensor not found
        """
        if tensor_name not in self.tensors:
            raise KeyError(f"Tensor '{tensor_name}' not found in GGUF file")

        tensor_info = self.tensors[tensor_name]
        abs_offset = self.tensor_data_offset + tensor_info.offset

        return memoryview(self.mmap[abs_offset:abs_offset + tensor_info.nbytes])

    def get_metadata_value(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)

    def list_tensors(self) -> List[str]:
        """Get list of all tensor names."""
        return list(self.tensors.keys())

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close file and memory map."""
        if self.mmap:
            self.mmap.close()
            self.mmap = None
        if self.file:
            self.file.close()
            self.file = None

    def __repr__(self) -> str:
        model_name = self.metadata.get('general.name', 'unknown')
        return f"GGUFReader(model='{model_name}', tensors={len(self.tensors)})"


def inspect_gguf(file_path: Union[str, Path]) -> None:
    """
    Print detailed information about a GGUF file.

    Args:
        file_path: Path to GGUF file
    """
    with GGUFReader(file_path) as reader:
        print("=" * 70)
        print(f"GGUF File: {file_path}")
        print("=" * 70)
        print()

        print("Metadata:")
        print("-" * 70)
        for key, value in sorted(reader.metadata.items()):
            if isinstance(value, list) and len(value) > 10:
                print(f"  {key}: [array with {len(value)} elements]")
            elif isinstance(value, str) and len(value) > 60:
                print(f"  {key}: {value[:57]}...")
            else:
                print(f"  {key}: {value}")
        print()

        print(f"Tensors: {len(reader.tensors)}")
        print("-" * 70)

        # Group tensors by type
        type_counts = {}
        for info in reader.tensors.values():
            type_name = info.ggml_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        print("Tensor types:")
        for type_name, count in sorted(type_counts.items()):
            print(f"  {type_name}: {count} tensors")
        print()

        # Calculate total size
        total_size = sum(info.nbytes for info in reader.tensors.values())
        print(f"Total tensor data: {total_size / (1024**3):.2f} GB")
        print()

        # Show first 10 tensors
        print("First 10 tensors:")
        for i, (name, info) in enumerate(list(reader.tensors.items())[:10]):
            shape_str = "x".join(map(str, info.shape))
            size_mb = info.nbytes / (1024**2)
            print(f"  {name}: [{shape_str}] {info.ggml_type.name} ({size_mb:.2f} MB)")

        if len(reader.tensors) > 10:
            print(f"  ... and {len(reader.tensors) - 10} more tensors")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gguf_parser.py <gguf_file>")
        sys.exit(1)

    inspect_gguf(sys.argv[1])
