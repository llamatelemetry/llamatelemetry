"""
Tests for GGUF parser.

Note: These tests require actual GGUF model files.
Download a small test model with:
    wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
"""

import pytest
from pathlib import Path
import struct
import tempfile

from llamatelemetry.gguf_parser import (
    GGUFReader,
    GGUFValueType,
    GGMLType,
    GGUFTensorInfo,
    inspect_gguf,
    GGML_TYPE_SIZE,
    GGML_BLOCK_SIZE,
)


class TestGGUFTensorInfo:
    """Test GGUFTensorInfo dataclass."""

    def test_numel_calculation(self):
        """Test element count calculation."""
        info = GGUFTensorInfo(
            name="test",
            n_dims=2,
            shape=[4096, 4096],
            ggml_type=GGMLType.F32,
            offset=0
        )
        assert info.numel == 4096 * 4096

    def test_nbytes_f32(self):
        """Test byte size for F32 tensor."""
        info = GGUFTensorInfo(
            name="test",
            n_dims=2,
            shape=[1024, 1024],
            ggml_type=GGMLType.F32,
            offset=0
        )
        expected = 1024 * 1024 * 4  # 4 bytes per element
        assert info.nbytes == expected

    def test_nbytes_q4_0(self):
        """Test byte size for Q4_0 quantized tensor."""
        info = GGUFTensorInfo(
            name="test",
            n_dims=2,
            shape=[1024, 1024],
            ggml_type=GGMLType.Q4_0,
            offset=0
        )
        # Q4_0: 18 bytes per 32 elements
        n_elements = 1024 * 1024
        n_blocks = (n_elements + 31) // 32
        expected = n_blocks * 18
        assert info.nbytes == expected

    def test_repr(self):
        """Test string representation."""
        info = GGUFTensorInfo(
            name="model.embed",
            n_dims=2,
            shape=[4096, 512],
            ggml_type=GGMLType.F16,
            offset=1024
        )
        repr_str = repr(info)
        assert "model.embed" in repr_str
        assert "4096x512" in repr_str
        assert "F16" in repr_str


class TestGGUFReader:
    """Test GGUF reader with minimal mock file."""

    @pytest.fixture
    def minimal_gguf_file(self):
        """Create a minimal valid GGUF file for testing."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.gguf') as f:
            # Magic: "GGUF" (0x46554747)
            f.write(struct.pack('<I', 0x46554747))

            # Version: 3
            f.write(struct.pack('<I', 3))

            # Tensor count: 1
            f.write(struct.pack('<Q', 1))

            # Metadata count: 2
            f.write(struct.pack('<Q', 2))

            # Metadata 1: general.name = "test_model"
            key = "general.name"
            f.write(struct.pack('<Q', len(key)))
            f.write(key.encode('utf-8'))
            f.write(struct.pack('<I', GGUFValueType.STRING))  # Type: STRING
            value = "test_model"
            f.write(struct.pack('<Q', len(value)))
            f.write(value.encode('utf-8'))

            # Metadata 2: general.alignment = 32
            key = "general.alignment"
            f.write(struct.pack('<Q', len(key)))
            f.write(key.encode('utf-8'))
            f.write(struct.pack('<I', GGUFValueType.UINT32))  # Type: UINT32
            f.write(struct.pack('<I', 32))

            # Tensor 1: "test_tensor", shape [2, 3], type F32
            name = "test_tensor"
            f.write(struct.pack('<Q', len(name)))
            f.write(name.encode('utf-8'))
            f.write(struct.pack('<I', 2))  # n_dims
            f.write(struct.pack('<Q', 3))  # dim 0 (reversed)
            f.write(struct.pack('<Q', 2))  # dim 1 (reversed)
            f.write(struct.pack('<I', GGMLType.F32))  # type
            f.write(struct.pack('<Q', 0))  # offset

            # Get current position and align to 32 bytes
            current_pos = f.tell()
            aligned_pos = ((current_pos + 31) // 32) * 32
            padding = aligned_pos - current_pos
            f.write(b'\x00' * padding)

            # Tensor data: 2x3 F32 = 24 bytes
            tensor_data = struct.pack('<6f', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
            f.write(tensor_data)

            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        temp_path.unlink()

    def test_reader_open_close(self, minimal_gguf_file):
        """Test opening and closing GGUF file."""
        reader = GGUFReader(minimal_gguf_file)
        assert reader.file is not None
        assert reader.mmap is not None
        reader.close()
        assert reader.file is None
        assert reader.mmap is None

    def test_context_manager(self, minimal_gguf_file):
        """Test context manager usage."""
        with GGUFReader(minimal_gguf_file) as reader:
            assert reader.file is not None
            assert reader.mmap is not None
        # File should be closed after context exit
        assert reader.file is None

    def test_metadata_parsing(self, minimal_gguf_file):
        """Test metadata parsing."""
        with GGUFReader(minimal_gguf_file) as reader:
            assert "general.name" in reader.metadata
            assert reader.metadata["general.name"] == "test_model"
            assert "general.alignment" in reader.metadata
            assert reader.metadata["general.alignment"] == 32

    def test_tensor_info_parsing(self, minimal_gguf_file):
        """Test tensor info parsing."""
        with GGUFReader(minimal_gguf_file) as reader:
            assert "test_tensor" in reader.tensors
            info = reader.tensors["test_tensor"]
            assert info.name == "test_tensor"
            assert info.shape == [2, 3]
            assert info.n_dims == 2
            assert info.ggml_type == GGMLType.F32

    def test_get_tensor_data(self, minimal_gguf_file):
        """Test reading tensor data."""
        with GGUFReader(minimal_gguf_file) as reader:
            data = reader.get_tensor_data("test_tensor")
            # Should be 2x3 = 6 floats = 24 bytes
            assert len(data) == 24

            # Convert to floats and verify
            floats = struct.unpack('<6f', data)
            assert floats == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    def test_get_nonexistent_tensor(self, minimal_gguf_file):
        """Test reading non-existent tensor."""
        with GGUFReader(minimal_gguf_file) as reader:
            with pytest.raises(KeyError):
                reader.get_tensor_data("nonexistent")

    def test_get_metadata_value(self, minimal_gguf_file):
        """Test getting metadata values."""
        with GGUFReader(minimal_gguf_file) as reader:
            assert reader.get_metadata_value("general.name") == "test_model"
            assert reader.get_metadata_value("nonexistent", "default") == "default"
            assert reader.get_metadata_value("nonexistent") is None

    def test_list_tensors(self, minimal_gguf_file):
        """Test listing tensor names."""
        with GGUFReader(minimal_gguf_file) as reader:
            tensors = reader.list_tensors()
            assert "test_tensor" in tensors
            assert len(tensors) == 1

    def test_invalid_file(self):
        """Test opening invalid file."""
        with pytest.raises(FileNotFoundError):
            GGUFReader("nonexistent.gguf")

    def test_repr(self, minimal_gguf_file):
        """Test string representation."""
        with GGUFReader(minimal_gguf_file) as reader:
            repr_str = repr(reader)
            assert "test_model" in repr_str
            assert "tensors=1" in repr_str


class TestGGMLTypes:
    """Test GGML type constants."""

    def test_block_sizes(self):
        """Test that all types have block sizes defined."""
        for ggml_type in [GGMLType.F32, GGMLType.F16, GGMLType.Q4_0, GGMLType.Q8_0]:
            assert ggml_type in GGML_BLOCK_SIZE
            assert GGML_BLOCK_SIZE[ggml_type] > 0

    def test_type_sizes(self):
        """Test that all types have sizes defined."""
        for ggml_type in [GGMLType.F32, GGMLType.F16, GGMLType.Q4_0, GGMLType.Q8_0]:
            assert ggml_type in GGML_TYPE_SIZE
            assert GGML_TYPE_SIZE[ggml_type] > 0

    def test_f32_properties(self):
        """Test F32 type properties."""
        assert GGML_TYPE_SIZE[GGMLType.F32] == 4
        assert GGML_BLOCK_SIZE[GGMLType.F32] == 1

    def test_q4_0_properties(self):
        """Test Q4_0 type properties."""
        assert GGML_TYPE_SIZE[GGMLType.Q4_0] == 18  # 18 bytes per block
        assert GGML_BLOCK_SIZE[GGMLType.Q4_0] == 32  # 32 elements per block


@pytest.mark.skipif(
    not Path("models/test.gguf").exists(),
    reason="No real GGUF model file available for integration testing"
)
class TestGGUFIntegration:
    """Integration tests with real GGUF files (optional)."""

    def test_real_model(self):
        """Test with real GGUF model file."""
        with GGUFReader("models/test.gguf") as reader:
            # Basic checks
            assert len(reader.metadata) > 0
            assert len(reader.tensors) > 0

            # Check common metadata
            assert "general.name" in reader.metadata or "general.architecture" in reader.metadata

            # Verify tensors are accessible
            for tensor_name in list(reader.tensors.keys())[:5]:
                data = reader.get_tensor_data(tensor_name)
                assert len(data) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
