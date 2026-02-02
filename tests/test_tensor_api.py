"""
Test suite for llamatelemetry v2.0 Tensor API
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from llamatelemetry.core import Tensor, DType, get_device_count, get_device_properties, matmul
except ImportError as e:
    pytest.skip(f"llamatelemetry native extension not built: {e}", allow_module_level=True)


class TestDevice:
    def test_get_device_count(self):
        """Test device count detection"""
        count = get_device_count()
        assert count > 0, "No CUDA devices found"
        print(f"✓ Found {count} CUDA device(s)")

    def test_get_device_properties(self):
        """Test device properties"""
        count = get_device_count()
        for device_id in range(count):
            props = get_device_properties(device_id)
            print(f"\nDevice {device_id}: {props.name}")
            print(f"  Compute Capability: {props.compute_capability_major}.{props.compute_capability_minor}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multiprocessors: {props.multiprocessor_count}")

            assert props.compute_capability_major >= 5, "GPU compute capability must be >= 5.0"


class TestTensor:
    def test_tensor_creation(self):
        """Test basic tensor creation"""
        t = Tensor([2, 3], dtype=DType.Float32, device=0)

        assert t.shape == [2, 3]
        assert t.dtype == DType.Float32
        assert t.device == 0
        assert t.ndim == 2
        assert t.numel() == 6

        print(f"✓ Created tensor: {t}")

    def test_zeros(self):
        """Test zeros tensor"""
        t = Tensor.zeros([4, 5], dtype=DType.Float32, device=0)

        assert t.shape == [4, 5]
        assert t.numel() == 20
        assert t.is_contiguous()

        print(f"✓ Created zeros tensor: {t}")

    def test_dtype_support(self):
        """Test different data types"""
        dtypes = [DType.Float32, DType.Float16, DType.Int32, DType.Int64, DType.UInt8]

        for dtype in dtypes:
            t = Tensor([2, 2], dtype=dtype, device=0)
            assert t.dtype == dtype
            print(f"✓ {dtype} supported")

    def test_multi_gpu(self):
        """Test multi-GPU support (if available)"""
        count = get_device_count()

        if count < 2:
            pytest.skip("Multi-GPU test requires at least 2 GPUs")

        t0 = Tensor.zeros([3, 3], device=0)
        t1 = t0.to(1)

        assert t0.device == 0
        assert t1.device == 1
        assert t0.shape == t1.shape

        print("✓ Multi-GPU transfer works")


class TestMatmul:
    def test_matmul_fp32(self):
        """Test float32 matrix multiplication"""
        A = Tensor.zeros([2, 3], dtype=DType.Float32, device=0)
        B = Tensor.zeros([3, 4], dtype=DType.Float32, device=0)

        C = matmul(A, B)

        assert C.shape == [2, 4]
        assert C.dtype == DType.Float32
        assert C.device == 0

        print(f"✓ Matmul FP32: {A.shape} @ {B.shape} = {C.shape}")

    def test_matmul_fp16(self):
        """Test float16 matrix multiplication"""
        A = Tensor.zeros([4, 8], dtype=DType.Float16, device=0)
        B = Tensor.zeros([8, 16], dtype=DType.Float16, device=0)

        C = A @ B  # Use @ operator

        assert C.shape == [4, 16]
        assert C.dtype == DType.Float16

        print(f"✓ Matmul FP16: {A.shape} @ {B.shape} = {C.shape}")

    def test_matmul_dimension_error(self):
        """Test dimension mismatch error"""
        A = Tensor.zeros([2, 3], dtype=DType.Float32, device=0)
        B = Tensor.zeros([5, 4], dtype=DType.Float32, device=0)

        with pytest.raises(RuntimeError, match="Incompatible dimensions"):
            C = matmul(A, B)

        print("✓ Dimension mismatch properly detected")

    def test_matmul_device_error(self):
        """Test device mismatch error"""
        count = get_device_count()
        if count < 2:
            pytest.skip("Multi-GPU test requires at least 2 GPUs")

        A = Tensor.zeros([2, 3], device=0)
        B = Tensor.zeros([3, 4], device=1)

        with pytest.raises(RuntimeError, match="same device"):
            C = matmul(A, B)

        print("✓ Device mismatch properly detected")


class TestMemory:
    def test_memory_allocation(self):
        """Test memory allocation and deallocation"""
        from llamatelemetry.core import Device

        initial_free = Device.get_free_memory(0)

        # Allocate large tensor
        t = Tensor.zeros([1000, 1000], dtype=DType.Float32, device=0)

        after_alloc_free = Device.get_free_memory(0)
        assert after_alloc_free < initial_free

        # Delete tensor
        del t

        # Memory should be freed (approximately)
        # Note: CUDA may cache freed memory
        print(f"✓ Memory allocation working (freed: {(initial_free - after_alloc_free) / 1024**2:.2f} MB)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
