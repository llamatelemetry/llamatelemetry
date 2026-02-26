"""
Unit tests for llamatelemetry v0.1.0 APIs

Tests quantization, unsloth, cuda, and inference APIs.
"""

import unittest
import sys
from pathlib import Path

# Add llamatelemetry to path
sys.path.insert(0, str(Path(__file__).parent.parent / "llamatelemetry"))


class TestQuantizationAPI(unittest.TestCase):
    """Test Quantization API"""

    def test_nf4_config(self):
        """Test NF4Config creation"""
        from quantization.nf4 import NF4Config

        config = NF4Config(blocksize=64, double_quant=True)
        self.assertEqual(config.blocksize, 64)
        self.assertTrue(config.double_quant)

    def test_gguf_quant_types(self):
        """Test GGUF quantization types"""
        from quantization.gguf import GGUFQuantType

        self.assertIsNotNone(GGUFQuantType.Q4_K_M)
        self.assertIsNotNone(GGUFQuantType.F16)
        self.assertEqual(GGUFQuantType.Q4_K_M.value, 25)

    def test_dynamic_quantizer(self):
        """Test DynamicQuantizer recommendation"""
        from quantization.dynamic import DynamicQuantizer, QuantStrategy

        quantizer = DynamicQuantizer(target_vram_gb=12.0, strategy=QuantStrategy.BALANCED)
        config = quantizer.recommend_config(model_size_gb=3.0, verbose=False)

        self.assertIn('quant_type', config)
        self.assertIn('expected_vram_gb', config)
        self.assertEqual(config['quant_type'], 'Q4_K_M')


class TestUnslothIntegrationAPI(unittest.TestCase):
    """Test Unsloth Integration API"""

    def test_unsloth_loader(self):
        """Test UnslothModelLoader creation"""
        try:
            import torch
        except Exception:
            raise unittest.SkipTest("torch not available")

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available for UnslothModelLoader")

        from unsloth.loader import UnslothModelLoader

        loader = UnslothModelLoader(max_seq_length=2048)
        self.assertEqual(loader.max_seq_length, 2048)

    def test_export_config(self):
        """Test ExportConfig"""
        from unsloth.exporter import ExportConfig

        config = ExportConfig(quant_type="Q4_K_M", merge_lora=True)
        self.assertEqual(config.quant_type, "Q4_K_M")
        self.assertTrue(config.merge_lora)

    def test_adapter_config(self):
        """Test AdapterConfig"""
        from unsloth.adapter import AdapterConfig

        config = AdapterConfig(r=16, lora_alpha=32)
        self.assertEqual(config.r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertEqual(len(config.target_modules), 7)


class TestCUDAOptimizationAPI(unittest.TestCase):
    """Test CUDA Optimization API"""

    def test_tensor_core_config(self):
        """Test TensorCoreConfig"""
        from cuda.tensor_core import TensorCoreConfig

        config = TensorCoreConfig(enabled=True, allow_tf32=True)
        self.assertTrue(config.enabled)
        self.assertTrue(config.allow_tf32)

    def test_cuda_graph(self):
        """Test CUDAGraph creation"""
        from cuda.graphs import CUDAGraph

        graph = CUDAGraph()
        self.assertFalse(graph.is_captured())

    def test_graph_pool(self):
        """Test GraphPool"""
        from cuda.graphs import GraphPool

        pool = GraphPool()
        self.assertEqual(len(pool.list_graphs()), 0)

    def test_kernel_config(self):
        """Test KernelConfig"""
        from cuda.triton_kernels import KernelConfig

        config = KernelConfig(name="test", block_size=128, num_warps=4)
        self.assertEqual(config.block_size, 128)
        self.assertEqual(config.num_warps, 4)

    def test_registered_kernels(self):
        """Test kernel registration"""
        from cuda.triton_kernels import list_kernels

        kernels = list_kernels()
        self.assertGreater(len(kernels), 0)


class TestAdvancedInferenceAPI(unittest.TestCase):
    """Test Advanced Inference API"""

    def test_flash_attention_config(self):
        """Test FlashAttentionConfig"""
        from inference.flash_attn import FlashAttentionConfig

        config = FlashAttentionConfig(enabled=True, causal=True)
        self.assertTrue(config.enabled)
        self.assertTrue(config.causal)

    def test_optimal_context_length(self):
        """Test context length estimation"""
        from inference.flash_attn import get_optimal_context_length

        ctx_len = get_optimal_context_length(
            model_size_b=3.0,
            available_vram_gb=12.0,
            use_flash_attention=True,
        )
        self.assertGreater(ctx_len, 0)
        self.assertIn(ctx_len, [512, 1024, 2048, 4096, 8192, 16384, 32768])

    def test_kv_cache_config(self):
        """Test KVCacheConfig"""
        from inference.kv_cache import KVCacheConfig

        config = KVCacheConfig(max_batch_size=8, max_seq_length=4096)
        self.assertEqual(config.max_batch_size, 8)
        self.assertEqual(config.max_seq_length, 4096)

    def test_kv_cache(self):
        """Test KVCache creation"""
        from inference.kv_cache import KVCache, KVCacheConfig

        config = KVCacheConfig()
        cache = KVCache(config)
        self.assertEqual(len(cache.cache), 0)

    def test_batch_config(self):
        """Test BatchConfig"""
        from inference.batch import BatchConfig

        config = BatchConfig(max_batch_size=8, dynamic_batching=True)
        self.assertEqual(config.max_batch_size, 8)
        self.assertTrue(config.dynamic_batching)


class TestAPIIntegration(unittest.TestCase):
    """Test API integration and workflow"""

    def test_all_imports(self):
        """Test that all APIs can be imported"""
        # Quantization
        from quantization import (
            quantize_nf4,
            NF4Quantizer,
            convert_to_gguf,
            DynamicQuantizer,
        )

        # Unsloth
        from unsloth import (
            load_unsloth_model,
            export_to_llamatelemetry,
            merge_lora_adapters,
        )

        # CUDA
        from cuda import (
            enable_tensor_cores,
            CUDAGraph,
            list_kernels,
        )

        # Inference
        from inference import (
            enable_flash_attention,
            KVCache,
            batch_inference_optimized,
        )

        # If we got here, all imports succeeded
        self.assertTrue(True)

    def test_quantization_strategies(self):
        """Test all quantization strategies"""
        from quantization.dynamic import QuantStrategy

        strategies = [
            QuantStrategy.AGGRESSIVE,
            QuantStrategy.BALANCED,
            QuantStrategy.QUALITY,
            QuantStrategy.MINIMAL,
        ]

        for strategy in strategies:
            self.assertIsNotNone(strategy)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestQuantizationAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestUnslothIntegrationAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestCUDAOptimizationAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedInferenceAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success/failure
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
