"""
llamatelemetry API Usage Examples

Quick examples demonstrating each new API module.
"""

import llamatelemetry
import torch


# ============================================================================
# 1. Quantization API Examples
# ============================================================================

def example_quantization():
    """Demonstrate quantization APIs."""
    print("=" * 70)
    print("1. QUANTIZATION API")
    print("=" * 70)

    # NF4 Quantization
    print("\n A) NF4 Quantization")
    from llamatelemetry.quantization import quantize_nf4, dequantize_nf4

    # Create a weight tensor
    weight = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    print(f"   Original size: {weight.nbytes / 1024**2:.2f} MB")

    # Quantize to NF4
    qweight, state = quantize_nf4(weight, blocksize=64, double_quant=True)
    print(f"   Quantized size: {qweight.nbytes / 1024**2:.2f} MB")
    print(f"   Compression: {weight.nbytes / qweight.nbytes:.2f}x")

    # Dequantize
    weight_restored = dequantize_nf4(qweight, state)
    print(f"   ✓ Quantization/dequantization complete")

    # GGUF Conversion
    print("\n B) GGUF Conversion")
    from llamatelemetry.quantization import GGUFQuantType

    print("   Supported quantization types:")
    for quant_type in ["Q4_K_M", "Q5_K_M", "Q8_0", "F16"]:
        print(f"     - {quant_type}")

    # Dynamic Quantization
    print("\n C) Dynamic Quantization")
    from llamatelemetry.quantization import DynamicQuantizer

    quantizer = DynamicQuantizer(target_vram_gb=12.0)
    config = quantizer.recommend_config(model_size_gb=3.0)

    print(f"   Recommended: {config['quant_type']}")
    print(f"   Expected VRAM: {config['expected_vram_gb']:.2f} GB")
    print(f"   Compression: {config['compression_ratio']:.1f}x")


# ============================================================================
# 2. Unsloth Integration Examples
# ============================================================================

def example_unsloth_integration():
    """Demonstrate Unsloth integration APIs."""
    print("\n" + "=" * 70)
    print("2. UNSLOTH INTEGRATION API")
    print("=" * 70)

    # Check if Unsloth is available
    from llamatelemetry.unsloth import check_unsloth_available

    if not check_unsloth_available():
        print("   ⚠ Unsloth not installed")
        print("   Install with: pip install unsloth")
        return

    # Load Unsloth model
    print("\n A) Load Unsloth Model")
    from llamatelemetry.unsloth import load_unsloth_model

    print("   Loading model...")
    # model, tokenizer = load_unsloth_model(
    #     "unsloth/Llama-3.2-1B-Instruct",
    #     max_seq_length=2048,
    # )
    print("   ✓ Model loading syntax demonstrated")

    # Export to GGUF
    print("\n B) Export to GGUF")
    from llamatelemetry.unsloth import export_to_llamatelemetry

    print("   Export syntax:")
    print("   export_to_llamatelemetry(model, tokenizer, 'model.gguf', quant_type='Q4_K_M')")

    # LoRA Adapter Management
    print("\n C) LoRA Adapter Management")
    from llamatelemetry.unsloth import merge_lora_adapters

    print("   Merge adapters: merged = merge_lora_adapters(model)")
    print("   ✓ Unsloth integration APIs ready")


# ============================================================================
# 3. CUDA Optimization Examples
# ============================================================================

def example_cuda_optimizations():
    """Demonstrate CUDA optimization APIs."""
    print("\n" + "=" * 70)
    print("3. CUDA OPTIMIZATION APIs")
    print("=" * 70)

    # Tensor Core Support
    print("\n A) Tensor Core Support")
    from llamatelemetry.cuda import check_tensor_core_support, enable_tensor_cores

    if check_tensor_core_support():
        config = enable_tensor_cores(dtype=torch.float16)
        print(f"   ✓ Tensor Cores enabled")
    else:
        print("   ⚠ Tensor Cores not supported")

    # CUDA Graphs
    print("\n B) CUDA Graphs")
    from llamatelemetry.cuda import CUDAGraph

    print("   Example: Capture and replay operations")
    graph = CUDAGraph()
    print("   graph = CUDAGraph()")
    print("   with graph.capture():")
    print("       output = model(input)")
    print("   graph.replay()  # Fast replay")

    # Triton Kernels
    print("\n C) Triton Kernel Integration")
    from llamatelemetry.cuda import list_kernels

    kernels = list_kernels()
    print(f"   Available kernels: {len(kernels)}")
    if kernels:
        print(f"     {', '.join(kernels[:3])}")


# ============================================================================
# 4. Advanced Inference Examples
# ============================================================================

def example_advanced_inference():
    """Demonstrate advanced inference APIs."""
    print("\n" + "=" * 70)
    print("4. ADVANCED INFERENCE APIs")
    print("=" * 70)

    # FlashAttention
    print("\n A) FlashAttention")
    from llamatelemetry.inference import check_flash_attention_available

    if check_flash_attention_available():
        print("   ✓ FlashAttention available")
        print("   Benefit: 2-3x faster for long contexts (>1024 tokens)")
    else:
        print("   ⚠ FlashAttention not installed")
        print("   Install: pip install flash-attn --no-build-isolation")

    # Optimal Context Length
    from llamatelemetry.inference import get_optimal_context_length

    ctx_len = get_optimal_context_length(
        model_size_b=3.0,
        available_vram_gb=12.0,
        use_flash_attention=True,
    )
    print(f"\n B) Optimal Context Length")
    print(f"   Recommended: {ctx_len} tokens")
    print(f"   (for 3B model with 12GB VRAM)")

    # KV-Cache Optimization
    print("\n C) KV-Cache Optimization")
    from llamatelemetry.inference import KVCache, KVCacheConfig

    config = KVCacheConfig(
        max_batch_size=8,
        max_seq_length=4096,
    )
    print(f"   KV-Cache config: {config.max_seq_length} tokens")

    # Batch Optimization
    print("\n D) Batch Inference Optimization")
    from llamatelemetry.inference import batch_inference_optimized

    print("   Optimized batching for maximum throughput")
    print("   results = batch_inference_optimized(prompts, model)")


# ============================================================================
# 5. Complete Workflow Example
# ============================================================================

def example_complete_workflow():
    """Demonstrate complete workflow."""
    print("\n" + "=" * 70)
    print("5. COMPLETE WORKFLOW")
    print("=" * 70)

    print("\n Typical llamatelemetry workflow:")
    print("""
    # 1. Fine-tune with Unsloth
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained("base_model")
    # ... training ...

    # 2. Export to GGUF
    from llamatelemetry.unsloth import export_to_llamatelemetry
    export_to_llamatelemetry(model, tokenizer, "model.gguf", quant_type="Q4_K_M")

    # 3. Deploy with llamatelemetry
    import llamatelemetry
    from llamatelemetry.cuda import enable_tensor_cores

    enable_tensor_cores()  # Optimize for T4
    engine = llamatelemetry.InferenceEngine()
    engine.load_model("model.gguf")

    # 4. Run inference
    result = engine.infer("What is AI?", max_tokens=200)
    print(f"{result.text} ({result.tokens_per_sec:.1f} tok/s)")
    """)


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    llamatelemetry API Usage Examples                      ║
║                   v2.1+ New API Demonstrations                    ║
╚═══════════════════════════════════════════════════════════════════╝
""")

    example_quantization()
    example_unsloth_integration()
    example_cuda_optimizations()
    example_advanced_inference()
    example_complete_workflow()

    print("\n" + "=" * 70)
    print("✓ API Examples Complete")
    print("=" * 70)
    print("\nFor more information:")
    print("  Documentation: https://llamatelemetry.github.io/")
    print("  GitHub: https://github.com/llamatelemetry/llamatelemetry")
    print("  Examples: https://github.com/llamatelemetry/llamatelemetry/tree/main/examples")


if __name__ == "__main__":
    main()
