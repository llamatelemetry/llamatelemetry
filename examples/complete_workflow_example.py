"""
Complete llamatelemetry Workflow Example: Unsloth to Deployment

This example demonstrates the complete workflow from Unsloth fine-tuning
to llamatelemetry deployment with quantization and optimization.

Requirements:
    - Tesla T4 GPU (16GB VRAM)
    - Python 3.11+
    - llamatelemetry installed: pip install git+https://github.com/llamatelemetry/llamatelemetry.git
    - Unsloth (optional, for fine-tuning): pip install unsloth

Workflow:
    1. Fine-tune model with Unsloth
    2. Export to GGUF with quantization
    3. Deploy with llamatelemetry for fast inference
    4. Optimize with CUDA features
"""

import llamatelemetry
from llamatelemetry.unsloth import export_to_llamatelemetry
from llamatelemetry.quantization import quantize_dynamic
from llamatelemetry.cuda import enable_tensor_cores, check_tensor_core_support
import torch


# ============================================================================
# PART 1: Fine-tune with Unsloth (Optional - skip if you have a model)
# ============================================================================

def finetune_with_unsloth():
    """Fine-tune a model with Unsloth."""
    print("=" * 70)
    print("PART 1: Fine-tuning with Unsloth")
    print("=" * 70)

    try:
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTTrainer, SFTConfig

        # Load base model
        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/Llama-3.2-1B-Instruct",
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
        )

        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
        )

        # Load dataset (example)
        dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")

        # Train (quick example - increase steps for real training)
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            processing_class=tokenizer,
            args=SFTConfig(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                max_steps=10,
                output_dir="./unsloth_output",
                logging_steps=1,
            ),
        )

        print("\n Training model...")
        trainer.train()

        print("✓ Fine-tuning complete!")
        return model, tokenizer

    except ImportError:
        print("⚠ Unsloth not installed, skipping fine-tuning")
        print("  Install with: pip install unsloth")
        return None, None


# ============================================================================
# PART 2: Export to GGUF with Quantization
# ============================================================================

def export_model_to_gguf(model, tokenizer):
    """Export fine-tuned model to GGUF format."""
    print("\n" + "=" * 70)
    print("PART 2: Export to GGUF with Quantization")
    print("=" * 70)

    if model is None:
        print("⚠ No model to export, loading pre-trained model")
        # Load a pre-trained GGUF model instead
        return "unsloth/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf"

    # Export using llamatelemetry's Unsloth integration
    output_path = "./model-q4_k_m.gguf"

    print(f"\n Exporting to {output_path}")
    print("  Quantization: Q4_K_M (recommended for Tesla T4)")

    export_to_llamatelemetry(
        model=model,
        tokenizer=tokenizer,
        output_path=output_path,
        quant_type="Q4_K_M",
        merge_lora=True,
        verbose=True,
    )

    print(f"✓ Model exported to {output_path}")
    return output_path


# ============================================================================
# PART 3: Deploy with llamatelemetry
# ============================================================================

def deploy_with_llamatelemetry(model_path):
    """Deploy model with llamatelemetry for fast inference."""
    print("\n" + "=" * 70)
    print("PART 3: Deploy with llamatelemetry")
    print("=" * 70)

    # Check Tensor Core support
    print("\n Checking GPU capabilities...")
    has_tensor_cores = check_tensor_core_support()

    if has_tensor_cores:
        print("  Enabling Tensor Core optimizations...")
        enable_tensor_cores(dtype=torch.float16, allow_tf32=True)

    # Initialize inference engine
    engine = llamatelemetry.InferenceEngine()

    print(f"\n Loading model: {model_path}")
    engine.load_model(
        model_path,
        silent=True,      # Suppress server output
        auto_start=True,  # Auto-start llama-server
    )

    print("✓ Model loaded and ready for inference")
    return engine


# ============================================================================
# PART 4: Run Inference with Optimizations
# ============================================================================

def run_optimized_inference(engine):
    """Run inference with various optimizations."""
    print("\n" + "=" * 70)
    print("PART 4: Optimized Inference")
    print("=" * 70)

    # Test prompts
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "What are the benefits of Tesla T4 GPUs?",
    ]

    print("\n Running single prompt inference...")
    result = engine.infer(
        prompts[0],
        max_tokens=150,
        temperature=0.7,
    )

    print(f"\n Prompt: {prompts[0]}")
    print(f"Response: {result.text[:200]}...")
    print(f"Performance: {result.tokens_per_sec:.1f} tokens/sec")

    # Batch inference
    print("\n Running batch inference...")
    results = engine.batch_infer(
        prompts,
        max_tokens=100,
        temperature=0.7,
    )

    for i, (prompt, result) in enumerate(zip(prompts, results), 1):
        print(f"\n Batch {i}: {result.tokens_per_sec:.1f} tok/s")

    # Performance metrics
    metrics = engine.get_metrics()
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"  Total requests: {metrics['throughput']['total_requests']}")
    print(f"  Total tokens: {metrics['throughput']['total_tokens']}")
    print(f"  Average speed: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")
    print(f"  Median latency: {metrics['latency']['p50_ms']:.1f} ms")


# ============================================================================
# PART 5: Advanced Features
# ============================================================================

def demonstrate_advanced_features():
    """Demonstrate advanced llamatelemetry features."""
    print("\n" + "=" * 70)
    print("PART 5: Advanced Features")
    print("=" * 70)

    # 1. Dynamic Quantization
    print("\n 1. Dynamic Quantization Recommendation")
    from llamatelemetry.quantization import DynamicQuantizer

    quantizer = DynamicQuantizer(target_vram_gb=12.0, strategy="balanced")
    config = quantizer.recommend_config(model_size_gb=1.0)

    print(f"   Recommended: {config['quant_type']}")
    print(f"   Expected VRAM: {config['expected_vram_gb']:.2f} GB")
    print(f"   Expected speed: {config['expected_speed_tps']:.1f} tok/s")

    # 2. Tensor Core Information
    print("\n 2. Tensor Core Capabilities")
    from llamatelemetry.cuda import get_tensor_core_info

    tc_info = get_tensor_core_info()
    print(f"   GPU: {tc_info.get('device', 'Unknown')}")
    print(f"   Tensor Cores: {tc_info.get('supported', False)}")
    print(f"   Architecture: {tc_info.get('architecture', 'N/A')}")
    print(f"   Speedup: {tc_info.get('estimated_speedup', 'N/A')}")

    # 3. FlashAttention for Long Context
    print("\n 3. FlashAttention for Long Context")
    from llamatelemetry.inference import get_optimal_context_length

    ctx_len = get_optimal_context_length(
        model_size_b=1.0,
        available_vram_gb=12.0,
        use_flash_attention=True,
    )
    print(f"   Optimal context length: {ctx_len} tokens")

    # 4. CUDA Graphs (conceptual)
    print("\n 4. CUDA Graphs Support")
    from llamatelemetry.cuda import CUDAGraph

    print("   CUDA Graphs available for reduced latency")
    print("   Benefit: 20-40% latency reduction for repeated operations")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run the complete workflow."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║       Complete llamatelemetry Workflow: Unsloth to Deployment            ║
║                   Tesla T4 Optimized Pipeline                     ║
╚═══════════════════════════════════════════════════════════════════╝
""")

    # Part 1: Fine-tune (optional)
    model, tokenizer = finetune_with_unsloth()

    # Part 2: Export to GGUF
    model_path = export_model_to_gguf(model, tokenizer)

    # Part 3: Deploy with llamatelemetry
    engine = deploy_with_llamatelemetry(model_path)

    # Part 4: Run inference
    run_optimized_inference(engine)

    # Part 5: Advanced features
    demonstrate_advanced_features()

    print("\n" + "=" * 70)
    print("✓ Workflow Complete!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Integrate into your application")
    print("  2. Optimize batch sizes for your use case")
    print("  3. Explore FlashAttention for longer contexts")
    print("  4. Try different quantization strategies")

    # Cleanup
    engine.unload_model()


if __name__ == "__main__":
    main()
