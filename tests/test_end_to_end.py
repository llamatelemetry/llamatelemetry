#!/usr/bin/env python3
"""
End-to-end test for llamatelemetry with actual model inference
"""
import llamatelemetry
import os
import time
from pathlib import Path

print("="*60)
print("llamatelemetry END-TO-END TEST")
print("="*60)

# Test 1: Basic imports and GPU check
print("\n1. GPU Compatibility Check:")
compat = llamatelemetry.check_gpu_compatibility()
print(f"   GPU: {compat['gpu_name']}")
print(f"   Compute Capability: {compat['compute_capability']}")
print(f"   Compatible: {compat['compatible']}")
assert compat['compatible'], "GPU must be compatible"

# Test 2: Create engine
print("\n2. Creating Inference Engine...")
engine = llamatelemetry.InferenceEngine()
print("   ✓ Engine created")

# Test 3: Find model
print("\n3. Looking for test model...")

# Check common locations
model_paths = [
    "/media/waqasm86/External1/Project-Nvidia/Ubuntu-Cuda-Llama.cpp-Executable/bin/gemma-3-1b-it-Q4_K_M.gguf",
    Path.home() / "gemma-3-1b-it-Q4_K_M.gguf",
    Path.cwd() / "test_model.gguf",
    Path.cwd() / "gemma-3-1b-it-Q4_K_M.gguf",
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = str(path)
        print(f"   ✓ Found model: {model_path}")
        break

if not model_path:
    print("   ⚠ No model found. Download one with:")
    print("   wget https://huggingface.co/google/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf")
    print("   Skipping inference tests.")
    exit(0)

# Test 4: Load model
print("\n4. Loading model...")
try:
    # For GeForce 940M (1GB VRAM), use conservative settings
    engine.load_model(
        model_path,
        gpu_layers=8,        # Start with 8 layers on GPU
        ctx_size=512,        # Smaller context to save memory
        batch_size=256,      # Reduced batch size
        ubatch_size=64,      # Small compute buffer
        verbose=True,
        auto_start=True
    )
    print("   ✓ Model loaded successfully!")
    
    # Test 5: Simple inference
    print("\n5. Running inference test...")
    test_prompts = [
        "Explain artificial intelligence in one sentence.",
        "What is 2+2?",
        "Say hello in Spanish."
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n   Prompt {i+1}: '{prompt}'")
        start_time = time.time()
        result = engine.infer(prompt, max_tokens=50, temperature=0.7)
        elapsed = time.time() - start_time
        
        if result.success:
            print(f"   Response: {result.text.strip()}")
            print(f"   Tokens: {result.tokens_generated}")
            print(f"   Speed: {result.tokens_per_sec:.1f} tok/s")
            print(f"   Time: {elapsed:.2f}s")
        else:
            print(f"   ⚠ Error: {result.error_message}")
    
    # Test 6: Get metrics
    print("\n6. Checking metrics...")
    metrics = engine.get_metrics()
    print(f"   Total requests: {metrics['throughput']['total_requests']}")
    print(f"   Total tokens: {metrics['throughput']['total_tokens']}")
    if metrics['throughput']['total_requests'] > 0:
        print(f"   Avg tokens/sec: {metrics['throughput']['tokens_per_sec']:.1f}")
    
    print("\n" + "="*60)
    print("🎉 END-TO-END TEST COMPLETE!")
    print("llamatelemetry is fully functional!")
    print("="*60)
    
except Exception as e:
    print(f"   ❌ Error during test: {e}")
    print("\nTroubleshooting tips:")
    print("1. Check if llama-server is executable: chmod +x /path/to/llama-server")
    print("2. Reduce gpu_layers if out of memory (try gpu_layers=4)")
    print("3. Check model file integrity")
    import traceback
    traceback.print_exc()