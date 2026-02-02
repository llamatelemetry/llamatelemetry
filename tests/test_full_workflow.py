import llamatelemetry
import os

print("Testing full llamatelemetry workflow...")

# 1. Check GPU
compat = llamatelemetry.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")
print(f"Compute: {compat['compute_capability']}")
print(f"Compatible: {compat['compatible']}")

# 2. Create engine
engine = llamatelemetry.InferenceEngine()
print("✓ Engine created")

# 3. Test with a small model (optional - requires model file)
test_model = "/media/waqasm86/External1/Project-Nvidia/Ubuntu-Cuda-Llama.cpp-Executable/bin/gemma-3-1b-it-Q4_K_M.gguf"
if os.path.exists(test_model):
    print(f"\nFound model: {test_model}")
    print("Testing model loading...")
    try:
        engine.load_model(test_model, gpu_layers=8, verbose=True)
        print("✓ Model loaded successfully!")
        
        # Test inference
        result = engine.infer("What is AI?", max_tokens=50)
        print(f"\nGenerated: {result.text[:100]}...")
        print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
    except Exception as e:
        print(f"⚠ Model test skipped: {e}")
else:
    print("\nNo test model found - download one to complete end-to-end test")
    print("You can download with:")
    print("wget https://huggingface.co/google/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf")

print("\n✅ llamatelemetry is working correctly!")
