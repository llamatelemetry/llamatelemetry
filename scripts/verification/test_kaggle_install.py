#!/usr/bin/env python3
"""
Test llamatelemetry v0.1.0 installation workflow (simulates Kaggle environment)
"""

print("=" * 70)
print("llamatelemetry v0.1.0 - Kaggle Installation Test")
print("=" * 70)
print()

# Step 1: Simulate pip install
print("📦 Step 1: Installing llamatelemetry v0.1.0 from GitHub...")
print("    Command: pip install --no-cache-dir --force-reinstall \\")
print("             git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0")
print()

# Step 2: Import and verify
print("📦 Step 2: Importing llamatelemetry...")
try:
    import llamatelemetry
    print(f"✅ llamatelemetry {llamatelemetry.__version__} imported successfully!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

print()

# Step 3: Check llamatelemetry status
print("📊 Step 3: Checking llamatelemetry status...")
try:
    from llamatelemetry import check_cuda_available, get_cuda_device_info
    from llamatelemetry.api.multigpu import gpu_count

    cuda_info = get_cuda_device_info()
    print(f"   CUDA Available: {check_cuda_available()}")
    print(f"   GPUs: {gpu_count()}")
    if cuda_info:
        print(f"   CUDA Version: {cuda_info.get('cuda_version', 'N/A')}")
        print(f"   GPU 0: {cuda_info.get('gpu_name', 'N/A')}")
except Exception as e:
    print(f"⚠️  Status check: {e}")

print()

# Step 4: Check binary paths
print("🔍 Step 4: Verifying binary installation...")
import os
from pathlib import Path

llamatelemetry_dir = Path(llamatelemetry.__file__).parent
binaries_dir = llamatelemetry_dir / "binaries" / "cuda12"
lib_dir = llamatelemetry_dir / "lib"

print(f"   Package dir: {llamatelemetry_dir}")
print(f"   Binaries dir: {binaries_dir}")
print(f"   Libraries dir: {lib_dir}")

if binaries_dir.exists():
    binaries = list(binaries_dir.glob("llama-*"))
    print(f"   ✅ Found {len(binaries)} binaries:")
    for binary in sorted(binaries)[:5]:  # Show first 5
        size_mb = binary.stat().st_size / (1024 * 1024)
        print(f"      - {binary.name} ({size_mb:.1f} MB)")
    if len(binaries) > 5:
        print(f"      ... and {len(binaries) - 5} more")
else:
    print("   ⚠️  Binaries directory not found (expected on first import)")

if lib_dir.exists():
    libs = list(lib_dir.glob("*.so*"))
    print(f"   ✅ Found {len(libs)} shared libraries")
else:
    print("   ⚠️  Libraries directory not found (expected on first import)")

print()

# Step 5: Test llama-server availability
print("🔍 Step 5: Testing llama-server...")
llama_server_path = os.environ.get("LLAMA_SERVER_PATH")
if llama_server_path:
    llama_server = Path(llama_server_path)
    if llama_server.exists():
        size_mb = llama_server.stat().st_size / (1024 * 1024)
        print(f"   ✅ llama-server found: {llama_server}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Executable: {os.access(llama_server, os.X_OK)}")
    else:
        print(f"   ❌ llama-server not found at: {llama_server_path}")
else:
    print("   ⚠️  LLAMA_SERVER_PATH not set (binaries will be downloaded on first use)")

print()

# Step 6: Test API imports
print("📦 Step 6: Testing API imports...")
try:
    from llamatelemetry import InferenceEngine, ServerManager
    from llamatelemetry.api.client import LlamaCppClient
    from llamatelemetry.api.multigpu import kaggle_t4_dual_config
    print("   ✅ Core classes imported successfully:")
    print("      - InferenceEngine")
    print("      - ServerManager")
    print("      - LlamaCppClient")
    print("      - kaggle_t4_dual_config")
except Exception as e:
    print(f"   ❌ API import failed: {e}")

print()

# Step 7: Check for updates
print("📡 Step 7: Checking for updates...")
try:
    from llamatelemetry import InferenceEngine
    InferenceEngine.check_for_updates()
except Exception as e:
    print(f"   ⚠️  Update check failed: {e}")

print()
print("=" * 70)
print("✅ Installation Test Complete!")
print("=" * 70)
print()
print("Next steps:")
print("1. Download a GGUF model from HuggingFace")
print("2. Load model: engine = InferenceEngine(); engine.load_model(model_path)")
print("3. Run inference: result = engine.infer('What is AI?')")
print()
