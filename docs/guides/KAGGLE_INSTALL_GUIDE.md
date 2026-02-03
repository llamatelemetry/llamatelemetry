# llamatelemetry v0.1.0 - Kaggle Installation Guide

Complete guide for installing and using llamatelemetry on Kaggle notebooks.

## 📋 Prerequisites

### Required Kaggle Settings
1. **Accelerator**: GPU T4 × 2 (dual Tesla T4)
2. **Internet**: Enabled (for package installation)
3. **Persistence**: Enabled (recommended, for downloaded models)

### System Requirements
- Python 3.11+ (pre-installed on Kaggle)
- CUDA 12.x (pre-installed on Kaggle)
- 2× Tesla T4 GPUs (15GB VRAM each, SM 7.5)

---

## 🚀 Quick Install (Recommended)

### Step 1: Install llamatelemetry

```python
# Install llamatelemetry v0.1.0 from GitHub (force fresh install, no cache)
print("📦 Installing llamatelemetry v0.1.0...")
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0

# Verify installation
import llamatelemetry
print(f"\n✅ llamatelemetry {llamatelemetry.__version__} installed!")
```

**What happens:**
- Python package (~100 KB) installs immediately
- On first `import llamatelemetry`, binaries (~1.4 GB) auto-download from:
  1. **HuggingFace CDN** (primary, faster)
  2. **GitHub Releases** (fallback)
- Downloads to `~/.cache/llamatelemetry/` then extracts to package directory
- SHA256 checksum verified automatically

### Step 2: Verify Installation

```python
# Check llamatelemetry status using available APIs
from llamatelemetry import check_cuda_available, get_cuda_device_info
from llamatelemetry.api.multigpu import gpu_count

cuda_info = get_cuda_device_info()
print(f"\n📊 llamatelemetry Status:")
print(f"   CUDA Available: {check_cuda_available()}")
print(f"   GPUs: {gpu_count()}")
if cuda_info:
    print(f"   CUDA Version: {cuda_info.get('cuda_version', 'N/A')}")
```

**Expected output:**
```
📊 llamatelemetry Status:
   CUDA Available: True
   GPUs: 2
   CUDA Version: 12.5
```

---

## 📥 Binary Download Process

### Auto-Download on First Import

When you first `import llamatelemetry`, the package automatically:

1. **Detects GPU**: Verifies Tesla T4 or compatible (SM 7.5+)
2. **Checks Cache**: Looks for binaries in `~/.cache/llamatelemetry/`
3. **Downloads** (if not cached):
   - **Primary**: HuggingFace CDN (`waqasm86/llamatelemetry-binaries`)
   - **Fallback**: GitHub Releases (`llamatelemetry/llamatelemetry/releases`)
4. **Verifies**: SHA256 checksum validation
5. **Extracts**: 13 binaries + libraries to package directory
6. **Configures**: Sets `LD_LIBRARY_PATH` and `LLAMA_SERVER_PATH`

### Download Details

| Source | File | Size | Speed (typical) |
|--------|------|------|-----------------|
| HuggingFace | llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz | 1.4 GB | ~2-5 MB/s |
| GitHub | llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz | 1.4 GB | ~1-3 MB/s |

**Download time**: 5-10 minutes (one-time, cached afterward)

### Manual Download (Advanced)

If auto-download fails, you can manually download and extract:

```python
from huggingface_hub import hf_hub_download
import tarfile
from pathlib import Path

# Download binary tarball
binary_path = hf_hub_download(
    repo_id="waqasm86/llamatelemetry-binaries",
    filename="v0.1.0/llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz",
    cache_dir="/kaggle/working/cache"
)

# Extract to llamatelemetry package directory
import llamatelemetry
package_dir = Path(llamatelemetry.__file__).parent

with tarfile.open(binary_path, "r:gz") as tar:
    tar.extractall(package_dir)

print(f"✅ Binaries extracted to {package_dir}")
```

---

## 🎯 Basic Usage

### Example 1: Quick Inference

```python
import llamatelemetry
from huggingface_hub import hf_hub_download

# Download model (1B-5B recommended for T4)
model_path = hf_hub_download(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",
    local_dir="/kaggle/working/models"
)

# Load model on GPU 0
engine = llamatelemetry.InferenceEngine()
engine.load_model(model_path, silent=True)

# Run inference
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
```

### Example 2: With OpenTelemetry (Observability)

```python
import llamatelemetry

# Enable telemetry
engine = llamatelemetry.InferenceEngine(
    enable_telemetry=True,
    telemetry_config={
        "service_name": "llamatelemetry-inference",
        "otlp_endpoint": "http://localhost:4317",  # Your OTLP endpoint
        "enable_graphistry": False,  # Set True for visualization
    },
)

# Use as normal - telemetry auto-collected
engine.load_model(model_path)
result = engine.infer("Explain quantum computing", max_tokens=150)
```

### Example 3: Split-GPU (LLM on GPU 0, Graphistry on GPU 1)

```python
from llamatelemetry.server import ServerManager

# Start llama-server on GPU 0 only
server = ServerManager()
server.start_server(
    model_path=model_path,
    gpu_layers=99,
    tensor_split="1.0,0.0",  # 100% GPU 0, 0% GPU 1
    flash_attn=1,
)

# GPU 1 is now free for RAPIDS/Graphistry visualization
# See Notebook 11 for complete workflow
```

---

## 🔧 Installation Options

### Option 1: Specific Version (Recommended)

```bash
pip install -q --no-cache-dir --force-reinstall \
    git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

### Option 2: Latest from Main Branch

```bash
pip install -q --no-cache-dir --force-reinstall \
    git+https://github.com/llamatelemetry/llamatelemetry.git@main
```

### Option 3: With Optional Dependencies

```bash
# Full installation (telemetry + graphistry + jupyter)
pip install -q --no-cache-dir --force-reinstall \
    'llamatelemetry[all] @ git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0'

# Just telemetry support
pip install -q --no-cache-dir --force-reinstall \
    'llamatelemetry[telemetry] @ git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0'

# Just graphistry support
pip install -q --no-cache-dir --force-reinstall \
    'llamatelemetry[graphistry] @ git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0'
```

---

## 📊 Verifying Installation

### Complete Verification Script

```python
#!/usr/bin/env python3
"""llamatelemetry v0.1.0 installation verification"""

import llamatelemetry
from pathlib import Path
import os

print(f"✅ llamatelemetry {llamatelemetry.__version__} installed")

# Check binaries
llamatelemetry_dir = Path(llamatelemetry.__file__).parent
binaries_dir = llamatelemetry_dir / "binaries" / "cuda12"

if binaries_dir.exists():
    binaries = list(binaries_dir.glob("llama-*"))
    print(f"✅ Found {len(binaries)} binaries")
else:
    print("⚠️  Binaries not found (will auto-download on first use)")

# Check llama-server
llama_server = os.environ.get("LLAMA_SERVER_PATH")
if llama_server and Path(llama_server).exists():
    print(f"✅ llama-server ready: {llama_server}")
else:
    print("⚠️  llama-server not configured (binaries pending download)")

# Check CUDA
from llamatelemetry import check_cuda_available, get_cuda_device_info
from llamatelemetry.api.multigpu import gpu_count

if check_cuda_available():
    print(f"✅ CUDA available, {gpu_count()} GPUs detected")
else:
    print("❌ CUDA not available")
```

---

## 🐛 Troubleshooting

### Issue 1: Import Fails with "No module named llamatelemetry"

**Solution**: Reinstall with `--force-reinstall`:
```bash
pip install -q --no-cache-dir --force-reinstall \
    git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

### Issue 2: Binary Download Times Out

**Symptoms**: Import hangs or fails during binary download

**Solutions**:
1. Check Kaggle internet is enabled
2. Try manual download from HuggingFace
3. Use GitHub fallback (automatically attempted)

### Issue 3: GPU Not Detected

**Symptoms**: `check_cuda_available()` returns `False`

**Solutions**:
1. Verify Kaggle accelerator is set to "GPU T4 × 2"
2. Check NVIDIA driver: `!nvidia-smi`
3. Ensure no other process is using GPUs

### Issue 4: Incompatible GPU

**Symptoms**: Error "GPU compute capability < 7.5"

**Solutions**:
- llamatelemetry v0.1.0 requires Tesla T4 (SM 7.5) or newer
- Use Kaggle with "GPU T4 × 2" setting
- Other platforms not supported in v0.1.0

---

## 📚 Next Steps

1. **Run Tutorial Notebooks**: Start with [Notebook 01 - Quick Start](https://github.com/llamatelemetry/llamatelemetry/blob/main/notebooks/01-quickstart-llamatelemetry-v0.1.0.ipynb)
2. **Download Models**: Get GGUF models from [waqasm86/llamatelemetry-models](https://huggingface.co/waqasm86/llamatelemetry-models)
3. **Read Documentation**: Visit [llamatelemetry.github.io](https://llamatelemetry.github.io)
4. **Join Community**: Report issues at [GitHub Issues](https://github.com/llamatelemetry/llamatelemetry/issues)

---

## 🔗 Resources

- **GitHub**: https://github.com/llamatelemetry/llamatelemetry
- **HuggingFace Binaries**: https://huggingface.co/waqasm86/llamatelemetry-binaries
- **HuggingFace Models**: https://huggingface.co/waqasm86/llamatelemetry-models
- **Documentation**: https://llamatelemetry.github.io
- **Tutorial Notebooks**: [notebooks/](https://github.com/llamatelemetry/llamatelemetry/tree/main/notebooks)

---

## 📄 License

MIT License - See [LICENSE](https://github.com/llamatelemetry/llamatelemetry/blob/main/LICENSE)
