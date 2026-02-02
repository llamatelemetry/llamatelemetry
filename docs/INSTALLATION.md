# llamatelemetry v2.2.0 - Installation Guide (Kaggle Only)

This guide covers installation for llamatelemetry v2.2.0, the CUDA 12-first backend for Unsloth on Kaggle.

**IMPORTANT:** llamatelemetry v2.2.0 is **Kaggle-specific only**. It is optimized for dual Tesla T4 GPUs (15GB × 2, SM 7.5) and is not designed for other environments.

---

## Table of Contents

- [Requirements](#requirements)
- [Quick Install (Kaggle)](#quick-install-kaggle)
- [Installation Methods](#installation-methods)
- [Binary Management](#binary-management)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Requirements

### Platform (Required)
| Component | Requirement |
|-----------|-------------|
| **Platform** | Kaggle notebooks ONLY (https://kaggle.com/code) |
| **GPU** | 2× NVIDIA Tesla T4 (15GB VRAM each, SM 7.5) |
| **Internet** | Enabled (for package installation) |
| **Persistence** | Files only |

### Software (Pre-installed on Kaggle)
| Component | Requirement |
|-----------|-------------|
| **Python** | 3.11+ (pre-installed) |
| **CUDA** | 12.x runtime (pre-installed) |
| **pip** | Latest (pre-installed) |

### Verify Requirements (In Kaggle Notebook)
```python
# Check Python version
!python --version  # Should show 3.11+

# Check CUDA and GPUs
!nvidia-smi  # Should show 2× Tesla T4 with CUDA 12.x

# Check pip
!pip --version
```

---

## Quick Install (Kaggle)

### One-Line Install (Recommended)
```bash
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0
```

### With Verification
```python
import llamatelemetry
print(f"✅ llamatelemetry {llamatelemetry.__version__} installed")
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

---

## Installation Methods

### Method 1: From GitHub (Recommended - Stable Release)

Install the stable v2.2.0 release directly from GitHub:

```bash
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0
```

**Why This Method:**
- ✅ Stable, tested release
- ✅ Reproducible builds
- ✅ Best for production
- ✅ Automatic binary download from GitHub Releases
- ✅ 62KB package + 961MB binaries (downloaded on first import)

### Method 2: From GitHub (Latest Development)

Install the latest development version:

```bash
!pip install -q git+https://github.com/llamatelemetry/llamatelemetry.git
```

**Why This Method:**
- ✅ Newest features and bug fixes
- ⚠️ May have breaking changes
- ⚠️ Less tested than stable release

### Method 3: From HuggingFace (Alternative Mirror)

Download pre-built wheel from HuggingFace:

```bash
# Download wheel from HuggingFace
!wget https://huggingface.co/waqasm86/llamatelemetry/resolve/main/llamatelemetry-2.2.0-py3-none-any.whl

# Install
!pip install llamatelemetry-2.2.0-py3-none-any.whl
```

**Why This Method:**
- ✅ Alternative if GitHub is blocked
- ✅ Same stable v2.2.0 release
- ✅ Hosted on HuggingFace mirror

**Note:** llamatelemetry is **NOT** available on PyPI. We distribute only via:
1. **GitHub** (primary): `git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0`
2. **HuggingFace** (mirror): `https://huggingface.co/waqasm86/llamatelemetry`

---

## Kaggle Installation (Complete Workflow)

### Standard Kaggle Notebook Setup

```python
# Cell 1: Configure Kaggle (Accelerator: GPU T4 × 2, Internet: Enabled)

# Cell 2: Install llamatelemetry
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0

# Cell 3: Verify installation
import llamatelemetry
print(f"llamatelemetry {llamatelemetry.__version__}")

# Cell 4: Check GPUs
!nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
```

### Kaggle with HuggingFace Models (1B-5B GGUF)

```python
# Cell 1: Install dependencies
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0
!pip install -q huggingface_hub

# Cell 2: Download small GGUF model (1B-5B)
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",  # 1.2 GB, fits in single T4
    local_dir="/kaggle/working/models"
)

# Cell 3: Start server with split-GPU architecture
from llamatelemetry.server import ServerManager

server = ServerManager()
server.start_server(
    model_path=model_path,
    gpu_layers=99,
    tensor_split="1.0,0.0",  # GPU 0: LLM, GPU 1: free for Graphistry
    flash_attn=1,
)
```

### Kaggle Settings (Required)

| Setting | Value | Why |
|---------|-------|-----|
| **Accelerator** | GPU T4 × 2 | llamatelemetry v2.2.0 requires dual T4 |
| **Internet** | Enabled | For pip install and binary download |
| **Persistence** | Files only | Keep downloaded models |
| **Python** | 3.11+ | Pre-installed on Kaggle |

---

## Binary Management

llamatelemetry includes pre-compiled C++ libraries (llama.cpp llama-server + NVIDIA NCCL) for CUDA 12. These are managed automatically.

### Distribution Strategy

```
GitHub Repository (Code Only):
├─ Python package: ~62 KB
└─ Source code only

GitHub Releases (Binary Package):
├─ llamatelemetry-v2.2.0-cuda12-kaggle-t4x2.tar.gz
├─ Size: ~961 MB
├─ Contents: llama.cpp binaries + NCCL libraries
└─ Downloaded automatically on first import

HuggingFace Mirror (Alternative):
├─ Wheel: llamatelemetry-2.2.0-py3-none-any.whl
└─ Same binaries downloaded from GitHub Releases
```

### Automatic Download (Default)

On first import, llamatelemetry downloads binaries (~961 MB) from GitHub Releases:

```python
import llamatelemetry  # Downloads to /kaggle/working/.cache/llamatelemetry/ or ~/.cache/llamatelemetry/
```

**What Gets Downloaded:**
- llama.cpp binaries (llama-server, llama-cli, llama-quantize, etc.)
- NVIDIA NCCL libraries (for multi-GPU support)
- CUDA 12 runtime libraries
- Total size: ~961 MB

### Manual Binary Download (If Automatic Fails)

```bash
# Download from GitHub Releases
!wget https://github.com/llamatelemetry/llamatelemetry/releases/download/v2.2.0/llamatelemetry-v2.2.0-cuda12-kaggle-t4x2.tar.gz

# Extract to Kaggle cache directory
!mkdir -p /kaggle/working/.cache/llamatelemetry
!tar -xzf llamatelemetry-v2.2.0-cuda12-kaggle-t4x2.tar.gz -C /kaggle/working/.cache/llamatelemetry/

# Verify binaries
!ls /kaggle/working/.cache/llamatelemetry/bin/
!ls /kaggle/working/.cache/llamatelemetry/lib/
```

### Binary Contents (Built-in C++ Libraries)

| Binary | Description | Source |
|--------|-------------|--------|
| `llama-server` | HTTP server with OpenAI API | llama.cpp |
| `llama-cli` | Command-line interface | llama.cpp |
| `llama-quantize` | GGUF quantization tool | llama.cpp |
| `llama-gguf` | GGUF metadata tool | llama.cpp |
| `llama-embedding` | Embedding extraction | llama.cpp |
| `llama-perplexity` | Perplexity calculation | llama.cpp |
| `libnccl.so.*` | Multi-GPU communication | NVIDIA NCCL |
| `libcudart.so.*` | CUDA runtime | CUDA 12 |

### Custom Binary Location (Advanced)

```python
import os
os.environ['LLCUDA_BINARY_PATH'] = '/custom/path/to/binaries'

import llamatelemetry
```

---

## Verification

### Basic Verification

```python
import llamatelemetry

# Check version
print(f"Version: {llamatelemetry.__version__}")

# Check available modules
from llamatelemetry.api import LlamaCppClient
from llamatelemetry.server import ServerManager, ServerConfig
from llamatelemetry.api.gguf import GGUFParser

print("✅ All core modules available")
```

### GPU Verification

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
```

### Multi-GPU Configuration

```python
from llamatelemetry.api import kaggle_t4_dual_config

config = kaggle_t4_dual_config()
print(f"CLI args: {config.to_cli_args()}")
```

### Full System Check

```python
def verify_llamatelemetry():
    """Complete llamatelemetry verification."""
    checks = []
    
    # Check 1: Import
    try:
        import llamatelemetry
        checks.append(("Import", True, llamatelemetry.__version__))
    except ImportError as e:
        checks.append(("Import", False, str(e)))
        return checks
    
    # Check 2: CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        checks.append(("CUDA", cuda_ok, f"{gpu_count} GPU(s)"))
    except Exception as e:
        checks.append(("CUDA", False, str(e)))
    
    # Check 3: Modules
    try:
        from llamatelemetry.api import LlamaCppClient
        from llamatelemetry.server import ServerManager
        checks.append(("Modules", True, "OK"))
    except Exception as e:
        checks.append(("Modules", False, str(e)))
    
    # Check 4: Binary path
    try:
        import os
        binary_path = os.path.expanduser("~/.cache/llamatelemetry/bin")
        exists = os.path.exists(binary_path)
        checks.append(("Binaries", exists, binary_path if exists else "Not found"))
    except Exception as e:
        checks.append(("Binaries", False, str(e)))
    
    return checks

# Run verification
for name, status, info in verify_llamatelemetry():
    icon = "✅" if status else "❌"
    print(f"{icon} {name}: {info}")
```

---

## Troubleshooting

### Common Issues

#### 1. Import Error

**Problem:** `ModuleNotFoundError: No module named 'llamatelemetry'`

**Solution:**
```bash
pip uninstall llamatelemetry
pip cache purge
pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0
```

#### 2. CUDA Not Found

**Problem:** `CUDA not available` or `nvidia-smi not found`

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# On Kaggle, ensure GPU is enabled in settings
```

#### 3. Binary Download Failed

**Problem:** Timeout or connection error during binary download

**Solution:**
```bash
# Manual download
wget https://github.com/llamatelemetry/llamatelemetry/releases/download/v2.2.0/llamatelemetry-v2.2.0-cuda12-kaggle-t4x2.tar.gz
mkdir -p ~/.cache/llamatelemetry
tar -xzf llamatelemetry-v2.2.0-cuda12-kaggle-t4x2.tar.gz -C ~/.cache/llamatelemetry/
```

#### 4. Permission Denied

**Problem:** Cannot execute binaries

**Solution:**
```bash
chmod +x ~/.cache/llamatelemetry/bin/*
```

#### 5. Wrong Python Version

**Problem:** Syntax errors or module issues

**Solution:**
```bash
# Check Python version
python --version

# Use Python 3.11+
python3.11 -m pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0
```

#### 6. Conflicting Packages

**Problem:** Version conflicts with other packages

**Solution:**
```bash
# Create fresh environment
python -m venv llamatelemetry-env
source llamatelemetry-env/bin/activate
pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0
```

###Getting Help

If issues persist:

1. Check [GitHub Issues](https://github.com/llamatelemetry/llamatelemetry/issues)
2. Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
3. Open a new issue with:
   - Python version (`!python --version`)
   - CUDA version (`!nvidia-smi`)
   - Error message (full traceback)
   - Kaggle notebook link (if possible)

### Distribution Clarification

**Where llamatelemetry is Available:**
- ✅ GitHub: `git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0` (Primary)
- ✅ HuggingFace: `https://huggingface.co/waqasm86/llamatelemetry` (Mirror)

**Where llamatelemetry is NOT Available:**
- ❌ PyPI (pypi.org) - We do not publish to PyPI
- ❌ piwheels (piwheels.org) - Not listed on piwheels

**Why Not PyPI?**
llamatelemetry v2.2.0 is Kaggle-specific with large binary dependencies (961 MB). GitHub Releases provides better distribution for large packages compared to PyPI's size limits.

---

## Next Steps

After installation:

1. **[Quick Start Guide](../QUICK_START.md)** - Get started in 5 minutes
2. **[Configuration Guide](CONFIGURATION.md)** - Server and client options
3. **[Tutorial Notebooks](../notebooks/README.md)** - Step-by-step tutorials
