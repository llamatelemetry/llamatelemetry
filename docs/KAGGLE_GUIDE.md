# llamatelemetry v0.1.0 - Kaggle Guide

Complete guide for using llamatelemetry on Kaggle's dual Tesla T4 GPU environment.

**Build Info:**
- Binary: `llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz` (961 MB)
- CUDA 12.5, SM 7.5 (Turing), llama.cpp b7760 (388ce82)
- Build Date: 2026-01-16

---

## Table of Contents

- [Kaggle Environment Overview](#kaggle-environment-overview)
- [Hardware Specifications](#hardware-specifications)
- [Setup Instructions](#setup-instructions)
- [Multi-GPU Configuration](#multi-gpu-configuration)
- [Model Recommendations](#model-recommendations)
- [Split-GPU Architecture](#split-gpu-architecture)
- [RAPIDS Integration](#rapids-integration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Kaggle Environment Overview

Kaggle provides free access to dual Tesla T4 GPUs, making it an excellent platform for LLM inference with llamatelemetry.

### Key Advantages

| Feature | Benefit |
|---------|---------|
| **2× Tesla T4** | 30GB total VRAM |
| **CUDA 12.x** | Latest CUDA features |
| **Free tier** | No cost for GPU hours |
| **Pre-installed packages** | RAPIDS, PyTorch, etc. |
| **Persistent storage** | Save outputs across sessions |

### Limitations

| Limitation | Workaround |
|------------|------------|
| 12-hour session limit | Save checkpoints frequently |
| No NVLink | Use efficient tensor-split |
| Internet required for packages | Cache downloads |
| 73GB disk space | Clean up unused files |

---

## Hardware Specifications

### Tesla T4 Specifications

```
┌─────────────────────────────────────────────────────────────────┐
│              KAGGLE DUAL TESLA T4 HARDWARE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   GPU 0: Tesla T4              GPU 1: Tesla T4                 │
│   ├─ VRAM: 15 GB               ├─ VRAM: 15 GB                  │
│   ├─ Compute: SM 7.5           ├─ Compute: SM 7.5              │
│   ├─ Architecture: Turing      ├─ Architecture: Turing         │
│   ├─ FP16: 65 TFLOPS           ├─ FP16: 65 TFLOPS              │
│   └─ Memory BW: 300 GB/s       └─ Memory BW: 300 GB/s          │
│                                                                 │
│   Total VRAM: 30 GB                                            │
│   Interconnect: PCIe (~15 GB/s, no NVLink)                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Verify Hardware

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"\nGPU {i}: {props.name}")
    print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute: {props.major}.{props.minor}")
```

---

## Setup Instructions

### Step 1: Notebook Settings

Configure your Kaggle notebook:

| Setting | Value |
|---------|-------|
| **Accelerator** | GPU T4 × 2 |
| **Internet** | On |
| **Persistence** | Files only |

### Step 2: Install llamatelemetry

```python
# Install llamatelemetry
!pip install -q git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0

# Verify installation
import llamatelemetry
print(f"llamatelemetry {llamatelemetry.__version__}")
```

### Step 3: Download Model

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",
    local_dir="/kaggle/working/models"
)
print(f"Model: {model_path}")
```

### Step 4: Start Server

```python
from llamatelemetry.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path=model_path,
    host="127.0.0.1",
    port=8080,
    n_gpu_layers=99,
    tensor_split="0.5,0.5",  # Dual T4
    flash_attn=True,
)

server = ServerManager()
server.start_with_config(config)
server.wait_until_ready()
print("Server ready!")
```

### Step 5: Use API

```python
from llamatelemetry.api.client import LlamaCppClient

client = LlamaCppClient()
response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

---

## Multi-GPU Configuration

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              MULTI-GPU INFERENCE ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   GPU 0: Tesla T4              GPU 1: Tesla T4                 │
│   ┌─────────────────┐          ┌─────────────────┐             │
│   │ Model Layers    │          │ Model Layers    │             │
│   │ 0 → N/2         │  ←───→   │ N/2 → N         │             │
│   │                 │  PCIe    │                 │             │
│   │ tensor-split    │          │ tensor-split    │             │
│   │ 0.5             │          │ 0.5             │             │
│   └─────────────────┘          └─────────────────┘             │
│                                                                 │
│   llama.cpp uses NATIVE CUDA tensor splitting                  │
│   (NOT NCCL - NCCL is for distributed training)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Quick Configuration

```python
from llamatelemetry.api import kaggle_t4_dual_config

config = kaggle_t4_dual_config(model_path="model.gguf")
print(config.to_cli_args())
# Output: -m model.gguf -ngl 99 --tensor-split 0.5,0.5 --split-mode layer -fa
```

### Manual Configuration

```python
from llamatelemetry.server import ServerConfig

config = ServerConfig(
    model_path="model.gguf",
    
    # GPU settings
    n_gpu_layers=99,           # All layers on GPU
    main_gpu=0,                # Primary GPU
    tensor_split="0.5,0.5",    # 50% each GPU
    split_mode="layer",        # Layer-wise split
    
    # Performance
    flash_attn=True,           # Flash attention
    context_size=4096,         # Context window
)
```

### Tensor Split Options

| Split | GPU 0 | GPU 1 | Best For |
|-------|-------|-------|----------|
| `"0.5,0.5"` | 50% | 50% | Default, equal load |
| `"0.48,0.48"` | 48% | 48% | Leave headroom |
| `"0.6,0.4"` | 60% | 40% | Unequal VRAM usage |
| `"1,0"` | 100% | 0% | Single GPU only |

### CLI Reference

```bash
./bin/llama-server \
    -m model.gguf \
    -ngl 99 \                    # All layers on GPU
    --tensor-split 0.5,0.5 \     # Split across GPUs
    --split-mode layer \         # Layer-wise split
    -fa \                        # Flash attention
    --host 0.0.0.0 \
    --port 8080
```

---

## Model Recommendations

### Model Size vs VRAM

| Model Size | Quantization | VRAM | Fits Kaggle? | Speed |
|------------|--------------|------|--------------|-------|
| 1-3B | Q4_K_M | 2-3 GB | ✅ Single T4 | ~50 tok/s |
| 7-8B | Q4_K_M | 5-6 GB | ✅ Single T4 | ~35 tok/s |
| 7-8B | Q8_0 | 8-9 GB | ✅ Single T4 | ~30 tok/s |
| 13B | Q4_K_M | 8-9 GB | ✅ Single T4 | ~25 tok/s |
| 13B | Q8_0 | 14-15 GB | ✅ Single T4 | ~20 tok/s |
| 32-34B | Q4_K_M | 20-22 GB | ✅ Dual T4 | ~15 tok/s |
| 70B | Q4_K_M | 40-42 GB | ❌ Too large | - |
| 70B | IQ3_XS | 25-27 GB | ✅ Dual T4 | ~8-10 tok/s |
| 70B | IQ2_XXS | 18-20 GB | ✅ Dual T4 | ~10-12 tok/s |

### Recommended Models

#### For Best Speed (Single T4)
```python
# Gemma 3 1B - Fast and capable
model_path = hf_hub_download(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",
)
```

#### For Best Quality (Single T4)
```python
# Qwen2.5 7B - Excellent quality
model_path = hf_hub_download(
    repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
    filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
)
```

#### For Maximum Capability (Dual T4)
```python
# Llama 3.1 70B IQ3_XS - Largest model that fits
model_path = hf_hub_download(
    repo_id="bartowski/Llama-3.1-70B-Instruct-GGUF",
    filename="Llama-3.1-70B-Instruct-IQ3_XS.gguf",
)
```

### I-Quant Types for Large Models

| Quant | Bits | 70B VRAM | Quality |
|-------|------|----------|---------|
| IQ4_XS | ~4.25 | ~35 GB | ⭐⭐⭐⭐ |
| IQ3_M | ~3.4 | ~28 GB | ⭐⭐⭐⭐ |
| IQ3_XS | ~3.0 | ~25 GB | ⭐⭐⭐ |
| IQ2_XXS | ~2.0 | ~18 GB | ⭐⭐ |
| IQ1_M | ~1.75 | ~16 GB | ⭐ |

---

## Split-GPU Architecture

Run LLM inference on GPU 0 while using GPU 1 for other workloads.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPLIT-GPU ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   GPU 0: Tesla T4              GPU 1: Tesla T4                 │
│   ┌─────────────────┐          ┌─────────────────┐             │
│   │  llama-server   │          │  RAPIDS cuDF    │             │
│   │  LLM Inference  │          │  cuGraph        │             │
│   │                 │          │  Graphistry     │             │
│   │  ~5-12 GB       │          │  ~2-8 GB        │             │
│   └─────────────────┘          └─────────────────┘             │
│                                                                 │
│   Use Case: LLM-powered graph analytics                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

```python
from llamatelemetry.server import ServerConfig
import os

# Configure llama-server for GPU 0 only
config = ServerConfig(
    model_path="model.gguf",
    n_gpu_layers=99,
    main_gpu=0,
    tensor_split="1,0",  # All on GPU 0
    flash_attn=True,
)

# Start server
from llamatelemetry.server import ServerManager
server = ServerManager()
server.start_with_config(config)
server.wait_until_ready()

# Use GPU 1 for RAPIDS
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cudf  # Now uses GPU 1
```

---

## RAPIDS Integration

Kaggle has RAPIDS pre-installed. Use it on GPU 1 while llama-server runs on GPU 0.

### Available Packages

| Package | Description |
|---------|-------------|
| `cudf` | GPU DataFrames |
| `cuml` | GPU ML algorithms |
| `cugraph` | GPU graph analytics |
| `cuspatial` | GPU spatial analysis |

### Example: LLM + cuGraph

```python
import os

# Step 1: Start llama-server on GPU 0
from llamatelemetry.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path="model.gguf",
    tensor_split="1,0",  # GPU 0 only
    n_gpu_layers=99,
)
server = ServerManager()
server.start_with_config(config)
server.wait_until_ready()

# Step 2: Use RAPIDS on GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cudf
import cugraph

# Create graph data
edges = cudf.DataFrame({
    'src': [0, 1, 2, 3],
    'dst': [1, 2, 3, 0],
})

G = cugraph.Graph()
G.from_cudf_edgelist(edges, source='src', destination='dst')

# Run PageRank
pagerank = cugraph.pagerank(G)
print(pagerank)

# Step 3: Use LLM to analyze results
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Restore

from llamatelemetry.api.client import LlamaCppClient
client = LlamaCppClient()

response = client.chat_completion(
    messages=[{
        "role": "user",
        "content": f"Analyze these PageRank results: {pagerank.to_pandas().to_dict()}"
    }],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

### RAPIDS Version Note

> **Important:** Do NOT upgrade `cuda-python` or `numba-cuda` on Kaggle.
> This can break RAPIDS packages. Use the pre-installed versions.

---

## Best Practices

### 1. GPU Memory Management

```python
import gc
import torch

def cleanup_gpu():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Call after large operations
cleanup_gpu()
```

### 2. Model Download Caching

```python
from huggingface_hub import hf_hub_download
import os

# Use Kaggle's persistent storage
CACHE_DIR = "/kaggle/working/models"
os.makedirs(CACHE_DIR, exist_ok=True)

# Check if already downloaded
model_file = f"{CACHE_DIR}/model.gguf"
if not os.path.exists(model_file):
    model_path = hf_hub_download(
        repo_id="...",
        filename="...",
        local_dir=CACHE_DIR,
    )
```

### 3. Session Time Management

```python
import time

SESSION_START = time.time()
MAX_SESSION_HOURS = 11.5  # Leave buffer before 12h limit

def check_session_time():
    elapsed = (time.time() - SESSION_START) / 3600
    remaining = MAX_SESSION_HOURS - elapsed
    print(f"Session time: {elapsed:.1f}h, Remaining: {remaining:.1f}h")
    return remaining > 0.5  # Warn if < 30 min left

if not check_session_time():
    print("⚠️ Session ending soon! Save your work.")
```

### 4. Context Size for Large Models

```python
# For 70B models, use smaller context
config = ServerConfig(
    model_path="70b-model.gguf",
    context_size=2048,    # Smaller context
    n_batch=128,          # Smaller batch
    tensor_split="0.48,0.48",  # Leave headroom
)
```

### 5. Error Handling

```python
from llamatelemetry.server import ServerManager, ServerConfig

server = ServerManager()

try:
    server.start_with_config(config)
    if not server.wait_until_ready(timeout=120):
        print("Server failed to start")
        print(server.get_logs())
except Exception as e:
    print(f"Error: {e}")
finally:
    server.stop()
```

---

## Troubleshooting

### GPU Not Detected

```python
# Check GPU availability
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPUs: {torch.cuda.device_count()}")

# Check nvidia-smi
!nvidia-smi
```

**Solution:** Ensure "GPU T4 × 2" is selected in notebook settings.

### Out of Memory (OOM)

```python
# Reduce memory usage
config = ServerConfig(
    model_path="model.gguf",
    context_size=2048,    # Reduce from 4096
    n_batch=256,          # Reduce from 512
    tensor_split="0.45,0.45",  # More headroom
)
```

### Server Won't Start

```python
# Check server logs
server = ServerManager()
server.start_with_config(config)
if not server.wait_until_ready(timeout=120):
    print("Logs:")
    print(server.get_logs())
```

### Slow Performance

```python
# Ensure flash attention is enabled
config = ServerConfig(
    model_path="model.gguf",
    flash_attn=True,      # Enable flash attention
    n_gpu_layers=99,      # Full GPU offload
)
```

### RAPIDS Conflicts

```bash
# DON'T upgrade these packages
# pip install --upgrade cuda-python  # ❌
# pip install --upgrade numba-cuda   # ❌

# Use pre-installed versions
import cudf
print(cudf.__version__)  # Use whatever is installed
```

---

## Tutorial Notebooks

See the complete tutorial series in [`notebooks/`](../notebooks/):

| # | Notebook | Focus |
|---|----------|-------|
| 01 | [Quick Start](../notebooks/01-quickstart-llamatelemetry-v0.1.0.ipynb) | 5-minute introduction |
| 03 | [Multi-GPU](../notebooks/03-multi-gpu-inference-llamatelemetry-v0.1.0.ipynb) | Dual T4 configuration |
| 06 | [Split-GPU](../notebooks/06-split-gpu-graphistry-llamatelemetry-v0.1.0.ipynb) | LLM + RAPIDS |
| 09 | [Large Models](../notebooks/09-large-models-kaggle-llamatelemetry-v0.1.0.ipynb) | 70B on Kaggle |

---

## Next Steps

- **[Installation Guide](INSTALLATION.md)** - Detailed installation
- **[Configuration Guide](CONFIGURATION.md)** - All configuration options
- **[API Reference](API_REFERENCE.md)** - Complete API documentation

## Kaggle Install + Verify (Recommended)

```python
# Install llamatelemetry v0.1.0 from GitHub (force fresh install, no cache)
print("📦 Installing llamatelemetry v0.1.0...")
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0

# Verify installation
import llamatelemetry
print(f"\n✅ llamatelemetry {llamatelemetry.__version__} installed!")

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

## Kaggle Install + Verify (Recommended)

```python
# Install llamatelemetry v0.1.0 from GitHub (force fresh install, no cache)
print("📦 Installing llamatelemetry v0.1.0...")
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0

# Verify installation
import llamatelemetry
print(f"\n✅ llamatelemetry {llamatelemetry.__version__} installed!")

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
