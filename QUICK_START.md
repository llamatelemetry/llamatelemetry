# llamatelemetry v0.1.0 - Quick Start Guide (Kaggle Only)

Get started with llamatelemetry on Kaggle dual T4 in 5 minutes.

## Prerequisites

**Platform:** Kaggle notebooks (https://kaggle.com/code)
**Required Settings:**
- Accelerator: GPU T4 × 2
- Internet: Enabled
- Python: 3.11+

**Model Requirements:**
- Size: 1B-5B parameters
- Format: GGUF (Q4_K_M quantization)
- Source: HuggingFace (Unsloth-compatible)

## Installation (Kaggle Notebook)

```bash
# In Kaggle notebook cell
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

## Optional: Enable OpenTelemetry

```python
import llamatelemetry

engine = llamatelemetry.InferenceEngine(
    enable_telemetry=True,
    telemetry_config={
        "service_name": "llamatelemetry-inference",
        "otlp_endpoint": "http://localhost:4317",
        "enable_graphistry": False,
    },
)
```

## Optional: Check for Updates

```python
from llamatelemetry import InferenceEngine

InferenceEngine.check_for_updates()
```

## Split-GPU Usage (Recommended: GPU 0 for LLM, GPU 1 for Graphistry)

### Step 1: Download Small GGUF Model (1B-5B)

```python
from huggingface_hub import hf_hub_download

# Download small model optimized for single T4
model_path = hf_hub_download(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",  # 1.2 GB, fits in ~1.5GB VRAM
    local_dir="/kaggle/working/models"
)
```

### Step 2: Start Server on GPU 0 (Split-GPU Architecture)

```python
from llamatelemetry.server import ServerManager

# Start llama-server on GPU 0 only (tensor_split="1.0,0.0")
server = ServerManager()
server.start_server(
    model_path=model_path,
    host="127.0.0.1",
    port=8080,
    gpu_layers=99,           # All layers on GPU
    tensor_split="1.0,0.0",  # 100% GPU 0, 0% GPU 1 (IMPORTANT!)
    flash_attn=1,            # FlashAttention enabled
)

# GPU 1 is now free for Graphistry visualization (see Notebook 11)
```

## Tensor-Split for Large Models (Advanced: Kaggle 2× T4)

### 1. Start Server
```bash
./bin/llama-server \
    -m model.gguf \
    -ngl 99 \
    --tensor-split 0.5,0.5 \
    --split-mode layer \
    -fa \
    --host 0.0.0.0 \
    --port 8080
```

### 2. Connect with Python
```python
from llamatelemetry.api.client import LlamaCppClient

client = LlamaCppClient("http://localhost:8080")

# Use OpenAI-compatible chat.create() API
response = client.chat.create(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

## Unsloth → llamatelemetry Workflow (Kaggle)

```python
# 1. Fine-tune with Unsloth (1B-3B models recommended)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True
)

# 2. Export to GGUF (Q4_K_M recommended for quality/speed)
model.save_pretrained_gguf(
    "/kaggle/working/my_model",
    tokenizer,
    quantization_method="q4_k_m"
)

# 3. Serve with llamatelemetry on GPU 0 (split-GPU architecture)
from llamatelemetry.server import ServerManager
server = ServerManager()
server.start_server(
    model_path="/kaggle/working/my_model-Q4_K_M.gguf",
    gpu_layers=99,
    tensor_split="1.0,0.0",  # GPU 0 only, leave GPU 1 for Graphistry
    flash_attn=1
)
```

## Key Modules (Kaggle-Specific)

| Module | Purpose | Kaggle-Optimized |
|--------|---------|------------------|
| `llamatelemetry.api.client` | OpenAI-compatible client | ✅ |
| `llamatelemetry.api.multigpu` | Split-GPU configuration | ✅ tensor_split="1.0,0.0" |
| `llamatelemetry.server` | llama-server manager | ✅ Built-in binaries |
| `llamatelemetry.graphistry` | Visualization connector | ✅ GPU 1 integration |

## Recommended Models (HuggingFace)

| Model | Size | Repo | File |
|-------|------|------|------|
| Gemma-3 1B | 1.0B | `unsloth/gemma-3-1b-it-GGUF` | `gemma-3-1b-it-Q4_K_M.gguf` |
| Llama-3.2 1B | 1.2B | `unsloth/Llama-3.2-1B-Instruct-GGUF` | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` |
| Gemma-2 2B | 2.0B | `unsloth/gemma-2-2b-it-GGUF` | `gemma-2-2b-it-Q4_K_M.gguf` |
| Qwen2.5 3B | 3.0B | `unsloth/Qwen2.5-3B-Instruct-GGUF` | `Qwen2.5-3B-Instruct-Q4_K_M.gguf` |

## Next Steps

- 📓 **[Notebook 01](notebooks/01-quickstart-llamatelemetry-v0.1.0.ipynb)** - Complete quickstart on Kaggle
- 📓 **[Notebook 11](notebooks/11-gguf-neural-network-graphistry-visualization.ipynb)** - GGUF visualization
- 📖 **[Full Documentation](README.md)** - All features and guides
- 🔗 **[GitHub Releases](https://github.com/llamatelemetry/llamatelemetry/releases)** - Download binaries

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
