# llamatelemetry v2.2.0

[![Version](https://img.shields.io/badge/version-2.2.0-blue.svg)](https://github.com/llamatelemetry/llamatelemetry/releases/tag/v2.2.0)
[![Python](https://img.shields.io/badge/python-3.11+-brightgreen.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-2×T4-orange.svg)](https://kaggle.com)
[![Documentation](https://img.shields.io/badge/docs-llamatelemetry.github.io-blue.svg)](https://llamatelemetry.github.io)

**CUDA-first OpenTelemetry Python SDK for LLM inference observability and explainability.**

llamatelemetry provides GPU-native telemetry for quantized LLM inference pipelines, integrating multi-GPU execution (NCCL), GGUF/llama.cpp runtimes, and real-time graph-based introspection via pygraphistry.

**Designed for production deployments, llamatelemetry enables:**

- End-to-end tracing of LLM inference requests
- Fine-grained GPU metrics (latency, tokens/sec, memory, NCCL overhead)
- Real-time visualization of internal model activity via GPU-accelerated graph analytics
- Vendor-neutral export using OpenTelemetry (OTLP)

Built to pair with CUDA inference backends and modern LLM stacks, llamatelemetry targets teams that need cost-efficient, on-prem or private LLM deployments with full transparency, auditability, and operational control.

🌐 **[Official Documentation](https://llamatelemetry.github.io/)** | 📖 **[Tutorial Notebooks](https://llamatelemetry.github.io/tutorials/index/)** | 🚀 **[Quick Start](https://llamatelemetry.github.io/guides/quickstart/)** | 🔧 **[API Reference](https://llamatelemetry.github.io/api/overview/)**

---

## 📖 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Multi-GPU Inference](#-multi-gpu-kaggle-2-t4)
- [Unsloth Integration](#-unsloth-integration)
- [Split-GPU Architecture](#-split-gpu-architecture)
- [Features](#-features)
- [Performance](#-performance)
- [Tutorial Notebooks](#-tutorial-notebooks)
- [Documentation](#-documentation)
- [Requirements](#-requirements)

---

## 🚀 Installation

### Quick Install (Kaggle Notebook)
```bash
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0
```

**Distribution Strategy:**
- ✅ **GitHub** (Primary): Direct pip install from repository
- ✅ **HuggingFace** (Mirror): Alternative at `waqasm86/llamatelemetry`
- ❌ **NOT on PyPI/piwheels** - We do not publish to PyPI

**Package Details:**
- Python code: ~62 KB (lightweight package)
- Built-in binaries: ~961 MB (llama.cpp + NCCL, auto-downloaded on first import from GitHub Releases)

### Verify Installation
```python
import llamatelemetry
print(f"llamatelemetry {llamatelemetry.__version__}")  # 2.2.0
```

📘 **[Full Installation Guide →](docs/INSTALLATION.md)** | 🎯 **Platform:** Kaggle only (2× Tesla T4)

---

## ⚡ Quick Start (Kaggle Dual T4)

### Prerequisites
- **Platform:** Kaggle notebook
- **GPUs:** 2× Tesla T4 (15GB VRAM each, SM 7.5)
- **Model Range:** 1B-5B parameters (GGUF Q4_K_M quantization)
- **Settings:** Internet enabled, GPU T4 × 2 selected

### Basic Inference (Single GPU 0)
```python
import llamatelemetry
from huggingface_hub import hf_hub_download

# Download small GGUF model (1B-5B range)
model_path = hf_hub_download(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",
    local_dir="/kaggle/working/models"
)

# Load on GPU 0 (15GB VRAM)
engine = llamatelemetry.InferenceEngine()
engine.load_model(model_path, silent=True)
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
```

### Split-GPU Architecture (GPU 0: LLM, GPU 1: Graphistry)
```python
from llamatelemetry.server import ServerManager

# Start llama-server on GPU 0 (100% allocation)
server = ServerManager()
server.start_server(
    model_path=model_path,
    gpu_layers=99,
    tensor_split="1.0,0.0",  # 100% GPU 0, 0% GPU 1
    flash_attn=1,
)

# GPU 1 now available for Graphistry visualization
# See Notebook 11 for complete visualization workflow
```

📘 **[Quick Start Guide →](QUICK_START.md)** | 📓 **[Notebook 01 →](notebooks/01-quickstart-llamatelemetry-v2.2.0.ipynb)**

---

## 🎯 Split-GPU Architecture (Kaggle 2× T4)

### Recommended: GPU 0 for LLM, GPU 1 for Graphistry
```
┌─────────────────────────────────────────────────────────────────┐
│         KAGGLE DUAL T4 SPLIT-GPU ARCHITECTURE (Optimized)       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   GPU 0: Tesla T4 (15GB VRAM, SM 7.5)                           │
│   ├─ llama.cpp llama-server (C++)                               │
│   ├─ GGUF Model: 1B-5B params (Q4_K_M)                          │
│   ├─ VRAM Usage: ~2-6 GB                                        │
│   ├─ Built-in: FlashAttention, CUDA Graphs                      │
│   └─ tensor-split: "1.0,0.0" (100% GPU 0)                       │
│                                                                 │
│   GPU 1: Tesla T4 (15GB VRAM, SM 7.5)                           │
│   ├─ Graphistry[ai] Python SDK                                  │
│   ├─ RAPIDS cuGraph (GPU-accelerated PageRank)                  │
│   ├─ Neural Network Visualization (929 nodes)                   │
│   ├─ VRAM Usage: ~0.5-2 GB                                      │
│   └─ Free VRAM: ~13 GB for analytics                            │
│                                                                 │
│   Built-in C++ Libraries: llama.cpp + NVIDIA NCCL               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Alternative: Tensor-Split for Large Models (Advanced)
```
┌─────────────────────────────────────────────────────────────────┐
│       KAGGLE DUAL T4 TENSOR-SPLIT (For models >15GB VRAM)       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   GPU 0: Tesla T4 (15GB)    GPU 1: Tesla T4 (15GB)              │
│   ├─ Model Layers 0-39      ├─ Model Layers 40-79               │
│   └─ ~14GB VRAM             └─ ~14GB VRAM                       │
│                                                                 │
│           ← tensor-split 0.5,0.5 (NCCL-based) →                 │
│                                                                 │
│   Total: 30GB VRAM for models up to 70B (IQ3_XS)                │
│   Note: Not recommended for 1B-5B models (use split-GPU)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Start Multi-GPU Server
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

### Python API
```python
from llamatelemetry.server import ServerManager
from llamatelemetry.api.multigpu import kaggle_t4_dual_config
from llamatelemetry.api.client import LlamaCppClient

# Get optimized configuration for Kaggle dual T4
config = kaggle_t4_dual_config()

# Start server with multi-GPU configuration
server = ServerManager()
tensor_split_str = ",".join(str(x) for x in config.tensor_split)
server.start_server(
    model_path="model.gguf",
    gpu_layers=config.n_gpu_layers,
    tensor_split=tensor_split_str,
    split_mode="layer",
    flash_attn=1 if config.flash_attention else 0,
)

# Use OpenAI-compatible API
client = LlamaCppClient("http://localhost:8080")
response = client.chat.create(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

> **Note:** llama.cpp uses **native CUDA tensor-split**, NOT NCCL.
> NCCL is available for PyTorch distributed workloads.

📘 **[Kaggle Multi-GPU Guide →](docs/KAGGLE_GUIDE.md)**

---

## 🔗 Unsloth Integration

Complete workflow from fine-tuning to deployment:

```python
# ═══════════════════════════════════════════════════════════════
# STEP 1: Fine-tune with Unsloth
# ═══════════════════════════════════════════════════════════════
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA and train...

# ═══════════════════════════════════════════════════════════════
# STEP 2: Export to GGUF
# ═══════════════════════════════════════════════════════════════
model.save_pretrained_gguf(
    "my_model",
    tokenizer,
    quantization_method="q4_k_m"  # Recommended for T4
)

# ═══════════════════════════════════════════════════════════════
# STEP 3: Deploy with llamatelemetry
# ═══════════════════════════════════════════════════════════════
from llamatelemetry.server import ServerManager, ServerConfig

server = ServerManager()
server.start_with_config(ServerConfig(
    model_path="my_model-Q4_K_M.gguf",
    n_gpu_layers=99,
    tensor_split="0.5,0.5",  # Dual T4
    flash_attn=True,
))
```

📘 **[Unsloth Integration Guide →](notebooks/05-unsloth-integration-llamatelemetry-v2.2.0.ipynb)**

---

## 🔧 Split-GPU Architecture

Run LLM inference on GPU 0 while using GPU 1 for RAPIDS/Graphistry analytics:

```
┌─────────────────┐      ┌─────────────────┐
│   GPU 0 (T4)    │      │   GPU 1 (T4)    │
├─────────────────┤      ├─────────────────┤
│ llama-server    │      │ RAPIDS cuDF     │
│ LLM Inference   │      │ cuGraph         │
│ ~5-12 GB        │      │ Graphistry      │
└─────────────────┘      └─────────────────┘
```

```python
from llamatelemetry import SplitGPUConfig

config = SplitGPUConfig(llm_gpu=0, graph_gpu=1)
# GPU 0: llama-server (LLM inference)
# GPU 1: RAPIDS cuGraph (graph visualization)
```

📘 **[Split-GPU Tutorial →](notebooks/06-split-gpu-graphistry-llamatelemetry-v2.2.0.ipynb)**

---

## 🎨 GGUF Architecture Visualization ⭐ NEW

**Visualize your GGUF models as interactive graphs** with Notebook 11:

```
┌─────────────────────────────────────────────────────────────────┐
│         GGUF NEURAL NETWORK ARCHITECTURE VISUALIZATION          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   📊 929 Nodes: Complete Llama-3.2-3B structure                 │
│   🔗 981 Edges: All connections and data flows                  │
│   🎯 896 Attention Heads: Multi-head attention visualized       │
│   📦 112 Quantization Blocks: Q4_K_M structure revealed         │
│   🌐 Interactive Graphistry Dashboards: Cloud + offline HTML    │
│                                                                 │
│   ✨ First comprehensive GGUF visualization tool                │
│   ✨ GPU-accelerated graph analytics (PageRank, centrality)     │
│   ✨ Dual-GPU architecture (inference + visualization)          │
│   ✨ Multi-scale: From overview to individual attention heads   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**What You Can Visualize:**
- Layer-by-layer transformer structure (35 nodes per layer)
- Attention head importance and connectivity
- Quantization block memory layout
- Information flow through the network
- Critical components via PageRank analysis

📘 **[GGUF Visualization Guide →](docs/GGUF_NEURAL_NETWORK_VISUALIZATION.md)** | 📓 **[Notebook 11 →](notebooks/11-gguf-neural-network-graphistry-vis-executed-2.ipynb)** | 📓 **[Notebook 12 →](notebooks/12-gguf-attention-mechanism-explorer-executed.ipynb)** | 📓 **[Notebook 13 →](notebooks/13-gguf-token-embedding-visualizer-executed-3.ipynb)**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Kaggle-Optimized** | Built specifically for Kaggle dual Tesla T4 (15GB × 2, SM 7.5) |
| **Small Models** | Optimized for 1B-5B params GGUF (Q4_K_M) on single T4 |
| **Split-GPU** | GPU 0: LLM inference, GPU 1: Graphistry visualization |
| **Built-in C++ Libraries** | llama.cpp llama-server + NVIDIA NCCL (no compilation needed) |
| **FlashAttention** | Built-in for all quantizations (2× speedup) |
| **Unsloth Backend** | CUDA 12-first inference for Unsloth-trained models |
| **Graphistry Dashboards** | Interactive neural network visualization (929 nodes) |
| **OpenAI API** | Full llama.cpp server compatibility |
| **GGUF Tools** | Parse, quantize, analyze GGUF files |
| **Auto-download** | 62KB package, 961MB binaries from GitHub Releases |

---

## 📊 Performance (Kaggle Single Tesla T4)

### Optimized for 1B-5B Models

| Model | Size | Quantization | VRAM | Tokens/sec | Recommended |
|-------|------|--------------|------|------------|-------------|
| Gemma-3 1B | 1.0B | Q4_K_M | ~1.2 GB | ~50 tok/s | ⭐ Best for fast inference |
| Llama-3.2 1B | 1.2B | Q4_K_M | ~1.3 GB | ~48 tok/s | ⭐ Excellent quality |
| Gemma-2 2B | 2.0B | Q4_K_M | ~1.8 GB | ~45 tok/s | ⭐ Balanced |
| Qwen2.5 3B | 3.0B | Q4_K_M | ~2.3 GB | ~40 tok/s | ⭐ High quality |
| Llama-3.2 3B | 3.2B | Q4_K_M | ~2.5 GB | ~38 tok/s | ⭐ Very capable |
| Gemma-3 4B | 4.0B | Q4_K_M | ~3.0 GB | ~35 tok/s | ⭐ Best quality |

**All tested on single Tesla T4 (15GB VRAM, SM 7.5) with FlashAttention enabled**

### VRAM Availability (Split-GPU Architecture)

```
Configuration: GPU 0 for LLM, GPU 1 for Graphistry

GPU 0 Usage:
├─ 1B model: ~1.2 GB → 13.8 GB free
├─ 2B model: ~1.8 GB → 13.2 GB free
├─ 3B model: ~2.5 GB → 12.5 GB free
├─ 4B model: ~3.0 GB → 12.0 GB free
└─ 5B model: ~3.8 GB → 11.2 GB free

GPU 1 Available:
├─ Graphistry: ~0.5-2 GB
├─ RAPIDS cuGraph: ~0.3 GB
└─ Free for analytics: ~13 GB
```

---

## 📓 Tutorial Notebooks

13 comprehensive Kaggle-ready tutorials in [`notebooks/`](notebooks/):

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Quick Start](notebooks/01-quickstart-llamatelemetry-v2.2.0.ipynb) | 5-minute introduction |
| 02 | [Server Setup](notebooks/02-llama-server-setup-llamatelemetry-v2.2.0.ipynb) | Advanced server configuration |
| 03 | [Multi-GPU](notebooks/03-multi-gpu-inference-llamatelemetry-v2.2.0.ipynb) | Dual T4 tensor-split |
| 04 | [GGUF Quantization](notebooks/04-gguf-quantization-llamatelemetry-v2.2.0.ipynb) | Complete quantization guide |
| 05 | [Unsloth Integration](notebooks/05-unsloth-integration-llamatelemetry-v2.2.0.ipynb) | Train → Export → Deploy |
| 06 | [Split-GPU + Graphistry](notebooks/06-split-gpu-graphistry-llamatelemetry-v2.2.0.ipynb) | LLM + RAPIDS analytics |
| 07 | [Knowledge Graph Extraction](notebooks/07-knowledge-graph-extraction-graphistry-v2.2.0.ipynb) | LLM-powered entity extraction + Graphistry |
| 08 | [Document Network Analysis](notebooks/08-document-network-analysis-graphistry-llamatelemetry-v2-2-0.ipynb) | Document similarity + GPU analytics |
| 09 | [Large Models](notebooks/09-large-models-kaggle-llamatelemetry-v2-2-0.ipynb) | Deploy 13B+ on dual T4 |
| 10 | [Complete Workflow](notebooks/10-complete-workflow-llamatelemetry-v2-2-0.ipynb) | Production end-to-end pipeline |
| 11 | [**GGUF Visualization**](notebooks/11-gguf-neural-network-graphistry-vis-executed-2.ipynb) | ⭐ Interactive architecture graphs |
| 12 | [Attention Mechanism Explorer](notebooks/12-gguf-attention-mechanism-explorer-executed.ipynb) | Q-K-V attention patterns + Graphistry |
| 13 | [Token Embedding Visualizer](notebooks/13-gguf-token-embedding-visualizer-executed-3.ipynb) | 3D embedding space + Plotly UMAP |

📘 **[Notebooks Index →](notebooks/README.md)**

---

## 📚 Documentation

### Core Documentation
| Document | Description |
|----------|-------------|
| [QUICK_START.md](QUICK_START.md) | Get started in 5 minutes |
| [INSTALL.md](INSTALL.md) | Detailed installation guide |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

### In-Depth Guides
| Document | Description |
|----------|-------------|
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Complete installation reference |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Server & client configuration |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Python API documentation |
| [docs/KAGGLE_GUIDE.md](docs/KAGGLE_GUIDE.md) | Kaggle-specific guide |
| [docs/GGUF_GUIDE.md](docs/GGUF_GUIDE.md) | GGUF format & quantization |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues & solutions |

### Contributing
| Document | Description |
|----------|-------------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md) | Building from source |

---

## 📋 Requirements

### Platform (Required)
- **Platform:** Kaggle notebooks (https://kaggle.com/code)
- **GPUs:** 2× Tesla T4 (15GB VRAM each, Compute Capability SM 7.5)
- **Python:** 3.11+ (pre-installed on Kaggle)
- **CUDA:** 12.x (pre-installed on Kaggle)

### Kaggle Settings (Required)
- **Accelerator:** GPU T4 × 2 (must select dual T4)
- **Internet:** Enabled (for package installation)
- **Persistence:** Enabled (for downloaded models)

### Model Requirements
- **Size:** 1B-5B parameters recommended
- **Format:** GGUF (from HuggingFace)
- **Quantization:** Q4_K_M (best quality/speed balance)
- **Source:** Unsloth-compatible models preferred

**Note:** llamatelemetry v2.2.0 is designed and tested exclusively for Kaggle dual T4 environment. Other platforms are not officially supported.

---

## 📦 Binary Package

| File | Size | Platform |
|------|------|----------|
| `llamatelemetry-v2.2.0-cuda12-kaggle-t4x2.tar.gz` | 961 MB | Kaggle 2× T4 |

**Build Info:**
- CUDA 12.5, SM 7.5 (Turing)
- llama.cpp b7760 (commit 388ce82)
- Build Date: 2026-01-16
- Contents: 13 binaries (llama-server, llama-cli, llama-quantize, etc.)

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/llamatelemetry/llamatelemetry.git
cd llamatelemetry
pip install -e ".[dev]"
pytest tests/
```

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

## 📓 Tutorial Notebooks (13 notebooks)

Complete tutorial series for llamatelemetry v2.2.0 on Kaggle dual T4 GPUs. Click the badges to open directly in Kaggle or view on GitHub.

| # | Notebook | Open in Kaggle | Description |
|---|----------|----------------|-------------|
| 01 | [Quick Start](notebooks/01-quickstart-llamatelemetry-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llamatelemetry/llamatelemetry/blob/main/notebooks/01-quickstart-llamatelemetry-v2.2.0.ipynb) | 5-minute introduction to llamatelemetry |
| 02 | [Llama Server Setup](notebooks/02-llama-server-setup-llamatelemetry-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llamatelemetry/llamatelemetry/blob/main/notebooks/02-llama-server-setup-llamatelemetry-v2.2.0.ipynb) | Server configuration & lifecycle |
| 03 | [Multi-GPU Inference](notebooks/03-multi-gpu-inference-llamatelemetry-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llamatelemetry/llamatelemetry/blob/main/notebooks/03-multi-gpu-inference-llamatelemetry-v2.2.0.ipynb) | Dual T4 tensor-split configuration |
| 04 | [GGUF Quantization](notebooks/04-gguf-quantization-llamatelemetry-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llamatelemetry/llamatelemetry/blob/main/notebooks/04-gguf-quantization-llamatelemetry-v2.2.0.ipynb) | K-quants, I-quants, GGUF parsing |
| 05 | [Unsloth Integration](notebooks/05-unsloth-integration-llamatelemetry-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llamatelemetry/llamatelemetry/blob/main/notebooks/05-unsloth-integration-llamatelemetry-v2.2.0.ipynb) | Fine-tune → GGUF → Deploy |
| 06 | [Split-GPU + Graphistry](notebooks/06-split-gpu-graphistry-llamatelemetry-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llamatelemetry/llamatelemetry/blob/main/notebooks/06-split-gpu-graphistry-llamatelemetry-v2.2.0.ipynb) | LLM on GPU 0 + RAPIDS on GPU 1 |
| 07 | [Knowledge Graph Extraction](notebooks/07-knowledge-graph-extraction-graphistry-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llamatelemetry/llamatelemetry/blob/main/notebooks/07-knowledge-graph-extraction-graphistry-v2.2.0.ipynb) | LLM entity extraction + graph visualization |
| 08 | [Document Network Analysis](notebooks/08-document-network-analysis-graphistry-llamatelemetry-v2-2-0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llamatelemetry/llamatelemetry/blob/main/notebooks/08-document-network-analysis-graphistry-llamatelemetry-v2-2-0.ipynb) | Document similarity networks with GPU analytics |
| 09 | [Large Models (13B+)](notebooks/09-large-models-kaggle-llamatelemetry-v2-2-0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llamatelemetry/llamatelemetry/blob/main/notebooks/09-large-models-kaggle-llamatelemetry-v2-2-0.ipynb) | Deploy large models on dual T4 with tensor-split |
| 10 | [Complete Workflow](notebooks/10-complete-workflow-llamatelemetry-v2-2-0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llamatelemetry/llamatelemetry/blob/main/notebooks/10-complete-workflow-llamatelemetry-v2-2-0.ipynb) | Production end-to-end: Setup → Model → Server → Analytics → API |
| 11 | [**GGUF Visualization** ⭐](notebooks/11-gguf-neural-network-graphistry-vis-executed-2.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llamatelemetry/llamatelemetry/blob/main/notebooks/11-gguf-neural-network-graphistry-vis-executed-2.ipynb) | **MOST IMPORTANT**: Dual-GPU architecture visualization with 8 interactive dashboards |
| 12 | [Attention Mechanism Explorer](notebooks/12-gguf-attention-mechanism-explorer-executed.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llamatelemetry/llamatelemetry/blob/main/notebooks/12-gguf-attention-mechanism-explorer-executed.ipynb) | Q-K-V attention patterns across all heads with Graphistry dashboards |
| 13 | [Token Embedding Visualizer](notebooks/13-gguf-token-embedding-visualizer-executed-3.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llamatelemetry/llamatelemetry/blob/main/notebooks/13-gguf-token-embedding-visualizer-executed-3.ipynb) | 3D embedding space exploration with GPU-accelerated UMAP + Plotly |

### 🎯 Learning Paths

| Path | Notebooks | Time | Focus |
|------|-----------|------|-------|
| **Quick Start** | 01 → 02 → 03 | 1 hour | Get running fast |
| **Full Course** | 01 → 13 (all) | 5.5 hours | Complete mastery + visualization |
| **Unsloth Focus** | 01 → 04 → 05 → 10 | 2 hours | Fine-tuning workflow |
| **Large Models** | 01 → 03 → 09 | 1.5 hours | 70B on Kaggle |
| **Visualization** ⭐ | 01 → 03 → 04 → 06 → 11 → 12 → 13 | 3.5 hours | Architecture + attention + embeddings |

📘 **[Full Notebook Guide →](notebooks/README.md)**
