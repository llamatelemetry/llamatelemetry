# llamatelemetry v0.1.0

**CUDA-first OpenTelemetry Python SDK for LLM inference observability and explainability**

llamatelemetry is a CUDA-first OpenTelemetry Python SDK for LLM inference observability and GPU telemetry built around llama.cpp GGUF and NCCL multi-GPU workflows.

Official repository: https://github.com/llamatelemetry/llamatelemetry  
Documentation: https://llamatelemetry.github.io/llamatelemetry/

llamatelemetry combines:
- **llama.cpp GGUF inference** - High-performance quantized model inference
- **NCCL-aware multi-GPU execution** - Dual T4 tensor parallelism and split-GPU workflows
- **OpenTelemetry traces + metrics** - Production-grade observability with OTLP export
- **GPU analytics and visualization** - RAPIDS cuGraph + Graphistry interactive dashboards

This repository is optimized for **Kaggle dual Tesla T4 notebooks** and small GGUF models (1B-5B parameters, Q4_K_M quantization). It ships lightweight Python code and downloads large CUDA binaries on first import.

## What You Get
- **LLM request tracing** with semantic attributes and distributed context propagation
- **GPU-aware metrics** (latency, tokens/sec, VRAM usage, temperature, power draw)
- **Split-GPU workflow** (GPU 0: inference, GPU 1: analytics/visualization)
- **Graph-based trace visualization** with Graphistry interactive dashboards
- **Real-time performance monitoring** with live Plotly dashboards
- **Production observability stack** with multi-layer telemetry collection
- **16 comprehensive tutorials** covering foundation → advanced → production workflows

## Quick Start (Kaggle Dual T4)
```python
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0

from huggingface_hub import hf_hub_download
from llamatelemetry.server import ServerManager

model_path = hf_hub_download(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",
    local_dir="/kaggle/working/models",
)

server = ServerManager()
server.start_server(
    model_path=model_path,
    gpu_layers=99,
    tensor_split="1.0,0.0",
    flash_attn=1,
)
```

## Documentation Map
Start here:
- `docs/INDEX.md`

Common entry points:
- `docs/INSTALLATION.md`
- `docs/QUICK_START_GUIDE.md`
- `docs/ARCHITECTURE.md`
- `docs/INTEGRATION_GUIDE.md`
- `docs/NOTEBOOKS_GUIDE.md`
- `docs/TROUBLESHOOTING.md`

Notebooks (16 comprehensive tutorials):
- `notebooks/README.md` - Notebooks overview
- Foundation: Notebooks 01-04 (Quick start, server setup, multi-GPU, quantization)
- Integration: Notebooks 05-06 (Unsloth, Graphistry split-GPU)
- Advanced: Notebooks 07-10 (Knowledge graphs, workflows)
- Deep Dive: Notebooks 11-13 (Neural network viz, attention, embeddings)
- **Observability Trilogy**: Notebooks 14-16 (OpenTelemetry, real-time monitoring, production stack) ⭐ **NEW**

## Project Layout (High Level)
- `llamatelemetry/` Python SDK
- `csrc/` CUDA/C++ kernels and bindings
- `docs/` Guides and references
- `notebooks/` Authoritative notebook specs
- `examples/` Minimal runnable examples
- `scripts/` Build and release utilities

## Support and Contribution
- `CONTRIBUTING.md`
- `CHANGELOG.md`
