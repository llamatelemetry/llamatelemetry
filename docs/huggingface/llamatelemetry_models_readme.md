---
license: mit
tags:
- llm
- gguf
- llama
- gemma
- mistral
- qwen
- inference
- opentelemetry
- observability
- kaggle
library_name: llamatelemetry
---

# llamatelemetry Models

Curated collection of GGUF models optimized for **llamatelemetry** on Kaggle dual Tesla T4 GPUs (2× 15GB VRAM).

## 🎯 About This Repository

This repository contains GGUF models tested and verified to work with:
- **llamatelemetry v0.1.0** - CUDA-first OpenTelemetry Python SDK for LLM inference observability
- **Platform**: Kaggle Notebooks (2× Tesla T4, 30GB total VRAM)
- **CUDA**: 12.5

## 📦 Available Models

> **Status**: Repository created, models coming soon!

### Planned Models (v0.1.0)

| Model | Size | Quantization | VRAM | Speed (tok/s) | Status |
|-------|------|--------------|------|---------------|--------|
| Gemma 3 1B Instruct | 1B | Q4_K_M | ~1.5GB | ~80 | 🔄 Coming soon |
| Gemma 3 3B Instruct | 3B | Q4_K_M | ~3GB | ~50 | 🔄 Coming soon |
| Llama 3.2 3B Instruct | 3B | Q4_K_M | ~3GB | ~50 | 🔄 Coming soon |
| Qwen 2.5 1.5B Instruct | 1.5B | Q4_K_M | ~2GB | ~70 | 🔄 Coming soon |
| Mistral 7B Instruct | 7B | Q4_K_M | ~6GB | ~25 | 🔄 Coming soon |

### Model Selection Criteria

Models in this repository are:
1. ✅ **Tested** on Kaggle dual T4 GPUs
2. ✅ **Verified** to fit in 15GB VRAM (single GPU)
3. ✅ **Compatible** with llamatelemetry's observability features
4. ✅ **Optimized** for GGUF + CUDA acceleration
5. ✅ **Documented** with performance benchmarks

## 🚀 Quick Start

### Install llamatelemetry

```bash
# On Kaggle with GPU T4 × 2
pip install --no-cache-dir --force-reinstall \
    git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

### Download and Run a Model

```python
import llamatelemetry
from llamatelemetry import InferenceEngine
from huggingface_hub import hf_hub_download

# Download model (example - not yet available)
model_path = hf_hub_download(
    repo_id="waqasm86/llamatelemetry-models",
    filename="gemma-3-1b-it-Q4_K_M.gguf",
    local_dir="/kaggle/working/models"
)

# Load model on GPU 0
engine = InferenceEngine()
engine.load_model(model_path, silent=True)

# Run inference with telemetry
result = engine.infer("Explain quantum computing in simple terms", max_tokens=150)
print(result.text)
```

## 📊 Recommended Models by Use Case

### For Fast Prototyping
- **Gemma 3 1B** - Fastest inference, good for testing
- **Qwen 2.5 1.5B** - Balance of speed and quality

### For Production Quality
- **Gemma 3 3B** - High quality, reasonable speed
- **Llama 3.2 3B** - Strong reasoning capabilities

### For Complex Tasks
- **Mistral 7B** - Best quality, slower but fits in single T4

## 🔗 Model Sources

Models are sourced from reputable providers:
- [Unsloth GGUF Models](https://huggingface.co/unsloth) - Optimized GGUF conversions
- [TheBloke GGUF Models](https://huggingface.co/TheBloke) - Community standard
- [Bartowski GGUF Models](https://huggingface.co/bartowski) - High-quality quants

All models are:
- ✅ Publicly available under permissive licenses
- ✅ Re-hosted here for convenience and verification
- ✅ Credited to original authors

## 🎯 Dual GPU Strategies

llamatelemetry supports multi-GPU workloads:

### Strategy 1: LLM on GPU 0, Observability on GPU 1

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
```

### Strategy 2: Model Sharding Across Both GPUs

```python
# Split large model across both T4s
server.start_server(
    model_path=large_model_path,
    gpu_layers=99,
    tensor_split="0.5,0.5",  # 50% GPU 0, 50% GPU 1
)
```

## 📚 Documentation & Links

- **GitHub**: https://github.com/llamatelemetry/llamatelemetry
- **Installation Guide**: [KAGGLE_INSTALL_GUIDE.md](https://github.com/llamatelemetry/llamatelemetry/blob/main/docs/guides/KAGGLE_INSTALL_GUIDE.md)
- **Binaries**: https://huggingface.co/waqasm86/llamatelemetry-binaries
- **Tutorials**: [notebooks/](https://github.com/llamatelemetry/llamatelemetry/tree/main/notebooks)

## 🆘 Getting Help

- **GitHub Issues**: https://github.com/llamatelemetry/llamatelemetry/issues
- **Documentation**: https://llamatelemetry.github.io (planned)

## 📄 License

This repository: MIT License

Individual models: See model cards for specific licenses (Apache 2.0, MIT, Gemma License, etc.)

---

**Maintained by**: [waqasm86](https://huggingface.co/waqasm86)  
**Status**: Repository initialized, models coming soon  
**Target Platform**: Kaggle dual Tesla T4 (CUDA 12.5)  
**Last Updated**: 2026-02-03
