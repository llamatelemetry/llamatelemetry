---
license: mit
---

# llamatelemetry

**CUDA-first OpenTelemetry Python SDK for LLM inference observability**

llamatelemetry is a specialized OpenTelemetry SDK designed for observing CUDA-accelerated LLM inference workloads on Kaggle and similar GPU environments.

## 🚀 Quick Start

```bash
# Install on Kaggle with GPU T4 × 2
pip install --no-cache-dir --force-reinstall \
    git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

```python
# Auto-downloads 1.4 GB CUDA binaries on first import
import llamatelemetry
from llamatelemetry import InferenceEngine

engine = InferenceEngine()
engine.load_model("path/to/model.gguf")
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
```

## 📦 Repositories

### Binaries
Pre-compiled CUDA binaries for llamatelemetry:
- **Organization**: [llamatelemetry/binaries](https://huggingface.co/llamatelemetry/binaries) (planned)
- **Personal**: [waqasm86/llamatelemetry-binaries](https://huggingface.co/waqasm86/llamatelemetry-binaries) (active)

### Models
GGUF models optimized for Tesla T4 GPUs:
- [waqasm86/llamatelemetry-models](https://huggingface.co/waqasm86/llamatelemetry-models) (planned)

## 🔗 Links

- **GitHub**: https://github.com/llamatelemetry/llamatelemetry
- **Releases**: https://github.com/llamatelemetry/llamatelemetry/releases
- **Installation Guide**: [KAGGLE_INSTALL_GUIDE.md](https://github.com/llamatelemetry/llamatelemetry/blob/main/docs/guides/KAGGLE_INSTALL_GUIDE.md)

## 📊 Features

- **CUDA-First**: Optimized for NVIDIA Tesla T4 GPUs (SM 7.5)
- **Multi-GPU**: Dual GPU support for split workloads
- **OpenTelemetry**: OTLP-based observability with vendor-neutral telemetry
- **Auto-Download**: Automatic binary provisioning from HuggingFace CDN
- **Kaggle-Ready**: Pre-configured for Kaggle dual T4 notebooks

## 📄 License

MIT License - See [LICENSE](https://github.com/llamatelemetry/llamatelemetry/blob/main/LICENSE)

---

**Version**: 0.1.1  
**Target**: Kaggle 2× Tesla T4 (CUDA 12.5)  
**Status**: Active Development
