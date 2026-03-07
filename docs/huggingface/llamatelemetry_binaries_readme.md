---
license: mit
tags:
- llm
- cuda
- inference
- opentelemetry
- observability
- telemetry
- gguf
- llama-cpp
- kaggle
- binaries
library_name: llamatelemetry
---

# llamatelemetry Binaries v0.1.1

Pre-compiled CUDA binaries for **llamatelemetry** - CUDA-first OpenTelemetry Python SDK for LLM inference observability.

## 📦 Available Binaries

| Version | File | Size | Target Platform | SHA256 |
|---------|------|------|-----------------|--------|
| v0.1.1 | llamatelemetry-v0.1.1-cuda12-kaggle-t4x2.tar.gz | 1.4 GB | Kaggle 2× Tesla T4, CUDA 12.5 | `31889a86116818be5a42a7bd4a20fde14be25f27348cabf2644259625374b355` |

## 🚀 Auto-Download (Recommended)

These binaries are automatically downloaded when you install llamatelemetry:

```bash
# Install on Kaggle with GPU T4 × 2
pip install --no-cache-dir --force-reinstall \
    git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

On first `import llamatelemetry`, the package will:
1. Detect your GPU (Tesla T4 required)
2. Check for cached binaries in `~/.cache/llamatelemetry/`
3. Download from HuggingFace CDN (this repo - fast, ~2-5 MB/s)
4. Fallback to GitHub Releases if needed
5. Verify SHA256 checksum: `31889a86116818be5a42a7bd4a20fde14be25f27348cabf2644259625374b355`
6. Extract 13 binaries + libraries to package directory
7. Configure environment variables

## 📥 Manual Download

### Using huggingface_hub

```python
from huggingface_hub import hf_hub_download

binary_path = hf_hub_download(
    repo_id="waqasm86/llamatelemetry-binaries",
    filename="v0.1.1/llamatelemetry-v0.1.1-cuda12-kaggle-t4x2.tar.gz",
    cache_dir="/kaggle/working/cache"
)

print(f"Downloaded to: {binary_path}")
```

### Direct Download URL

```bash
wget https://huggingface.co/waqasm86/llamatelemetry-binaries/resolve/main/v0.1.1/llamatelemetry-v0.1.1-cuda12-kaggle-t4x2.tar.gz
```

### Verify Checksum

```bash
# Download checksum file
wget https://huggingface.co/waqasm86/llamatelemetry-binaries/resolve/main/v0.1.1/llamatelemetry-v0.1.1-cuda12-kaggle-t4x2.tar.gz.sha256

# Verify
sha256sum -c llamatelemetry-v0.1.1-cuda12-kaggle-t4x2.tar.gz.sha256
```

## 📊 Build Information

| Property | Value |
|----------|-------|
| **Version** | 0.1.1 |
| **CUDA Version** | 12.5 |
| **Compute Capability** | SM 7.5 (Tesla T4) |
| **llama.cpp Version** | b7760 (commit 388ce82) |
| **Build Date** | 2026-02-03 |
| **Target Platform** | Kaggle dual Tesla T4 GPUs (2× 15GB VRAM) |
| **Binaries Included** | 13 (llama-server, llama-cli, llama-bench, etc.) |
| **Libraries** | CUDA shared libraries + dependencies |

## 🔧 What's Inside

The binary bundle contains:

### Executables (13 binaries)
- `llama-server` - OpenAI-compatible API server
- `llama-cli` - CLI inference tool
- `llama-bench` - Benchmarking utility
- `llama-quantize` - Model quantization tool
- And 9 more utilities

### Shared Libraries
- CUDA 12.5 shared libraries
- cuBLAS, cuDNN dependencies
- llama.cpp runtime libraries

## 🔗 Links

- **GitHub Repository**: https://github.com/llamatelemetry/llamatelemetry
- **GitHub Releases**: https://github.com/llamatelemetry/llamatelemetry/releases/tag/v0.1.1
- **Installation Guide**: [KAGGLE_INSTALL_GUIDE.md](https://github.com/llamatelemetry/llamatelemetry/blob/main/docs/guides/KAGGLE_INSTALL_GUIDE.md)
- **Models Repository**: https://huggingface.co/waqasm86/llamatelemetry-models
- **Documentation**: https://llamatelemetry.github.io (planned)

## 🎯 Supported Platforms

| Platform | GPU | CUDA | Status |
|----------|-----|------|--------|
| Kaggle Notebooks | 2× Tesla T4 (SM 7.5) | 12.5 | ✅ Supported |
| Google Colab | Tesla T4 (SM 7.5) | 12.x | 🔄 Planned |
| Local Workstation | Tesla T4, RTX 4000+ | 12.x+ | 🔄 Planned |
| Other GPUs | SM < 7.5 | Any | ❌ Not supported |

## 📄 License

MIT License - See [LICENSE](https://github.com/llamatelemetry/llamatelemetry/blob/main/LICENSE)

## 🆘 Troubleshooting

### Binary Download Fails

1. **Check internet connection** in Kaggle notebook settings
2. **Retry import**: `import llamatelemetry` (automatic retry logic)
3. **Manual download**: Use `hf_hub_download()` method above
4. **GitHub fallback**: Binaries also available at GitHub Releases

### GPU Not Detected

```python
from llamatelemetry import check_cuda_available, get_cuda_device_info

print(f"CUDA Available: {check_cuda_available()}")
print(f"GPU Info: {get_cuda_device_info()}")
```

Expected on Kaggle T4 × 2:
```
CUDA Available: True
GPU Info: {'gpu_name': 'Tesla T4', 'cuda_version': '12.5', 'compute_capability': '7.5'}
```

### Incompatible GPU Error

llamatelemetry v0.1.1 requires Tesla T4 (SM 7.5) or newer. If you see "GPU compute capability < 7.5", you're running on an incompatible GPU.

**Solution**: Use Kaggle with "GPU T4 × 2" accelerator setting.

---

**Maintained by**: [waqasm86](https://huggingface.co/waqasm86)  
**Version**: 0.1.1  
**Last Updated**: 2026-02-03  
**Status**: Active Development
