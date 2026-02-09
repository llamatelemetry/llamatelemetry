# llamatelemetry Documentation

**CUDA-first OpenTelemetry Python SDK for LLM inference observability and explainability**

---

## Welcome

llamatelemetry combines high-performance CUDA inference, production-grade observability, and interactive GPU analytics into a unified platform optimized for Kaggle dual Tesla T4 notebooks.

## Quick Links

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Quick Start__

    ---

    Get running in 5 minutes on Kaggle dual T4

    [:octicons-arrow-right-24: Quick Start Guide](QUICK_START_GUIDE.md)

-   :material-download:{ .lg .middle } __Installation__

    ---

    Install llamatelemetry and dependencies

    [:octicons-arrow-right-24: Installation Guide](INSTALLATION.md)

-   :material-notebook:{ .lg .middle } __Notebooks__

    ---

    16 comprehensive tutorials (5.5 hours)

    [:octicons-arrow-right-24: Notebooks Guide](NOTEBOOKS_GUIDE.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Complete API documentation

    [:octicons-arrow-right-24: API Reference](API_REFERENCE.md)

</div>

## What You Get

llamatelemetry provides:

- **LLM request tracing** with semantic attributes and distributed context propagation
- **GPU-aware metrics** (latency, tokens/sec, VRAM usage, temperature, power draw)
- **Split-GPU workflow** (GPU 0: inference, GPU 1: analytics/visualization)
- **Graph-based trace visualization** with Graphistry interactive dashboards
- **Real-time performance monitoring** with live Plotly dashboards
- **Production observability stack** with multi-layer telemetry collection

## Key Features

### :material-gpu: CUDA Inference
- llama.cpp GGUF inference with 29 quantization types
- NCCL-aware multi-GPU execution (dual T4 tensor parallelism)
- FlashAttention v2, KV-cache optimization
- Continuous batching and optimization
- 1B-70B parameter model support

### :material-chart-line: LLM Observability
- OpenTelemetry traces, metrics, and logs
- GPU-native resource detection
- OTLP export (gRPC/HTTP) to Grafana, Jaeger, DataDog
- llama.cpp `/metrics` endpoint integration
- PyNVML GPU monitoring

### :material-graph: Visualization & Analytics
- RAPIDS cuGraph + Graphistry integration
- Interactive trace graphs (2D network visualization)
- Real-time Plotly dashboards (2D/3D)
- Knowledge graph extraction
- Neural network architecture visualization

## Documentation Structure

### Getting Started
- [Installation](INSTALLATION.md) - Install llamatelemetry
- [Quick Start](QUICK_START_GUIDE.md) - 5-minute setup
- [Kaggle Guide](KAGGLE_GUIDE.md) - Kaggle-specific optimization

### Core Documentation
- [Architecture](ARCHITECTURE.md) - System design
- [Configuration](CONFIGURATION.md) - Server configuration
- [Integration Guide](INTEGRATION_GUIDE.md) - OpenTelemetry integration
- [API Reference](API_REFERENCE.md) - Complete API docs

### Guides & Tutorials
- [Notebooks Guide](NOTEBOOKS_GUIDE.md) - 16 comprehensive tutorials
- [GGUF Guide](GGUF_GUIDE.md) - Quantization and GGUF format
- [Quick Reference](QUICK_REFERENCE.md) - Quick lookup tables

### Advanced Topics
- [Build Guide](BUILD_GUIDE.md) - Build from source
- [GGUF Neural Network Visualization](GGUF_NEURAL_NETWORK_VISUALIZATION.md) - Architecture viz

### Help & Support
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
- [GitHub Releases](GITHUB_RELEASE_GUIDE.md) - Release notes

## Quick Start Example

```python
# Install
!pip install -q --no-cache-dir git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0

# Download model
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="bartowski/Qwen2.5-3B-Instruct-GGUF",
    filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    local_dir="/kaggle/working/models",
)

# Start server
from llamatelemetry.server import ServerManager
server = ServerManager()
server.start_server(
    model_path=model_path,
    gpu_layers=99,
    tensor_split="1.0,0.0",
    flash_attn=1,
)

# Run inference
from llamatelemetry.api import LlamaCppClient
client = LlamaCppClient("http://127.0.0.1:8090")
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "What is CUDA?"}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

## Platform Requirements

- **Platform:** Kaggle dual Tesla T4 (30GB VRAM total)
- **Python:** 3.11+
- **CUDA:** 12.5
- **Model Range:** 1B-70B parameters
- **Recommended:** 1B-5B parameters (Q4_K_M quantization)

## Repository

- **GitHub:** [llamatelemetry/llamatelemetry](https://github.com/llamatelemetry/llamatelemetry)
- **Issues:** [GitHub Issues](https://github.com/llamatelemetry/llamatelemetry/issues)
- **Changelog:** [CHANGELOG.md](https://github.com/llamatelemetry/llamatelemetry/blob/main/CHANGELOG.md)

## License

MIT License - See [LICENSE](https://github.com/llamatelemetry/llamatelemetry/blob/main/LICENSE) for details.
