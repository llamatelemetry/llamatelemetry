# llamatelemetry v0.1.0

**CUDA-first Python SDK for GGUF inference, telemetry, and Kaggle workflows**

`llamatelemetry` is a CUDA-first Python SDK for local LLM inference and observability around `llama.cpp`/GGUF. It provides a high-level inference engine, a robust `llama-server` lifecycle manager, OpenTelemetry instrumentation, Kaggle presets, Graphistry/RAPIDS hooks, and optional CUDA optimization utilities.

## What you get

- High-level `InferenceEngine` for loading and running GGUF models
- `ServerManager` for starting and monitoring `llama-server`
- OpenAI-compatible `LlamaCppClient` for endpoint-level control
- Model registry, GGUF metadata parsing, and quantization helpers
- OpenTelemetry tracing + GPU metrics collection
- Kaggle dual-T4 presets and environment helpers
- 18 notebooks with cell-by-cell walkthroughs

## Install

```bash
pip install --no-cache-dir --force-reinstall \
  git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

## Quick start

```python
import llamatelemetry as lt

engine = lt.InferenceEngine(enable_telemetry=False)
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
result = engine.infer("Explain CUDA in one sentence.", max_tokens=64)
print(result.text)
```

## Kaggle quickstart (dual T4)

```python
import llamatelemetry as lt
from llamatelemetry.api import kaggle_t4_dual_config

cfg = kaggle_t4_dual_config()
print(cfg)

engine = lt.InferenceEngine(enable_telemetry=False)
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
print(engine.generate("Kaggle dual-GPU test", max_tokens=32).text)
```

## Documentation

- Docs site: https://llamatelemetry.github.io/
- Get Started: `docs/INSTALLATION.md`, `docs/QUICK_START_GUIDE.md`
- Core docs: `docs/ARCHITECTURE.md`, `docs/API_REFERENCE.md`
- Notebooks: `notebooks/` (18 notebooks)

## Project layout

- `llamatelemetry/` Python package
- `csrc/` CUDA/C++ sources
- `docs/` documentation for mkdocs
- `notebooks/` Kaggle-focused notebooks
- `examples/` runnable examples
- `tests/` test suite

## Contributing

- `CONTRIBUTING.md`
- `CHANGELOG.md`
