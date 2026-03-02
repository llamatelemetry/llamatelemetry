# Architecture Overview

`llamatelemetry` is a Python orchestration SDK around `llama-server` with optional telemetry and GPU analytics layers.

## High-level layers

1. **Bootstrap/runtime** — automatic discovery and download of CUDA binaries and libs
2. **Inference facade** — `InferenceEngine` provides a high-level API
3. **Server lifecycle** — `ServerManager` starts, monitors, and stops `llama-server`
4. **Client API** — `LlamaCppClient` for OpenAI-compatible endpoints
5. **Model tooling** — GGUF registry, metadata inspection, and quantization helpers
6. **Optional telemetry** — OpenTelemetry traces and GPU metrics
7. **Optional analytics** — Graphistry/RAPIDS hooks and knowledge graphs

## Typical request flow

1. `InferenceEngine.load_model()` resolves model path and downloads if needed
2. `ServerManager.start_server()` launches or attaches to `llama-server`
3. `InferenceEngine.infer()` sends a request to `/completion`
4. The response maps into `InferResult`
5. Optional telemetry spans/metrics are emitted

## Repository layout (high level)

- `llamatelemetry/` Python package
- `csrc/` CUDA/C++ sources
- `docs/` documentation
- `notebooks/` Kaggle-focused notebooks
- `examples/` runnable examples
- `tests/` test suite
- `releases/` source and binary artifacts
