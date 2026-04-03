# Architecture

## High-Level Overview
`llamatelemetry` is a Python SDK that wraps `llama-server` (llama.cpp) for local GGUF inference and adds optional GPU-native telemetry and analytics. The primary user entry point is `InferenceEngine`, which handles model selection, server lifecycle, inference requests, and metrics/telemetry hooks.

## Core Layers
1. **Bootstrap & Runtime Setup**
   - Auto-configures `LD_LIBRARY_PATH`, resolves `llama-server`, and can bootstrap binaries/models on first import.
   - Files: `llamatelemetry/__init__.py`, `llamatelemetry/_internal/bootstrap.py`, `llamatelemetry/server.py`.

2. **Inference Facade**
   - `InferenceEngine` manages model loading (registry/HF/local), auto-configuration, server start, and inference calls.
   - Files: `llamatelemetry/__init__.py`, `llamatelemetry/models.py`, `llamatelemetry/utils.py`.

3. **Server Lifecycle**
   - `ServerManager` finds, downloads, starts, health-checks, and stops `llama-server` processes.
   - Files: `llamatelemetry/server.py`.

4. **HTTP Client API**
   - `LlamaCppClient` provides full OpenAI-compatible and native llama.cpp endpoints.
   - Files: `llamatelemetry/api/client.py`, `llamatelemetry/api/__init__.py`.

5. **Model Tooling**
   - GGUF parsing, quantization helpers, model registry, and suitability reporting.
   - Files: `llamatelemetry/gguf_parser.py`, `llamatelemetry/api/gguf.py`, `llamatelemetry/_internal/registry.py`, `llamatelemetry/models.py`.

6. **Telemetry & Observability (Optional)**
   - OpenTelemetry tracing + GPU metrics collection; optional OTLP export and Graphistry trace export.
   - Files: `llamatelemetry/telemetry/*`.

7. **Kaggle & Multi-GPU Helpers (Optional)**
   - Kaggle presets, secrets, environment setup; multi-GPU/NCCL helpers for tensor split.
   - Files: `llamatelemetry/kaggle/*`, `llamatelemetry/api/multigpu.py`, `llamatelemetry/api/nccl.py`.

## Primary Data Flow (Inference)
1. `InferenceEngine.load_model()` resolves model path (registry/HF/local) — `llamatelemetry/models.py`.
2. Auto-configuration computes GPU layers, ctx size, batch sizes — `llamatelemetry/utils.py`.
3. `ServerManager.start_server()` launches `llama-server` with CLI args — `llamatelemetry/server.py`.
4. `InferenceEngine.infer()` sends HTTP request to `/completion` — `llamatelemetry/__init__.py`.
5. Optional telemetry annotates span + GPU metrics — `llamatelemetry/telemetry/*`.

## Secondary Data Flow (Client API)
- `LlamaCppClient` provides explicit control of llama.cpp endpoints including OpenAI-compatible chat/completions/embeddings, tokenization, slots, LoRA, etc.
- Files: `llamatelemetry/api/client.py`.

## Native Extension Layer
- `llamatelemetry_cpp` provides CUDA tensor/device primitives and ops (matmul).
- Files: `csrc/*`, `core/__init__.py`.

## Key Abstractions
- `InferenceEngine` — high-level UX for inference and server lifecycle.
- `ServerManager` — server binary discovery + process control.
- `LlamaCppClient` — complete HTTP API wrapper.
- `GpuMetricsCollector` / `InferenceTracerProvider` — telemetry instrumentation.
- `KaggleEnvironment` — one-line Kaggle preset + secret loading.
