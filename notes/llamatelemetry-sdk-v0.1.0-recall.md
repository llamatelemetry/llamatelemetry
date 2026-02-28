# llamatelemetry v0.1.0 SDK Recall Notes

Generated: 2026-02-28
Local path: /media/waqasm86/External1/Project-Nvidia-Office/Project-Llamatelemetry/llamatelemetry

## Purpose
llamatelemetry is a CUDA-first Python SDK focused on llama.cpp GGUF inference with built-in observability. It targets Kaggle dual Tesla T4 (SM 7.5) as the primary runtime, adds multi-GPU (NCCL) support, and layers OpenTelemetry + Graphistry on top of inference workloads.

## Core Entry Points
- `llamatelemetry.InferenceEngine` in `llamatelemetry/__init__.py` for high-level inference and optional telemetry.
- `llamatelemetry.ServerManager` in `llamatelemetry/server.py` for llama-server lifecycle management.
- `llamatelemetry.api.LlamaCppClient` in `llamatelemetry/api/client.py` for complete llama.cpp server API coverage (OpenAI-compatible endpoints + native endpoints).
- `llamatelemetry.kaggle.KaggleEnvironment` in `llamatelemetry/kaggle/environment.py` for zero-boilerplate Kaggle setup.

## Bootstrap and Runtime Setup
- Import-time auto-configuration in `llamatelemetry/__init__.py` sets `LD_LIBRARY_PATH` and `LLAMA_SERVER_PATH` when bundled binaries exist.
- Hybrid bootstrap in `llamatelemetry/_internal/bootstrap.py` downloads CUDA binaries and models on first import when missing.
- Bootstrap is optimized for Kaggle dual T4 (SM 7.5) and validates compute capability.

## Inference Flow (High Level)
1. `InferenceEngine.load_model()` resolves model name/path using `llamatelemetry.models.load_model_smart`.
2. Optional auto-configuration via `llamatelemetry.utils.auto_configure_for_model` chooses GPU layers, context size, and batch settings.
3. `ServerManager.start_server()` locates or downloads `llama-server`, validates GPU compatibility, and launches the server.
4. `InferenceEngine.infer()` posts to `/completion` and records metrics/telemetry on success.

## Model Management
- `llamatelemetry.models.ModelInfo` parses GGUF metadata and recommends settings by VRAM.
- `load_model_smart()` supports registry names, local paths, and HuggingFace `repo:file` syntax.
- `llamatelemetry.gguf_parser` and `llamatelemetry/api/gguf.py` provide low-level GGUF inspection and conversion utilities.

## Telemetry Layer (OpenTelemetry)
- `llamatelemetry.telemetry.setup_telemetry()` creates a TracerProvider and MeterProvider.
- `telemetry/resource.py` builds a GPU-aware OTel resource with CUDA, NCCL, and platform attributes.
- `telemetry/metrics.py` collects GPU metrics via `nvidia-smi` and exposes inference counters/histograms.
- `telemetry/tracer.py` provides inference span annotations and a no-op fallback if OTel is unavailable.
- `telemetry/exporter.py` configures OTLP HTTP/gRPC exporters and console fallback.
- Optional Graphistry export via `telemetry/graphistry_export.py` for trace visualization.

## Multi-GPU + NCCL
- `llamatelemetry.api.multigpu` defines `MultiGPUConfig`, `SplitMode`, GPU discovery, and VRAM estimates.
- `llamatelemetry.api.nccl` provides NCCL discovery and ctypes bindings for collectives.
- Kaggle presets encode common split ratios (50/50, 60/40, etc.) via `kaggle/presets.py`.

## Kaggle Convenience Layer
- `kaggle/environment.py` provides `KaggleEnvironment.setup()` to auto-detect GPUs, load secrets, and pick presets.
- `kaggle/presets.py` defines `ServerPreset` and `TensorSplitMode` to standardize configuration.
- `kaggle/secrets.py` and `kaggle/gpu_context.py` support secret loading and GPU isolation for RAPIDS/Graphistry.

## Graphistry Integration
- `graphistry/connector.py` handles Graphistry registration and graph visualization basics.
- `graphistry/rapids.py`, `graphistry/workload.py`, and `graphistry/viz.py` provide data prep and graph export utilities.

## Unsloth Integration
- `unsloth/loader.py` and `unsloth/exporter.py` provide a pathway from Unsloth fine-tuning to GGUF export.
- `unsloth/adapter.py` handles LoRA adapter management and merging.

## Advanced Inference Helpers
- `inference/` offers flash-attention, KV-cache optimizations, and batch inference helpers.
- `cuda/` contains CUDA graph and tensor-core utilities.

## Package Metadata
- `pyproject.toml` sets Python >= 3.11, core deps, and optional extras for telemetry and graphistry.
- Version is pinned to `0.1.0` and references llama.cpp binary `v0.1.0`.

## Notable Design Traits
- Emphasis on Kaggle dual T4 runtime and auto-downloaded CUDA bundles.
- Optional telemetry is non-blocking with graceful fallbacks.
- API surface is broad: high-level engine, low-level API client, and Kaggle presets.
