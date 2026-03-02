# API Reference

This file summarizes the public APIs for `llamatelemetry`. For full reference and notebooks, see https://llamatelemetry.github.io/.

## Core

- `InferenceEngine`
- `InferResult`
- `ServerManager`

## API client

- `LlamaCppClient` (OpenAI-compatible HTTP API)
- `ChatCompletionsAPI`, `EmbeddingsClientAPI`, `ModelsClientAPI`, `SlotsClientAPI`

## Model tooling

- `ModelInfo`, `ModelManager`
- `load_model_smart`, `download_model`
- `gguf_parser.GGUFReader`
- `api.gguf` helper functions

## Telemetry

- `setup_telemetry`
- `GpuMetricsCollector`
- `InferenceTracerProvider`

## Kaggle

- `kaggle_t4_dual_config`
- `KaggleSecrets`, `KagglePipeline`

## CUDA and inference optimizations

- CUDA graphs, tensor cores, Triton kernels
- Batch inference, KV cache, FlashAttention helpers
