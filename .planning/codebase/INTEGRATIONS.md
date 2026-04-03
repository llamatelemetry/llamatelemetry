# Integrations

## llama.cpp / llama-server
- HTTP API endpoints: `/completion`, `/v1/chat/completions`, `/v1/embeddings`, `/health`, `/metrics`, `/slots`, `/props`.
- Managed by `ServerManager` and used by `InferenceEngine` and `LlamaCppClient`.
- Files: `llamatelemetry/server.py`, `llamatelemetry/__init__.py`, `llamatelemetry/api/client.py`.

## Hugging Face Hub
- Model downloads (GGUF) and binary bundles via Hugging Face repos.
- Files: `llamatelemetry/_internal/bootstrap.py`, `llamatelemetry/models.py`, `llamatelemetry/_internal/registry.py`.

## GitHub Releases
- Binary bundle download fallback for `llama-server`.
- Files: `llamatelemetry/server.py`, `llamatelemetry/_internal/bootstrap.py`.

## OpenTelemetry (OTel)
- Tracing + metrics via `opentelemetry-api`/`opentelemetry-sdk` with optional OTLP exporters.
- Files: `llamatelemetry/telemetry/__init__.py`, `llamatelemetry/telemetry/tracer.py`, `llamatelemetry/telemetry/metrics.py`, `llamatelemetry/telemetry/exporter.py`.

## OTLP Export (gRPC/HTTP)
- Vendor-neutral telemetry export with `opentelemetry-exporter-otlp-*`.
- Files: `llamatelemetry/telemetry/exporter.py`, `llamatelemetry/telemetry/__init__.py`.

## Graphistry / RAPIDS
- Optional graph visualization + GPU analytics integration.
- Files: `llamatelemetry/graphistry/*`, `llamatelemetry/telemetry/graphistry_export.py`.

## Kaggle Environment
- Kaggle secrets and environment detection, GPU presets.
- Files: `llamatelemetry/kaggle/environment.py`, `llamatelemetry/kaggle/secrets.py`, `llamatelemetry/kaggle/presets.py`.

## NVIDIA Driver / CUDA Toolkit
- CUDA detection + GPU stats via `nvidia-smi`, `nvcc`.
- Files: `llamatelemetry/utils.py`, `llamatelemetry/telemetry/metrics.py`, `llamatelemetry/telemetry/monitor.py`.

## NCCL (Multi-GPU)
- NCCL config/availability helpers for multi-GPU inference.
- Files: `llamatelemetry/api/nccl.py`, `llamatelemetry/api/multigpu.py`.
