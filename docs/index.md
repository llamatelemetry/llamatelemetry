---
title: llamatelemetry - CUDA-first OpenTelemetry Python SDK
description: llamatelemetry is a CUDA-first OpenTelemetry Python SDK for LLM inference observability and GPU telemetry with llama.cpp GGUF, NCCL multi-GPU, and OTLP export.
keywords: llamatelemetry, OpenTelemetry, LLM observability, GPU telemetry, llama.cpp, GGUF, NCCL, OTLP, CUDA, tracing, metrics
---

# llamatelemetry

llamatelemetry is a CUDA-first OpenTelemetry Python SDK for LLM inference observability and GPU telemetry. It focuses on high-performance llama.cpp GGUF inference, NCCL-aware multi-GPU execution, and OTLP-compatible tracing and metrics.

## What You Can Do
- Trace LLM inference with OpenTelemetry spans and semantic attributes
- Collect GPU-aware metrics like latency, tokens/sec, VRAM usage, and power draw
- Run split-GPU workflows (inference on GPU 0, analytics/visualization on GPU 1)
- Visualize traces with Graphistry and interactive dashboards

## Quick Links
- Install and setup: `INSTALLATION.md`
- Quick start: `QUICK_START_GUIDE.md`
- Architecture overview: `ARCHITECTURE.md`
- Integration guide: `INTEGRATION_GUIDE.md`
- API reference: `API_REFERENCE.md`

## Why llamatelemetry
If you need production-grade LLM observability with GPU-native metrics and OpenTelemetry export, llamatelemetry provides a focused Python SDK designed for CUDA and llama.cpp pipelines.

## Repository
Official GitHub repository: https://github.com/llamatelemetry/llamatelemetry
Documentation: https://llamatelemetry.github.io/llamatelemetry/
