#!/usr/bin/env bash
set -euo pipefail

# Requires GitHub CLI and an authenticated session: gh auth login
# Adds SEO-relevant topics to the repository.

REPO="llamatelemetry/llamatelemetry"

gh repo edit "$REPO" --add-topic \
  llamatelemetry \
  opentelemetry \
  llm-observability \
  gpu-telemetry \
  cuda \
  llama-cpp \
  gguf \
  nccl \
  otlp \
  tracing \
  metrics \
  llm \
  observability \
  inference
