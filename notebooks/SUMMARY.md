# Notebook Summary

The 18 notebooks form a progressive curriculum. Early notebooks introduce the
high-level SDK and llama.cpp lifecycle. Later notebooks add model analysis,
third-party integrations, and production observability.

## 01. Quickstart

Introduces `InferenceEngine`, CUDA detection, registry-based model loading,
single and batch inference, engine metrics, and the OpenAI-compatible
`LlamaCppClient`. This is the entry point for all other tracks.

## 02. llama-server Setup and Introspection

Uses `ServerManager` directly to control the server lifecycle. It enables and
queries the health, model properties, slot state, and Prometheus metrics
endpoints, then demonstrates clean shutdown.

## 03. Multi-GPU Inference

Builds `MultiGPUConfig` and `NCCLConfig` objects for Kaggle's dual Tesla T4
runtime. It demonstrates tensor splitting and `split_gpu_session` for explicit
GPU workload isolation.

## 04. GGUF Quantization and Suitability

Uses the GGUF utilities to inspect metadata, produce model reports, estimate
file and VRAM requirements, compare quantization options, and recommend a
quantization for Kaggle hardware.

## 05. Unsloth Integration

Shows the supported bridge from Unsloth model loading to `ExportConfig` and
`UnslothExporter`. The expensive model load and GGUF export steps are explicit
so users can adapt them to their own fine-tuned model.

## 06. Split-GPU Graphistry

Demonstrates the architecture used throughout the graph track: LLM inference
on one GPU and graph construction or analytics on the other. It builds a small
knowledge graph and optionally renders it with Graphistry.

## 07. Knowledge Graph Extraction

Defines typed entities and relationships, converts them with
`GraphistryBuilders.knowledge_graph`, inspects the resulting node and edge
tables, and optionally publishes an interactive graph.

## 08. Document Network Analysis

Creates a graph from pairwise document similarity scores. It demonstrates
threshold filtering and prepares the resulting network for visual exploration.

## 09. Large Models on Kaggle

Checks available GPU resources, evaluates whether a GGUF model is suitable for
the runtime, compares server presets, and passes the selected preset into the
model-loading workflow.

## 10. Complete Kaggle Workflow

Combines Kaggle Secrets, server presets, `ServerManager`, telemetry setup, the
instrumented client, inference, and server metrics. It is the reference
end-to-end operational workflow.

## 11. GGUF Neural Network Visualization

Reads a GGUF report and builds an embedding nearest-neighbor graph. The graph
provides a visual way to inspect local neighborhood structure associated with
model representations.

## 12. Attention Mechanism Explorer

Generates an attention matrix, converts weighted token relationships into a
graph, identifies the strongest attention pairs, and compares graph density at
different thresholds.

## 13. Token Embedding Visualizer

Builds a nearest-neighbor graph from token embeddings and checks whether known
clusters remain connected. The notebook can use generated data or embeddings
returned by a compatible llama.cpp model.

## 14. OpenTelemetry LLM Observability

Initializes Grafana-compatible OTLP export and uses
`InstrumentedLlamaCppClient` to trace chat, completion, and embedding requests.
It is the baseline notebook for distributed tracing.

## 15. Real-Time Performance Monitoring

Uses `PerformanceMonitor` to collect inference records, calculate summaries,
pull llama.cpp Prometheus metrics, and export records to a DataFrame for
analysis or visualization.

## 16. Production Observability Pipeline

Builds a production-oriented inference loop around preset-based server startup,
OTLP configuration, instrumented requests, health checks, and deterministic
cleanup.

## 17. Weights & Biases Integration

Adds W&B experiment tracking to the Kaggle pipeline. It logs configuration,
inference performance, generated outputs, and server health while retaining the
SDK telemetry path.

## 18. OTel and Graphistry Trace Glue

Demonstrates the post-processing layer between telemetry and visualization. It
converts exported spans into inference records, tabular data, graph nodes and
edges, and a latency time series suitable for Graphistry or other dashboards.

## Coverage

| Area | Notebooks |
|---|---|
| Core inference and server management | 01, 02, 03, 09, 10 |
| GGUF analysis and export | 04, 05, 11 |
| Graph construction and visualization | 06, 07, 08, 11, 12, 13, 18 |
| OpenTelemetry and runtime monitoring | 10, 14, 15, 16, 18 |
| External experiment platforms | 17 |

For ordering, prerequisites, and direct notebook links, see
[`README.md`](README.md). For a side-by-side feature comparison, see
[`VISUAL-COMPARISON.md`](VISUAL-COMPARISON.md).
