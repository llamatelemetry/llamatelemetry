# Notebook Comparison

Use this matrix to select a notebook by workload, hardware requirement, and
integration. "Optional" means the core explanatory cells can run without the
service, while publishing or live inference requires it.

| # | Primary topic | Model/server | Dual GPU | Graph data | OTel/metrics | External service |
|---|---|---:|---:|---:|---:|---|
| 01 | SDK quickstart and inference | Yes | No | No | Engine metrics | None |
| 02 | Server lifecycle and endpoints | Yes | No | No | `/metrics` | None |
| 03 | Multi-GPU and NCCL configuration | Yes | Yes | No | No | None |
| 04 | GGUF sizing and quantization | Optional | No | No | No | None |
| 05 | Unsloth loading and GGUF export | Optional | No | No | No | Hugging Face/Unsloth |
| 06 | Split-GPU graph workflow | Optional | Yes | Knowledge graph | No | Graphistry optional |
| 07 | Entity-relationship graph | No | No | Knowledge graph | No | Graphistry optional |
| 08 | Document similarity network | No | No | Similarity graph | No | Graphistry optional |
| 09 | Large-model capacity planning | Optional | Recommended | No | No | None |
| 10 | Complete Kaggle pipeline | Yes | Recommended | No | OTLP + server metrics | Grafana/OTLP optional |
| 11 | GGUF and embedding graph | GGUF input | No | kNN graph | No | Graphistry optional |
| 12 | Attention graph exploration | No | No | Attention graph | No | Graphistry optional |
| 13 | Token embedding neighborhoods | Optional | No | kNN graph | No | Graphistry optional |
| 14 | Instrumented LLM requests | Yes | No | No | OTLP traces | Grafana/OTLP optional |
| 15 | Runtime performance monitoring | Yes | No | No | Monitor + `/metrics` | None |
| 16 | Production observability loop | Yes | Recommended | No | OTLP + health | Grafana/OTLP optional |
| 17 | Experiment tracking | Yes | Recommended | No | SDK telemetry | Weights & Biases |
| 18 | Trace-to-graph transformation | No | No | Trace graph | OTel span records | Graphistry optional |

## Capability Map

| Capability | Best starting notebook | Follow-up notebooks |
|---|---|---|
| First successful inference | 01 | 02, 03 |
| Direct `llama-server` control | 02 | 10, 16 |
| Dual-T4 configuration | 03 | 06, 09 |
| Quantization selection | 04 | 09, 11 |
| Fine-tuned model export | 05 | 01, 10 |
| Graphistry integration | 06 | 07, 08, 11-13, 18 |
| End-to-end Kaggle deployment | 10 | 16, 17 |
| OpenTelemetry tracing | 14 | 16, 18 |
| Performance analysis | 15 | 16, 17 |
| Production monitoring | 16 | 18 |
| Experiment tracking | 17 | 15, 16 |
| Trace graph visualization | 18 | 14, 16 |

## Execution Profile

| Profile | Notebooks | Notes |
|---|---|---|
| Lightweight/data-only | 07, 08, 12, 18 | Primarily constructs or transforms local tabular/graph data |
| Model inspection | 04, 11, 13 | Requires a GGUF file or compatible embeddings for real model analysis |
| Live inference | 01, 02, 03, 09, 10, 14-17 | Requires the llama.cpp runtime and a compatible GGUF model |
| Dual-GPU focused | 03, 06 | Designed around Kaggle's two-T4 accelerator option |
| Credential-dependent | 10, 14, 16, 17 | External export or tracking requires Kaggle Secrets |

See [`README.md`](README.md) for setup and learning paths, and
[`SUMMARY.md`](SUMMARY.md) for detailed notebook descriptions.
