# llamatelemetry Kaggle Notebooks

This directory contains the official 18-notebook curriculum for
`llamatelemetry v0.1.1`. The notebooks cover GGUF inference, llama.cpp server
management, dual-GPU workflows, graph analytics, model inspection, and
OpenTelemetry observability.

The notebooks are clean source notebooks: outputs and execution counts are
removed so they can be imported into Kaggle and run from top to bottom.

## Runtime

- Kaggle Python 3.11 environment
- Internet access enabled
- GPU accelerator enabled where indicated
- Dual Tesla T4 recommended for multi-GPU and split-GPU notebooks
- Kaggle Secrets configured for optional Hugging Face, Graphistry, Grafana,
  and Weights & Biases integrations

Each notebook installs the SDK from the `v0.1.1` Git tag. The first import may
download the matching llama.cpp CUDA runtime bundle.

## Curriculum

### Foundation

| # | Notebook | Main outcome | GPU |
|---|---|---|---|
| 01 | [`01-quickstart-llamatelemetry-v0-1-1-e1.ipynb`](01-quickstart-llamatelemetry-v0-1-1-e1.ipynb) | Load a registry model, run single and batch inference, inspect engine metrics | Required |
| 02 | [`02-llama-server-setup-llamatelemetry-v0-1-1-e1.ipynb`](02-llama-server-setup-llamatelemetry-v0-1-1-e1.ipynb) | Manage `llama-server` and query `/health`, `/props`, `/slots`, and `/metrics` | Required |
| 03 | [`03-multi-gpu-inference-llamatelemetry-v0-1-1-e1.ipynb`](03-multi-gpu-inference-llamatelemetry-v0-1-1-e1.ipynb) | Configure dual-T4 tensor splitting, NCCL, and isolated GPU sessions | Dual T4 |
| 04 | [`04-gguf-quantization-llamatelemetry-v0-1-1-e1.ipynb`](04-gguf-quantization-llamatelemetry-v0-1-1-e1.ipynb) | Inspect GGUF metadata, estimate size, and select a suitable quantization | Optional |

### Integration

| # | Notebook | Main outcome | Extra dependency |
|---|---|---|---|
| 05 | [`05-unsloth-integration-llamatelemetry-v0-1-1-e1.ipynb`](05-unsloth-integration-llamatelemetry-v0-1-1-e1.ipynb) | Load an Unsloth model and configure GGUF export | Unsloth |
| 06 | [`06-split-gpu-graphistry-llamatelemetry-v0-1-1-e1.ipynb`](06-split-gpu-graphistry-llamatelemetry-v0-1-1-e1.ipynb) | Reserve one GPU for inference and one for graph workloads | Graphistry, dual T4 |
| 07 | [`07-knowledge-graph-extraction-graphistry-v0-1-1-e1.ipynb`](07-knowledge-graph-extraction-graphistry-v0-1-1-e1.ipynb) | Build entity and relationship DataFrames for an interactive knowledge graph | Graphistry optional |
| 08 | [`08-document-network-analysis-graphistry-e1.ipynb`](08-document-network-analysis-graphistry-e1.ipynb) | Construct and filter a document-similarity network | Graphistry optional |
| 09 | [`09-large-models-kaggle-llamatelemetry-e3.ipynb`](09-large-models-kaggle-llamatelemetry-e3.ipynb) | Evaluate model fit and choose Kaggle server presets | GPU recommended |
| 10 | [`10-complete-workflow-llamatelemetry-v0-1-1-e1.ipynb`](10-complete-workflow-llamatelemetry-v0-1-1-e1.ipynb) | Run the complete preset, server, OTLP, client, and metrics workflow | GPU, OTLP optional |

### Model Visualization

| # | Notebook | Main outcome | Data source |
|---|---|---|---|
| 11 | [`11-gguf-neural-network-graphistry-vis-executed-e1.ipynb`](11-gguf-neural-network-graphistry-vis-executed-e1.ipynb) | Combine GGUF metadata with an embedding nearest-neighbor graph | GGUF model |
| 12 | [`12-gguf-attention-mechanism-explorer-executed-e1.ipynb`](12-gguf-attention-mechanism-explorer-executed-e1.ipynb) | Convert an attention matrix into a thresholded graph | Simulated attention |
| 13 | [`13-gguf-token-embedding-visualizer-executed-e1.ipynb`](13-gguf-token-embedding-visualizer-executed-e1.ipynb) | Explore token embedding neighborhoods and clusters | Synthetic or llama.cpp embeddings |

### Observability

| # | Notebook | Main outcome | Integration |
|---|---|---|---|
| 14 | [`14-opentelemetry-llm-observability-e5.ipynb`](14-opentelemetry-llm-observability-e5.ipynb) | Trace completions, chat requests, and embeddings with an instrumented client | OpenTelemetry/OTLP |
| 15 | [`15-rt-performance-monitoring-llamatelemetry-e3.ipynb`](15-rt-performance-monitoring-llamatelemetry-e3.ipynb) | Record inference performance and parse live llama.cpp metrics | `PerformanceMonitor` |
| 16 | [`16-production-observability-llamatelemetry-e2.ipynb`](16-production-observability-llamatelemetry-e2.ipynb) | Assemble a production-style health, telemetry, and inference loop | OpenTelemetry/OTLP |
| 17 | [`17-llamatelemetry-wandb-kaggle-notebook-e2.ipynb`](17-llamatelemetry-wandb-kaggle-notebook-e2.ipynb) | Log inference and server health data to a W&B run | Weights & Biases |
| 18 | [`18-otel-graphistry-trace-glue-e2.ipynb`](18-otel-graphistry-trace-glue-e2.ipynb) | Convert exported spans into graph and latency DataFrames | OpenTelemetry + Graphistry |

## Recommended Paths

- Fast start: `01 -> 02 -> 03`
- Model planning: `01 -> 04 -> 09`
- Graph analytics: `01 -> 06 -> 07 -> 08 -> 11 -> 12 -> 13`
- Production observability: `01 -> 02 -> 10 -> 14 -> 15 -> 16 -> 18`
- Experiment tracking: `01 -> 10 -> 17`

## Kaggle Secrets

Only configure secrets required by the notebook being run. Common names used
by the SDK workflows include:

| Service | Secret names |
|---|---|
| Hugging Face | `HF_TOKEN` |
| Graphistry | `GRAPHISTRY_USERNAME`, `GRAPHISTRY_PASSWORD` |
| Generic OTLP | `OTLP_ENDPOINT`, `OTLP_TOKEN` |
| Grafana OTLP | `GRAFANA_OTLP_ENDPOINT`, `GRAFANA_OTLP_HEADERS`, `GRAFANA_OTLP_TOKEN` |
| Weights & Biases | `WANDB_API_KEY` |

Never place credentials directly in notebook cells or committed output.

## Usage Notes

1. Create or open a Kaggle notebook and enable the required accelerator.
2. Upload the selected `.ipynb` file or import it from GitHub.
3. Enable internet access and configure only the required Kaggle Secrets.
4. Run cells in order. Restart the runtime if a dependency installation asks
   for it.
5. Stop the server or call the notebook cleanup cell before changing models.

See [`SUMMARY.md`](SUMMARY.md) for detailed notebook descriptions and
[`VISUAL-COMPARISON.md`](VISUAL-COMPARISON.md) for a feature matrix.
