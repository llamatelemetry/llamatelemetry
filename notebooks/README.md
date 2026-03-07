# LlamaTelemetry Kaggle Notebooks

This folder contains the Kaggle notebook set for **llamatelemetry v0.1.1**.
Use the notebooks in order for a smooth learning path.

## Notebook Index

| # | Notebook | Focus |
|---|----------|-------|
| 01 | `01-quickstart-llamatelemetry-v0-1-1-e1.ipynb` | Quickstart: install, GPU check, load model, run inference |
| 02 | `02-llama-server-setup-llamatelemetry-v0-1-1-e1.ipynb` | ServerManager setup + /health /props /slots /metrics |
| 03 | `03-multi-gpu-inference-llamatelemetry-v0-1-1-e1.ipynb` | Multi-GPU config (SplitMode, NCCLConfig, split_gpu_session) |
| 04 | `04-gguf-quantization-llamatelemetry-v0-1-1-e1.ipynb` | GGUF reports, suitability checks, quantization matrix |
| 05 | `05-unsloth-integration-llamatelemetry-v0-1-1-e1.ipynb` | Unsloth fine-tune + GGUF export workflow |
| 06 | `06-split-gpu-graphistry-llamatelemetry-v0-1-1-e1.ipynb` | Split GPU session + Graphistry knowledge graph basics |
| 07 | `07-knowledge-graph-extraction-graphistry-v0-1-1-e1.ipynb` | Knowledge graph builder with entities and relationships |
| 08 | `08-document-network-analysis-graphistry-e1.ipynb` | Document similarity network graph |
| 09 | `09-large-models-kaggle-llamatelemetry-e3.ipynb` | Large model suitability + server presets |
| 10 | `10-complete-workflow-llamatelemetry-v0-1-1-e1.ipynb` | End-to-end Kaggle pipeline (OTLP + preset + client) |
| 11 | `11-gguf-neural-network-graphistry-vis-executed-e1.ipynb` | GGUF metadata + embedding kNN graph visualization |
| 12 | `12-gguf-attention-mechanism-explorer-executed-e1.ipynb` | Attention weight matrix graph exploration |
| 13 | `13-gguf-token-embedding-visualizer-executed-e1.ipynb` | Token embedding kNN visualization with clusters |
| 14 | `14-opentelemetry-llm-observability-e5.ipynb` | OTLP setup + InstrumentedLlamaCppClient |
| 15 | `15-rt-performance-monitoring-llamatelemetry-e3.ipynb` | PerformanceMonitor + /metrics parsing + DataFrame export |
| 16 | `16-production-observability-llamatelemetry-e2.ipynb` | Production observability pipeline setup |
| 17 | `17-llamatelemetry-wandb-kaggle-notebook-e2.ipynb` | W&B logging + telemetry pipeline on Kaggle |
| 18 | `18-otel-graphistry-trace-glue-e2.ipynb` | Trace-to-graph helpers (spans to DataFrames to graphs) |

## Learning Path

- **Getting started:** Notebooks 01-02
- **Multi-GPU and models:** Notebooks 03-05, 09
- **Graph visualization:** Notebooks 06-08, 11-13
- **Observability:** Notebooks 10, 14-16, 18
- **Integrations:** Notebooks 05 (Unsloth), 17 (W&B)
