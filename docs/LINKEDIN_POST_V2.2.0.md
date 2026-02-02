# LinkedIn Blog Post — llamatelemetry v2.2.0

---

## Post Text

---

From GeForce 940M to Kaggle dual Tesla T4 — llamatelemetry v2.2.0 is here.

Last time I wrote about llamatelemetry, it was a proof of concept: local CUDA 12 inference on a GeForce 940M, VRAM down from 820 MB to 200 MB, 42 tok/s on a consumer GPU. The core hypothesis was right — GGUF quantization + llama.cpp makes serious LLM inference accessible without cloud spend.

v2.2.0 flips the script. Same philosophy, different scale.

**What changed:**

The target moved from a single consumer GPU to Kaggle's dual Tesla T4 environment (15 GB × 2, SM 7.5). The architecture is now split-GPU by design: GPU 0 runs llama-server for inference, GPU 1 runs RAPIDS + visualization. A 62 KB pip-installable Python package that auto-downloads 961 MB of pre-compiled CUDA 12.5 binaries on first import. Zero build step for the end user.

13 tutorial notebooks. From a 5-minute quickstart to production-grade pipelines — and three notebooks at the end that are the real reason this release exists.

**The Visualization Trilogy (Notebooks 11–13):**

This is where it gets interesting. Three notebooks that make the internals of a quantized 3B-parameter transformer visible and explorable — something that's surprisingly hard to do in practice.

Notebook 11 — GGUF Neural Network Architecture. 929 nodes, 981 edges. Every layer, every attention block, every quantization region of Llama 3.2 3B (Q4_K_M) mapped into an interactive graph. 8 Graphistry dashboards. PageRank on transformer components. You can see which parts of the network carry the most information flow.

Notebook 12 — Attention Mechanism Explorer. 896 attention heads (28 layers × 32 heads) decomposed into Q-K-V matrices. The attention patterns are extracted via llama.cpp inference on GPU 0, then visualized as interactive graphs on GPU 1. The key finding: early layers attend broadly, later layers sharpen. Quantization (Q4_K_M) preserves this structure — the 7.8× compression doesn't destroy the attention hierarchy.

Notebook 13 — Token Embedding Visualizer. 42 words across 7 semantic categories, each producing a real 3072-dimensional embedding vector via the /v1/embeddings API. GPU-accelerated UMAP (cuML on GPU 1) projects them to 3D. The clusters are real — colors, animals, emotions, countries separate cleanly in the embedding space. This is what Llama 3.2 actually learned, visible in an interactive Plotly plot you can rotate and filter.

**Why this matters:**

Transformers-Explainer (Georgia Tech / Polo Club) does something similar for GPT-2 in a browser — and it's excellent for learning the fundamentals. But it's FP32, it's GPT-2 (124M params), and the visualizations are fixed.

Notebooks 11–13 are the production counterpart. Quantized models. Customizable Jupyter notebooks. GPU-accelerated analytics. And the visualization targets a model 24× larger than GPT-2.

**The numbers that matter:**

- Llama 3.2 3B: 14.7 GB (FP32) → 1.88 GB (Q4_K_M) — 7.8× compression
- Inference: 30–50 tok/s on a single T4, depending on model size
- Notebook 11: 929 nodes, 981 edges, 8 interactive dashboards
- Notebook 12: 896 attention heads analyzed across 28 layers
- Notebook 13: 3072D → 3D UMAP in seconds on GPU

**How to run it:**

Everything runs on Kaggle. Set accelerator to GPU T4 × 2, run one pip install, open Notebook 01. The full visualization trilogy is Notebooks 11 → 12 → 13, or follow the Visualization Track: 01 → 03 → 04 → 06 → 11 → 12 → 13 (about 3.5 hours end to end).

Open source. MIT license. 13 notebooks. All on Kaggle.

github.com/llamatelemetry/llamatelemetry

---

#llamatelemetry #CUDA #LLM #GGUF #llama_cpp #Kaggle #NVIDIA #Tesla_T4 #RAPIDS #Graphistry #Plotly #Quantization #NeuralNetwork #Visualization #Transformers #OpenSource #DeepLearning #GPU #MachineLearning #AI

---
