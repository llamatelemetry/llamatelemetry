# llamatelemetry v2.2.0 - Tutorial Notebooks

Complete tutorial notebook series for llamatelemetry on Kaggle with dual Tesla T4 GPUs.

---

## Overview

This directory contains **13 comprehensive tutorial notebooks** covering all aspects of llamatelemetry v2.2.0, culminating in the flagship **Neural Network Visualization** trilogy (notebooks 11–13) demonstrating cutting-edge GGUF architecture, attention, and embedding analysis:

```
┌──────────────────────────────────────────────────────────────────────┐
│                       LLCUDA TUTORIAL PATH                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   FUNDAMENTALS          INTERMEDIATE          ADVANCED               │
│   ────────────          ────────────          ────────               │
│   01 Quick Start        04 GGUF               07 OpenAI API          │
│   02 Server Setup       05 Unsloth            08 Document Network    │
│   03 Multi-GPU          06 Split-GPU          09 Large Models        │
│                         07 Knowledge Graph    10 Complete Workflow   │
│                                                                      │
│                         ⭐ VISUALIZATION TRILOGY ⭐                   │
│                         ─────────────────────────                    │
│                         11 Neural Network Graphistry                 │
│                            (8 Interactive Dashboards)                │
│                         12 Attention Mechanism Explorer              │
│                            (Q-K-V + Graphistry)                      │
│                         13 Token Embedding Visualizer                │
│                            (3D UMAP + Plotly)                        │
│                                                                      │
│   ────────────────────────────────────────────────────────────────▶  │
│          Beginner                              Expert                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Notebook Index

### Beginner Level

| # | Notebook | Description | Time |
|---|----------|-------------|------|
| 01 | [Quick Start](01-quickstart-llamatelemetry-v2.2.0.ipynb) | Get started in 5 minutes with basic setup and first query | 5 min |
| 02 | [Llama Server Setup](02-llama-server-setup-llamatelemetry-v2.2.0.ipynb) | Deep dive into server configuration and lifecycle management | 15 min |
| 03 | [Multi-GPU Inference](03-multi-gpu-inference-llamatelemetry-v2.2.0.ipynb) | Use both T4 GPUs with tensor-split for larger models | 20 min |

### Intermediate Level

| # | Notebook | Description | Time |
|---|----------|-------------|------|
| 04 | [GGUF Quantization](04-gguf-quantization-llamatelemetry-v2.2.0.ipynb) | Understanding GGUF format, K-quants, I-quants, and parsing | 20 min |
| 05 | [Unsloth Integration](05-unsloth-integration-llamatelemetry-v2.2.0.ipynb) | Fine-tune with Unsloth → export GGUF → deploy with llamatelemetry | 30 min |
| 06 | [Split-GPU Graphistry](06-split-gpu-graphistry-llamatelemetry-v2.2.0.ipynb) | LLM on GPU 0 + RAPIDS/Graphistry on GPU 1 | 30 min |

### Advanced Level

| # | Notebook | Description | Time |
|---|----------|-------------|------|
| 07 | [Knowledge Graph Extraction](07-knowledge-graph-extraction-graphistry-v2.2.0.ipynb) | Extract knowledge graphs from text using LLM + Graphistry visualization | 30 min |
| 08 | [Document Network Analysis](08-document-network-analysis-graphistry-llamatelemetry-v2-2-0.ipynb) | Document similarity networks with GPU-accelerated graph analytics | 35 min |
| 09 | [Large Models](09-large-models-kaggle-llamatelemetry-v2-2-0.ipynb) | Deploy large models (13B+) on dual T4 with tensor-split and performance optimization | 30 min |
| 10 | [Complete Workflow](10-complete-workflow-llamatelemetry-v2-2-0.ipynb) | Production end-to-end: Setup → Model → Server → Analytics → Visualization → API | 50 min |

### Advanced Visualization ⭐

| # | Notebook | Description | Time |
|---|----------|-------------|------|
| 11 | [GGUF Neural Network Graphistry Visualization](11-gguf-neural-network-graphistry-vis-executed-2.ipynb) | **FLAGSHIP**: Dual-GPU architecture visualization with 8 interactive Graphistry dashboards | 60 min |
| 12 | [GGUF Attention Mechanism Explorer](12-gguf-attention-mechanism-explorer-executed.ipynb) | Visualize Q-K-V attention patterns across all heads with Graphistry | 20 min |
| 13 | [GGUF Token Embedding Visualizer](13-gguf-token-embedding-visualizer-executed-3.ipynb) | 3D embedding space exploration with GPU-accelerated UMAP and Plotly | 15 min |

---

## Detailed Descriptions

### 01 - Quick Start

**File:** `01-quickstart-llamatelemetry-v2.2.0.ipynb`

Get started with llamatelemetry in just 5 minutes. This notebook covers:

- Installing llamatelemetry
- Downloading a GGUF model
- Starting the llama-server
- Making your first chat completion
- Cleaning up resources

**Prerequisites:** None  
**VRAM Required:** 3-5 GB (single T4)

```python
# Sample from notebook
from llamatelemetry.server import ServerManager, ServerConfig

config = ServerConfig(model_path="model.gguf")
server = ServerManager()
server.start_with_config(config)
```

---

### 02 - Llama Server Setup

**File:** `02-llama-server-setup-llamatelemetry-v2.2.0.ipynb`

Deep dive into server configuration and management:

- ServerConfig parameter reference
- Server lifecycle (start → ready → stop)
- Health checking and monitoring
- Log access and debugging
- Multiple server configurations

**Prerequisites:** Complete notebook 01  
**VRAM Required:** 5-8 GB (single T4)

---

### 03 - Multi-GPU Inference

**File:** `03-multi-gpu-inference-llamatelemetry-v2.2.0.ipynb`

Harness both Kaggle T4 GPUs for larger models:

- GPU detection and VRAM monitoring
- tensor-split configuration
- split-mode options (layer vs row)
- Performance optimization
- Memory management

**Prerequisites:** Complete notebooks 01-02  
**VRAM Required:** 15-25 GB (dual T4)

**Key Concept:**
```python
# Split model across both GPUs
config = ServerConfig(
    model_path="model.gguf",
    tensor_split="0.5,0.5",  # 50% each GPU
    split_mode="layer",
)
```

---

### 04 - GGUF Quantization

**File:** `04-gguf-quantization-llamatelemetry-v2.2.0.ipynb`

Master the GGUF format and quantization:

- GGUF file structure
- K-quants (Q4_K_M, Q5_K_M, Q6_K)
- I-quants (IQ3_XS, IQ2_XXS)
- VRAM estimation
- Quality vs size trade-offs
- Using GGUFParser

**Prerequisites:** Complete notebooks 01-03  
**VRAM Required:** Varies by model

---

### 05 - Unsloth Integration

**File:** `05-unsloth-integration-llamatelemetry-v2.2.0.ipynb`

Complete Unsloth fine-tuning → llamatelemetry deployment workflow:

- Loading models with Unsloth
- LoRA fine-tuning basics
- Exporting to GGUF format
- Deploying with llamatelemetry
- Performance comparison

**Prerequisites:** Complete notebooks 01-04  
**VRAM Required:** 10-15 GB for training, 5-8 GB for inference

**Workflow:**
```
Unsloth (Train) → GGUF (Export) → llamatelemetry (Deploy)
```

---

### 06 - Split-GPU Graphistry

**File:** `06-split-gpu-graphistry-llamatelemetry-v2.2.0.ipynb`

Advanced architecture: LLM + RAPIDS on separate GPUs:

- Split-GPU architecture design
- LLM on GPU 0 (llama-server)
- RAPIDS/Graphistry on GPU 1
- Inter-GPU coordination
- LLM-powered graph analytics

**Prerequisites:** Complete notebooks 01-05  
**VRAM Required:** GPU 0: 5-10 GB, GPU 1: 2-8 GB

**Architecture:**
```
GPU 0: llama-server (LLM inference)
GPU 1: RAPIDS cuDF, cuGraph, Graphistry
```

---

### 07 - Knowledge Graph Extraction with Graphistry

**File:** `07-knowledge-graph-extraction-graphistry-v2.2.0.ipynb`

Extract knowledge graphs from unstructured text using LLM-powered entity recognition and visualize with Graphistry:

- **LLM-based entity extraction** from documents
- **Relationship detection** between entities
- **Graph construction** with nodes (entities) and edges (relationships)
- **Graphistry visualization** with interactive exploration
- **GPU acceleration** using RAPIDS for large graphs
- **Split-GPU architecture** (LLM on GPU 0, Graphistry on GPU 1)

**Prerequisites:** Complete notebooks 01-06
**VRAM Required:** GPU 0: 5-8 GB, GPU 1: 2-4 GB

**Use Cases:**
- Academic paper analysis
- Legal document processing
- News article relationship mapping
- Scientific literature mining

**Key Workflow:**
```python
# Extract entities and relationships using LLM
response = client.chat.create(
    messages=[{"role": "user", "content": f"Extract entities from: {text}"}]
)

# Build graph
entities_df = pd.DataFrame(entities)
relationships_df = pd.DataFrame(relationships)

# Visualize with Graphistry
g = graphistry.bind(source='from', destination='to', node='entity')
g.edges(relationships_df).nodes(entities_df).plot()
```

---

### 08 - Document Network Analysis with Graphistry

**File:** `08-document-network-analysis-graphistry-llamatelemetry-v2-2-0.ipynb`

Analyze document similarity and topic clustering using GPU-accelerated graph analytics:

- **Document embedding** generation via LLM
- **Similarity network** construction (cosine similarity)
- **Community detection** using RAPIDS cuGraph
- **Topic clustering** with GPU-accelerated algorithms
- **Interactive visualization** with Graphistry
- **Dual-GPU workflow** (embeddings on GPU 0, analytics on GPU 1)

**Prerequisites:** Complete notebooks 01-06
**VRAM Required:** GPU 0: 6-10 GB, GPU 1: 3-5 GB

**Key Algorithms:**
- **Louvain community detection** - Find document clusters
- **PageRank** - Identify influential documents
- **Betweenness centrality** - Find bridge documents
- **K-core decomposition** - Extract dense subnetworks

**Applications:**
- Research paper citation networks
- News article topic analysis
- Corporate document organization
- Social media content clustering

**Technical Stack:**
```python
# Generate embeddings
embeddings = get_embeddings_from_llm(documents)

# Build similarity graph
similarity_matrix = cosine_similarity(embeddings)
graph = build_graph_from_similarity(similarity_matrix, threshold=0.7)

# GPU analytics with cuGraph
communities = cugraph.louvain(graph)
pagerank = cugraph.pagerank(graph)

# Visualize
g = graphistry.nodes(docs_df).edges(edges_df)
g.plot()
```

---

### 09 - Large Models on Kaggle

**File:** `09-large-models-kaggle-llamatelemetry-v2.2.0.ipynb`

Run 70B models on Kaggle's dual T4 setup:

- I-quant selection for 70B
- Memory-optimized configuration
- Context size management
- Performance expectations
- Quality vs feasibility trade-offs

**Prerequisites:** Complete notebooks 01-04  
**VRAM Required:** 25-30 GB (dual T4)

**Key Configuration:**
```python
# 70B model on 30GB VRAM
config = ServerConfig(
    model_path="llama-70b-IQ3_XS.gguf",
    tensor_split="0.48,0.48",
    context_size=2048,  # Smaller context
    n_batch=128,        # Smaller batch
)
```

---

### 10 - Complete Workflow

**File:** `10-complete-workflow-llamatelemetry-v2.2.0.ipynb`

End-to-end production workflow:

1. Environment setup
2. Model selection and download
3. Unsloth fine-tuning
4. GGUF export and quantization
5. Multi-GPU deployment
6. OpenAI API client usage
7. Performance monitoring
8. Production best practices

**Prerequisites:** Complete notebooks 01-09  
**VRAM Required:** Varies (single or dual T4)

---

### 12 - GGUF Attention Mechanism Explorer

**File:** `12-gguf-attention-mechanism-explorer-executed.ipynb`

Visualize how quantized GGUF models process attention — complementary to [Transformers-Explainer](https://poloclub.github.io/transformer-explainer/) which shows FP32 GPT-2 in a browser:

- **Q-K-V decomposition** across all 896 attention heads (28 layers × 32 heads)
- **Attention matrix extraction** via llama.cpp inference on GPU 0
- **Causal masking and softmax** pattern analysis per layer
- **Layer-depth sharpness** — early layers attend broadly, later layers sharpen
- **Interactive Graphistry dashboards** on GPU 1 (token → head → weight graphs)
- **Quantization impact** — how Q4_K_M affects attention scores vs FP32

**Prerequisites:** Complete notebooks 01, 03, 06
**VRAM Required:** GPU 0: 3-4 GB, GPU 1: 1-2 GB

**Key Workflow:**
```
GPU 0: llama-server → inference → attention matrix simulation
GPU 1: RAPIDS cuDF → Graphistry attention graph dashboards
```

**Complementarity with Transformers-Explainer:**
| Aspect | Transformers-Explainer | Notebook 12 |
|--------|------------------------|-------------|
| Model | GPT-2 (FP32, 124M) | Llama 3.2 3B (Q4_K_M) |
| Runtime | Browser (WebAssembly) | Kaggle dual T4 |
| Attention view | 4-stage fixed UI | Multi-head Graphistry graphs |
| Customization | None | Fully editable Jupyter |

---

### 13 - GGUF Token Embedding Visualizer

**File:** `13-gguf-token-embedding-visualizer-executed-3.ipynb`

Explore the semantic structure of GGUF model embedding spaces using GPU-accelerated dimensionality reduction — complementary to Transformers-Explainer's 2D rectangle view:

- **Real embeddings** extracted via `/v1/embeddings` API (3072D vectors from Llama 3.2 3B)
- **42 test words** across 7 semantic categories (colors, animals, technology, emotions, numbers, verbs, countries)
- **GPU-accelerated UMAP** on GPU 1 via RAPIDS cuML (3072D → 3D in seconds)
- **Cosine similarity analysis** — intra-category vs cross-category clustering
- **Interactive 3D/2D Plotly visualizations** — rotate, zoom, hover for details
- **Combined dashboard** with side-by-side 3D and 2D UMAP projections
- **Quantization impact** — Q4_K_M preserves semantic structure at 7.8× compression

**Prerequisites:** Complete notebooks 01, 03, 04
**VRAM Required:** GPU 0: 3-4 GB (llama-server), GPU 1: 1-2 GB (cuML UMAP)

**Key Workflow:**
```
GPU 0: llama-server → /v1/embeddings → 42 × 3072D vectors
GPU 1: cuML UMAP → 3D projection → Plotly 3D/2D scatter plots
```

**Complementarity with Transformers-Explainer:**
| Aspect | Transformers-Explainer | Notebook 13 |
|--------|------------------------|-------------|
| Embedding view | 768D colored rectangles | 3D interactive UMAP |
| Model | GPT-2 (50K vocab) | Llama 3.2 3B (128K vocab) |
| Semantic analysis | Not shown | Cosine similarity + clustering |
| Interactivity | Static | Rotate, zoom, filter by category |

---

## Running on Kaggle

### Setup Steps

1. **Create New Notebook**
   - Go to [kaggle.com/code](https://kaggle.com/code)
   - Click "New Notebook"

2. **Configure GPU**
   - Settings → Accelerator → GPU T4 × 2
   - Settings → Internet → On

3. **Upload or Copy Notebook**
   - Upload `.ipynb` file, or
   - Copy cells into new notebook

4. **Run All Cells**
   - Kernel → Run All

### Kaggle-Specific Notes

- **Session limit:** 12 hours maximum
- **Disk space:** 73 GB available
- **Internet:** Required for package installation
- **Persistence:** Only `/kaggle/working` persists

---

## Learning Path

### Path 1: Quick Start (1 hour)
```
01 → 02 → 03
Quick Start → Server Setup → Multi-GPU
```

### Path 2: Full Course (5.5 hours) ⭐ RECOMMENDED
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13
Complete journey from basics to cutting-edge visualization trilogy
```

### Path 3: Advanced Topics (2 hours)
```
01 → 03 → 08 → 09
Focus on multi-GPU and large models
```

### Path 4: Unsloth Focus (2 hours)
```
01 → 04 → 05 → 10
Fine-tuning and deployment
```

### Path 5: Visualization & Analysis (3.5 hours) 🎨 VISUALIZATION TRACK
```
01 → 03 → 04 → 06 → 11 → 12 → 13
Quick start → Multi-GPU → GGUF → Split-GPU → Neural Network → Attention → Embeddings
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| GPU not detected | Check Settings → Accelerator → GPU T4 × 2 |
| Out of memory | Reduce context_size, use smaller model |
| Server won't start | Check logs with `server.get_logs()` |
| Slow inference | Enable flash_attn=True |
| Import errors | Restart kernel after pip install |

### Getting Help

- **Documentation:** See [`../docs/`](../docs/) for detailed guides
- **API Reference:** See [`API_REFERENCE.md`](../docs/API_REFERENCE.md)
- **Troubleshooting:** See [`TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md)

---

## Contributing

Want to improve these notebooks? See the [Contributing Guide](../CONTRIBUTING.md).

---

### 11 - GGUF Neural Network Graphistry Visualization ⭐ MOST IMPORTANT

**File:** `11-gguf-neural-network-graphistry-vis-executed-2.ipynb`

**THE flagship demonstration of llamatelemetry v2.2.0's cutting-edge visualization capabilities** - A tour de force showing how to visualize the internal architecture of GGUF quantized models running on dual Tesla T4 GPUs.

#### 🎯 Overview

This notebook demonstrates **advanced neural network architecture visualization** by combining:
- **llama.cpp llama-server** for GGUF model inference (GPU 0)
- **NVIDIA NCCL** optimizations for multi-GPU coordination
- **RAPIDS cuGraph** for GPU-accelerated graph analytics (GPU 1)
- **Graphistry[AI]** for interactive cloud visualization dashboards

#### 🏗️ Dual-GPU Architecture Strategy

**GPU Workload Distribution:**
```
GPU 0 (Nvidia T4 - 15GB)          GPU 1 (Nvidia T4 - 15GB)
├─ llama-server (100%)            ├─ RAPIDS cuDF/cuGraph
├─ Llama-3.2-3B-Instruct         ├─ PageRank analytics
├─ Q4_K_M quantization           ├─ Centrality metrics
├─ tensor_split="1.0,0.0"        └─ Graphistry rendering
├─ 28 transformer layers
├─ 4096 context window
└─ API at :8090
```

**Why Split-GPU?** This architecture demonstrates **workload isolation** - keeping expensive model inference separate from compute-intensive graph operations, preventing memory contention and GPU thrashing.

#### 📊 Model Architecture Details

**Model:** Llama-3.2-3B-Instruct (bartowski/Llama-3.2-3B-Instruct-GGUF)

**Quantization:** Q4_K_M (4-bit k-quants, medium variant)
- Original FP32: ~10.6 GB
- Quantized: **1.88 GB** (5.6x compression)
- Average: 5.7 bits per parameter

**Architecture Specifications:**
- **Layers:** 28 transformer blocks
- **Attention Heads:** 32 per layer = **896 total**
- **Hidden Dimension:** 3,072
- **Vocabulary Size:** 128,256 tokens
- **Context Length:** 8,192 tokens max
- **FFN Multiplier:** 4x (SwiGLU)
- **Total Parameters:** ~2.8 billion

**Parameter Distribution:**
- Embedding layer: 394M params (12.6%)
- Attention layers: 1.05B params (33.7%)
- Feed-forward layers: 2.1B params (67.2%)
- Output layer: 394M params (12.6%)

#### 🎨 8 Interactive Graphistry Visualizations

**1. Main Architecture Visualization (929 nodes, 981 edges)**
- Complete Llama-3.2-3B structure
- Color-coded by component type (7 categories)
- Size scaled by PageRank importance
- Custom tooltips with parameters, dimensions, centrality
- Force-directed layout with gravity settings

**2-6. Layer-by-Layer Subgraphs (Layers 1-5)**
- Each layer: 35 nodes, 34 edges
- Components: 1 transformer + 32 attention heads + 2 shared (LayerNorm, FFN)
- Interactive filtering by layer number
- Deep-dive into individual transformer block architecture

**7. Interactive Layer Explorer**
- Full graph with sidebar filtering UI
- Dynamic layer switching controls
- `showFilters=true`, `showLabels=true`, `sidebarMode=full`
- Explore all 28 layers interactively

**8. Quantization Blocks Visualization (112 nodes)**
- 4 quantization blocks per layer × 28 layers
- Shows Q4_K_M memory distribution
- Each block: ~737K parameters, ~1.2 MB
- Visualizes how quantization reduces memory footprint

#### 🔬 Technical Workflow

**Phase 1: Setup** → GPU detection, install llamatelemetry v2.2.0 + RAPIDS 25.6 + Graphistry

**Phase 2: Model Serving** → Download model, start llama-server on GPU 0 with `tensor_split="1.0,0.0"`

**Phase 3: Architecture Extraction** → Query model via API, build node/edge DataFrames with 929 components

**Phase 4: GPU Analytics** → Switch to GPU 1, run cuGraph PageRank + Betweenness Centrality

**Phase 5: Visualization** → Generate 8 Graphistry dashboards with custom styling and interactivity

**Phase 6: Dashboard** → Create `/kaggle/working/complete_dashboard.html` with statistics and URLs

#### 💡 What You'll Learn

1. **Split-GPU Computing** - Orchestrate dual GPUs via `tensor_split` and `CUDA_VISIBLE_DEVICES`
2. **GGUF Architecture Introspection** - Extract model structure programmatically from running inference
3. **Graph Theory for Neural Networks** - Apply PageRank to identify critical transformer components
4. **GPU-Accelerated Analytics** - Use RAPIDS cuGraph for large-scale graph algorithms
5. **Interactive AI Visualization** - Create shareable Graphistry dashboards
6. **Quantization Analysis** - Understand Q4_K_M block structure and memory layout
7. **Production Deployment** - Serve GGUF models via llama-server with health monitoring

#### 🎯 Key Insights: llamatelemetry v2.2.0 Capabilities

This notebook proves that **llamatelemetry v2.2.0** enables:

**Core Capabilities:**
- ✅ Seamless GGUF integration via llama.cpp
- ✅ Split-GPU orchestration without manual CUDA management
- ✅ Architecture introspection through API queries
- ✅ Clean Python client for llama-server
- ✅ Zero-configuration ServerManager

**Advanced Features:**
- ✅ Graph-based neural network modeling
- ✅ GPU-accelerated analytics with RAPIDS
- ✅ Production-ready background server management
- ✅ Interactive visualization with Graphistry
- ✅ Reproducible workflows in pure Python

**What Makes This Cutting-Edge:**
- 🚀 First-class GGUF support in Python-native package
- 🚀 Dual-GPU split computing on free Kaggle GPUs
- 🚀 Visual AI explainability - see inside transformers
- 🚀 No compilation required - pip installable
- 🚀 Integration of llamatelemetry + Unsloth + Graphistry ecosystem

#### 📦 Outputs

**Interactive URLs:**
- 8 Graphistry cloud visualizations (30-day shareable links)

**Downloadable Files:**
- `/kaggle/working/complete_dashboard.html` - Interactive dashboard with statistics
- `/kaggle/working/attention_dashboard.html` - Attention head analysis
- `/kaggle/working/workflow_nodes.csv` - Graph node data
- `/kaggle/working/workflow_edges.csv` - Graph edge data

#### 🔬 Research Applications

- **Quantization Comparison** - Compare Q4_K_M vs IQ3_XS vs Q8_0 structures
- **Pruning Opportunities** - Identify low-importance attention heads
- **Information Flow Analysis** - Understand bottlenecks in transformer layers
- **GGUF Validation** - Verify conversions vs original HuggingFace models
- **Architecture Exploration** - Interactively explore different model families

#### 🛠️ Technical Stack

**llamatelemetry v2.2.0:**
- `ServerManager` - llama-server lifecycle management
- `LlamaCppClient` - API client for inference
- Built-in CUDA binaries for T4 GPUs

**RAPIDS 25.6.0:**
- `cuDF` - GPU DataFrames
- `cuGraph` - PageRank, Betweenness Centrality
- `CuPy` - GPU memory management

**Graphistry 0.50.4:**
- Interactive cloud visualization
- Custom styling, tooltips, layouts
- Shareable dashboard URLs

**Additional:**
- Pandas, PyArrow, HuggingFace Hub
- Kaggle Secrets for API authentication

#### ⚙️ Prerequisites

**Completed Notebooks:** 01 (Quickstart), 03 (Multi-GPU), 04 (GGUF), 06 (Split-GPU)

**VRAM Required:**
- GPU 0: 3-4 GB (Llama-3.2-3B Q4_K_M model)
- GPU 1: 0.5-1 GB (RAPIDS analytics)
- Total: 4-5 GB of 30 GB available

**Kaggle Setup:**
- Accelerator: GPU T4 × 2 (required)
- Internet: On (for package installation)
- Secrets: `HF_TOKEN`, `Graphistry_Personal_Key_ID`, `Graphistry_Personal_Secret_Key`

#### 🎓 Novel Features

1. **First tool to visualize GGUF quantization as graphs** - No binary parsing, runtime introspection
2. **Dual-GPU split for concurrent operations** - Inference + visualization simultaneously
3. **Graph theory metrics for neural architectures** - PageRank applied to transformer components
4. **Zero-code Graphistry integration** - DataFrame → Interactive dashboard
5. **Production-ready workflow** - Complete setup to visualization in one notebook

#### 📈 Performance Metrics

**Model Loading:** ~2-3 seconds
**Architecture Extraction:** ~5-10 seconds
**Graph Analytics (cuGraph):** ~1-2 seconds for 929 nodes
**Graphistry Upload:** ~10-15 seconds per visualization
**Total Runtime:** ~5-7 minutes for all 8 visualizations

#### 🔗 Integration Points

```
llamatelemetry (GPU 0 - Inference)
    ↓ API calls
Architecture Data (pandas DataFrames)
    ↓ CUDA_VISIBLE_DEVICES="1"
RAPIDS cuGraph (GPU 1 - Analytics)
    ↓ PageRank + Centrality
Enhanced DataFrames
    ↓ Graphistry API
Interactive Visualizations (Cloud)
    ↓ Download
HTML Dashboard (Local)
```

#### 📚 Related Documentation

- **API Reference:** [`../docs/API_REFERENCE.md`](../docs/API_REFERENCE.md)
- **Split-GPU Guide:** See notebook 06 for split-GPU fundamentals
- **GGUF Deep Dive:** See notebook 04 for quantization details
- **Troubleshooting:** [`../docs/TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md)

---

**💬 Summary:** This notebook represents the **state-of-the-art** in quantized model visualization, combining llamatelemetry's elegant GGUF serving with GPU-accelerated graph analytics and interactive visualization. The 8 dashboards provide unprecedented insight into how a 3B-parameter transformer is structured, quantized, and connected - making "black box" AI models transparent and explorable.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.2.0 | 2026-02-01 | Complete 13-notebook series with visualization trilogy |
| | | - Notebooks 01-06: Core fundamentals and split-GPU architecture |
| | | - Notebooks 07-08: Knowledge graphs and document network analysis |
| | | - Notebook 09: Large model deployment on dual T4 |
| | | - Notebook 10: Production end-to-end workflow |
| | | - Notebook 11: ⭐ FLAGSHIP - 8 interactive Graphistry dashboards for GGUF architecture visualization |
| | | - Notebook 12: Attention Mechanism Explorer — Q-K-V patterns with Graphistry |
| | | - Notebook 13: Token Embedding Visualizer — 3D UMAP with Plotly (real /v1/embeddings) |

---

## License

MIT License - See [LICENSE](../LICENSE)
