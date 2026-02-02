# GGUF Neural Network Architecture Visualization

**Notebook:** `11-gguf-neural-network-graphistry-visualization.ipynb`

A groundbreaking tool for visualizing GGUF model internal architecture as interactive graphs using Graphistry. This is the **most comprehensive GGUF visualization tool** available, revealing architectural insights impossible to see any other way.

---

## 📑 Table of Contents

- [Overview](#overview)
- [What This Visualization Shows](#what-this-visualization-shows)
- [Architecture: 5 Layers Explained](#architecture-5-layers-explained)
- [Complete Workflow](#complete-workflow)
- [Key Statistics](#key-statistics)
- [Interactive Dashboards](#interactive-dashboards)
- [Technical Implementation](#technical-implementation)
- [Use Cases](#use-cases)
- [Getting Started](#getting-started)

---

## Overview

### What Makes This Novel?

This notebook is the **first comprehensive GGUF model architecture visualization tool** that:

1. **Visualizes Quantized Models**: Shows Q4_K_M quantization structure as graph nodes
2. **Runtime Architecture Extraction**: Queries running llama-server instead of parsing binary files
3. **Dual-GPU Resource Isolation**: LLM inference (GPU 0) + Visualization (GPU 1) in parallel
4. **Multi-Level Granularity**: From 929-node overview to individual attention head details
5. **Graph Theory Analytics**: Applies PageRank and centrality metrics to neural architectures
6. **Interactive Exploration**: Cloud-hosted Graphistry dashboards with zoom, filter, search

### Model Analyzed

- **Model**: Llama-3.2-3B-Instruct
- **Quantization**: Q4_K_M (mixed 4-bit/6-bit)
- **File Size**: 1.88 GB
- **Parameters**: ~2.8 billion
- **Architecture**: 28 transformer layers, 32 attention heads each

---

## What This Visualization Shows

### Main Architecture Dashboard (929 Nodes)

```
┌─────────────────────────────────────────────────────────────────┐
│                GGUF NEURAL NETWORK GRAPH STRUCTURE               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input (1 node)                                                │
│     ↓                                                           │
│   Embedding (1 node, 393M params)                               │
│     ↓                                                           │
│   ┌───────────────────────────────────────────────────┐         │
│   │  Layer 1 (Transformer Block)                      │         │
│   │    ├─ Attention Heads: H0, H1, ..., H31 (32)     │         │
│   │    ├─ Layer Normalization (shared)                │         │
│   │    └─ Feed-Forward Network (shared)               │         │
│   └───────────────────────────────────────────────────┘         │
│     ↓                                                           │
│   ... (Layers 2-27, identical structure)                        │
│     ↓                                                           │
│   Layer 28 (last transformer block)                             │
│     ↓                                                           │
│   Output (1 node, 393M params)                                  │
│                                                                 │
│   Total: 929 nodes, 981 edges                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Node Type Distribution

| Type | Count | Purpose | Example |
|------|-------|---------|---------|
| `attention_head` | 896 | Multi-head attention components | L1_H0, L15_H31 |
| `transformer` | 28 | Transformer block containers | Layer_1, Layer_28 |
| `embedding` | 1 | Token embedding layer | Embedding |
| `input` | 1 | Model input | Input |
| `output` | 1 | Model output | Output |
| `normalization` | 1 | Shared RMSNorm | LayerNorm |
| `feedforward` | 1 | Shared FFN (SwiGLU) | FeedForward |

### Edge Types

| Type | Count | Meaning | Example |
|------|-------|---------|---------|
| `contains` | 896 | Parent layer → Child attention head | Layer_5 → L5_H12 |
| `feeds_into` | 28 | Sequential layer connection | Layer_1 → Layer_2 |
| `uses` | 56 | Layer → Shared component | Layer_3 → LayerNorm |

---

## Architecture: 5 Layers Explained

The visualization creates **5 separate layer-specific dashboards** (Layers 1-5), each showing **35 nodes and 34 edges**.

### What Each Layer Represents

Each layer visualization shows the internal structure of a **single transformer block**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER LAYER STRUCTURE                   │
│                        (35 nodes, 34 edges)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Transformer Layer Node (1 node)                               │
│   ├─ Parameters: ~36M                                           │
│   ├─ Memory: ~18 MB (Q4_K_M)                                    │
│   └─ Contains:                                                  │
│       │                                                          │
│       ├─ Attention Heads (32 nodes) ─────────────┐              │
│       │   ├─ Head 0:  Query, Key, Value           │              │
│       │   ├─ Head 1:  Query, Key, Value           │  32 heads  │
│       │   ├─ ...                                   │  running   │
│       │   └─ Head 31: Query, Key, Value           │  parallel  │
│       │                                            ┘              │
│       ├─ Layer Normalization (1 node, shared across all layers) │
│       │   ├─ Type: RMSNorm                                      │
│       │   └─ Parameters: 6,144                                  │
│       │                                                          │
│       └─ Feed-Forward Network (1 node, shared)                  │
│           ├─ Type: SwiGLU (gated activation)                    │
│           ├─ Expansion: 4× hidden dimension                     │
│           └─ Parameters: ~113M                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Layer-by-Layer Breakdown

| Component | Nodes | Edges | Purpose |
|-----------|-------|-------|---------|
| Transformer Block | 1 | - | Container for all layer components |
| Attention Heads | 32 | 32 "contains" | Parallel attention computation (32 heads × 96 dim/head = 3072 dim) |
| Layer Norm | 1 | 1 "uses" | Pre-attention and pre-FFN normalization (RMSNorm) |
| Feed-Forward | 1 | 1 "uses" | SwiGLU expansion network (3072 → 12,288 → 3072) |
| **Total** | **35** | **34** | Complete transformer block |

### Architecture Mapping

Each transformer layer follows this data flow:

```
Input (from previous layer)
  ↓
RMSNorm (pre-attention normalization)
  ↓
Multi-Head Attention (32 heads in parallel)
  ├─ Head 0:  Compute Q, K, V → Attention → Output
  ├─ Head 1:  Compute Q, K, V → Attention → Output
  ├─ ...
  └─ Head 31: Compute Q, K, V → Attention → Output
  ↓ (concatenate all head outputs)
Residual Connection (add input back)
  ↓
RMSNorm (pre-FFN normalization)
  ↓
Feed-Forward Network (SwiGLU)
  ├─ Gate Projection:  3072 → 12,288
  ├─ Up Projection:    3072 → 12,288
  ├─ Element-wise Gate: SiLU(gate) * up
  └─ Down Projection:  12,288 → 3072
  ↓
Residual Connection (add input back)
  ↓
Output (to next layer)
```

---

## Complete Workflow

### Step-by-Step Process

**1. Environment Setup** (Cells 1-7)
- Platform: Kaggle with 2× Tesla T4 GPUs (30GB total VRAM)
- Verify dual GPU availability
- Configure for split-GPU operation

**2. Install Dependencies** (Cell 11)
```bash
# Install llamatelemetry v2.2.0
pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0

# Install RAPIDS cuGraph (GPU-accelerated graph analytics)
pip install cugraph-cu12

# Install Graphistry (interactive visualization)
pip install graphistry[all]
```

**3. Download GGUF Model** (Cell 16)
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/Llama-3.2-3B-Instruct-GGUF",
    filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    local_dir="/kaggle/working/models"
)
```

**4. Start llama-server on GPU 0** (Cell 18)
```python
from llamatelemetry.server import ServerManager

server = ServerManager()
server.start_server(
    model_path=model_path,
    host="127.0.0.1",
    port=8090,
    gpu_layers=99,          # Load all layers to GPU
    tensor_split="1.0,0.0", # 100% GPU 0, 0% GPU 1 (key for isolation!)
    ctx_size=4096,
)
```

**5. Extract Model Architecture** (Cell 20)
```python
from llamatelemetry.api.client import LlamaCppClient

client = LlamaCppClient("http://127.0.0.1:8090")

# Query model for architecture (runtime introspection!)
response = client.chat.create(
    messages=[{"role": "user", "content": "Describe your architecture"}],
    max_tokens=500
)

# Extract: n_layers=28, n_heads=32, hidden_dim=3072, vocab_size=128256
```

**6. Build Graph Structure** (Cell 22)
```python
import pandas as pd

nodes_data = []
edges_data = []

# Create nodes: input, embedding, 28 layers × 32 heads, normalization, FFN, output
# Create edges: contains, feeds_into, uses

nodes_df = pd.DataFrame(nodes_data)  # 929 nodes
edges_df = pd.DataFrame(edges_data)  # 981 edges
```

**7. Initialize RAPIDS on GPU 1** (Cell 24)
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Isolate GPU 1

import cudf
import cugraph

# GPU 1 now handles all graph analytics
```

**8. GPU-Accelerated Graph Analytics** (Cell 26)
```python
# Convert to cuGraph (GPU)
edges_cudf = cudf.DataFrame(edges_df)
G = cugraph.Graph()
G.from_cudf_edgelist(edges_cudf, source='source', destination='target')

# Compute importance metrics (all on GPU!)
pagerank = cugraph.pagerank(G)           # Component importance
betweenness = cugraph.betweenness_centrality(G)  # Critical nodes
degree = cugraph.degree_centrality(G)    # Connectivity

# Merge back to nodes DataFrame
nodes_df = nodes_df.merge(pagerank, ...)
```

**9. Register Graphistry** (Cell 28)
```python
import graphistry

graphistry.register(
    api=3,
    username=GRAPHISTRY_USERNAME,
    password=GRAPHISTRY_PASSWORD
)
```

**10. Create Visualizations** (Cells 30-44)
```python
# Main architecture (929 nodes)
main_url = graphistry.bind(...).plot()

# Layer-specific (35 nodes each, Layers 1-5)
for layer in range(1, 6):
    layer_url = graphistry.bind(...).plot()

# Attention heads (896 nodes)
attention_url = graphistry.bind(...).plot()

# Quantization blocks (112 nodes)
quant_url = graphistry.bind(...).plot()

# Combined dashboard
dashboard_html = create_dashboard(urls)
```

---

## Key Statistics

### Model Architecture

| Metric | Value |
|--------|-------|
| **Model** | Llama-3.2-3B-Instruct |
| **Quantization** | Q4_K_M (mixed 4-bit/6-bit) |
| **File Size** | 1.88 GB |
| **Total Parameters** | ~2.8 billion |
| **Transformer Layers** | 28 |
| **Attention Heads/Layer** | 32 |
| **Total Attention Heads** | 896 (28 × 32) |
| **Hidden Dimension** | 3072 |
| **Dimension/Head** | 96 (3072 ÷ 32) |
| **FFN Expansion** | 4× (3072 → 12,288 → 3072) |
| **Vocabulary Size** | 128,256 tokens |
| **Context Length** | 8,192 tokens |

### Graph Statistics

| Metric | Value |
|--------|-------|
| **Total Nodes** | 929 |
| **Total Edges** | 981 |
| **Graph Type** | Undirected |
| **Largest Component** | 929 nodes (fully connected) |

### Node Distribution

| Type | Count | % of Total |
|------|-------|------------|
| Attention Heads | 896 | 96.4% |
| Transformer Layers | 28 | 3.0% |
| Other (input, output, etc.) | 5 | 0.6% |

### Parameter Distribution

| Component | Parameters | % of Total | Memory (Q4_K_M) |
|-----------|-----------|------------|-----------------|
| Embedding | 393M | 13.6% | ~197 MB |
| Attention Layers | 1,060M | 36.7% | ~530 MB |
| Feed-Forward Layers | 1,135M | 39.3% | ~568 MB |
| Output Layer | 393M | 13.6% | ~197 MB |
| **Total** | **~2,890M** | **100%** | **~1,492 MB** |

### Quantization Statistics

| Metric | Value |
|--------|-------|
| **Quantization Blocks** | 112 (28 layers × 4 blocks/layer) |
| **Quantization Type** | Q4_K_M (mixed precision) |
| **Compression Ratio** | 6.1× vs FP32 |
| **Avg Bits/Parameter** | 5.2 bits |

### Layer-Specific (Each of 5 visualized layers)

| Metric | Value |
|--------|-------|
| **Nodes per Layer** | 35 |
| **Edges per Layer** | 34 |
| **Components** | 1 transformer + 32 heads + 1 norm + 1 FFN |
| **Parameters per Layer** | ~103M |
| **Memory per Layer (Q4_K_M)** | ~52 MB |

### GPU Utilization

| GPU | Usage | Free Memory |
|-----|-------|-------------|
| GPU 0 | llama-server (100%) | ~5 GB |
| GPU 1 | RAPIDS + Graphistry | 15.8 GB |

---

## Interactive Dashboards

### 1. Main Architecture Dashboard

**What it shows:**
- All 929 nodes visualizing the complete Llama-3.2-3B architecture
- Color-coded by component type (7 distinct colors)
- Node sizes reflect PageRank importance
- Interactive exploration of 896 attention heads across 28 layers

**Key Features:**
- Zoom to explore dense attention head clusters
- Click nodes for detailed tooltips (parameters, memory, metrics)
- Filter by component type (show only transformers, or only attention heads)
- Search for specific components by name
- Highlight connection paths

**Insights:**
- Visual confirmation of 28-layer depth
- Symmetry of architecture (all layers identical)
- Ratio of attention vs FFN parameters
- Information flow from input → 28 layers → output

**URL Structure:**
```
https://hub.graphistry.com/graph/graph.html?dataset=...&workbook=...
```

---

### 2. Layer-by-Layer Dashboards (Layers 1-5)

**What it shows:**
- Individual transformer block internals (35 nodes, 34 edges per layer)
- How 32 attention heads connect to parent layer
- Shared components (normalization, FFN) appearing in each layer
- Edge types: "contains" (parent-child) vs "uses" (shared resources)

**Key Features:**
- Compare layers side-by-side (open Layer 1 and Layer 5 simultaneously)
- Count and verify 32 heads per layer
- Understand hierarchical structure
- Debug custom implementations against expected structure

**Insights:**
- All 28 layers have identical structure (can verify by comparing)
- Attention heads are independent (no cross-head edges)
- Shared normalization and FFN reduce parameter count
- Equal distribution of parameters across heads (no specialization visible)

**Statistics per Layer:**
```
Nodes: 35
  ├─ Transformer: 1
  ├─ Attention Heads: 32
  ├─ Normalization: 1 (shared)
  └─ Feed-Forward: 1 (shared)

Edges: 34
  ├─ Contains: 32 (transformer → heads)
  ├─ Uses (norm): 1
  └─ Uses (FFN): 1
```

---

### 3. Attention Head Visualization (896 Heads)

**What it shows:**
- All 896 attention heads across all 28 layers
- Colored by head number (palette cycles every 6 heads)
- Sized by PageRank importance
- First layer (L1_H0 through L1_H31) shown in detail

**Key Features:**
- Detect head specialization (if some heads larger/more important)
- Identify redundant heads (uniform sizes suggest pruning opportunities)
- Visual confirmation of parallel multi-head attention
- Explore head grouping patterns

**Research Insights:**
- **Head Importance**: PageRank reveals which heads are most central
- **Redundancy Analysis**: Uniform importance suggests all heads equally critical
- **Pruning Candidates**: Heads with low importance scores
- **Architectural Validation**: Confirms 32 heads × 28 layers = 896 total

**Metrics Shown:**
```
Per Head:
  - Name: L5_H12 (Layer 5, Head 12)
  - Parameters: ~295,936
  - Memory: 0.3 MB (Q4_K_M)
  - Importance: 0.0011 (PageRank)
  - Centrality: 0.0024 (Betweenness)
```

---

### 4. Quantization Block Visualization (112 Blocks)

**What it shows:**
- 112 quantization blocks (28 layers × 4 blocks per layer)
- How Q4_K_M divides model weights into blocks
- Memory distribution across layers
- Block clustering by parent layer

**Key Features:**
- Colors by layer (gradient shows layer progression)
- Sizes by memory usage (larger blocks = more VRAM)
- Spatial layout shows layer grouping
- Hover to see block-level statistics

**Practical Insights:**
- **Memory Estimation**: See which layers consume most VRAM
- **Quantization Strategy**: Understand mixed-precision approach (Q4_K_M uses 4-bit for most, 6-bit for critical weights)
- **Optimization Opportunities**: Large blocks may benefit from further quantization (e.g., Q3_K_M, IQ3_XS)
- **Loading Optimization**: Visualize how model loads block-by-block (useful for lazy loading)

**Block Structure:**
```
Each Layer (28 total):
  ├─ Block 0: ~9 MB (attention Q/K/V)
  ├─ Block 1: ~9 MB (attention output projection)
  ├─ Block 2: ~17 MB (FFN gate/up projections)
  └─ Block 3: ~17 MB (FFN down projection)

Total: 112 blocks, ~1,492 MB
```

---

### 5. Interactive Layer Switcher

**What it shows:**
- Unified view with all layers in one graph
- Sidebar controls to filter by layer number
- Toggle layers on/off to compare structures
- Component filtering across all layers

**Key Features:**
- **Layer Comparison**: Enable Layers 1, 5, 15, 28 simultaneously
- **Depth Visualization**: See entire 28-layer stack
- **Component Filtering**: Show only attention heads across all layers
- **Pattern Detection**: Spot anomalies or architectural variations

**Use Cases:**
- **Teaching**: Step through layers 1→2→3 to explain progressive depth
- **Debugging**: Verify all layers have identical structure
- **Performance**: Identify which layers are slowest (via importance metrics)
- **Optimization**: Compare different quantization strategies layer-by-layer

---

### 6. Complete Dashboard (All-in-One HTML)

**What it shows:**
- Integrated HTML dashboard with all 8 visualizations
- Statistics overview panel
- Navigation menu between visualizations
- Downloadable for offline use

**Contents:**
1. Main Architecture (929 nodes)
2. Layer 1 Detail (35 nodes)
3. Layer 2 Detail (35 nodes)
4. Layer 3 Detail (35 nodes)
5. Layer 4 Detail (35 nodes)
6. Layer 5 Detail (35 nodes)
7. Attention Heads Focus (896 nodes)
8. Quantization Blocks (112 nodes)

**Statistics Panel:**
```
Model: Llama-3.2-3B-Instruct-Q4_K_M
Total Layers: 28
Attention Heads: 896 (32 per layer)
Total Nodes: 929
Total Edges: 981
Quantization Blocks: 112
File Size: 1.88 GB
```

**Download:**
- File: `/kaggle/working/complete_dashboard.html`
- Size: ~15 MB (includes embedded visualizations)
- Offline: Works without internet connection
- Shareable: Email to colleagues or embed in documentation

---

## Technical Implementation

### Dual-GPU Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL-GPU SPLIT ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   GPU 0: Tesla T4 (15GB)          GPU 1: Tesla T4 (15GB)        │
│   ├─────────────────────┐         ├─────────────────────┐       │
│   │  llama-server       │         │  RAPIDS cuGraph     │       │
│   │  ─────────────      │         │  ─────────────      │       │
│   │  ✓ Model loaded     │         │  ✓ Graph analytics  │       │
│   │  ✓ Inference ready  │         │  ✓ PageRank         │       │
│   │  ✓ OpenAI API       │         │  ✓ Centrality       │       │
│   │  ✓ VRAM: ~10 GB     │         │  ✓ Degree           │       │
│   │                     │         │  ✓ VRAM: ~0.5 GB    │       │
│   │  tensor_split:      │         │  ✓ Free: 15.8 GB    │       │
│   │  "1.0,0.0"          │         │                     │       │
│   │  (100% GPU 0)       │         │  Graphistry         │       │
│   │                     │         │  ─────────────      │       │
│   │                     │         │  ✓ Visualization    │       │
│   │                     │         │  ✓ Cloud upload     │       │
│   │                     │         │  ✓ HTML generation  │       │
│   └─────────────────────┘         └─────────────────────┘       │
│                                                                 │
│   Benefit: No memory contention between LLM and visualization   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

**llamatelemetry v2.2.0:**
- **ServerManager**: Manages llama.cpp server lifecycle
- **LlamaCppClient**: OpenAI-compatible API client
- **Purpose**: Provides runtime access to GGUF model
- **Key Feature**: `tensor_split="1.0,0.0"` isolates GPU 0

**RAPIDS cuGraph (GPU-accelerated):**
- **PageRank**: `cugraph.pagerank(G)` → Component importance
- **Betweenness**: `cugraph.betweenness_centrality(G)` → Critical nodes
- **Degree**: `cugraph.degree_centrality(G)` → Connectivity
- **Performance**: 10-100× faster than NetworkX on CPU

**Graphistry Cloud:**
- **Binding**: Maps DataFrame columns to visual properties
- **Layout**: GPU-accelerated force-directed layout
- **Rendering**: WebGL in browser
- **Hosting**: Cloud URLs (15-day retention) + downloadable HTML

**Python Libraries:**
- **pandas**: DataFrame manipulation
- **cuDF**: GPU-accelerated DataFrame (RAPIDS)
- **pyarrow**: Efficient data transfer to Graphistry
- **requests**: HTTP communication with llama-server

### Data Pipeline

```
1. GGUF File Download (HuggingFace)
   ↓
2. llama-server Startup (GPU 0)
   ├─ Load model: tensor_split="1.0,0.0"
   └─ Start API: http://127.0.0.1:8090
   ↓
3. Architecture Extraction
   ├─ Query via LlamaCppClient
   ├─ Extract: n_layers, n_heads, hidden_dim
   └─ Fallback to known specs if query fails
   ↓
4. Graph Construction (Python)
   ├─ Create nodes_df: 929 rows × 10 columns
   ├─ Create edges_df: 981 rows × 5 columns
   └─ Add metadata: parameters, memory, types
   ↓
5. GPU Analytics (GPU 1)
   ├─ Convert to cuDF/cuGraph
   ├─ Compute: PageRank, Betweenness, Degree
   └─ Merge metrics back to nodes_df
   ↓
6. Visualization Preparation
   ├─ Define colors by node type
   ├─ Define icons (FontAwesome)
   ├─ Create HTML tooltips
   └─ Scale node sizes by importance
   ↓
7. Graphistry Upload
   ├─ Bind: nodes_df, edges_df to visual properties
   ├─ Generate cloud URLs (8 visualizations)
   └─ Create downloadable HTML dashboard
   ↓
8. Interactive Exploration (User Browser)
   └─ WebGL rendering, zoom, filter, search
```

### Code Snippets

**Dual-GPU Server Start:**
```python
from llamatelemetry.server import ServerManager

server = ServerManager()
server.start_server(
    model_path=model_path,
    host="127.0.0.1",
    port=8090,
    gpu_layers=99,          # Load all layers to GPU
    tensor_split="1.0,0.0", # 100% GPU 0, 0% GPU 1 (CRITICAL!)
    ctx_size=4096,
)
```

**Graph Construction:**
```python
import pandas as pd

nodes_data = []
edges_data = []

# Input node
nodes_data.append({'node_id': 0, 'name': 'Input', 'type': 'input', ...})

# Embedding node
nodes_data.append({'node_id': 1, 'name': 'Embedding', 'type': 'embedding', ...})

# 28 transformer layers
for layer in range(1, 29):
    # Layer node
    nodes_data.append({
        'node_id': current_id,
        'name': f'Layer_{layer}',
        'type': 'transformer',
        'layer': layer,
        'parameters': 4 * hidden_dim**2 + 4 * hidden_dim,
        'size_mb': (4 * hidden_dim**2 * 2) / (1024**2) / 4,  # Q4_K_M ~4× compression
    })

    # 32 attention heads
    for head in range(32):
        nodes_data.append({
            'node_id': current_id + 1 + head,
            'name': f'L{layer}_H{head}',
            'type': 'attention_head',
            'layer': layer,
            'head': head,
            'parameters': hidden_dim * hidden_dim // 32,
            'size_mb': (hidden_dim * hidden_dim * 2) / (1024**2) / 32 / 4,
        })

    # Edges: transformer contains heads
    for head in range(32):
        edges_data.append({
            'source': current_id,
            'target': current_id + 1 + head,
            'type': 'contains',
            'weight': 1.0,
        })

nodes_df = pd.DataFrame(nodes_data)
edges_df = pd.DataFrame(edges_data)
```

**GPU-Accelerated PageRank:**
```python
import cudf
import cugraph

# Convert to GPU DataFrames
edges_cudf = cudf.DataFrame(edges_df)

# Build graph on GPU
G = cugraph.Graph()
G.from_cudf_edgelist(edges_cudf, source='source', destination='target', edge_attr='weight')

# Compute PageRank on GPU (fast!)
pagerank = cugraph.pagerank(G)
pagerank = pagerank.sort_values('pagerank', ascending=False)

# Merge back to nodes (convert to pandas first)
pagerank_pd = pagerank.to_pandas().rename(columns={'vertex': 'node_id', 'pagerank': 'importance'})
nodes_df = nodes_df.merge(pagerank_pd, on='node_id', how='left')
```

**Graphistry Visualization:**
```python
import graphistry

# Register with Graphistry Cloud
graphistry.register(api=3, username=USERNAME, password=PASSWORD)

# Define visual encodings
nodes_df['color'] = nodes_df['type'].map({
    'input': '#FF6B6B',
    'output': '#4ECDC4',
    'embedding': '#FFD166',
    'transformer': '#06D6A0',
    'attention_head': '#118AB2',
    'normalization': '#EF476F',
    'feedforward': '#073B4C',
})

nodes_df['icon'] = nodes_df['type'].map({
    'input': 'sign-in-alt',
    'output': 'sign-out-alt',
    'embedding': 'layer-group',
    'transformer': 'microchip',
    'attention_head': 'eye',
    'normalization': 'balance-scale',
    'feedforward': 'arrows-alt-h',
})

nodes_df['point_size'] = 15 + (nodes_df['importance'] / nodes_df['importance'].max()) * 65

# Create visualization
g = graphistry.bind(
    source='source',
    destination='target',
    node='node_id',
    point_title='tooltip',
    point_color='color',
    point_size='point_size',
    point_icon='icon',
).settings(url_params={
    'play': 0,
    'pointSize': 2.5,
    'edgeOpacity': 0.6,
    'bg': '%23FFFFFF',
    'strongGravity': 'true',
})

plotter = g.edges(edges_df).nodes(nodes_df)
url = plotter.plot(render=False, name="GGUF Neural Network Architecture")
print(f"Dashboard URL: {url}")
```

---

## Use Cases

### For Researchers

**1. Quantization Impact Analysis**
- Compare Q4_K_M vs IQ3_XS vs Q8_0 quantization block structures
- Identify which layer types are most affected by quantization
- Measure importance score changes across quantization methods

**2. Architecture Comparison**
- Visualize Llama vs Gemma vs Qwen architectures side-by-side
- Identify architectural differences (head count, FFN size, normalization type)
- Validate GGUF conversions match original PyTorch models

**3. Pruning & Distillation**
- Use PageRank to identify least important attention heads
- Find redundant layers (low centrality scores)
- Design pruning strategies based on graph metrics

**4. Information Flow Analysis**
- Trace paths through the network (input → layer 15 → output)
- Identify bottleneck layers (high betweenness centrality)
- Understand residual connection impact on graph structure

---

### For Practitioners

**1. Deployment Planning**
- Estimate memory requirements per layer (VRAM planning)
- Identify which layers to offload to CPU if GPU memory limited
- Optimize quantization block loading order

**2. Performance Optimization**
- Find layers with highest parameter count (likely slowest)
- Optimize tensor-split ratios based on layer sizes
- Identify opportunities for layer fusion

**3. Model Debugging**
- Verify GGUF conversion preserved architecture
- Check parameter counts match expected values
- Validate attention head configuration (32 heads × 28 layers = 896 total)

**4. Custom Model Development**
- Use as template for building custom architectures
- Verify new models follow expected graph structure
- Compare custom model to reference architectures

---

### For Education

**1. Visual Learning**
- Students see transformer architecture as interactive graph (not just text)
- Zoom into individual layers to understand components
- Explore attention heads in parallel

**2. Parameter Counting**
- Hover over nodes to see exact parameter counts
- Understand why 3B parameter model has ~2.8B actual parameters
- Learn how quantization reduces memory (Q4_K_M saves 6.1× vs FP32)

**3. Architectural Concepts**
- **Multi-Head Attention**: See 32 heads running in parallel
- **Residual Connections**: Understand "feeds_into" edges
- **Layer Normalization**: See shared component across all layers
- **Feed-Forward Networks**: Visualize 4× expansion (SwiGLU)

**4. Comparative Analysis**
- Compare small models (3B) to large models (70B) visually
- Understand scaling laws (2× parameters ≠ 2× performance)
- Learn trade-offs between model size and quality

---

## Getting Started

### Prerequisites

**Hardware:**
- Kaggle account with dual T4 GPUs enabled (or local dual-GPU setup)
- Minimum 30GB total VRAM (2× 15GB GPUs)

**Software:**
- Python 3.11+
- CUDA 12.x
- Graphistry account (free tier available at https://hub.graphistry.com)

**Kaggle Settings:**
- Accelerator: GPU T4 × 2
- Internet: Enabled
- Secrets: Add Graphistry credentials

### Quick Start

**1. Open Notebook on Kaggle:**
```
https://www.kaggle.com/code/[username]/11-gguf-neural-network-graphistry-visualization
```

**2. Add Graphistry Secrets:**
- Go to Settings → Secrets
- Add secret: `GRAPHISTRY_USERNAME` = your_username
- Add secret: `GRAPHISTRY_PASSWORD` = your_password

**3. Run All Cells:**
- Kernel → Run All (takes ~15-20 minutes)

**4. Explore Dashboards:**
- Click generated URLs in output cells
- Download `/kaggle/working/complete_dashboard.html` for offline use

### Customization

**Analyze Different Models:**
```python
# Change model in Cell 16
MODEL_REPO = "unsloth/Llama-3.1-8B-Instruct-GGUF"
MODEL_FILE = "Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Update architecture specs in Cell 22
n_layers = 32  # For Llama-8B
n_heads = 32
hidden_dim = 4096
```

**Visualize All 28 Layers:**
```python
# In Cell 34, change range
for layer_num in range(1, n_layers + 1):  # All layers instead of just 1-5
    ...
```

**Add Custom Metrics:**
```python
# In Cell 26, add more cuGraph metrics
clustering = cugraph.clustering_coefficient(G)
triangle_count = cugraph.triangle_count(G)
```

**Change Color Scheme:**
```python
# In Cell 30, modify type_colors
type_colors = {
    'transformer': '#FF0000',  # Red transformers
    'attention_head': '#0000FF',  # Blue heads
    ...
}
```

---

## FAQ

**Q: Why only visualize Layers 1-5 instead of all 28?**
A: Graphistry cloud has rendering limits. For all 28 layers, use the "Interactive Layer Switcher" visualization which lets you filter layers dynamically.

**Q: Can I run this locally instead of Kaggle?**
A: Yes! Replace `'/kaggle/working/'` paths with local paths. Ensure you have 2 GPUs and CUDA 12.x installed.

**Q: Does this work with non-GGUF models?**
A: No, this is specifically for GGUF quantized models loaded via llama-server. For PyTorch/HF models, use different visualization tools.

**Q: How long does the notebook take to run?**
A: ~15-20 minutes total:
  - Installation: 5 min
  - Model download: 3 min
  - Graph construction: 2 min
  - GPU analytics: 1 min
  - Graphistry upload: 5 min

**Q: Can I visualize 70B models?**
A: Yes! The graph structure scales linearly. Llama-70B has 80 layers × 64 heads = 5,120 attention heads. Just update architecture specs and use dual T4 with IQ3_XS quantization.

**Q: Why use PageRank for neural networks?**
A: PageRank identifies "important" nodes based on connectivity. In neural networks, components with high PageRank are more central to information flow, making them critical for model performance.

**Q: Are the visualizations static or interactive?**
A: **Fully interactive!** You can zoom, pan, click, filter, search, and explore in real-time. The downloadable HTML is also interactive (no internet required).

---

## Conclusion

This notebook represents a **breakthrough in GGUF model understanding**:

- **First tool** to visualize GGUF quantization structure as graphs
- **Novel application** of graph theory (PageRank, centrality) to neural architectures
- **Dual-GPU innovation** separating inference from visualization workloads
- **Multi-scale exploration** from 929-node overview to 35-node layer details
- **Production-ready dashboards** for research, deployment, and education

By transforming opaque binary GGUF files into **interactive, explorable graph structures**, this tool reveals architectural insights impossible to see any other way.

---

**Notebook:** [`11-gguf-neural-network-graphistry-visualization.ipynb`](../notebooks/11-gguf-neural-network-graphistry-visualization.ipynb)

**Related Docs:**
- [GGUF Guide](GGUF_GUIDE.md) - GGUF format and quantization
- [Kaggle Guide](KAGGLE_GUIDE.md) - Dual T4 configuration
- [API Reference](API_REFERENCE.md) - llamatelemetry Python API
