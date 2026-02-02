# 🚀 New llamatelemetry v2.2.0 Kaggle Notebooks

**Complementary Educational Tools for [Transformers-Explainer](https://poloclub.github.io/transformer-explainer/)**

---

## 📚 Notebook Series Overview

These 5 new notebooks extend the flagship **Notebook 11 (GGUF Neural Network Visualization)** to provide deep, customizable analysis of GGUF models as educational complements to the transformers-explainer web tool.

### Key Differentiators

| Feature | Transformers-Explainer | llamatelemetry Notebooks (12-16) |
|---------|------------------------|--------------------------|
| **Runtime** | Browser (WebAssembly) | Kaggle Dual T4 GPUs |
| **Models** | GPT-2 only (FP32) | 1B-8B GGUF (Q4_K_M/Q5_K_M) |
| **Speed** | 2-5s inference | <1s GPU-accelerated |
| **Quantization** | Not shown | **Visible in all analyses** |
| **Customization** | Fixed UI | **Fully editable Kaggle notebooks** |
| **Visualization** | D3.js in browser | **Graphistry + RAPIDS** |
| **Scale** | 124M parameters | **1B-8B parameters** |

---

## 📓 Notebook Summaries

### ✅ Notebook 12: GGUF Attention Mechanism Explorer
**Status: Created** ✓

**Purpose**: Visualize attention patterns in quantized GGUF models

**Key Features:**
- Extract attention weights from llama.cpp inference
- Visualize Q-K-V decomposition across all heads
- Interactive Graphistry dashboards on GPU 1
- Compare attention patterns across layers
- Analyze quantization impact on attention scores

**What It Shows That Transformers-Explainer Doesn't:**
- Post-quantization attention patterns (Q4_K_M vs FP32)
- Models beyond GPT-2 (Gemma, Llama 3.2, Qwen 2.5)
- GPU-accelerated analysis of 32+ layer models
- Production model attention behavior

**Use Cases:**
- Debug attention in fine-tuned models
- Compare attention heads across models
- Understand quantization trade-offs
- Educational: Show how attention works in production models

---

### ✅ Notebook 13: GGUF Token Embedding Visualizer
**Status: Created** ✓

**Purpose**: Explore semantic structure of embedding spaces

**Key Features:**
- Extract embeddings from GGUF models (768D-4096D)
- GPU-accelerated UMAP/t-SNE for 3D projections
- Semantic clustering analysis
- Cosine similarity networks
- Interactive 3D exploration with Plotly + Graphistry

**What It Shows That Transformers-Explainer Doesn't:**
- 3D interactive embedding space (vs 2D colored rectangles)
- Semantic clustering (similar words group together)
- Quantization impact on embedding quality
- Word analogies (king - man + woman ≈ queen)
- Cross-model embedding comparisons

**Use Cases:**
- Understand how models represent concepts
- Visualize semantic relationships
- Compare embeddings across quantizations
- Debugging token representation issues

---

### 📋 Notebook 14: GGUF Layer-by-Layer Inference Tracker

**Purpose**: Track hidden states through transformer layers

**Key Features:**
- Capture hidden states at each layer (0 → 32)
- Visualize activation patterns and magnitudes
- Track residual connection contributions
- Layer norm impact analysis
- Interactive layer explorer with Graphistry

**Architecture:**
```
Input Tokens
    ↓
Layer 0: Attention + MLP → Hidden State 0
    ↓
Layer 1: Attention + MLP → Hidden State 1
    ↓
  ... (track all intermediate states)
    ↓
Layer N: Final Hidden State → Logits
```

**What It Shows:**
- How information propagates through layers
- Which layers contribute most to final prediction
- Residual connection importance
- Layer-wise feature evolution

**Code Structure:**
```python
# Pseudo-code for implementation

# Step 1: Run inference with layer-wise logging
hidden_states = {}
for layer_idx in range(n_layers):
    # Extract hidden state after layer
    hidden_state = extract_layer_output(model, layer_idx, input_tokens)
    hidden_states[layer_idx] = hidden_state

# Step 2: Analyze hidden state magnitudes
norms_per_layer = {
    layer: np.linalg.norm(hidden_states[layer], axis=-1)
    for layer in hidden_states
}

# Step 3: Visualize with Graphistry
# Create node for each (layer, token) pair
nodes = create_layer_token_nodes(hidden_states)
edges = create_layer_connections(nodes)  # Layer i → Layer i+1
g = graphistry.edges(edges).nodes(nodes).plot()
```

**Customization Options:**
- Select specific layers to visualize
- Focus on specific tokens
- Compare multiple prompts side-by-side
- Heatmap view of activation patterns

---

### 📋 Notebook 15: GGUF Multi-Head Attention Comparator

**Purpose**: Compare attention behavior across heads and layers

**Architecture:**
```
Model: 32 Layers × 32 Heads = 1024 Attention Heads

Visualization Strategy:
1. Head Clustering: Group similar attention patterns
2. Head Specialization: Identify role of each head
3. Layer Analysis: How heads evolve across depth
4. Comparative View: Side-by-side head matrices
```

**Key Features:**
- Simultaneous visualization of all 12-32 attention heads
- Clustering heads by behavior (local, global, syntactic)
- Interactive head comparison dashboard
- Attention pattern taxonomy

**What It Shows:**
- Head specialization (which heads do what?)
- Redundancy analysis (do all heads matter?)
- Layer-wise attention evolution
- Most important heads for specific tasks

**Implementation Approach:**
```python
# Step 1: Extract all attention matrices
attention_data = {}
for layer in range(n_layers):
    for head in range(n_heads):
        attn_matrix = extract_attention(layer, head, prompt)
        attention_data[(layer, head)] = attn_matrix

# Step 2: Cluster heads by behavior
from sklearn.cluster import KMeans
flattened_patterns = [mat.flatten() for mat in attention_data.values()]
kmeans = KMeans(n_clusters=5)
head_clusters = kmeans.fit_predict(flattened_patterns)

# Step 3: Visualize clusters
# Cluster 0: Local attention (diagonal)
# Cluster 1: Global attention (uniform)
# Cluster 2: Positional attention (position-based)
# Cluster 3: Syntactic attention (grammar-aware)
# Cluster 4: Semantic attention (meaning-focused)

# Step 4: Create dashboard with Graphistry
for cluster_id in range(5):
    heads_in_cluster = get_heads(cluster_id)
    visualize_cluster(heads_in_cluster, cluster_label)
```

**Insights:**
- Not all heads are equally important
- Early layers: more uniform attention
- Late layers: specialized, peaked attention
- Some heads can be pruned without performance loss

---

### 📋 Notebook 16: GGUF Quantization Impact Analyzer

**Purpose**: Quantitatively measure quantization effects on model behavior

**Comparison Matrix:**
```
Quantization Types Tested:
- Q8_0:    8.5 bits/weight (near-lossless)
- Q5_K_M:  5.69 bits/weight (high quality)
- Q4_K_M:  4.85 bits/weight (recommended)
- Q3_K_M:  3.91 bits/weight (aggressive)
- IQ3_XS:  3.30 bits/weight (extreme)
```

**Key Features:**
- Side-by-side output comparison (FP32 → Q4_K_M → IQ3_XS)
- Perplexity measurement across quantizations
- Attention weight precision analysis
- Embedding similarity preservation
- Performance vs quality trade-off charts

**Analysis Dimensions:**
1. **Output Quality**: Compare generated text quality
2. **Attention Preservation**: Cosine similarity of attention matrices
3. **Embedding Space**: How quantization affects embeddings
4. **Inference Speed**: Throughput improvement (tokens/sec)
5. **VRAM Usage**: Memory savings (GB)

**Implementation:**
```python
# Step 1: Download multiple quantizations
models = {
    'Q8_0': load_model_smart("llama-3.2-3b-Q8_0"),
    'Q5_K_M': load_model_smart("llama-3.2-3b-Q5_K_M"),
    'Q4_K_M': load_model_smart("llama-3.2-3b-Q4_K_M"),
    'IQ3_XS': load_model_smart("llama-3.2-3b-IQ3_XS")
}

# Step 2: Run same prompt through all models
prompt = "The transformer architecture uses"
results = {}
for quant_type, model_path in models.items():
    # Start server with this quantization
    server.restart_server(model_path)

    # Generate text
    response = client.chat.completions.create(...)

    # Extract attention (if possible)
    attention = extract_attention_matrices(...)

    results[quant_type] = {
        'text': response.choices[0].message.content,
        'attention': attention,
        'latency': response_time,
        'tokens_per_sec': throughput
    }

# Step 3: Comparative analysis
for q1, q2 in combinations(quant_types, 2):
    # Text similarity (BLEU/ROUGE)
    text_sim = compute_text_similarity(results[q1]['text'], results[q2]['text'])

    # Attention similarity (if extracted)
    attn_sim = cosine_similarity(
        results[q1]['attention'].flatten(),
        results[q2]['attention'].flatten()
    )

    print(f"{q1} vs {q2}:")
    print(f"  Text similarity: {text_sim:.3f}")
    print(f"  Attention similarity: {attn_sim:.3f}")

# Step 4: Visualize trade-offs with Plotly
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[bits_per_weight[q] for q in quant_types],
    y=[quality_scores[q] for q in quant_types],
    mode='markers+text',
    text=quant_types,
    marker=dict(size=15)
))
fig.update_layout(
    title="Quantization Trade-off: Quality vs Size",
    xaxis_title="Bits per Weight",
    yaxis_title="Quality Score"
)
fig.show()
```

**Key Findings (Expected):**
- Q5_K_M vs Q8_0: <1% quality loss, 33% smaller
- Q4_K_M vs Q5_K_M: ~2% quality loss, 15% smaller
- IQ3_XS vs Q4_K_M: ~5-10% quality loss, 32% smaller
- Speed improvements: 10-20% faster inference for lower quantizations

---

## 🎯 Educational Progression

### Learning Path: From Browser to Production

1. **Start**: [Transformers-Explainer](https://poloclub.github.io/transformer-explainer/)
   - Learn Q-K-V mechanics (4-stage breakdown)
   - Understand softmax, causal masking
   - Fixed GPT-2, educational interface

2. **Notebook 12**: Attention Mechanism Explorer
   - Same concepts, but with **quantized models**
   - Explore **multiple model architectures**
   - See how **production models** behave

3. **Notebook 13**: Token Embedding Visualizer
   - Go deeper into **embedding spaces**
   - **3D interactive** exploration
   - Understand **semantic structure**

4. **Notebook 14**: Layer-by-Layer Tracker
   - Follow information **through all layers**
   - Understand **residual connections**
   - See **layer specialization**

5. **Notebook 15**: Multi-Head Comparator
   - Compare **all attention heads**
   - Identify **head roles**
   - Learn which heads **matter most**

6. **Notebook 16**: Quantization Analyzer
   - Understand **production trade-offs**
   - Measure **quantization impact**
   - Make informed **deployment decisions**

---

## 🛠️ Technical Implementation Guide

### Common Setup Pattern

All notebooks follow this structure:

```python
# 1. Environment Verification
!nvidia-smi -L  # Dual T4 check

# 2. Install llamatelemetry v2.2.0
!pip install git+https://github.com/llamatelemetry/llamatelemetry.git \
    graphistry[all] cudf-cu12 cugraph-cu12

# 3. Download GGUF Model
from llamatelemetry.models import load_model_smart
model_path = load_model_smart("gemma-3-1b-Q4_K_M")

# 4. Start llama-server on GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
server = ServerManager()
server.start_server(model_path, gpu_layers=99)

# 5. Run Analysis on GPU 0
client = LlamaCppClient()
response = client.chat.completions.create(...)

# 6. Switch to GPU 1 for Visualization
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cudf, cugraph, graphistry

# 7. Create Interactive Dashboard
g = graphistry.edges(edges_df).nodes(nodes_df).plot()
```

### Key Dependencies

```python
# Core
llamatelemetry>=2.2.0
huggingface_hub

# Visualization
graphistry[all]
plotly
matplotlib

# GPU Analytics (GPU 1)
cudf-cu12
cugraph-cu12
cuml-cu12  # For UMAP/t-SNE

# Utils
pandas
numpy
scikit-learn
```

---

## 📊 Comparison Matrix

| Notebook | Transformers-Explainer Feature | llamatelemetry Enhancement |
|----------|-------------------------------|-------------------|
| **12** | Q·K^T → Softmax visualization | **Quantized attention patterns** |
| **13** | 768D embeddings as rectangles | **3D UMAP projection + clustering** |
| **14** | Single layer view | **All 32 layers tracked** |
| **15** | One head at a time | **All heads compared simultaneously** |
| **16** | FP32 only | **5 quantization types compared** |

---

## 🚀 Getting Started

### 1. Upload to Kaggle

```bash
# Each notebook is self-contained
# Upload .ipynb files to Kaggle Notebooks
# Enable Dual T4 Accelerator
```

### 2. Set Secrets

```python
# Add Kaggle Secrets:
# - Graphistry_Personal_Key_ID
# - Graphistry_Username
```

### 3. Run Notebooks in Order

**Recommended Sequence:**
1. Notebook 12 (Attention) - 15 min
2. Notebook 13 (Embeddings) - 10 min
3. Notebook 14 (Layers) - 20 min
4. Notebook 15 (Heads) - 25 min
5. Notebook 16 (Quantization) - 30 min

**Total Time**: ~2 hours for complete understanding

---

## 🎓 Educational Use Cases

### For Students
- **Visual Learning**: See how transformers work with real models
- **Interactive Exploration**: Modify code, see results immediately
- **Scalability**: Understand production vs toy models

### For Researchers
- **Model Analysis**: Debug attention patterns in custom models
- **Quantization Research**: Measure precision-quality trade-offs
- **Comparative Studies**: Evaluate multiple architectures

### For Engineers
- **Production Decisions**: Choose quantization for deployment
- **Performance Tuning**: Identify bottleneck layers/heads
- **Model Debugging**: Visualize unexpected behaviors

---

## 🔗 Related Resources

- **Transformers-Explainer**: https://poloclub.github.io/transformer-explainer/
- **llamatelemetry GitHub**: https://github.com/llamatelemetry/llamatelemetry
- **Graphistry Docs**: https://github.com/graphistry/pygraphistry
- **RAPIDS cuML**: https://docs.rapids.ai/api/cuml/stable/

---

## 📝 Citation

If you use these notebooks in your research or education:

```bibtex
@software{llcuda_transformer_notebooks,
  title={llamatelemetry Transformer Visualization Notebooks},
  author={Your Name},
  year={2026},
  note={Complementary educational tools for transformers-explainer},
  url={https://github.com/llamatelemetry/llamatelemetry}
}
```

---

## 🤝 Contributing

**Want to add more notebooks?**

Ideas for future notebooks:
- Notebook 17: Cross-Attention Visualizer (for encoder-decoder models)
- Notebook 18: KV-Cache Optimization Explorer
- Notebook 19: LoRA Fine-tuning Visualizer
- Notebook 20: Model Pruning Impact Analyzer

**Submit issues/PRs**: https://github.com/llamatelemetry/llamatelemetry/issues

---

## 📄 License

MIT License - Free for educational and commercial use

---

**Created with ❤️ using llamatelemetry v2.2.0, Graphistry, and RAPIDS**
