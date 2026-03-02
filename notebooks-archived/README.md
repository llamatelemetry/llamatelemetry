# llamatelemetry Notebooks

**16 comprehensive Jupyter notebooks** covering LLM inference, observability, and visualization on Kaggle dual Tesla T4.

---

## 📚 All Notebooks (5.5 hours total)

### **Phase 1: Foundation (Beginner)** - 65 minutes
| # | Notebook | Time | Topics |
|---|----------|------|--------|
| 01 | Quick Start | 10 min | Basic inference, model loading |
| 02 | Server Setup | 15 min | Configuration, optimization |
| 03 | Multi-GPU Inference | 20 min | Tensor parallelism, dual T4 |
| 04 | GGUF Quantization | 20 min | 29 quantization types, VRAM estimation |

### **Phase 2: Integration (Intermediate)** - 60 minutes
| # | Notebook | Time | Topics |
|---|----------|------|--------|
| 05 | Unsloth Integration | 30 min | Fine-tuning, LoRA, GGUF export |
| 06 | Split-GPU Graphistry | 30 min | Concurrent LLM + RAPIDS |

### **Phase 3: Advanced Applications** - 65 minutes
| # | Notebook | Time | Topics |
|---|----------|------|--------|
| 07 | Knowledge Graph Extraction | 35 min | LLM-powered knowledge graphs |
| 08 | Document Network Analysis | 30 min | Document similarity, clustering |

### **Phase 4: Optimization & Production** - 120 minutes
| # | Notebook | Time | Topics |
|---|----------|------|--------|
| 09 | Large Models on Kaggle | 35 min | 70B models on dual T4 |
| 10 | Complete Workflow | 45 min | End-to-end production pipeline |
| 11 | Neural Network Visualization | 40 min | 929-node architecture graphs |

### **Phase 5: Observability Trilogy** ⭐ **NEW** - 120 minutes
| # | Notebook | Time | Topics |
|---|----------|------|--------|
| 12 | Attention Mechanism Explorer | 25 min | Q-K-V decomposition, 896 heads |
| 13 | Token Embedding Visualizer | 30 min | 3D UMAP embedding space |
| 14 | OpenTelemetry Observability | 45 min | Full OTel integration, OTLP export |
| 15 | **Real-Time Monitoring** ⭐ | 30 min | Live Plotly dashboards, llama.cpp /metrics |
| 16 | **Production Stack** ⭐ | 45 min | Complete observability (Graphistry + Plotly 2D/3D) |

---

## 🎯 Recommended Learning Paths

### **Path 1: Quick Start** (1 hour)
```
01 → 02 → 03
```
Perfect for: Getting started with LLM inference on Kaggle

### **Path 2: Observability Expert** ⭐ **RECOMMENDED** (2.5 hours)
```
01 → 03 → 14 → 15 → 16
```
Perfect for: Building production observability systems

### **Path 3: Complete Foundation** (3 hours)
```
01 → 02 → 03 → 04 → 05 → 06 → 10
```
Perfect for: Comprehensive understanding of all features

### **Path 4: Visualization Specialist** (3.5 hours)
```
01 → 03 → 06 → 11 → 12 → 13 → 16
```
Perfect for: Advanced visualization and analytics

---

## ⭐ New Notebooks 15 & 16 Highlights

### **Notebook 15: Real-Time Performance Monitoring**
- **Live Plotly FigureWidget dashboards** with auto-updating charts
- **llama.cpp /metrics endpoint** integration (Prometheus format)
- **PyNVML GPU monitoring** (utilization, memory, temperature, power)
- **Request queue monitoring** via /slots endpoint
- **Background metrics collection** with threading
- **6-panel real-time dashboard** (tokens/sec, GPU %, latency, memory, slots, temp/power)

**Objectives:**
- ✅ CUDA Inference (GPU 0) - Continuous workload
- ✅ LLM Observability (GPU 0) - llama.cpp + GPU metrics
- ✅ Plotly Visualizations (GPU 1) - Live dashboards

---

### **Notebook 16: Production Observability Stack** 🏆
**The flagship comprehensive notebook integrating all three core objectives**

**Multi-Layer Observability:**
1. **OpenTelemetry SDK** - TracerProvider, MeterProvider, LoggerProvider
2. **llama.cpp Native Metrics** - /metrics, /slots, /health endpoints
3. **GPU Monitoring** - PyNVML (utilization, memory, temp, power, clocks)
4. **GGUF Model Introspection** - Attention weights, embeddings, layer activations

**Unified Visualization Dashboard:**
- **Section 1:** Request Trace Graphs (Graphistry 2D) - Distributed tracing
- **Section 2:** Performance Metrics (Plotly 2D) - 6-chart comprehensive view
- **Section 3:** Model Internals 3D (Plotly 3D) - Token embeddings, attention heatmaps
- **Section 4:** Real-Time Monitoring (Plotly Gauges) - Live indicators

**Objectives:**
- ✅ CUDA Inference (GPU 0) - Production pipeline
- ✅ LLM Observability (GPU 0) - Multi-source telemetry
- ✅ Unified Visualizations (GPU 1) - Graphistry 2D + Plotly 2D/3D

---

## 📁 File Organization

This directory contains:
- **Executed notebooks** (with outputs): `*-e1.ipynb`, `*-e2.ipynb`
- **Specification files**: `*-SPEC-*.md` (implementation guides)
- **Index files**: `README.md`, `14-15-16-INDEX.md`

---

## 🚀 Quick Start

1. **Installation:**
   ```python
   !pip install -q --no-cache-dir git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
   ```

2. **Start with Notebook 01** for basic setup

3. **Jump to Notebooks 14-16** for production observability

---

## 📖 Additional Resources

- **Documentation:** `../docs/NOTEBOOKS_GUIDE.md`
- **Kaggle Guide:** `../docs/KAGGLE_GUIDE.md`
- **API Reference:** `../docs/API_REFERENCE.md`
- **Troubleshooting:** `../docs/TROUBLESHOOTING.md`

---

**Total Learning Time:** 5.5 hours
**Platform:** Kaggle dual Tesla T4 (30GB VRAM)
**Python:** 3.11+
**CUDA:** 12.5
