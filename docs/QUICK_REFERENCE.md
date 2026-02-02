# llamatelemetry v2.2.0 - Quick Reference Card

**Version**: 2.2.0
**Target**: Kaggle 2× Tesla T4 (SM 7.5)
**Build**: CUDA 12.5, llama.cpp b7760 (388ce82)
**Date**: February 1, 2026

---

## 🚀 Quick Start (30 seconds on Kaggle)

1. **Set accelerator**: Settings → Accelerator → GPU T4 × 2
2. **Install**: `!pip install -q --no-cache-dir git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0`
3. **Run**: Open [Notebook 01](../notebooks/01-quickstart-llamatelemetry-v2.2.0.ipynb) and execute all cells
4. **Done!** First inference in under 5 minutes

---

## 📓 Tutorial Notebooks (13 total)

| # | Notebook | Focus | Time |
|---|----------|-------|------|
| 01 | Quick Start | Install + first inference | 5 min |
| 02 | Server Setup | Configuration + lifecycle | 15 min |
| 03 | Multi-GPU | Dual T4 tensor-split | 20 min |
| 04 | GGUF Quantization | K-quants, I-quants | 20 min |
| 05 | Unsloth Integration | Fine-tune → Export → Deploy | 30 min |
| 06 | Split-GPU Graphistry | LLM + RAPIDS on separate GPUs | 30 min |
| 07 | Knowledge Graph | LLM entity extraction | 30 min |
| 08 | Document Network | Similarity + community detection | 35 min |
| 09 | Large Models | 13B+ on dual T4 | 30 min |
| 10 | Complete Workflow | Production end-to-end | 50 min |
| 11 | **GGUF Visualization** ⭐ | Architecture (929 nodes, 8 dashboards) | 60 min |
| 12 | **Attention Explorer** | Q-K-V patterns + Graphistry | 20 min |
| 13 | **Embedding Visualizer** | 3D UMAP + Plotly | 15 min |

---

## 💻 Essential Commands

### Installation
```python
# Kaggle (recommended)
!pip install -q --no-cache-dir git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0
```

### Basic Inference
```python
import llamatelemetry
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",
    local_dir="/kaggle/working/models"
)

engine = llamatelemetry.InferenceEngine()
engine.load_model(model_path, silent=True)
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
```

### Split-GPU (LLM + Visualization)
```python
from llamatelemetry.server import ServerManager

server = ServerManager()
server.start_server(
    model_path=model_path,
    gpu_layers=99,
    tensor_split="1.0,0.0",  # GPU 0 only
)
# GPU 1 free for RAPIDS / Graphistry / Plotly
```

### Unsloth Workflow
```python
# 1. Fine-tune
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/gemma-3-1b-it", load_in_4bit=True)
# ... add LoRA, train ...

# 2. Export
model.save_pretrained_gguf("my_model", tokenizer, quantization_method="q4_k_m")

# 3. Deploy
server.start_server(model_path="my_model-Q4_K_M.gguf", gpu_layers=99)
```

---

## 📊 Performance (Tesla T4)

| Model | Speed | VRAM |
|-------|-------|------|
| Gemma 3-1B Q4_K_M | **45 tok/s** | 1.2 GB |
| Llama 3.2-3B Q4_K_M | **30 tok/s** | 2.0 GB |
| Qwen 2.5-7B Q4_K_M | **18 tok/s** | 5.0 GB |
| Llama 3.1-8B Q4_K_M | **15 tok/s** | 5.5 GB |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU not detected | Settings → Accelerator → GPU T4 × 2 |
| Out of memory | Reduce `ctx_size`, use smaller model or lower quantization |
| Server won't start | Check logs: `server.get_logs()` |
| Slow inference | Enable `flash_attn=True` in `start_server()` |
| Import errors | Restart kernel after `pip install` |
| Binaries not found | First import triggers auto-download (~961 MB) |

---

## 📊 Performance (Single Tesla T4)

| Model | Size | Quantization | VRAM | Tokens/sec |
|-------|------|--------------|------|------------|
| Gemma-3 1B | 1.0B | Q4_K_M | ~1.2 GB | ~50 tok/s |
| Llama-3.2 3B | 3.2B | Q4_K_M | ~2.5 GB | ~38 tok/s |
| Gemma-3 4B | 4.0B | Q4_K_M | ~3.0 GB | ~35 tok/s |
| Qwen2.5 7B | 7.0B | Q4_K_M | ~5.0 GB | ~18 tok/s |
| Llama-3.1 8B | 8.0B | Q4_K_M | ~5.5 GB | ~15 tok/s |

---

## 🎯 Learning Paths

| Path | Notebooks | Time |
|------|-----------|------|
| Quick Start | 01 → 02 → 03 | 1 hour |
| Full Course ⭐ | 01 → 13 (all) | 5.5 hours |
| Unsloth Focus | 01 → 04 → 05 → 10 | 2 hours |
| Large Models | 01 → 03 → 09 | 1.5 hours |
| Visualization | 01 → 03 → 04 → 06 → 11 → 12 → 13 | 3.5 hours |

---

## 🔗 Links

- **llamatelemetry**: https://github.com/llamatelemetry/llamatelemetry
- **Documentation**: https://llamatelemetry.github.io/
- **Unsloth**: https://github.com/unslothai/unsloth
- **llama.cpp**: https://github.com/ggml-org/llama.cpp
- **Graphistry**: https://hub.graphistry.com

---

## ✅ Pre-flight Checklist

- [ ] Kaggle account created
- [ ] Accelerator set to GPU T4 × 2
- [ ] Internet enabled in notebook settings
- [ ] Secrets added: `HF_TOKEN`, `Graphistry_Personal_Key_ID`, `Graphistry_Personal_Secret_Key`
- [ ] Notebook 01 opened and all cells executed successfully
