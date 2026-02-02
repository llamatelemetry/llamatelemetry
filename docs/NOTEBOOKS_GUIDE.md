# Kaggle Notebooks Guide - llamatelemetry v2.2.0

**Updated:** 2026-02-01
**Purpose:** Guide to the 13 llamatelemetry v2.2.0 notebooks for Kaggle dual Tesla T4.

---

## ✅ Scope (v2.2.0)

- **Platform:** Kaggle notebooks only (GPU T4 x2)
- **Models:** 1B-5B GGUF (Q4_K_M recommended)
- **Architecture:** Split-GPU (GPU 0: LLM, GPU 1: Graphistry/RAPIDS)
- **Distribution:** GitHub Releases (no PyPI)

---

## 📚 Notebook Series (13 total)

1. `01-quickstart-llamatelemetry-v2.2.0.ipynb`
2. `02-llama-server-setup-llamatelemetry-v2.2.0.ipynb`
3. `03-multi-gpu-inference-llamatelemetry-v2.2.0.ipynb`
4. `04-gguf-quantization-llamatelemetry-v2.2.0.ipynb`
5. `05-unsloth-integration-llamatelemetry-v2.2.0.ipynb`
6. `06-split-gpu-graphistry-llamatelemetry-v2-2-0.ipynb`
7. `07-knowledge-graph-extraction-graphistry-v2.2.0.ipynb`
8. `08-document-network-analysis-graphistry-llamatelemetry-v2-2-0.ipynb`
9. `09-large-models-kaggle-llamatelemetry-v2-2-0.ipynb`
10. `10-complete-workflow-llamatelemetry-v2-2-0.ipynb`
11. `11-gguf-neural-network-graphistry-vis-executed-2.ipynb`
12. `12-gguf-attention-mechanism-explorer-executed.ipynb`
13. `13-gguf-token-embedding-visualizer-executed-3.ipynb`

---

## 🧰 Recommended Kaggle Settings

- Accelerator: **GPU T4 x2**
- Internet: **Enabled**
- Python: **3.11+**

---

## ✅ Install (Kaggle)

```python
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0
```

---

## 🔗 Related Docs

- `docs/INSTALLATION.md`
- `docs/KAGGLE_GUIDE.md`
- `docs/API_REFERENCE.md`
- `notebooks/README.md`
