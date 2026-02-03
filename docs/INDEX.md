# llamatelemetry Project - Documentation Index (v0.1.0)

## Overview

llamatelemetry v0.1.0 is a CUDA 12-first inference stack for **Kaggle dual Tesla T4** (SM 7.5). It is optimized for **small GGUF models (1B-5B)** and a split-GPU workflow (GPU 0: LLM, GPU 1: Graphistry/RAPIDS).

**Target Platform:** Kaggle notebooks only (GPU T4 x2)

---

## Quick Start (Kaggle)

1. Install llamatelemetry:
   ```python
   !pip install -q --no-cache-dir git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
   ```
2. Start inference:
   ```python
   from llamatelemetry.server import ServerManager
   server = ServerManager()
   server.start_server(model_path="model.gguf", gpu_layers=99, tensor_split="1.0,0.0")
   ```
3. Follow notebooks: `notebooks/README.md`

---

## Core Guides (v0.1.0)

- `docs/INSTALLATION.md` — Kaggle-only install guide
- `docs/KAGGLE_GUIDE.md` — full Kaggle workflow
- `docs/API_REFERENCE.md` — Python API reference
- `docs/CONFIGURATION.md` — server/config settings
- `docs/QUICK_START_GUIDE.md` — 5-minute quick start
- `docs/INTEGRATION_GUIDE.md` — llama-server detection flow (legacy notes included)

---

## Project Guides (organized)

- `guides/QUICK_START.md` — end-user quick start
- `guides/INSTALL.md` — install details (short form)
- `guides/KAGGLE_INSTALL_GUIDE.md` — Kaggle-specific install flow

## HuggingFace Setup + Ops

- `huggingface/HUGGINGFACE_WAQASM86_SETUP_COMPLETE.md` — primary setup summary
- `huggingface/HUGGINGFACE_SETUP_INSTRUCTIONS.md` — org setup steps
- `huggingface/llamatelemetry_org_card.md` — organization card content

## Reports + Reference

- `reports/DEPLOYMENT_SUMMARY.md` — deployment status
- `reports/UPLOAD_VERIFICATION_REPORT.md` — upload verification
- `reference/LLAMATELEMETRY_QUICK_REFERENCE.md` — quick links

---

## Notebooks

- `notebooks/README.md` — 13-notebook tutorial path
- `notebooks/01-quickstart-llamatelemetry-v0.1.0.ipynb` → `notebooks/13-gguf-token-embedding-visualizer-executed-3.ipynb`

---

## Release / Distribution

- `docs/GITHUB_RELEASE_GUIDE.md` — GitHub release steps (v0.1.0)
- `docs/RELEASE_FILES_OVERVIEW.txt` — binary bundle overview (note: file may be legacy in parts)

**Distribution:** GitHub Releases (primary) + optional HuggingFace mirror. **No PyPI** for v0.1.0.

---

## Legacy References (pre-0.1.0)

These are kept for historical context:

- `docs/PYPI_PACKAGE_GUIDE.md`
- `docs/FINAL_WORKFLOW_GUIDE.md`
- `docs/GITHUB_RELEASE_NOTES_SIMPLIFIED.md`
- `docs/BUILD_GUIDE.md`
- `docs/RELEASE_SUMMARY.md`
