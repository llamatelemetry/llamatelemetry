# GitHub Release Guide (v2.2.0)

This guide covers publishing llamatelemetry v2.2.0 binaries to GitHub Releases.

**Scope:** v2.2.0 is Kaggle-only (dual Tesla T4). Distribution is via GitHub Releases (primary) and optional HuggingFace mirror. PyPI is not used.

---

## 1) Prepare the Binary Bundle

**Expected bundle name:**
- `llamatelemetry-v2.2.0-cuda12-kaggle-t4x2.tar.gz`

**Expected contents:**
- `bin/` (llama-server + tools)
- `lib/` (CUDA libraries)

---

## 2) Create the GitHub Release

1. Go to the releases page: `https://github.com/llamatelemetry/llamatelemetry/releases`
2. Click **Draft a new release**
3. **Tag:** `v2.2.0`
4. **Title:** `llamatelemetry v2.2.0 - Kaggle Dual T4 CUDA 12 Binaries`
5. **Description:** (example below)
6. **Upload assets:** `llamatelemetry-v2.2.0-cuda12-kaggle-t4x2.tar.gz`
7. Publish the release

**Suggested description (short):**
```
CUDA 12-first inference bundle for Kaggle dual Tesla T4 (SM 7.5).
Optimized for 1B-5B GGUF models. Includes llama.cpp + NCCL.
```

---

## 3) Verify the Release

In a Kaggle notebook:

```python
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0
import llamatelemetry
print(llamatelemetry.__version__)
```

Then confirm the binaries auto-download on first import.

---

## 4) Optional: HuggingFace Mirror

If mirroring the bundle to HF, ensure the same filename and checksum are used.

---

## Notes

- GitHub Releases is the primary distribution channel for v2.2.0.
- Do not publish to PyPI for v2.2.0.
