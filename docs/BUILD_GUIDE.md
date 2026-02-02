# llamatelemetry v2.2.0 Build Guide (Kaggle Dual T4)

This guide is scoped to **llamatelemetry v2.2.0** and the **Kaggle dual Tesla T4** target.

**Runtime Target:** Kaggle notebooks only (GPU T4 x2)

---

## When You Need This

Only use this guide if you are **rebuilding binaries** for the v2.2.0 Kaggle target. Most users should rely on the prebuilt GitHub Release bundle.

---

## Build Summary (High-Level)

1. Build llama.cpp with CUDA 12.x for **SM 7.5 (T4)**.
2. Package `bin/` and `lib/` into:
   - `llamatelemetry-v2.2.0-cuda12-kaggle-t4x2.tar.gz`
3. Upload the bundle to GitHub Releases.

---

## Recommended Build Environment

- Kaggle notebook (GPU T4 x2)
- CUDA 12.x
- Python 3.11+

---

## Verification (Kaggle)

```python
!pip install -q --no-cache-dir git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0
import llamatelemetry
print(llamatelemetry.__version__)
```

---

For release steps, see `docs/GITHUB_RELEASE_GUIDE.md`.
