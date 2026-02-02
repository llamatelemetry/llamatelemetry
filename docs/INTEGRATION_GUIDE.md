# llamatelemetry v2.2.0 Integration Guide (Kaggle Dual T4)

This guide explains how llamatelemetry v2.2.0 finds and starts `llama-server` in **Kaggle dual T4** notebooks.

---

## 1) Where `llama-server` Comes From

- On first import, llamatelemetry downloads the binary bundle:
  - `llamatelemetry-v2.2.0-cuda12-kaggle-t4x2.tar.gz`
- It is extracted to `~/.cache/llamatelemetry`.

---

## 2) Detection Order

`ServerManager.find_llama_server()` searches in this order:

1. `LLAMA_SERVER_PATH` environment variable
2. `llamatelemetry` package binaries
3. `~/.cache/llamatelemetry/`
4. PATH lookup
5. Download bundle (last resort)

---

## 3) Typical Kaggle Usage

```python
from llamatelemetry.server import ServerManager

server = ServerManager()
server.start_server(
    model_path="model.gguf",
    gpu_layers=99,
    tensor_split="1.0,0.0",
    flash_attn=1,
)
```

---

## 4) Troubleshooting

- **Binary missing**: delete `~/.cache/llamatelemetry` and re-import llamatelemetry.
- **Wrong GPU**: ensure Kaggle is set to GPU T4 x2.

---

For installation, see `docs/INSTALLATION.md`.
