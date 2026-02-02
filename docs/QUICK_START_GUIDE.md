# llamatelemetry v2.2.0 - Quick Start (Kaggle Dual T4)

Get running in 5 minutes on Kaggle dual Tesla T4 GPUs.

## Requirements

- **Platform:** Kaggle notebooks only (GPU T4 x2)
- **Python:** 3.11+
- **CUDA:** 12.x runtime (pre-installed on Kaggle)
- **Internet:** Enabled

**Note:** llamatelemetry v2.2.0 is distributed via GitHub (not PyPI).

---

## Step 1: Install llamatelemetry

```bash
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0
```

On first import, llamatelemetry auto-downloads the CUDA binaries (~961 MB) from GitHub Releases.

---

## Step 2: Download a GGUF Model (1B-5B)

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",
    local_dir="/kaggle/working/models",
)
```

---

## Step 3: Start llama-server (GPU 0)

```python
from llamatelemetry.server import ServerManager

server = ServerManager()
server.start_server(
    model_path=model_path,
    gpu_layers=99,
    tensor_split="1.0,0.0",  # GPU 0 for LLM, GPU 1 free for Graphistry/RAPIDS
    flash_attn=1,
)
```

---

## Step 4: Run Inference

```python
import llamatelemetry

engine = llamatelemetry.InferenceEngine()
engine.load_model(model_path, auto_start=False)
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
```

---

## Step 5: Cleanup

```python
server.stop_server()
```

---

## Next Steps

- **Multi-GPU tensor-split:** `docs/KAGGLE_GUIDE.md`
- **Visualization trilogy:** `notebooks/README.md`
- **Server configuration:** `docs/API_REFERENCE.md`
