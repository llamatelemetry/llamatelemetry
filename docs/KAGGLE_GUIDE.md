# Kaggle Guide (Dual T4)

This workflow targets Kaggle notebooks with **GPU T4 x2**. `llamatelemetry` is optimized for this environment and can automatically configure split-GPU workloads.

## 1) Enable GPUs in Kaggle

- Notebook Settings → Accelerator → GPU (T4 x2)

## 2) Install

```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

## 3) Validate the environment

```python
from llamatelemetry import detect_cuda
print(detect_cuda())
```

## 4) Use Kaggle presets

```python
from llamatelemetry.api import kaggle_t4_dual_config

cfg = kaggle_t4_dual_config()
print(cfg)
```

## 5) Load a GGUF model

```python
import llamatelemetry as lt

engine = lt.InferenceEngine(enable_telemetry=False)
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
```

## 6) Run inference

```python
result = engine.infer("Summarize CUDA graphs.", max_tokens=96)
print(result.text)
```

## 7) Optional: telemetry and metrics

```python
engine = lt.InferenceEngine(
    enable_telemetry=True,
    telemetry_config={
        "service_name": "llamatelemetry-kaggle",
        "enable_llama_metrics": True,
    },
)
```
