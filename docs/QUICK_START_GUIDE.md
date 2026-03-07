# Quick Start Guide

This walkthrough loads a small GGUF model, launches `llama-server`, and runs inference.

## 1) Install

```bash
pip install --no-cache-dir --force-reinstall \
  git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

## 2) Verify CUDA

```python
from llamatelemetry import detect_cuda

cuda_info = detect_cuda()
print(cuda_info)
```

## 3) Load a model

```python
import llamatelemetry as lt

engine = lt.InferenceEngine(enable_telemetry=False)
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
```

## 4) Run inference

```python
result = engine.generate("What is CUDA?", max_tokens=64)
print(result.text)
```

## 5) Batch inference

```python
prompts = [
    "Explain tensor cores in one sentence.",
    "What is llama.cpp?",
    "How does GGUF quantization work?",
]
results = engine.batch_infer(prompts, max_tokens=64)
for r in results:
    print(r.text)
```

## 6) Fetch metrics

```python
metrics = engine.get_metrics()
print(metrics)
```

## 7) Clean up

```python
engine.unload_model()
```
