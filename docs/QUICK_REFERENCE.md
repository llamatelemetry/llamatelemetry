# Quick Reference

## Install

```bash
pip install --no-cache-dir --force-reinstall \
  git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

## Quick inference

```python
import llamatelemetry as lt
engine = lt.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
print(engine.infer("Hello", max_tokens=32).text)
```

## Telemetry

```python
from llamatelemetry.telemetry import setup_telemetry
setup_telemetry(service_name="llamatelemetry", enable_llama_metrics=True)
```

## Kaggle preset

```python
from llamatelemetry.api import kaggle_t4_dual_config
print(kaggle_t4_dual_config())
```
