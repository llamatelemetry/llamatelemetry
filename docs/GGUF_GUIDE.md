# GGUF Guide

This guide covers GGUF model inspection, validation, and quantization utilities.

## Inspecting a GGUF file

```python
from llamatelemetry.gguf_parser import GGUFReader

with GGUFReader("model.gguf") as reader:
    print(reader.metadata)
    print(len(reader.tensors))
```

## Validating and summarizing

```python
from llamatelemetry.api.gguf import validate_gguf, get_model_summary

print(validate_gguf("model.gguf"))
print(get_model_summary("model.gguf"))
```

## Quantization helpers

```python
from llamatelemetry.api.gguf import get_recommended_quant

print(get_recommended_quant(vram_gb=8, params_b=3))
```
