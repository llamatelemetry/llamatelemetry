# llamatelemetry v2.2.0 - GGUF Guide

Complete guide to GGUF format and quantization for llamatelemetry.

---

## Table of Contents

- [What is GGUF?](#what-is-gguf)
- [Quantization Types](#quantization-types)
- [K-Quants](#k-quants)
- [I-Quants](#i-quants)
- [VRAM Requirements](#vram-requirements)
- [Quality vs Size Comparison](#quality-vs-size-comparison)
- [Converting from Unsloth](#converting-from-unsloth)
- [GGUF Parsing with llamatelemetry](#gguf-parsing-with-llamatelemetry)
- [Best Practices](#best-practices)
- [Model Sources](#model-sources)

---

## What is GGUF?

GGUF (GPT-Generated Unified Format) is the standard format for running LLMs with llama.cpp. It's optimized for:

- **Efficient loading** - Fast startup times
- **Quantization** - Reduced memory usage
- **Portability** - Single file, no dependencies
- **Metadata** - Model info embedded in file

### GGUF vs Other Formats

| Format | Engine | Quantization | Use Case |
|--------|--------|--------------|----------|
| **GGUF** | llama.cpp | K/I-quants | CPU/GPU inference |
| SafeTensors | PyTorch/HF | FP16/BF16 | Training, fine-tuning |
| ONNX | ONNX Runtime | INT8 | Production deployment |
| TensorRT | NVIDIA | INT8/FP16 | NVIDIA optimized |

---

## Quantization Types

### Overview

Quantization reduces model precision to save memory:

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUANTIZATION SPECTRUM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   FP32 ─→ FP16 ─→ Q8 ─→ Q6 ─→ Q5 ─→ Q4 ─→ Q3 ─→ Q2 ─→ IQ1     │
│    │        │      │     │     │     │     │     │      │       │
│  32 bit  16 bit  8 bit 6 bit 5 bit 4 bit 3 bit 2 bit  1.x bit   │
│   100%    50%   25%   19%   16%   12%   9%   6%     5%          │
│                                                                 │
│  ◀───── Higher Quality          Smaller Size ────▶              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Two Quantization Families

| Family | Types | Best For |
|--------|-------|----------|
| **K-Quants** | Q4_K_M, Q5_K_M, Q6_K | Most models, best balance |
| **I-Quants** | IQ3_XS, IQ4_XS | Very large models (70B+) |

---

## K-Quants

K-Quants use k-means clustering for better quality at each bit level.

### K-Quant Types

| Type | Bits/Weight | Size Factor | Quality | Speed |
|------|-------------|-------------|---------|-------|
| **Q8_0** | 8.5 | 0.53× | ⭐⭐⭐⭐⭐ | Fastest |
| **Q6_K** | 6.6 | 0.41× | ⭐⭐⭐⭐⭐ | Fast |
| **Q5_K_M** | 5.7 | 0.35× | ⭐⭐⭐⭐ | Fast |
| **Q5_K_S** | 5.5 | 0.34× | ⭐⭐⭐⭐ | Fast |
| **Q4_K_M** | 4.8 | 0.30× | ⭐⭐⭐⭐ | Fast |
| **Q4_K_S** | 4.5 | 0.28× | ⭐⭐⭐ | Fastest |
| **Q3_K_M** | 3.9 | 0.24× | ⭐⭐⭐ | Medium |
| **Q3_K_S** | 3.5 | 0.22× | ⭐⭐ | Medium |
| **Q2_K** | 3.4 | 0.21× | ⭐⭐ | Slow |

### Recommended K-Quants

| Scenario | Recommended | Why |
|----------|-------------|-----|
| **Quality-critical** | Q6_K | Near-lossless quality |
| **Balanced (default)** | Q4_K_M | Best quality/size ratio |
| **Memory-limited** | Q4_K_S | Smaller, acceptable quality |
| **Extreme compression** | Q3_K_M | Usable quality |

### K-Quant Naming Convention

```
Q{bits}_K_{size}

bits = quantization bits (2-8)
K    = k-means quantization
size = S (small), M (medium), L (large) - refers to quality
```

---

## I-Quants

I-Quants (Importance Matrix Quants) provide even more compression by using importance-weighted quantization.

### I-Quant Types

| Type | Bits/Weight | Size Factor | Quality | Use Case |
|------|-------------|-------------|---------|----------|
| **IQ4_XS** | 4.25 | 0.26× | ⭐⭐⭐⭐ | Quality at 4-bit |
| **IQ4_NL** | 4.5 | 0.28× | ⭐⭐⭐⭐ | Non-linear 4-bit |
| **IQ3_M** | 3.4 | 0.21× | ⭐⭐⭐⭐ | Best 3-bit |
| **IQ3_S** | 3.25 | 0.20× | ⭐⭐⭐ | Smaller 3-bit |
| **IQ3_XS** | 3.0 | 0.19× | ⭐⭐⭐ | 70B on 30GB |
| **IQ3_XXS** | 2.75 | 0.17× | ⭐⭐ | Extreme 3-bit |
| **IQ2_M** | 2.5 | 0.16× | ⭐⭐ | 2-bit medium |
| **IQ2_S** | 2.3 | 0.14× | ⭐⭐ | 2-bit small |
| **IQ2_XS** | 2.2 | 0.14× | ⭐ | 2-bit extra small |
| **IQ2_XXS** | 2.0 | 0.12× | ⭐ | 70B on 24GB |
| **IQ1_M** | 1.75 | 0.11× | ⭐ | Extreme 1-bit |
| **IQ1_S** | 1.5 | 0.09× | ⚠️ | Research only |

### When to Use I-Quants

| Model Size | VRAM Limit | Recommendation |
|------------|------------|----------------|
| 7-13B | 8GB | Q4_K_M (K-quant) |
| 32-34B | 24GB | Q4_K_M (K-quant) |
| 70B | 30GB | IQ3_XS (I-quant) |
| 70B | 24GB | IQ2_XXS (I-quant) |
| 70B | 48GB+ | Q4_K_M (K-quant) |

---

## VRAM Requirements

### Formula

```
VRAM ≈ (Parameters × BitsPerWeight / 8) + ContextBuffer

Where:
- Parameters: Model parameter count (e.g., 7B = 7 billion)
- BitsPerWeight: From quantization type
- ContextBuffer: ~1-4 GB depending on context size
```

### Practical VRAM Table

| Model | Q8_0 | Q6_K | Q4_K_M | Q3_K_M | IQ3_XS |
|-------|------|------|--------|--------|--------|
| **1B** | 1.5 GB | 1.2 GB | 1.0 GB | 0.8 GB | - |
| **3B** | 4.0 GB | 3.2 GB | 2.5 GB | 2.0 GB | - |
| **7B** | 8.5 GB | 6.8 GB | 5.5 GB | 4.5 GB | - |
| **8B** | 9.5 GB | 7.6 GB | 6.0 GB | 5.0 GB | - |
| **13B** | 14.5 GB | 11.5 GB | 9.0 GB | 7.5 GB | - |
| **32B** | 36 GB | 28 GB | 22 GB | 18 GB | - |
| **70B** | 75 GB | 60 GB | 42 GB | 35 GB | 25 GB |

### Kaggle Dual T4 (30GB) Fit Guide

| ✅ Fits | ❌ Doesn't Fit |
|---------|----------------|
| 70B IQ3_XS (25GB) | 70B Q4_K_M (42GB) |
| 70B IQ2_XXS (18GB) | 70B Q3_K_M (35GB) |
| 32B Q4_K_M (22GB) | 70B Q8_0 (75GB) |
| 13B Q8_0 (14.5GB) | - |

---

## Quality vs Size Comparison

### 7B Model Comparison

| Quantization | File Size | VRAM | Perplexity | Quality Loss |
|--------------|-----------|------|------------|--------------|
| FP16 | 14.0 GB | 15+ GB | 5.50 | 0% |
| Q8_0 | 7.5 GB | 8.5 GB | 5.51 | ~0.2% |
| Q6_K | 5.8 GB | 6.8 GB | 5.53 | ~0.5% |
| Q5_K_M | 5.0 GB | 6.0 GB | 5.57 | ~1.3% |
| Q4_K_M | 4.2 GB | 5.5 GB | 5.62 | ~2.2% |
| Q3_K_M | 3.3 GB | 4.5 GB | 5.80 | ~5.5% |
| Q2_K | 2.8 GB | 3.8 GB | 6.10 | ~11% |

### Quality Tiers

```
┌─────────────────────────────────────────────────────────────────┐
│                        QUALITY TIERS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Tier 1 (Production)     │ Q6_K, Q8_0                         │
│   ─────────────────────   │ < 1% quality loss                  │
│                                                                 │
│   Tier 2 (Recommended)    │ Q4_K_M, Q5_K_M                     │
│   ─────────────────────   │ 1-3% quality loss                  │
│                                                                 │
│   Tier 3 (Acceptable)     │ Q3_K_M, IQ3_XS                     │
│   ─────────────────────   │ 3-6% quality loss                  │
│                                                                 │
│   Tier 4 (Extreme)        │ Q2_K, IQ2_XXS                      │
│   ─────────────────────   │ 10%+ quality loss                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Converting from Unsloth

### Unsloth Fine-tuned Model to GGUF

```python
# After fine-tuning with Unsloth
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(...)

# Train your model...

# Save as GGUF
model.save_pretrained_gguf(
    "output_dir",
    tokenizer,
    quantization_method="q4_k_m",  # or other quant type
)
```

### Available Quantization Methods in Unsloth

```python
# K-Quants
quantization_method = "q8_0"      # 8-bit
quantization_method = "q6_k"      # 6-bit k-means
quantization_method = "q5_k_m"    # 5-bit k-means medium
quantization_method = "q4_k_m"    # 4-bit k-means medium (recommended)
quantization_method = "q3_k_m"    # 3-bit k-means medium

# Multiple at once
model.save_pretrained_gguf(
    "output_dir",
    tokenizer,
    quantization_method=["q4_k_m", "q8_0"],  # Creates both
)
```

### Manual Conversion with llama.cpp

```bash
# 1. Convert to GGUF F16
python convert_hf_to_gguf.py /path/to/model --outtype f16

# 2. Quantize
./bin/llama-quantize input.gguf output.gguf Q4_K_M
```

---

## GGUF Parsing with llamatelemetry

llamatelemetry provides a built-in GGUF parser for model inspection.

### Basic Usage

```python
from llamatelemetry.api.gguf import GGUFParser

# Parse GGUF file
parser = GGUFParser("model.gguf")
info = parser.parse()

# Model information
print(f"Parameters: {info['parameters']:,}")
print(f"File size: {info['file_size'] / 1024**3:.2f} GB")
print(f"Quantization: {info['quantization']}")
print(f"Architecture: {info['architecture']}")
print(f"Context size: {info['context_size']}")
```

### Quantization Info Table

```python
from llamatelemetry.api.gguf import QUANT_TYPE_INFO

# Print all supported quantization types
for qtype, info in QUANT_TYPE_INFO.items():
    print(f"{qtype:12} | {info['bits']:.2f} bits | {info['description']}")
```

Output:
```
Q8_0         | 8.50 bits | 8-bit round-to-nearest
Q6_K         | 6.56 bits | 6-bit k-quant
Q5_K_M       | 5.69 bits | 5-bit k-quant medium
Q4_K_M       | 4.83 bits | 4-bit k-quant medium
Q3_K_M       | 3.91 bits | 3-bit k-quant medium
IQ4_XS       | 4.25 bits | 4-bit i-quant extra small
IQ3_XS       | 3.00 bits | 3-bit i-quant extra small
IQ2_XXS      | 2.00 bits | 2-bit i-quant double extra small
```

### VRAM Estimation

```python
from llamatelemetry.api.gguf import GGUFParser

parser = GGUFParser("model.gguf")
info = parser.parse()

# Estimate VRAM for different context sizes
for ctx in [2048, 4096, 8192]:
    vram = parser.estimate_vram(context_size=ctx)
    print(f"Context {ctx}: {vram:.2f} GB VRAM")
```

### Multi-GPU Fit Check

```python
from llamatelemetry.api.gguf import GGUFParser

parser = GGUFParser("70b-model.gguf")

# Check if fits on Kaggle dual T4
total_vram = 30  # 15 GB × 2
estimated = parser.estimate_vram(context_size=4096)

if estimated <= total_vram * 0.9:  # 90% threshold
    print(f"✅ Model fits ({estimated:.1f} GB / {total_vram} GB)")
else:
    print(f"❌ Model too large ({estimated:.1f} GB / {total_vram} GB)")
```

---

## Best Practices

### 1. Choosing Quantization

```python
def recommend_quant(model_params_b: float, vram_gb: float) -> str:
    """Recommend quantization based on model size and VRAM."""
    
    # Rough estimates for VRAM per billion parameters
    vram_per_b = {
        "Q8_0": 1.1,
        "Q6_K": 0.85,
        "Q5_K_M": 0.75,
        "Q4_K_M": 0.65,
        "Q3_K_M": 0.55,
        "IQ3_XS": 0.40,
        "IQ2_XXS": 0.30,
    }
    
    for quant, factor in vram_per_b.items():
        estimated = model_params_b * factor + 2  # +2 GB for context
        if estimated <= vram_gb * 0.9:
            return quant
    
    return "Model too large"

# Example
print(recommend_quant(70, 30))  # -> IQ3_XS
print(recommend_quant(7, 15))   # -> Q4_K_M
```

### 2. Quality Validation

```python
# Test quantization quality with sample prompts
test_prompts = [
    "What is 25 * 48?",           # Math
    "Explain quantum computing",   # Knowledge
    "Write a haiku about AI",      # Creativity
]

for prompt in test_prompts:
    response = client.generate(prompt=prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response.text}\n")
```

### 3. Speed vs Quality Trade-off

| Priority | Recommendation |
|----------|----------------|
| **Speed** | Q4_K_S - fastest inference |
| **Quality** | Q6_K - near-lossless |
| **Balance** | Q4_K_M - best trade-off |
| **Memory** | IQ3_XS - for large models |

---

## Model Sources

### Recommended Repositories

| Repository | Focus |
|------------|-------|
| [TheBloke](https://huggingface.co/TheBloke) | Classic GGUF models |
| [bartowski](https://huggingface.co/bartowski) | Latest models, good I-quants |
| [mradermacher](https://huggingface.co/mradermacher) | Wide variety |
| [Unsloth](https://huggingface.co/unsloth) | Optimized inference models |
| [QuantFactory](https://huggingface.co/QuantFactory) | Multiple quant options |

### Download Example

```python
from huggingface_hub import hf_hub_download

# Download Q4_K_M for general use
model_path = hf_hub_download(
    repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
    filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    local_dir="./models",
)

# Download IQ3_XS for 70B on limited VRAM
model_path = hf_hub_download(
    repo_id="bartowski/Llama-3.1-70B-Instruct-GGUF",
    filename="Llama-3.1-70B-Instruct-IQ3_XS.gguf",
    local_dir="./models",
)
```

---

## Next Steps

- **[Kaggle Guide](KAGGLE_GUIDE.md)** - Running GGUF on Kaggle
- **[Configuration Guide](CONFIGURATION.md)** - Server configuration
- **[API Reference](API_REFERENCE.md)** - GGUF parsing API
