# llamatelemetry Examples

This directory contains example scripts and notebooks for using llamatelemetry.

## Kaggle Examples (v2.2.0)

### Gemma 3-1B + Unsloth Tutorial
Complete tutorial for using llamatelemetry v2.2.0 with Gemma 3-1B on Kaggle dual Tesla T4.

**Features demonstrated:**
- Binary auto-download on first import
- Loading GGUF models from HuggingFace
- Quantization API usage
- Unsloth integration workflow
- Inference and batch inference
- Performance metrics and optimization
- Verified 134 tok/s on Tesla T4

**Notebooks:**
1. [llamatelemetry_v2_1_0_gemma3_1b_unsloth_colab.ipynb](../notebooks/llamatelemetry_v2_1_0_gemma3_1b_unsloth_colab.ipynb) - Tutorial notebook
2. [llamatelemetry_v2_1_0_gemma3_1b_unsloth_colab_executed.ipynb](../notebooks/llamatelemetry_v2_1_0_gemma3_1b_unsloth_colab_executed.ipynb) - Executed with results

**Recommended Runtime:**
- Python 3.11+
- Dual Tesla T4 GPUs (Kaggle)

**Usage:**
1. Open the notebook from `notebooks/` in Kaggle
2. Select Accelerator → GPU T4 x2
3. Run all cells

### Legacy Examples (v1.x)

For legacy v1.x examples, see the [archive/v1.x/examples](../archive/v1.x/examples) directory.

## Local Examples

More examples coming soon for local development and production deployments.

## Contributing

Have an example you'd like to share? Submit a pull request!

---

**Links:**
- [llamatelemetry on GitHub](https://github.com/llamatelemetry/llamatelemetry)
- [Documentation](https://llamatelemetry.github.io/)
