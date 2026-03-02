# Build Guide

This guide covers building `llamatelemetry` from source.

## Python package

```bash
pip install -e .
```

## CUDA/C++ extensions

```bash
cmake -S . -B build
cmake --build build
```

## Notes

- The SDK is optimized for CUDA 12 and Kaggle T4 GPUs.
- Ensure you have a compatible CUDA toolkit and compiler.
