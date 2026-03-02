# Installation

## Requirements

- Python `>= 3.11`
- Linux recommended (Kaggle/Ubuntu optimized)
- NVIDIA GPU with CUDA drivers for high performance
- Optional: OpenTelemetry, Graphistry/RAPIDS, Jupyter, Unsloth

## Install from GitHub (recommended)

```bash
pip install --no-cache-dir --force-reinstall \
  git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

## Install from source (dev)

```bash
git clone https://github.com/llamatelemetry/llamatelemetry.git
cd llamatelemetry
pip install -e .
```

## Optional dependency groups

`pyproject.toml` defines optional extras. You can install them manually in notebook environments as well.

- `telemetry`: OpenTelemetry API/SDK and OTLP exporters
- `graphistry`: Graphistry + pandas integration
- `jupyter`: Jupyter widgets and visualization helpers
- `dev`: testing and formatting tools

## Verify installation

```python
import llamatelemetry as lt

print(lt.__version__)  # expected: 0.1.0
print(lt.check_cuda_available())
```

## Runtime bootstrap behavior

On first import, the SDK may download runtime binaries and shared libraries. This is expected:

- `llamatelemetry/binaries/` contains the `llama-server` runtime
- `llamatelemetry/lib/` contains shared libraries
- `llamatelemetry/models/` is the default model cache

## Environment variables

- `LLAMA_SERVER_PATH` overrides binary discovery
- `LLAMA_CPP_DIR` points to a custom `llama.cpp` build
- `CUDA_VISIBLE_DEVICES` restricts visible GPUs
- `LD_LIBRARY_PATH` is updated by the SDK to include bundled libs

## Windows note

`llamatelemetry` is primarily tested on Linux/Kaggle. Windows users can still experiment, but some workflows and tests are Linux-first.
