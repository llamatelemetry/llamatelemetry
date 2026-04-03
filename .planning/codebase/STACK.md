# Stack

## Summary
- Python SDK targeting Python 3.11+ with optional Jupyter/telemetry/graphistry extras.
- Native CUDA/C++ extension via `pybind11` (`llamatelemetry_cpp`).
- Runtime depends on `llama-server` (llama.cpp) binaries with CUDA 12.x support.

## Languages & Runtime
- Python 3.11+ (primary SDK) — `pyproject.toml`.
- C++/CUDA (native extension) — `csrc/`, `CMakeLists.txt`.

## Core Dependencies (runtime)
- `numpy`, `requests`, `huggingface_hub`, `tqdm` — `pyproject.toml`.
- `opentelemetry-api`, `opentelemetry-sdk` — `pyproject.toml`.

## Optional Dependencies
- Telemetry exporters/instrumentation: `opentelemetry-exporter-otlp-*`, `opentelemetry-instrumentation` — `pyproject.toml`.
- Graphistry/RAPIDS support: `pygraphistry`, `pandas` — `pyproject.toml`.
- Jupyter UI: `ipywidgets`, `IPython`, `matplotlib`, `pandas` — `pyproject.toml`.
- Dev tooling: `pytest`, `pytest-cov`, `black`, `mypy`, `pybind11` — `pyproject.toml`.

## Native/CUDA Components
- Pybind extension module `llamatelemetry_cpp` — `csrc/bindings.cpp`.
- Core tensor/device APIs — `csrc/core/`, `core/__init__.py`.
- CUDA ops: `matmul`, `batched_matmul` — `csrc/ops/`.

## External Binaries
- `llama-server` binary (llama.cpp) auto-discovered and/or auto-downloaded — `llamatelemetry/server.py`, `llamatelemetry/_internal/bootstrap.py`.
- Bundled CUDA libs and binaries are expected in package directories (`llamatelemetry/binaries`, `llamatelemetry/lib`) — `llamatelemetry/__init__.py`.

## Build & Packaging
- `setuptools`/`wheel` build — `pyproject.toml`.
- Build config for native extension — `CMakeLists.txt` + `csrc/`.
- Package discovery excludes large artifacts (`binaries`, `lib`, `models`) — `pyproject.toml`.

## Configuration Surface
- Environment variables: `LLAMA_SERVER_PATH`, `LLAMA_CPP_DIR`, `CUDA_VISIBLE_DEVICES`, `LD_LIBRARY_PATH` — `llamatelemetry/__init__.py`, `llamatelemetry/utils.py`.
- OTEL exporter env vars: `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS` — `llamatelemetry/telemetry/__init__.py`.

## Entry Points / Primary API
- `InferenceEngine` (high-level inference facade) — `llamatelemetry/__init__.py`.
- `ServerManager` (server lifecycle) — `llamatelemetry/server.py`.
- `LlamaCppClient` (full HTTP API client) — `llamatelemetry/api/client.py`.
- Telemetry setup — `llamatelemetry/telemetry/__init__.py`.
