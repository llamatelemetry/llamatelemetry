# Structure

## Top-Level Layout (selected)
- `llamatelemetry/` — Python SDK package (primary).
- `core/` — Python wrapper for native CUDA extension (`llamatelemetry_cpp`).
- `csrc/` — C++/CUDA sources for the native extension.
- `docs/` — mkdocs documentation, guides, references.
- `examples/` — runnable examples.
- `notebooks/` — Jupyter/Kaggle notebooks.
- `tests/` — pytest-based test suite.
- `releases/` — release artifacts and metadata.
- `dist/` — built distributions (if present).

## Python Package Layout (`llamatelemetry/`)
- `__init__.py` — InferenceEngine, auto-bootstrap, env setup, quick helpers.
- `_internal/` — bootstrap + model registry.
- `api/` — LlamaCppClient, GGUF tools, multi-GPU config, NCCL utilities.
- `telemetry/` — OpenTelemetry tracing, metrics, exporters, instrumentors.
- `kaggle/` — Kaggle presets, secrets, environment setup.
- `graphistry/` — Graphistry/RAPIDS visualization and workload helpers.
- `inference/` — advanced inference helpers (FlashAttention, KV cache, batching).
- `quantization/` — quantization helpers (GGUF, NF4, dynamic).
- `cuda/` — CUDA helper utilities (graphs, tensor core, triton kernels).
- `chat.py` — chat history + OpenAI-compatible chat handling.
- `embeddings.py` — embeddings wrapper helpers.
- `models.py` — model discovery and metadata parsing.
- `gguf_parser.py` — GGUF file parsing (memory-mapped).
- `utils.py` — environment detection + auto-configuration.

## Native Extension
- `csrc/bindings.cpp` — pybind bindings for `llamatelemetry_cpp`.
- `csrc/core/*` — device + tensor implementations.
- `csrc/ops/*` — math ops (matmul/batched).

## Tests
- `tests/test_llamatelemetry.py` — SDK import, server manager, model loading.
- `tests/test_new_apis.py` — new API surfaces (telemetry, GGUF, multi-GPU).
- `tests/test_tensor_api.py` — CUDA tensor extension tests.
- `tests/test_gguf_parser.py` — GGUF parser validation.
- `tests/test_full_workflow.py` — end-to-end flows.
