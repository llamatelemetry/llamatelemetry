# Troubleshooting

## CUDA not detected

Symptoms:

- `detect_cuda()` returns `available: False`
- `llama-server` fails to start

Fixes:

- Ensure NVIDIA drivers are installed
- Run `nvidia-smi` and confirm GPUs are visible
- Verify you are not in a CPU-only runtime

## `llama-server` not found

Symptoms:

- `ServerManager.find_llama_server()` returns `None`
- `InferenceEngine.load_model` raises a runtime error

Fixes:

- Set `LLAMA_SERVER_PATH` to the binary location
- Set `LLAMA_CPP_DIR` if you built llama.cpp manually
- Reinstall the package to trigger bootstrap download

## Missing shared libraries

Symptoms:

- `llama-server` fails with `libnccl.so` or other missing libs

Fixes:

- Ensure `LD_LIBRARY_PATH` includes the `llamatelemetry/lib` directory
- Re-import `llamatelemetry` to re-run bootstrap

## Model download issues

Symptoms:

- Registry download fails or hangs

Fixes:

- Verify internet access in the runtime
- Provide a local GGUF path instead
- Use HuggingFace token if the model is gated

## OpenTelemetry not available

Symptoms:

- `setup_telemetry()` returns `(None, None)`

Fixes:

- Install OTel SDK/exporters:
  - `pip install opentelemetry-api opentelemetry-sdk`
  - `pip install opentelemetry-exporter-otlp-proto-grpc`

## Kaggle-specific issues

- Ensure the notebook accelerator is set to GPU (T4 x2)
- Avoid `pip install` in every cell; use a single install cell
