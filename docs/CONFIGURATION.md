# Configuration

`llamatelemetry` relies on environment variables and explicit function arguments for configuration.

## Environment variables

- `LLAMA_SERVER_PATH` — explicit path to `llama-server` binary
- `LLAMA_CPP_DIR` — custom llama.cpp build directory
- `CUDA_VISIBLE_DEVICES` — GPU visibility control
- `LD_LIBRARY_PATH` — includes bundled libraries

## InferenceEngine options

```python
engine = InferenceEngine(
    server_url="http://127.0.0.1:8090",
    enable_telemetry=True,
    telemetry_config={
        "service_name": "llamatelemetry",
        "otlp_endpoint": "http://localhost:4317",
        "enable_llama_metrics": True,
    },
)
```

## Model loading options

```python
engine.load_model(
    model_name_or_path="gemma-3-1b-Q4_K_M",
    gpu_layers=40,
    ctx_size=2048,
    auto_start=True,
    n_parallel=2,
)
```
