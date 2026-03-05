# Bug Report: multi_gpu_config leaks into llama-server CLI

## Summary
Passing `multi_gpu_config` into `InferenceEngine.load_model()` causes llama-server to receive an invalid flag:

```
--multi-gpu-config
```

This results in:

```
error: invalid argument: --multi-gpu-config
```

## Repro (Kaggle Dual T4)
```python
from llamatelemetry.api.multigpu import MultiGPUConfig, SplitMode
from llamatelemetry.api.nccl import NCCLConfig
import llamatelemetry as lt

nccl_config = NCCLConfig(gpu_ids=[0, 1])
nccl_config.apply_env()

multi_gpu_config = MultiGPUConfig(
    n_gpu_layers=-1,
    split_mode=SplitMode.LAYER,
    tensor_split=[0.5, 0.5],
    ctx_size=4096,
    batch_size=1024,
    ubatch_size=256,
)

engine = lt.InferenceEngine(enable_telemetry=False)
engine.load_model(
    "llama-3.2-3b-Q4_K_M",
    auto_start=True,
    multi_gpu_config=multi_gpu_config,
    nccl_config=nccl_config,
    enable_metrics=True,
    enable_props=True,
    enable_slots=True,
)
```

## Expected
`multi_gpu_config` should be consumed by the SDK and translated into supported
llama-server flags like:

- `--split-mode`
- `--tensor-split`
- `--main-gpu`

## Actual
`multi_gpu_config` remains in `kwargs` and gets converted into:

```
--multi-gpu-config
```

which llama-server rejects.

## Suggested Fix
Ensure `multi_gpu_config` and `nccl_config` are not forwarded as raw kwargs when
building the llama-server CLI.

Candidate fixes:
- `InferenceEngine.load_model`: `kwargs.pop("multi_gpu_config", None)` and `kwargs.pop("nccl_config", None)` before calling `ServerManager.start_server`.
- Or, filter these in `ServerManager.start_server` before CLI arg construction.
