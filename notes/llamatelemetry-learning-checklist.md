# llamatelemetry Learning Checklist

Generated: 2026-02-28T10:09:29
Project root: /media/waqasm86/External1/Project-Nvidia-Office/Project-Llamatelemetry/llamatelemetry

## Phase 1: Understand the intent
1. Read README and summarize the primary user promise in 3 lines.
2. Scan CHANGELOG and list major capabilities in v0.1.0.
3. Note Python version and dependencies in pyproject.toml.

## Phase 2: Core runtime
1. Trace how `InferenceEngine` boots: env vars, bootstrap, model loading.
2. Map the server lifecycle in `ServerManager`.
3. Identify where inference requests are sent and responses parsed.

## Phase 3: Telemetry
1. Explain how `setup_telemetry` wires TracerProvider and MeterProvider.
2. List all metrics collected in `telemetry/metrics.py`.
3. Identify where spans are annotated with LLM attributes.

## Phase 4: API client
1. List which llama.cpp endpoints are supported in `api/client.py`.
2. Confirm how streaming is handled and what dependencies are optional.

## Phase 5: Multi-GPU and NCCL
1. Summarize `MultiGPUConfig` flags and how they map to CLI args.
2. Identify NCCL discovery and setup functions in `api/nccl.py`.

## Phase 6: Kaggle flow
1. Read `KaggleEnvironment.setup()` and document the exact flow.
2. List presets and recommended settings for dual T4.

## Phase 7: Integrations
1. Graphistry: note registration and graph export utilities.
2. Unsloth: note the path from fine-tune to GGUF export.

## Phase 8: Docs + notebooks
1. Skim docs for the user-facing narrative and API expectations.
2. Run through at least one notebook to verify runtime assumptions.

## Output you should have when done
1. A 1-page architecture summary.
2. A flowchart of model loading → server start → inference → telemetry.
3. A shortlist of files you would modify for a new feature.
