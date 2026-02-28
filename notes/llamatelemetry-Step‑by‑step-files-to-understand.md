# Llamatelemetry Step-By-Step Files To Understand

Step‑by‑step path to understand the entire project (pragmatic order)

  1. Start with the big picture
     Read README.md, CHANGELOG.md, docs/ entry points, and mkdocs.yml to understand the intent, audience, and scope.
  2. Understand the package entry point
     Read llamatelemetry/__init__.py end‑to‑end. This shows the runtime bootstrap, environment setup, and main public API surface.
  3. Learn the inference lifecycle
     Read llamatelemetry/server.py (server management) and the InferenceEngine class in llamatelemetry/__init__.py.
     Read llamatelemetry/models.py and llamatelemetry/gguf_parser.py to understand how GGUF models are found, parsed, and configured.
  5. Trace the telemetry stack
     Read llamatelemetry/telemetry/__init__.py, then telemetry/tracer.py, telemetry/metrics.py, telemetry/resource.py, and telemetry/exporter.py.
     This shows how OTel is integrated and how GPU metrics are collected.
  6. Review the low‑level API client
     Read llamatelemetry/api/client.py and llamatelemetry/api/__init__.py.
     This is the complete llama.cpp server API surface, including OpenAI‑compatible endpoints.
  7. Understand multi‑GPU and NCCL
     Read llamatelemetry/api/multigpu.py and llamatelemetry/api/nccl.py.
     This explains split‑GPU behavior and NCCL wiring.
  8. Learn the Kaggle workflow (core use case)
     Read llamatelemetry/kaggle/environment.py, kaggle/presets.py, kaggle/gpu_context.py, and kaggle/secrets.py.
     This is the “zero‑boilerplate” path the project is designed around.
  9. Explore Graphistry and Unsloth integrations
     Read llamatelemetry/graphistry/* and llamatelemetry/unsloth/* for optional integrations.
  10. Review advanced inference helpers
     Read llamatelemetry/inference/* and llamatelemetry/cuda/* to understand optional performance features.
  11. Walk the scripts and notebooks
     Read scripts/, notebooks/, and notebooks-local/ to see practical workflows and packaging logic.
  12. Close the loop with tests and examples
     Review tests/ and examples/ to see how the SDK is meant to be used and validated.
