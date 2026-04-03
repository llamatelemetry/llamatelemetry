# Concerns & Risks

## Import-Time Side Effects
- `llamatelemetry/__init__.py` mutates environment variables and may trigger bootstrap downloads on import. This can surprise users and complicate reproducibility.

## Large Binary Downloads
- Bootstrap can download ~GB-scale CUDA binaries and GGUF models. This is brittle in constrained or offline environments and may stall imports or tests.
- Files: `llamatelemetry/_internal/bootstrap.py`, `llamatelemetry/server.py`.

## Security: Tar Extraction
- `tarfile.extractall()` is used without path traversal validation. This can be risky if archives are compromised.
- Files: `llamatelemetry/_internal/bootstrap.py`, `llamatelemetry/server.py`.

## Broad Exception Handling
- Many components swallow exceptions silently (telemetry, download checks, server probes), which can hide failure causes and make debugging harder.
- Files: `llamatelemetry/__init__.py`, `llamatelemetry/telemetry/*`, `llamatelemetry/utils.py`.

## Platform/Hardware Assumptions
- Heavy assumptions about NVIDIA GPUs (Tesla T4) and CUDA tooling presence (`nvidia-smi`, `nvcc`).
- Files: `llamatelemetry/_internal/bootstrap.py`, `llamatelemetry/utils.py`, `llamatelemetry/telemetry/metrics.py`.

## API Surface Fragmentation
- Functionality overlaps across `gguf_parser.py`, `api/gguf.py`, and `models.py`, which may lead to drift and duplicated logic.

## Native Extension Build Complexity
- The C++/CUDA extension (`llamatelemetry_cpp`) requires a working CUDA toolchain; installation may fail on environments without it.
- Files: `csrc/*`, `CMakeLists.txt`, `core/__init__.py`.

## Test Fragility
- Tests depend on internet access and GPU availability; end-to-end tests may be flaky in CI or local machines without CUDA.
- Files: `tests/*`.
