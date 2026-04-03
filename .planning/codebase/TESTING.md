# Testing

## Test Framework
- `pytest` with `pytest-cov` optional.
- Config in `pyproject.toml` (`[tool.pytest.ini_options]`).

## Test Suites
- `tests/test_llamatelemetry.py` — import/version, GPU detection, server manager, binary download, inference engine workflow.
- `tests/test_new_apis.py` — API surface for GGUF tools, telemetry, multi-GPU helpers.
- `tests/test_tensor_api.py` — native CUDA tensor API basics.
- `tests/test_gguf_parser.py` — GGUF parser correctness.
- `tests/test_full_workflow.py` — end-to-end workflow coverage.
- `tests/test_end_to_end.py` — integration-level tests (may require server/binaries).

## Environment Dependencies
- Many tests assume access to `nvidia-smi` or CUDA devices.
- Some tests may download large binaries or GGUF models; network access is expected.

## Typical Commands
- Run all tests: `pytest`
- Run a single file: `pytest tests/test_llamatelemetry.py`

## Known Constraints
- GPU/NCCL tests are environment-sensitive and may skip or fail on CPU-only machines.
- Tests that download binaries may be flaky without internet access.
