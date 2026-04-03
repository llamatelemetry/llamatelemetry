# Conventions

## Language & Style
- Python 3.11+ with type hints and docstrings.
- Formatting via `black` (line length 88) — `pyproject.toml`.
- `dataclasses` used for API response models and configs — `llamatelemetry/api/client.py`, `llamatelemetry/api/multigpu.py`.

## Error Handling
- Many modules use best-effort behavior with broad exception handling and silent fallback (e.g., telemetry, bootstrap, optional deps).
- Network and GPU detection failures typically degrade gracefully instead of raising.

## Optional Dependencies
- Optional imports are wrapped in try/except and gated by availability checks.
- Example: `llamatelemetry/telemetry/__init__.py`, `llamatelemetry/telemetry/instrumentor.py`.

## Environment & Side Effects
- Import-time side effects set env vars (`LD_LIBRARY_PATH`, `LLAMA_SERVER_PATH`) and may bootstrap binaries — `llamatelemetry/__init__.py`.
- Kaggle-specific config flows are centralized in `llamatelemetry/kaggle/*`.

## Public API Surface
- Primary API exposed via `llamatelemetry/__init__.py` (`InferenceEngine`, `quick_infer`, helpers).
- Full HTTP client APIs live under `llamatelemetry/api/` and are re-exported in `llamatelemetry/api/__init__.py`.

## Configuration Patterns
- Uses simple dict/config methods (e.g., `MultiGPUConfig.to_cli_args()` and `.to_dict()`).
- Model registry is a static dict in `llamatelemetry/_internal/registry.py`.

## Testing
- `pytest` is the standard test runner.
- Tests often rely on environment availability (CUDA, network) and may skip if missing.
