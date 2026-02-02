# Contributing to llamatelemetry

Thank you for your interest in contributing to llamatelemetry! This guide will help you get started.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

---

## Code of Conduct

This project follows a standard code of conduct. Please be respectful and constructive in all interactions.

---

## Getting Started

### Types of Contributions

We welcome:

- 🐛 **Bug fixes** - Fix issues and improve stability
- ✨ **Features** - Add new functionality
- 📚 **Documentation** - Improve guides and API docs
- 🧪 **Tests** - Expand test coverage
- 📓 **Notebooks** - Tutorial and example notebooks
- 🔧 **Tooling** - Build and CI improvements

### First-Time Contributors

Good first issues are labeled with `good first issue`. Start there!

---

## Development Setup

### Prerequisites

- Python 3.9+
- CUDA 12.x (for GPU features)
- Git

### Clone Repository

```bash
git clone https://github.com/llamatelemetry/llamatelemetry.git
cd llamatelemetry
```

### Create Virtual Environment

```bash
# Create venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dev dependencies
pip install -e ".[dev]"
```

### Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### Verify Setup

```bash
# Run tests
pytest tests/

# Check code style
ruff check llamatelemetry/
mypy llamatelemetry/
```

---

## Project Structure

```
llamatelemetry/
├── llamatelemetry/               # Main package
│   ├── __init__.py
│   ├── server.py         # ServerManager, ServerConfig
│   ├── api/
│   │   ├── client.py     # LlamaCppClient
│   │   ├── gguf.py       # GGUFParser
│   │   └── multigpu.py   # Multi-GPU utilities
│   └── utils/
├── core/                 # Core utilities
├── csrc/                 # C/C++ source (if any)
├── docs/                 # Documentation
│   ├── INSTALLATION.md
│   ├── CONFIGURATION.md
│   ├── API_REFERENCE.md
│   ├── KAGGLE_GUIDE.md
│   ├── GGUF_GUIDE.md
│   └── TROUBLESHOOTING.md
├── examples/             # Example scripts
├── notebooks/            # Tutorial notebooks
│   └── README.md         # Notebook index
├── scripts/              # Build and utility scripts
├── tests/                # Test suite
├── CHANGELOG.md
├── CONTRIBUTING.md       # This file
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `llamatelemetry/server.py` | Server management |
| `llamatelemetry/api/client.py` | API client |
| `llamatelemetry/api/gguf.py` | GGUF parsing |
| `llamatelemetry/api/multigpu.py` | Multi-GPU config |

---

## Making Changes

### Branch Naming

```bash
# Feature branches
git checkout -b feature/your-feature-name

# Bug fixes
git checkout -b fix/issue-description

# Documentation
git checkout -b docs/what-you-documented
```

### Commit Messages

Use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `chore`: Maintenance

Examples:
```
feat(server): add timeout parameter to ServerConfig
fix(client): handle connection timeout gracefully
docs(readme): update installation instructions
test(gguf): add parser edge case tests
```

### Keep Changes Focused

- One feature/fix per PR
- Small, reviewable changes
- Update tests and docs together

---

## Testing

### Run All Tests

```bash
pytest tests/
```

### Run Specific Tests

```bash
# Run specific test file
pytest tests/test_server.py

# Run specific test
pytest tests/test_server.py::test_server_config

# Run with coverage
pytest --cov=llamatelemetry tests/
```

### Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests (requires GPU)
pytest tests/integration/

# Skip slow tests
pytest -m "not slow"
```

### Writing Tests

```python
# tests/test_example.py
import pytest
from llamatelemetry.server import ServerConfig

def test_server_config_defaults():
    """Test ServerConfig default values."""
    config = ServerConfig(model_path="test.gguf")
    assert config.host == "127.0.0.1"
    assert config.port == 8080
    assert config.n_gpu_layers == -1

def test_server_config_validation():
    """Test ServerConfig validation."""
    with pytest.raises(ValueError):
        ServerConfig(model_path="")  # Empty path

@pytest.mark.gpu
def test_multi_gpu_inference():
    """Test multi-GPU inference (requires 2 GPUs)."""
    # This test only runs with --gpu flag
    pass
```

---

## Code Style

### Python Style

We use:
- **Ruff** for linting and formatting
- **mypy** for type checking
- **Black** style (via Ruff)

### Ruff Configuration

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

### Run Linters

```bash
# Check style
ruff check llamatelemetry/

# Auto-fix
ruff check --fix llamatelemetry/

# Format
ruff format llamatelemetry/

# Type check
mypy llamatelemetry/
```

### Code Guidelines

```python
# Use type hints
def estimate_vram(
    model_params: int,
    bits_per_weight: float,
    context_size: int = 4096,
) -> float:
    """
    Estimate VRAM requirements.
    
    Args:
        model_params: Number of model parameters.
        bits_per_weight: Bits per weight (e.g., 4.0 for Q4).
        context_size: Context window size.
        
    Returns:
        Estimated VRAM in GB.
    """
    ...

# Use dataclasses for config
from dataclasses import dataclass

@dataclass
class ServerConfig:
    model_path: str
    host: str = "127.0.0.1"
    port: int = 8080
```

---

## Documentation

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview |
| `docs/INSTALLATION.md` | Installation guide |
| `docs/CONFIGURATION.md` | Configuration reference |
| `docs/API_REFERENCE.md` | API documentation |
| `docs/KAGGLE_GUIDE.md` | Kaggle-specific guide |
| `docs/GGUF_GUIDE.md` | GGUF and quantization |
| `docs/TROUBLESHOOTING.md` | Common issues |

### Documentation Style

- Use clear, concise language
- Include code examples
- Add tables for reference data
- Use ASCII diagrams for architecture
- Link between documents

### Docstrings

Use Google-style docstrings:

```python
def chat_completion(
    self,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> ChatCompletionResponse:
    """
    Send a chat completion request.
    
    Args:
        messages: List of message dicts with 'role' and 'content'.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0-2.0).
        
    Returns:
        ChatCompletionResponse with generated content.
        
    Raises:
        ConnectionError: If server is not reachable.
        TimeoutError: If request times out.
        
    Example:
        >>> client = LlamaCppClient()
        >>> response = client.chat_completion(
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response.choices[0].message.content)
    """
```

---

## Pull Request Process

### Before Submitting

1. ✅ Tests pass: `pytest tests/`
2. ✅ Linting passes: `ruff check llamatelemetry/`
3. ✅ Type checks pass: `mypy llamatelemetry/`
4. ✅ Documentation updated
5. ✅ CHANGELOG.md updated

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manually tested

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

### Review Process

1. Submit PR against `main` branch
2. CI runs tests and linting
3. Maintainer reviews code
4. Address feedback
5. Squash and merge

---

## Release Process

### Version Numbering

We use semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release tag
4. Build and publish

```bash
# Update version
# pyproject.toml: version = "2.3.0"

# Update changelog
# Add release notes under ## [2.3.0] - YYYY-MM-DD

# Commit and tag
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release v2.3.0"
git tag v2.3.0
git push origin main --tags
```

---

## Questions?

- Open a GitHub issue for bugs or features
- Start a discussion for questions
- Check existing issues before creating new ones

Thank you for contributing! 🚀
