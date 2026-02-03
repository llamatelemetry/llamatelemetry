# llamatelemetry v0.1.0 - Installation Guide

## Requirements

- **Python:** 3.11+
- **CUDA:** 12.x runtime
- **GPU:** 2x NVIDIA Tesla T4 (15GB each, SM 7.5)
- **Platform:** Kaggle notebooks only (dual T4)
- **Internet:** Enabled (for package + binary download)

## Installation Methods

### Method 1: From GitHub (Recommended)
```bash
pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

### Method 2: Latest Development
```bash
pip install git+https://github.com/llamatelemetry/llamatelemetry.git
```

### Method 3: Development Install
```bash
git clone https://github.com/llamatelemetry/llamatelemetry.git
cd llamatelemetry
pip install -e .
```
**Note:** llamatelemetry v0.1.0 is not published to PyPI. GitHub is the primary distribution channel.

## Binary Download

On first import, llamatelemetry automatically downloads CUDA binaries (~961 MB) from GitHub Releases:

```python
import llamatelemetry  # Downloads binaries to ~/.cache/llamatelemetry/
```

### Manual Binary Download
```bash
wget https://github.com/llamatelemetry/llamatelemetry/releases/download/v0.1.0/llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz
mkdir -p ~/.cache/llamatelemetry
tar -xzf llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz -C ~/.cache/llamatelemetry/
```

## Verification

```python
import llamatelemetry
print(f"Version: {llamatelemetry.__version__}")  # 0.1.0

from llamatelemetry.api import kaggle_t4_dual_config
config = kaggle_t4_dual_config()
print(f"Multi-GPU config: {config.to_cli_args()}")
```

## Troubleshooting

### GPU Not Detected
```bash
nvidia-smi  # Verify GPU is available
```

### Binary Download Failed
Check internet connection and try manual download above.

### Import Errors
```bash
pip uninstall llamatelemetry
pip cache purge
pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```
