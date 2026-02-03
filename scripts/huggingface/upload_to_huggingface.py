#!/usr/bin/env python3
"""
Upload llamatelemetry v0.1.0 binaries to HuggingFace
Following the llcuda reference pattern for CDN/fast downloads
"""

import os
from huggingface_hub import HfApi, create_repo

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "waqasm86/llamatelemetry-binaries"  # Following llcuda pattern
REPO_TYPE = "model"  # HuggingFace model repository for binary hosting
VERSION = "v0.1.0"

# Files to upload
RELEASE_DIR = "/media/waqasm86/External1/Project-Nvidia-Office/llamatelemetry/releases/v0.1.0"
FILES_TO_UPLOAD = [
    "llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz",
    "llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz.sha256",
]

def main():
    print("🚀 Starting HuggingFace upload for llamatelemetry v0.1.0")
    print(f"📦 Repository: {REPO_ID}")

    # Initialize HF API
    api = HfApi(token=HF_TOKEN)

    # Create repository if it doesn't exist
    try:
        print(f"\n🔨 Creating repository {REPO_ID}...")
        create_repo(
            repo_id=REPO_ID,
            token=HF_TOKEN,
            repo_type=REPO_TYPE,
            exist_ok=True,
            private=False,
        )
        print("✅ Repository created/verified")
    except Exception as e:
        print(f"⚠️  Repository creation: {e}")

    # Upload files
    for filename in FILES_TO_UPLOAD:
        file_path = os.path.join(RELEASE_DIR, filename)

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        file_size = os.path.getsize(file_path) / (1024**3)  # GB
        print(f"\n📤 Uploading {filename} ({file_size:.2f} GB)...")

        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=f"{VERSION}/{filename}",
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=HF_TOKEN,
            )
            print(f"✅ Uploaded: {filename}")
        except Exception as e:
            print(f"❌ Upload failed for {filename}: {e}")

    # Upload README for the repository
    readme_content = f"""---
license: mit
tags:
- llm
- cuda
- inference
- opentelemetry
- observability
- telemetry
- gguf
- llama-cpp
- kaggle
- binaries
---

# llamatelemetry Binaries v{VERSION[1:]}

Pre-compiled CUDA binaries for llamatelemetry - CUDA-first OpenTelemetry Python SDK for LLM inference observability.

## 📦 Available Binaries

| File | Size | Target | SHA256 |
|------|------|--------|--------|
| llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz | 1.4 GB | Kaggle 2× T4, CUDA 12.5 | 31889a86... |

## 🚀 Auto-Download

These binaries are automatically downloaded when you install llamatelemetry:

```bash
pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

On first `import llamatelemetry`, the package will:
1. Check for local binaries in `~/.cache/llamatelemetry/`
2. Download from HuggingFace CDN (fast) or GitHub Releases (fallback)
3. Extract to the package directory
4. Verify SHA256 checksum

## 📥 Manual Download

```python
from huggingface_hub import hf_hub_download

binary_path = hf_hub_download(
    repo_id="waqasm86/llamatelemetry-binaries",
    filename="v0.1.0/llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz",
    cache_dir="/kaggle/working/cache"
)
```

## 🔗 Links

- **GitHub**: https://github.com/llamatelemetry/llamatelemetry
- **Documentation**: https://llamatelemetry.github.io
- **PyPI**: Not published (install from GitHub)

## 📊 Build Info

- **CUDA Version**: 12.5
- **Compute Capability**: SM 7.5 (Tesla T4)
- **llama.cpp**: b7760 (commit 388ce82)
- **Build Date**: 2026-02-03
- **Target Platform**: Kaggle dual T4 GPUs

## 📄 License

MIT License - See https://github.com/llamatelemetry/llamatelemetry/blob/main/LICENSE
"""

    print(f"\n📝 Uploading README.md...")
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=HF_TOKEN,
        )
        print("✅ README.md uploaded")
    except Exception as e:
        print(f"⚠️  README upload: {e}")

    print(f"\n✅ Upload complete!")
    print(f"🔗 Repository: https://huggingface.co/{REPO_ID}")
    print(f"🔗 Files: https://huggingface.co/{REPO_ID}/tree/main/{VERSION}")

if __name__ == "__main__":
    main()
