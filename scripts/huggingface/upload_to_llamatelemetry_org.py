#!/usr/bin/env python3
"""
Upload binaries to llamatelemetry/binaries (correct organization)
Run this AFTER creating the repository
"""

import os
from huggingface_hub import HfApi
import sys

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "llamatelemetry/binaries"
REPO_TYPE = "model"
VERSION = "v0.1.0"

RELEASE_DIR = "/media/waqasm86/External1/Project-Nvidia-Office/llamatelemetry/releases/v0.1.0"
FILES_TO_UPLOAD = [
    "llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz",
    "llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz.sha256",
]

def main():
    print("=" * 70)
    print(f"Uploading Binaries to {REPO_ID}")
    print("=" * 70)
    print()
    
    api = HfApi(token=HF_TOKEN)
    
    # Check if repository exists
    print("🔍 Checking if repository exists...")
    try:
        files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
        print(f"✅ Repository found with {len(files)} existing files")
    except Exception as e:
        print(f"❌ Repository not found: {e}")
        print()
        print("Please run: python3 scripts/huggingface/create_org_binaries_repo.py first")
        sys.exit(1)
    
    # Upload binary files
    for filename in FILES_TO_UPLOAD:
        file_path = os.path.join(RELEASE_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        
        file_size = os.path.getsize(file_path) / (1024**3)
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
            sys.exit(1)
    
    # Upload README
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

Official pre-compiled CUDA binaries for **llamatelemetry** - CUDA-first OpenTelemetry Python SDK for LLM inference observability.

## 📦 Available Binaries

| File | Size | Target | SHA256 |
|------|------|--------|--------|
| llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz | 1.4 GB | Kaggle 2× T4, CUDA 12.5 | 31889a86... |

## 🚀 Auto-Download

These binaries are automatically downloaded when you install llamatelemetry:

```bash
pip install --no-cache-dir --force-reinstall \\
    git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

On first `import llamatelemetry`, the package will:
1. Check for local binaries in `~/.cache/llamatelemetry/`
2. Download from HuggingFace CDN (this repo - fast, ~2-5 MB/s)
3. Fallback to GitHub Releases if needed (~1-3 MB/s)
4. Extract to the package directory
5. Verify SHA256 checksum

## 📥 Manual Download

```python
from huggingface_hub import hf_hub_download

binary_path = hf_hub_download(
    repo_id="{REPO_ID}",
    filename="v0.1.0/llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz",
    cache_dir="/kaggle/working/cache"
)
```

## 🔗 Links

- **GitHub**: https://github.com/llamatelemetry/llamatelemetry
- **Releases**: https://github.com/llamatelemetry/llamatelemetry/releases
- **Installation Guide**: https://github.com/llamatelemetry/llamatelemetry/blob/main/docs/guides/KAGGLE_INSTALL_GUIDE.md
- **Documentation**: https://llamatelemetry.github.io (planned)

## 📊 Build Info

- **CUDA Version**: 12.5
- **Compute Capability**: SM 7.5 (Tesla T4)
- **llama.cpp**: b7760 (commit 388ce82)
- **Build Date**: 2026-02-03
- **Target Platform**: Kaggle dual T4 GPUs

## 📄 License

MIT License - See https://github.com/llamatelemetry/llamatelemetry/blob/main/LICENSE

---

**Maintained by**: llamatelemetry organization  
**Version**: {VERSION[1:]}  
**Status**: Active
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
    
    print()
    print("=" * 70)
    print("✅ UPLOAD COMPLETE!")
    print("=" * 70)
    print()
    print(f"🔗 Repository: https://huggingface.co/{REPO_ID}")
    print(f"🔗 Files: https://huggingface.co/{REPO_ID}/tree/main/{VERSION}")
    print()
    print("Next steps:")
    print("1. Update llamatelemetry/_internal/bootstrap.py line 39:")
    print(f'   HF_BINARIES_REPO = "{REPO_ID}"')
    print("2. Commit and push to GitHub")
    print("3. Test the installation workflow")

if __name__ == "__main__":
    main()
