"""
llamatelemetry v2.2.0 Bootstrap Module - Kaggle Dual T4 GPUs

This module handles first-time setup for llamatelemetry v2.2.0:
- Verifies GPU is Tesla T4 or compatible (SM 7.5+)
- Downloads Kaggle dual T4 CUDA 12.5 binaries (~961 MB)
- Downloads llamatelemetry native extension if needed

Designed for Kaggle 2× T4 and modern GPUs with Tensor Core support.
"""

import os
import sys
import json
import shutil
import tarfile
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple
import subprocess

# Note: Do NOT import from llamatelemetry here to avoid circular imports
# Use BINARY_VERSION constant instead for version info

try:
    from huggingface_hub import hf_hub_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# Configuration for llamatelemetry v2.2.0 (Kaggle dual T4)
BINARY_VERSION = "2.2.0"
PRIMARY_BINARY_BUNDLE = f"llamatelemetry-v{BINARY_VERSION}-cuda12-kaggle-t4x2.tar.gz"
GITHUB_RELEASE_URL = "https://github.com/llamatelemetry/llamatelemetry/releases/download"

# HuggingFace repos (faster CDN, better for large files)
HF_BINARIES_REPO = "waqasm86/llamatelemetry-binaries"  # Binary bundles (~961 MB)
HF_MODELS_REPO = "waqasm86/llamatelemetry-models"      # GGUF models

# SHA256 checksums for integrity verification
BINARY_CHECKSUMS = {
    "llamatelemetry-v2.2.0-cuda12-kaggle-t4x2.tar.gz": "489f3df54bac24d3801af3149c346402bea099d2bd8793897aa606c5c8af0025",
}

# Binary bundle preference order (HuggingFace primary → GitHub fallback)
BINARY_BUNDLE_CANDIDATES = [
    {
        "version": BINARY_VERSION,
        "filename": PRIMARY_BINARY_BUNDLE,
        "label": "HuggingFace",
        "source": "huggingface",
        "hf_path": f"v{BINARY_VERSION}/{PRIMARY_BINARY_BUNDLE}",
    },
    {
        "version": BINARY_VERSION,
        "filename": PRIMARY_BINARY_BUNDLE,
        "label": "GitHub",
        "source": "github",
    },
]

# Legacy constant retained for downstream tooling/documentation
T4_BINARY_BUNDLE = PRIMARY_BINARY_BUNDLE
T4_NATIVE_BUNDLE = "llamatelemetry-v2-native-t4.tar.gz"        # ~100 MB

# Minimum compute capability for llamatelemetry v2.2.0
MIN_COMPUTE_CAPABILITY = 7.5  # Tesla T4, RTX 20xx+, A100, H100

# Paths
PACKAGE_DIR = Path(__file__).parent.parent
BINARIES_DIR = PACKAGE_DIR / "binaries"
LIB_DIR = PACKAGE_DIR / "lib"
MODELS_DIR = PACKAGE_DIR / "models"
CACHE_DIR = Path.home() / ".cache" / "llamatelemetry"


def detect_gpu_compute_capability() -> Optional[Tuple[str, str]]:
    """
    Detect NVIDIA GPU compute capability using nvidia-smi.

    Returns:
        Tuple of (gpu_name, compute_capability) or None if no GPU found
        Example: ("Tesla T4", "7.5")
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            # Take first GPU
            line = result.stdout.strip().split("\n")[0]
            gpu_name, compute_cap = line.split(",")
            return gpu_name.strip(), compute_cap.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    return None


def detect_platform() -> str:
    """
    Detect execution platform (local, colab, kaggle).

    Returns:
        Platform name: "colab", "kaggle", or "local"
    """
    # Legacy environment check (not supported in v2.2.0 runtime)
    try:
        import google.colab

        return "colab"
    except ImportError:
        pass

    # Check for Kaggle
    if os.path.exists("/kaggle"):
        return "kaggle"

    return "local"


def verify_gpu_compatibility(gpu_name: str, compute_cap: str) -> bool:
    """
    Verify GPU is compatible with llamatelemetry v2.2.0 (SM 7.5).

    Args:
        gpu_name: GPU name from nvidia-smi
        compute_cap: Compute capability (e.g., "7.5", "8.0")

    Returns:
        True if compatible, False otherwise

    Raises:
        RuntimeError if GPU is not compatible
    """
    try:
        cc_float = float(compute_cap)
    except (ValueError, TypeError):
        raise RuntimeError(f"Invalid compute capability: {compute_cap}")

    gpu_lower = gpu_name.lower()

    # Check minimum requirement
    if cc_float < MIN_COMPUTE_CAPABILITY:
        print()
        print("=" * 70)
        print("❌ INCOMPATIBLE GPU DETECTED")
        print("=" * 70)
        print()
        print(f"  Your GPU: {gpu_name} (SM {compute_cap})")
        print(f"  Required: Tesla T4 (SM 7.5)")
        print()
        print("  llamatelemetry v2.2.0 is designed exclusively for Tesla T4 (SM 7.5)")
        print()
        print("  Compatible environment:")
        print("    - Kaggle notebooks (dual Tesla T4)")
        print()
        print("  llamatelemetry v2.2.0 requires Kaggle dual Tesla T4 (SM 7.5)")
        print()
        print("=" * 70)
        raise RuntimeError(f"GPU compute capability {compute_cap} < {MIN_COMPUTE_CAPABILITY} (minimum required)")

    # Tesla T4 verification
    if cc_float == 7.5 and "t4" in gpu_lower:
        print(f"  ✅ Tesla T4 detected - Perfect for llamatelemetry v2.2.0!")
    elif cc_float == 7.5:
        print(f"  ⚠️  {gpu_name} (SM {compute_cap}) - May work but not tested")
        print(f"      llamatelemetry v2.2.0 is optimized exclusively for Tesla T4")
    else:
        print(f"  ⚠️  {gpu_name} (SM {compute_cap}) - Not tested")
        print(f"      llamatelemetry v2.2.0 is designed for Tesla T4 (SM 7.5)")

    return True


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> None:
    """
    Download file with progress bar.

    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress bar
    """
    import urllib.request

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            mb_downloaded = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r{desc}: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
            )
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Download failed: {e}")


def extract_tarball(tarball_path: Path, dest_dir: Path) -> None:
    """
    Extract tarball to destination directory.

    Args:
        tarball_path: Path to tarball
        dest_dir: Destination directory
    """
    print(f"📦 Extracting {tarball_path.name}...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tarball_path, "r:gz") as tar:
        # First, check what's in the tarball
        members = tar.getmembers()
        print(f"Found {len(members)} files in archive")

        # Extract all
        tar.extractall(dest_dir)

        # List extracted files for debugging
        extracted_files = list(dest_dir.rglob("*"))
        print(f"Extracted {len(extracted_files)} files to {dest_dir}")

    print("✅ Extraction complete!")


def locate_bin_and_lib_dirs(extract_root: Path) -> Optional[Tuple[Path, Path]]:
    """Locate bin/ and lib/ directories within an extracted archive."""

    queue = deque([extract_root])
    visited = set()

    while queue:
        current = queue.popleft()

        try:
            current_resolved = current.resolve()
        except FileNotFoundError:
            continue

        if current_resolved in visited:
            continue

        visited.add(current_resolved)

        bin_dir = current / "bin"
        lib_dir = current / "lib"

        if bin_dir.exists() and lib_dir.exists():
            return bin_dir, lib_dir

        for child in current.iterdir():
            if child.is_dir():
                queue.append(child)

    return None


def verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """Verify SHA256 checksum of a file."""
    import hashlib
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest()
    return actual_hash == expected_hash


def download_from_huggingface(hf_path: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """
    Download file from HuggingFace Hub with resume support.
    
    Args:
        hf_path: Path within the HuggingFace repo (e.g., "v2.2.0/file.tar.gz")
        dest_path: Local destination path
        desc: Description for logging
        
    Returns:
        True if successful, False otherwise
    """
    if not HF_AVAILABLE:
        return False
    
    try:
        print(f"📥 {desc} from HuggingFace Hub...")
        print(f"   Repo: {HF_BINARIES_REPO}")
        print(f"   File: {hf_path}")
        
        # hf_hub_download handles caching, resume, and progress automatically
        downloaded_path = hf_hub_download(
            repo_id=HF_BINARIES_REPO,
            filename=hf_path,
            repo_type="model",
            local_dir=dest_path.parent,
            local_dir_use_symlinks=False,
        )
        
        # Move to expected location if needed
        downloaded = Path(downloaded_path)
        if downloaded != dest_path:
            shutil.copy2(downloaded, dest_path)
        
        return True
    except Exception as e:
        print(f"   ⚠️  HuggingFace download failed: {e}")
        return False


def download_t4_binaries() -> None:
    """
    Download and install Kaggle 2× T4 optimized CUDA 12.5 binaries for llamatelemetry v2.2.0.

    Download sources (tried in order):
    1. HuggingFace Hub (waqasm86/llamatelemetry-binaries) - faster CDN, resume support
    2. GitHub Releases - fallback
    
    This version uses the v2.2.0 binaries built for Kaggle dual T4.
    Includes:
    - llama-server with multi-GPU support
    - libggml-cuda.so with FlashAttention
    - 13 binaries total (llama-cli, llama-quantize, etc.)

    Total size: ~961 MB
    Build: CUDA 12.5, SM 7.5 (Turing), llama.cpp b7760 (388ce82)
    """
    # Check if binaries already exist
    llama_server = BINARIES_DIR / "cuda12" / "llama-server"
    if llama_server.exists() and llama_server.stat().st_size > 0:
        print("✅ T4 binaries already installed")
        return

    print("=" * 70)
    print("🎯 llamatelemetry v2.2.0 First-Time Setup - Kaggle 2× T4 Multi-GPU")
    print("=" * 70)
    print()

    # Detect GPU and verify compatibility
    gpu_info = detect_gpu_compute_capability()
    platform = detect_platform()

    if gpu_info:
        gpu_name, compute_cap = gpu_info
        print(f"🎮 GPU Detected: {gpu_name} (Compute {compute_cap})")

        # Verify SM 7.5+ compatibility
        try:
            verify_gpu_compatibility(gpu_name, compute_cap)
        except RuntimeError:
            # GPU not compatible, abort
            raise
    else:
        print("❌ No NVIDIA GPU detected")
        print()
        print("llamatelemetry v2.2.0 requires Tesla T4 (SM 7.5) on Kaggle dual-GPU")
        raise RuntimeError("No compatible NVIDIA GPU found")

    print(f"🌐 Platform: {platform.capitalize()}")
    print()

    # Download T4 binary bundle (HuggingFace primary, GitHub fallback)
    print("📦 Downloading Kaggle 2× T4 binaries (~961 MB)...")
    print("    Features: FlashAttention + Tensor Cores + Multi-GPU tensor-split")
    print()

    install_success = False
    bundle_used: Optional[Dict[str, str]] = None
    failures = []

    for idx, bundle in enumerate(BINARY_BUNDLE_CANDIDATES):
        version = bundle["version"]
        bundle_name = bundle["filename"]
        source = bundle.get("source", "github")
        cache_tarball = CACHE_DIR / bundle_name

        print(f"➡️  Attempt {idx + 1}: {bundle['label']} ({bundle_name})")

        # Download if not cached
        if not cache_tarball.exists():
            download_ok = False
            
            if source == "huggingface":
                # Try HuggingFace first (faster CDN, resume support)
                hf_path = bundle.get("hf_path", f"v{version}/{bundle_name}")
                download_ok = download_from_huggingface(
                    hf_path, cache_tarball, f"Downloading v{version}"
                )
            
            if not download_ok:
                # Fall back to GitHub
                bundle_url = f"{GITHUB_RELEASE_URL}/v{version}/{bundle_name}"
                print(f"   Source: {bundle_url}")
                print(f"📥 Downloading binaries v{version} (~961 MB)...")
                try:
                    download_file(bundle_url, cache_tarball, f"Downloading T4 binaries v{version}")
                    download_ok = True
                except Exception as e:
                    failures.append(f"Download failed for v{version} ({source}): {e}")
                    print(f"   ⚠️  Download error: {e}")
                    continue
            
            if not download_ok:
                continue
        else:
            print(f"✅ Using cached archive: {cache_tarball}")

        # Verify SHA256 checksum if available
        expected_hash = BINARY_CHECKSUMS.get(bundle_name)
        if expected_hash:
            print(f"🔐 Verifying SHA256 checksum...")
            if not verify_sha256(cache_tarball, expected_hash):
                print(f"   ❌ Checksum mismatch! Deleting corrupted file.")
                cache_tarball.unlink()
                failures.append(f"SHA256 verification failed for {bundle_name}")
                continue
            print(f"   ✅ Checksum verified")

        temp_extract_dir = CACHE_DIR / f"extract_{version}"
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
        temp_extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            extract_tarball(cache_tarball, temp_extract_dir)
            located = locate_bin_and_lib_dirs(temp_extract_dir)
            if not located:
                raise RuntimeError("Expected 'bin' and 'lib' directories were not found in the archive")

            bin_dir, lib_dir = located
            print(f"  Found bin/ and lib/ under {bin_dir.parent}")

            cuda12_dir = BINARIES_DIR / "cuda12"
            cuda12_dir.mkdir(parents=True, exist_ok=True)
            copied_bins = 0
            for item in bin_dir.iterdir():
                if item.is_file():
                    dest = cuda12_dir / item.name
                    shutil.copy2(item, dest)
                    copied_bins += 1
                    if not item.suffix or item.suffix == '.sh':
                        try:
                            dest.chmod(0o755)
                        except Exception:
                            pass

            LIB_DIR.mkdir(parents=True, exist_ok=True)
            copied_libs = 0
            for item in lib_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, LIB_DIR / item.name)
                    copied_libs += 1

            print(f"  Copied {copied_bins} binaries to {cuda12_dir}")
            print(f"  Copied {copied_libs} libraries to {LIB_DIR}")

            install_success = True
            bundle_used = bundle
            break

        except Exception as e:
            failures.append(f"Extraction failed for v{version}: {e}")
            print(f"   ⚠️  Extraction error: {e}")
        finally:
            shutil.rmtree(temp_extract_dir, ignore_errors=True)

    if not install_success:
        failure_report = "\n".join(failures) if failures else "Unknown error"
        raise RuntimeError(
            "Unable to install llamatelemetry binaries. Tried all available bundles but each failed:\n"
            f"{failure_report}"
        )

    print("✅ Binaries installed successfully!")
    if bundle_used and bundle_used["version"] != BINARY_VERSION:
        print(
            f"ℹ️  Primary bundle unavailable. Installed fallback binaries v{bundle_used['version']} instead."
        )
    print()


def download_default_model() -> None:
    """
    Download default model (Gemma 3 1B) from Hugging Face.
    """
    if not HF_AVAILABLE:
        print("⚠️  huggingface_hub not available, skipping model download")
        print("   Install with: pip install huggingface_hub")
        return

    # Check if model already exists
    model_file = MODELS_DIR / "google_gemma-3-1b-it-Q4_K_M.gguf"
    if model_file.exists() and model_file.stat().st_size > 700_000_000:  # > 700 MB
        print("✅ Model already downloaded")
        return

    print("📥 Downloading default model from Hugging Face...")
    print(f"   Repository: {HF_REPO_ID}")
    print(f"   Model: google_gemma-3-1b-it-Q4_K_M.gguf (769 MB)")
    print(f"   This is a one-time download")
    print()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="google_gemma-3-1b-it-Q4_K_M.gguf",
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
        )
        print()
        print(f"✅ Model downloaded: {downloaded_path}")
    except Exception as e:
        print(f"⚠️  Model download failed: {e}")
        print("   You can manually download models later")

    print()


def bootstrap() -> None:
    """
    Main bootstrap function for llamatelemetry v2.2.0 - called on first import.

    Downloads T4-optimized binaries from GitHub Releases on first import.
    Uses v2.2.0 binaries for Kaggle dual T4.
    Models are downloaded on-demand when load_model() is called.

    Raises:
        RuntimeError: If GPU is not compatible (SM < 7.5) or download fails
    """
    # Check if binaries already installed
    llama_server = BINARIES_DIR / "cuda12" / "llama-server"

    if llama_server.exists() and llama_server.stat().st_size > 0:
        # Binaries already installed
        return

    # Download T4 binaries from GitHub Releases
    print()
    download_t4_binaries()

    # Verify installation
    if not llama_server.exists():
        raise RuntimeError(
            "Binary installation failed. Please check your internet connection and try again:\n"
            "pip install --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git"
        )


if __name__ == "__main__":
    bootstrap()
