"""
llamatelemetry.server - Server Management for llama-server

This module provides automatic management of llama-server lifecycle,
including finding, starting, and stopping the server process.
"""

import tarfile
import requests
import shutil
from pathlib import Path
import sys
import time
from typing import Optional, Dict, Any
import os
import subprocess
import time
import signal
from pathlib import Path
import requests


class ServerManager:
    """
    Manages llama-server lifecycle automatically.

    This class handles:
    - Finding llama-server executable in common locations
    - Starting llama-server with appropriate parameters
    - Health checking and waiting for server readiness
    - Stopping server gracefully

    Examples:
        >>> manager = ServerManager()
        >>> manager.start_server(
        ...     model_path="/path/to/model.gguf",
        ...     gpu_layers=99
        ... )
        >>> # Server is now running
        >>> manager.stop_server()
    """

    # Binary bundles used when bootstrap-installed binaries are missing
    _BINARY_RELEASE_BASE = "https://github.com/llamatelemetry/llamatelemetry/releases/download"
    _BINARY_BUNDLES = [
        {"version": "0.1.0", "filename": "llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz", "label": "primary"},
    ]

    def __init__(self, server_url: str = "http://127.0.0.1:8090"):
        """
        Initialize the server manager.

        Args:
            server_url: URL where server will be accessible
        """
        self.server_url = server_url
        self.server_process: Optional[subprocess.Popen] = None
        self._server_path: Optional[Path] = None

    def find_llama_server(self) -> Optional[Path]:
        """Locate the llama-server executable or download it if missing."""

        def _validate(candidate: Optional[Path]) -> Optional[Path]:
            if not candidate:
                return None
            path = Path(candidate)
            if path.exists() and path.is_file():
                os.chmod(path, 0o755)
                self._setup_library_path(path)
                self._server_path = path
                return path
            return None

        if self._server_path and self._server_path.exists():
            return self._server_path

        # 1) Explicit env override
        env_path = os.getenv("LLAMA_SERVER_PATH")
        if _validate(env_path):
            return self._server_path

        # 2) Package bootstrap directory (installed via llamatelemetry bootstrap)
        package_dir = Path(__file__).resolve().parent
        bootstrap_candidates = [
            package_dir / "binaries" / "bin" / "llama-server",
            package_dir / "binaries" / "llama-server",
            package_dir / "bin" / "llama-server",
        ]

        for candidate in bootstrap_candidates:
            if _validate(candidate):
                return self._server_path

        # 3) User-provided llama.cpp directory
        llama_cpp_dir = os.getenv("LLAMA_CPP_DIR")
        if llama_cpp_dir:
            custom_candidates = [
                Path(llama_cpp_dir) / "bin" / "llama-server",
                Path(llama_cpp_dir) / "llama-server",
            ]
            for candidate in custom_candidates:
                if _validate(candidate):
                    return self._server_path

        # 4) Cache directory (where we download binaries)
        cache_dir = Path.home() / ".cache" / "llamatelemetry"
        cache_candidates = [
            cache_dir / "llama-server",
            cache_dir / "bin" / "llama-server",
        ]
        for candidate in cache_candidates:
            if _validate(candidate):
                return self._server_path

        # 5) Common project directories inside repo (dev installs)
        repo_root = Path(__file__).resolve().parents[2]
        repo_candidates = [
            repo_root / "Ubuntu-Cuda-Llama.cpp-Executable" / "llama-server",
            repo_root / "Ubuntu-Cuda-Llama.cpp-Executable" / "bin" / "llama-server",
        ]
        for candidate in repo_candidates:
            if _validate(candidate):
                return self._server_path

        # 6) System PATH lookup
        system_paths = os.environ.get("PATH", "").split(os.pathsep)
        for path_str in system_paths:
            if not path_str:
                continue
            candidate = Path(path_str) / "llama-server"
            if _validate(candidate):
                return self._server_path

        # 7) Download fresh bundle as last resort
        downloaded = self._download_llama_server()
        if downloaded:
            return _validate(Path(downloaded))

        return None
    def _setup_library_path(self, server_path: Path):
        """
        Setup LD_LIBRARY_PATH for the llama-server executable.

        Args:
            server_path: Path to llama-server executable
        """
        # Find lib directory relative to server binary
        lib_dir = server_path.parent.parent / "lib"

        if lib_dir.exists():
            lib_path_str = str(lib_dir.absolute())
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

            if lib_path_str not in current_ld_path:
                if current_ld_path:
                    os.environ["LD_LIBRARY_PATH"] = f"{lib_path_str}:{current_ld_path}"
                else:
                    os.environ["LD_LIBRARY_PATH"] = lib_path_str

    def check_server_health(self, timeout: float = 2.0) -> bool:
        """
        Check if llama-server is responding to health checks.

        Args:
            timeout: Request timeout in seconds

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _detect_platform(self):
        """
        Detect the current platform (kaggle or local).

        Returns:
            Dictionary with platform information
        """
        platform_info = {
            "platform": "local",
            "gpu_name": None,
            "compute_capability": None,
        }

        # Check for Kaggle
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle"):
            platform_info["platform"] = "kaggle"
            platform_info["gpu_name"] = "Tesla T4"
            platform_info["compute_capability"] = 7.5
            return platform_info

        return platform_info


    def _download_llama_server(self):
        """
        Download and extract the pre-built llama-server binary.
        Returns the path to the downloaded binary.
        """
        def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
            base_path = path.resolve()
            for member in tar.getmembers():
                member_path = (path / member.name).resolve()
                if not str(member_path).startswith(str(base_path)):
                    raise RuntimeError("Unsafe tar entry detected during extraction")
            tar.extractall(path)

        print("llama-server not found. Downloading pre-built CUDA binary...")

        # Determine cache directory based on platform
        platform_info = self._detect_platform()
        if platform_info["platform"] == "colab":
            cache_dir = Path("/content/.cache/llamatelemetry")
        elif platform_info["platform"] == "kaggle":
            cache_dir = Path("/kaggle/working/.cache/llamatelemetry")
        else:
            cache_dir = Path.home() / ".cache" / "llamatelemetry"

        cache_dir.mkdir(parents=True, exist_ok=True)

        errors = []

        for bundle in self._BINARY_BUNDLES:
            tar_filename = bundle["filename"]
            tar_path = cache_dir / tar_filename
            extract_dir = cache_dir / f"extracted_{bundle['version']}"
            download_url = f"{self._BINARY_RELEASE_BASE}/v{bundle['version']}/{tar_filename}"

            print(
                f"➡️  Attempting {bundle['label']} bundle (v{bundle['version']}) from {download_url}"
            )

            try:
                response = requests.get(download_url, stream=True, timeout=60)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(tar_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            sys.stdout.write(
                                f"\rDownloading: {percent:.1f}% ({downloaded}/{total_size} bytes)"
                            )
                            sys.stdout.flush()

                print("\n✓ Download complete")

            except requests.exceptions.RequestException as e:
                message = f"Download failed for v{bundle['version']}: {e}"
                print(f"⚠️  {message}")
                errors.append(message)
                continue

            print("Extracting binary...")
            try:
                shutil.rmtree(extract_dir, ignore_errors=True)
                extract_dir.mkdir(exist_ok=True)

                with tarfile.open(tar_path, "r:gz") as tar:
                    _safe_extract(tar, extract_dir)

                possible_paths = list(extract_dir.rglob("llama-server"))
                if not possible_paths:
                    possible_paths = list(extract_dir.rglob("bin/llama-server"))

                if not possible_paths:
                    raise FileNotFoundError("Could not find llama-server in downloaded archive")

                server_binary = possible_paths[0]
                final_path = cache_dir / "llama-server"
                shutil.copy2(server_binary, final_path)
                os.chmod(final_path, 0o755)

                if tar_path.exists():
                    tar_path.unlink()

                shutil.rmtree(extract_dir, ignore_errors=True)

                if bundle["label"] == "fallback":
                    print("ℹ️  Primary bundle unavailable; using fallback binaries.")

                print(f"✓ Binary installed at: {final_path}")
                return str(final_path)

            except (tarfile.TarError, OSError, FileNotFoundError) as e:
                message = f"Extraction failed for v{bundle['version']}: {e}"
                print(f"⚠️  {message}")
                errors.append(message)
                shutil.rmtree(extract_dir, ignore_errors=True)
                continue

        raise RuntimeError(
            "Failed to download and extract llama-server binaries.\n" + "\n".join(errors)
        )

    def start_server(
        self,
        model_path: str,
        port: int = 8090,
        host: str = "127.0.0.1",
        gpu_layers: int = 99,
        ctx_size: int = 2048,
        n_parallel: int = 1,
        batch_size: int = 512,
        ubatch_size: int = 128,
        timeout: int = 60,
        verbose: bool = True,
        skip_gpu_check: bool = False,
        silent: bool = False,
        **kwargs,
    ) -> bool:
        """
        Start llama-server with specified configuration.

        Args:
            model_path: Path to GGUF model file
            port: Server port (default: 8090)
            host: Server host (default: 127.0.0.1)
            gpu_layers: Number of layers to offload to GPU (default: 99)
            ctx_size: Context size (default: 2048)
            n_parallel: Number of parallel sequences (default: 1)
            batch_size: Logical maximum batch size (default: 512)
            ubatch_size: Physical maximum batch size (default: 128)
            timeout: Max seconds to wait for server startup (default: 60)
            verbose: Print status messages (default: True)
            skip_gpu_check: Skip GPU compatibility check (default: False)
            silent: Suppress all llama-server output (default: False)
            **kwargs: Additional server arguments (flash_attn, cache_ram, fit, etc.)

        Returns:
            True if server started successfully, False otherwise

        Raises:
            FileNotFoundError: If llama-server executable not found
            RuntimeError: If server fails to start or GPU is incompatible
        """
        # Check GPU compatibility (only if using GPU layers)
        if gpu_layers > 0 and not skip_gpu_check:
            from .utils import check_gpu_compatibility

            compat = check_gpu_compatibility(min_compute_cap=5.0)

            if verbose:
                print(f"GPU Check:")
                print(f"  Platform: {compat['platform']}")
                if compat["gpu_name"]:
                    print(f"  GPU: {compat['gpu_name']}")
                if compat["compute_capability"]:
                    print(f"  Compute Capability: {compat['compute_capability']}")

            if not compat["compatible"]:
                error_msg = f"GPU Compatibility Error: {compat['reason']}\n"
                if compat["compute_capability"] and compat["compute_capability"] < 5.0:
                    error_msg += (
                        f"\nYour GPU (compute capability {compat['compute_capability']}) is not supported.\n"
                        f"llamatelemetry requires NVIDIA GPU with compute capability 5.0 or higher.\n"
                        f"Supported GPUs: Maxwell, Pascal, Volta, Turing, Ampere, Ada Lovelace.\n\n"
                        f"Options:\n"
                        f"1. Use CPU-only mode: engine.load_model(model_path, gpu_layers=0)\n"
                        f"2. Upgrade to a newer GPU\n"
                        f"3. Use skip_gpu_check=True to override (may cause runtime errors)"
                    )
                else:
                    error_msg += "\nTo skip this check, use skip_gpu_check=True"
                raise RuntimeError(error_msg)

            if verbose and compat["compatible"]:
                print(f"  Status: ✓ Compatible")

        # Check if server is already running
        if self.check_server_health(timeout=1.0):
            if verbose:
                print(f"✓ llama-server already running at {self.server_url}")
            return True

        # Find llama-server executable
        if self._server_path is None:
            self._server_path = self.find_llama_server()

        if self._server_path is None:
            # Auto-download llama-server binary
            self._server_path = self._download_llama_server()

        # Verify the binary exists and is executable
        if not os.path.exists(self._server_path):
            raise FileNotFoundError(
                f"llama-server not found at downloaded location: {self._server_path}\n"
                "Try setting LLAMA_SERVER_PATH environment variable manually."
            )

        # Make sure it's executable
        os.chmod(self._server_path, 0o755)

        # Verify model exists
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Build command
        cmd = [
            str(self._server_path),
            "-m",
            str(model_path_obj.absolute()),
            "--host",
            host,
            "--port",
            str(port),
            "-ngl",
            str(gpu_layers),
            "-c",
            str(ctx_size),
            "--parallel",
            str(n_parallel),
            "-b",
            str(batch_size),
            "-ub",
            str(ubatch_size),
        ]

        # Add additional arguments with proper parameter mapping
        param_map = {
            "flash_attn": "-fa",
            "cache_ram": "--cache-ram",
            "fit": "-fit",
        }
        # Flags that should be passed without a value
        flag_only = {
            "embeddings",
            "embedding",
        }

        for key, value in kwargs.items():
            if key in flag_only:
                if value is True:
                    cmd.append("--embeddings")
                # If value is False/None, skip the flag entirely
                continue
            if key.startswith("-"):
                # Already formatted parameter
                cmd.extend([key, str(value)])
            elif key in param_map:
                # Use mapped parameter name
                cmd.extend([param_map[key], str(value)])
            else:
                # Convert underscores to hyphens for standard parameters
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        if verbose:
            print(f"Starting llama-server...")
            print(f"  Executable: {self._server_path}")
            print(f"  Model: {model_path_obj.name}")
            print(f"  GPU Layers: {gpu_layers}")
            print(f"  Context Size: {ctx_size}")
            print(f"  Server URL: {self.server_url}")

        # Start server process
        try:
            if silent:
                # Suppress all output
                self.server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            else:
                # Capture output for error reporting
                self.server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to start llama-server: {e}")

        # Wait for server to be ready
        if verbose:
            print(f"Waiting for server to be ready...", end="", flush=True)

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_server_health(timeout=1.0):
                if verbose:
                    elapsed = time.time() - start_time
                    print(f" ✓ Ready in {elapsed:.1f}s")
                return True

            # Check if process died
            if self.server_process.poll() is not None:
                # Read stderr only if it's not DEVNULL (silent mode)
                if self.server_process.stderr is not None:
                    stderr = self.server_process.stderr.read().decode(
                        "utf-8", errors="ignore"
                    )
                    raise RuntimeError(
                        f"llama-server process died unexpectedly.\nError output:\n{stderr}"
                    )
                else:
                    raise RuntimeError(
                        f"llama-server process died unexpectedly. Run with silent=False for error details."
                    )

            if verbose:
                print(".", end="", flush=True)
            time.sleep(1)

        # Timeout reached
        if verbose:
            print(" ✗ Timeout")
        self.stop_server()
        raise RuntimeError(f"Server failed to start within {timeout} seconds")

    def stop_server(self, timeout: float = 10.0) -> bool:
        """
        Stop the running llama-server gracefully.

        Args:
            timeout: Max seconds to wait for graceful shutdown

        Returns:
            True if server stopped successfully, False otherwise
        """
        if self.server_process is None:
            return True

        if self.server_process.poll() is not None:
            # Already stopped
            self.server_process = None
            return True

        try:
            # Try graceful shutdown first (SIGTERM)
            self.server_process.terminate()

            # Wait for graceful shutdown
            try:
                self.server_process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                self.server_process.kill()
                self.server_process.wait(timeout=5.0)

            self.server_process = None
            return True

        except Exception as e:
            print(f"Error stopping server: {e}")
            return False

    def get_server_info(self) -> Dict[str, Any]:
        """
        Get information about the running server.

        Returns:
            Dictionary with server information
        """
        info = {
            "running": False,
            "url": self.server_url,
            "process_id": None,
            "executable": str(self._server_path) if self._server_path else None,
        }

        if self.server_process and self.server_process.poll() is None:
            info["running"] = True
            info["process_id"] = self.server_process.pid

        return info

    def restart_server(self, model_path: str, **kwargs) -> bool:
        """
        Restart the server with new configuration.

        Args:
            model_path: Path to GGUF model file
            **kwargs: Server configuration parameters

        Returns:
            True if restart successful, False otherwise
        """
        self.stop_server()
        time.sleep(1)  # Brief pause before restart
        return self.start_server(model_path, **kwargs)

    def __del__(self):
        """Cleanup: stop server when manager is destroyed."""
        if self.server_process is not None:
            self.stop_server()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: stop server."""
        self.stop_server()
        return False
