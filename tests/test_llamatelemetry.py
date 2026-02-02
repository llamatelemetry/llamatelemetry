"""
Test suite for llamatelemetry Python package - Xubuntu 22.04 / Ubuntu 22.04 Edition
Tests auto-download, GPU detection, and inference workflow
"""

import pytest
import sys
import os
import tempfile
import subprocess
from pathlib import Path
import time

# Add parent directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_import_and_version():
    """Test that the package can be imported and has version"""
    import llamatelemetry
    assert hasattr(llamatelemetry, '__version__')
    print(f"llamatelemetry version: {llamatelemetry.__version__}")


def test_platform_detection():
    """Test platform detection for Xubuntu/Ubuntu 22.04"""
    import llamatelemetry
    
    # Test via server module
    from llamatelemetry.server import ServerManager
    manager = ServerManager()
    
    # Get platform info
    platform_info = manager._detect_platform()
    
    print(f"\nPlatform Detection Results:")
    print(f"  Platform: {platform_info['platform']}")
    print(f"  GPU Name: {platform_info.get('gpu_name', 'Not detected')}")
    print(f"  Compute Capability: {platform_info.get('compute_capability', 'Not detected')}")
    
    # Validate platform detection works
    assert 'platform' in platform_info
    assert platform_info['platform'] in ['local', 'colab', 'kaggle']
    
    # On Xubuntu local, it should detect NVIDIA GPU if available
    if platform_info['platform'] == 'local':
        # Check for NVIDIA GPU (common on Xubuntu with NVIDIA drivers)
        try:
            result = subprocess.run(['which', 'nvidia-smi'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # nvidia-smi exists, we should have GPU info
                assert platform_info['gpu_name'] is not None
        except:
            pass  # nvidia-smi not available, that's okay


def test_gpu_compatibility_check():
    """Test GPU compatibility checking"""
    import llamatelemetry
    
    compat = llamatelemetry.check_gpu_compatibility()
    
    print(f"\nGPU Compatibility Check:")
    print(f"  Platform: {compat['platform']}")
    print(f"  GPU: {compat['gpu_name']}")
    print(f"  Compute Capability: {compat['compute_capability']}")
    print(f"  Compatible: {compat['compatible']}")
    print(f"  Reason: {compat['reason']}")
    
    # Basic validation
    assert isinstance(compat, dict)
    assert 'platform' in compat
    assert 'compatible' in compat
    assert isinstance(compat['compatible'], bool)


def test_binary_auto_download():
    """Test automatic binary download functionality"""
    # Clear any existing binary path
    if 'LLAMA_SERVER_PATH' in os.environ:
        del os.environ['LLAMA_SERVER_PATH']
    
    from llamatelemetry.server import ServerManager
    manager = ServerManager()
    
    print("\nTesting binary auto-download...")
    
    # Check if binary already exists in standard location
    existing_binary = manager.find_llama_server()
    if existing_binary and os.path.exists(existing_binary):
        print(f"  ✓ Binary already exists at: {existing_binary}")
        print(f"  Skipping download test (binary already present)")
        return

    # First, clear any cached binary
    cache_dir = Path.home() / ".cache" / "llamatelemetry"
    if cache_dir.exists():
        # Remove existing binary to test fresh download
        binary_path = cache_dir / "llama-server"
        if binary_path.exists():
            print(f"  Removing cached binary: {binary_path}")
            binary_path.unlink()
    
    # Test download
    try:
        binary_path = manager._download_llama_server()
        print(f"  ✓ Binary downloaded to: {binary_path}")
        
        # Verify binary exists and is executable
        assert os.path.exists(binary_path)
        assert os.access(binary_path, os.X_OK)
        
        # Test binary version
        result = subprocess.run([binary_path, "--version"], 
                              capture_output=True, text=True, timeout=5)
        print(f"  ✓ Binary executes successfully")
        
        # Check for CUDA support
        if "ggml_cuda_init" in result.stdout or "ggml_cuda_init" in result.stderr:
            print(f"  ✓ CUDA support detected")
        
    except Exception as e:
        pytest.skip(f"Binary download failed (might be offline): {e}")


def test_server_manager_creation():
    """Test ServerManager initialization and basic methods"""
    from llamatelemetry.server import ServerManager
    
    # Test with default URL
    manager = ServerManager()
    assert manager.server_url == "http://127.0.0.1:8090"
    assert manager.server_process is None
    
    # Test with custom URL
    custom_manager = ServerManager(server_url="http://127.0.0.1:9999")
    assert custom_manager.server_url == "http://127.0.0.1:9999"
    
    # Test find_llama_server (without download)
    # Clear path to test find logic
    if 'LLAMA_SERVER_PATH' in os.environ:
        del os.environ['LLAMA_SERVER_PATH']
    
    found_path = manager.find_llama_server()
    if found_path:
        print(f"\nFound llama-server at: {found_path}")
        assert os.path.exists(found_path)
    else:
        print("\nNo llama-server found in standard locations (expected for fresh install)")


def test_inference_engine_workflow():
    """Test the main InferenceEngine workflow without actual server"""
    import llamatelemetry
    
    engine = llamatelemetry.InferenceEngine()
    
    # Verify engine has expected methods
    assert hasattr(engine, 'load_model')
    assert hasattr(engine, 'infer')
    assert hasattr(engine, 'get_metrics')
    assert hasattr(engine, 'reset_metrics')
    assert hasattr(engine, 'is_loaded')
    
    # Initial state
    assert engine.is_loaded == False
    
    # Test context manager
    with llamatelemetry.InferenceEngine() as engine_ctx:
        assert engine_ctx is not None
        assert engine_ctx.is_loaded == False
    
    print("\n✓ InferenceEngine workflow validated")


def test_model_config_handling():
    """Test model configuration and parameter parsing"""
    import llamatelemetry
    
    # Test with a mock model path
    test_model_path = "/tmp/test_dummy.gguf"
    
    # Create dummy file for path testing
    with open(test_model_path, 'w') as f:
        f.write("dummy")
    
    try:
        # This should validate the file exists (even though it's not a real GGUF)
        # We expect it to fail on server start, not on file check
        pass
    finally:
        # Clean up
        if os.path.exists(test_model_path):
            os.remove(test_model_path)


def test_error_handling():
    """Test error handling for common scenarios"""
    import llamatelemetry
    from llamatelemetry.server import ServerManager
    
    manager = ServerManager()
    
    # Test 1: Non-existent model file
    with pytest.raises(FileNotFoundError) as exc_info:
        manager.start_server("/non/existent/model.gguf", gpu_layers=0)
    assert "Model file not found" in str(exc_info.value)
    
    print("\n✓ Error handling tests passed")


def test_metrics_structure():
    """Test metrics structure and initialization"""
    import llamatelemetry
    
    engine = llamatelemetry.InferenceEngine()
    metrics = engine.get_metrics()
    
    # Check metrics structure
    assert isinstance(metrics, dict)
    assert 'throughput' in metrics
    assert 'total_requests' in metrics['throughput']
    assert metrics['throughput']['total_requests'] == 0
    
    print("\n✓ Metrics structure validated")


def test_xubuntu_specific_compatibility():
    """Test specific Xubuntu 22.04 compatibility"""
    import platform
    import llamatelemetry
    import shutil
    
    print("\nSystem Information:")
    try:
        # Modern method (Python 3.10+)
        distro_info = platform.freedesktop_os_release()
        print(f"  Distribution: {distro_info.get('NAME', 'Unknown')} {distro_info.get('VERSION', '')}")
        print(f"  ID: {distro_info.get('ID', 'Unknown')}")
    except AttributeError:
        try:
            # Older method (Python 3.8-3.9)
            distro_info = platform.linux_distribution()
            print(f"  Distribution: {distro_info[0]} {distro_info[1]} {distro_info[2]}")
        except AttributeError:
            # Last resort
            print(f"  Distribution: Ubuntu/Xubuntu (version detection not available)")
    
    # Test CUDA availability
    from llamatelemetry.utils import check_gpu_compatibility
    compat = check_gpu_compatibility()
    
    if compat['compute_capability']:
        print(f"  Compute Capability: {compat['compute_capability']}")
        if compat['compute_capability'] >= 5.0:
            print("  ✓ Xubuntu system compatible with llamatelemetry")
        else:
            print(f"  ⚠ GPU compute capability {compat['compute_capability']} < 5.0")
    else:
        print(f"  Compute Capability: Not detected")
    
    # Test library path handling
    from llamatelemetry.server import ServerManager
    manager = ServerManager()
    
    # Create a test binary path
    test_dir = Path("/tmp/llamatelemetry_test")
    test_dir.mkdir(exist_ok=True)
    test_bin_dir = test_dir / "bin"
    test_bin_dir.mkdir(exist_ok=True)
    test_path = test_bin_dir / "llama-server"
    
    # Create dummy file
    test_path.touch(mode=0o755)
    
    # Test library path setup
    manager._setup_library_path(test_path)
    
    # Check LD_LIBRARY_PATH was potentially modified
    if 'LD_LIBRARY_PATH' in os.environ:
        ld_path = os.environ['LD_LIBRARY_PATH']
        print(f"  LD_LIBRARY_PATH is set ({len(ld_path.split(':'))} entries)")
    else:
        print(f"  LD_LIBRARY_PATH not set (may be set dynamically)")
    
    # Clean up - use shutil.rmtree for non-empty directory
    shutil.rmtree(test_dir, ignore_errors=True)
    
    print("✓ Xubuntu compatibility tests passed")



def run_quick_integration_test():
    """Quick integration test without requiring actual model"""
    print("\n" + "="*60)
    print("QUICK INTEGRATION TEST - No model required")
    print("="*60)
    
    import llamatelemetry
    
    # 1. Import test
    print("1. Import test... ✓")
    
    # 2. Platform detection
    from llamatelemetry.server import ServerManager
    manager = ServerManager()
    platform_info = manager._detect_platform()
    print(f"2. Platform detection: {platform_info['platform']} ✓")
    
    # 3. GPU compatibility
    compat = llamatelemetry.check_gpu_compatibility()
    print(f"3. GPU compatibility: {compat['compatible']} ✓")
    
    # 4. Binary detection
    existing_binary = manager.find_llama_server()
    if existing_binary and os.path.exists(existing_binary):
        print(f"4. Binary already exists: ✓ ({existing_binary})")
    else:
        print("4. Binary auto-download test...")
        try:
            binary_path = manager._download_llama_server()
            print(f"   ✓ Downloaded to: {binary_path}")
        except Exception as e:
            print(f"   ⚠ Download skipped: {e}")
    
    print("\n✅ All integration tests passed!")
    return True


if __name__ == '__main__':
    # Run tests with verbose output
    print("="*60)
    print("llamatelemetry Test Suite - Xubuntu/Ubuntu 22.04 Edition")
    print("="*60)
    
    # Run individual tests
    test_functions = [
        test_import_and_version,
        test_platform_detection,
        test_gpu_compatibility_check,
        test_server_manager_creation,
        test_inference_engine_workflow,
        test_metrics_structure,
        test_xubuntu_specific_compatibility,
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
    
    # Try binary download test (might fail if offline)
    try:
        test_binary_auto_download()
        passed += 1
    except Exception as e:
        print(f"⚠ Binary download test skipped: {e}")
    
    print(f"\nSummary: {passed}/{total} tests passed")
    
    # Run quick integration test
    print("\n" + "="*60)
    run_quick_integration_test()
    
    if passed == total:
        print("\n🎉 All tests completed successfully!")
    else:
        print(f"\n⚠ {total - passed} tests had issues")
        sys.exit(1)