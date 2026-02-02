# llamatelemetry v2.2.0 - Troubleshooting Guide

Common issues and solutions for llamatelemetry on Kaggle and other environments.

---

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Installation Issues](#installation-issues)
- [GPU Issues](#gpu-issues)
- [Server Issues](#server-issues)
- [Memory Issues](#memory-issues)
- [Model Issues](#model-issues)
- [API Issues](#api-issues)
- [Performance Issues](#performance-issues)
- [Kaggle-Specific Issues](#kaggle-specific-issues)
- [Getting Help](#getting-help)

---

## Quick Diagnostics

Run this diagnostic script to check your environment:

```python
#!/usr/bin/env python3
"""llamatelemetry diagnostic script."""

import sys
print(f"Python: {sys.version}")

# Check llamatelemetry
try:
    import llamatelemetry
    print(f"✅ llamatelemetry: {llamatelemetry.__version__}")
except ImportError:
    print("❌ llamatelemetry: not installed")

# Check CUDA
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
except ImportError:
    print("❌ PyTorch: not installed")

# Check binary
import subprocess
result = subprocess.run(["which", "llama-server"], capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ llama-server: {result.stdout.strip()}")
else:
    print("❌ llama-server: not found in PATH")

# Check disk space
import shutil
total, used, free = shutil.disk_usage("/")
print(f"ℹ️  Disk space: {free // 1024**3} GB free")
```

---

## Installation Issues

### llamatelemetry Not Found

**Error:**
```
ModuleNotFoundError: No module named 'llamatelemetry'
```

**Solution:**
```python
# Install llamatelemetry
!pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v2.2.0

# Restart kernel after installation
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
```

---

### Binary Not Found

**Error:**
```
FileNotFoundError: llama-server not found
```

**Solution:**
```python
from llamatelemetry.server import ServerManager

# Download binaries
server = ServerManager()
server.ensure_binaries()

# Or manually
!llamatelemetry-download-binaries
```

---

### Permission Denied on Binary

**Error:**
```
PermissionError: [Errno 13] Permission denied: './bin/llama-server'
```

**Solution:**
```bash
# Make binary executable
chmod +x ./bin/llama-server

# Or from Python
import os
import stat
os.chmod("./bin/llama-server", stat.S_IRWXU)
```

---

### pip Install Timeout

**Error:**
```
ReadTimeoutError: HTTPSConnectionPool timed out
```

**Solution:**
```python
# Install with longer timeout
!pip install --default-timeout=100 git+https://github.com/llamatelemetry/llamatelemetry.git
```

---

## GPU Issues

### CUDA Not Available

**Error:**
```python
>>> torch.cuda.is_available()
False
```

**Solutions:**

1. **Kaggle:** Ensure GPU accelerator is selected
   - Settings → Accelerator → GPU T4 × 2

2. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   ```

3. **Reinstall PyTorch with CUDA:**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

---

### Wrong GPU Count

**Error:**
```python
>>> torch.cuda.device_count()
1  # Expected 2
```

**Solution:**
```python
# Check CUDA_VISIBLE_DEVICES
import os
print(os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))

# Reset to use all GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
```

---

### GPU Already in Use

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate...
```

**Solution:**
```python
import gc
import torch

# Kill existing processes
def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

cleanup_gpu()

# Check what's using GPU
!nvidia-smi
```

---

## Server Issues

### Server Won't Start

**Error:**
```
Server failed to start within timeout
```

**Diagnostics:**
```python
from llamatelemetry.server import ServerManager, ServerConfig

server = ServerManager()
server.start_with_config(config)

# Check if process started
print(f"Process running: {server.is_running()}")

# Get logs
print("Server logs:")
print(server.get_logs())
```

**Common Causes:**

| Cause | Solution |
|-------|----------|
| Binary not found | Run `server.ensure_binaries()` |
| Model not found | Check model path exists |
| Port in use | Use different port |
| Out of memory | Use smaller model or context |
| Invalid config | Check ServerConfig parameters |

---

### Port Already in Use

**Error:**
```
Address already in use (os error 98)
```

**Solution:**
```python
# Use different port
config = ServerConfig(
    model_path="model.gguf",
    port=8081,  # Try different port
)

# Or kill process using port
!lsof -ti:8080 | xargs kill -9
```

---

### Server Health Check Fails

**Error:**
```
Health check timeout
```

**Solution:**
```python
# Increase timeout
server.wait_until_ready(timeout=180)  # 3 minutes

# Check if server is responding
import requests
try:
    r = requests.get("http://127.0.0.1:8080/health", timeout=5)
    print(f"Status: {r.status_code}")
except requests.exceptions.ConnectionError:
    print("Server not responding")
```

---

## Memory Issues

### CUDA Out of Memory

**Error:**
```
CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. **Reduce context size:**
   ```python
   config = ServerConfig(
       model_path="model.gguf",
       context_size=2048,  # Reduce from 4096
   )
   ```

2. **Use smaller quantization:**
   ```python
   # Use Q4_K_M instead of Q8_0
   model_path = "model-Q4_K_M.gguf"  # Not model-Q8_0.gguf
   ```

3. **Leave headroom:**
   ```python
   config = ServerConfig(
       tensor_split="0.45,0.45",  # Not 0.5,0.5
   )
   ```

4. **Reduce batch size:**
   ```python
   config = ServerConfig(
       n_batch=256,  # Reduce from 512
   )
   ```

---

### Model Too Large for VRAM

**Error:**
```
Model requires 42 GB but only 30 GB available
```

**Solutions:**

| Current State | Solution |
|---------------|----------|
| Q4_K_M doesn't fit | Use Q3_K_M or IQ3_XS |
| 70B doesn't fit | Use IQ2_XXS or smaller model |
| Single GPU | Enable dual GPU with tensor_split |

```python
# Example: 70B on dual T4
config = ServerConfig(
    model_path="70b-IQ3_XS.gguf",  # Use I-quant
    tensor_split="0.48,0.48",
    context_size=2048,
)
```

---

### Memory Leak Over Time

**Symptoms:**
- VRAM usage slowly increases
- Eventually OOM after many requests

**Solution:**
```python
# Periodic cleanup
import gc
import torch

def periodic_cleanup():
    gc.collect()
    torch.cuda.empty_cache()

# Call every N requests
request_count = 0
for request in requests:
    response = process(request)
    request_count += 1
    if request_count % 100 == 0:
        periodic_cleanup()
```

---

## Model Issues

### Model File Not Found

**Error:**
```
Error: unable to load model: model.gguf not found
```

**Solution:**
```python
import os

# Check if file exists
model_path = "/kaggle/working/models/model.gguf"
print(f"Exists: {os.path.exists(model_path)}")

# List directory
print(os.listdir("/kaggle/working/models/"))
```

---

### Invalid GGUF Format

**Error:**
```
Error: invalid gguf magic
```

**Causes:**
- Corrupted download
- Not a GGUF file
- Wrong file format

**Solution:**
```python
# Re-download model
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",
    force_download=True,  # Force re-download
    local_dir="/kaggle/working/models",
)
```

---

### Wrong Quantization for Hardware

**Symptoms:**
- Very slow inference
- Unexpected memory usage

**Solution:**
```python
from llamatelemetry.api.gguf import GGUFParser

# Check model info
parser = GGUFParser(model_path)
info = parser.parse()
print(f"Quantization: {info['quantization']}")
print(f"Estimated VRAM: {parser.estimate_vram()} GB")
```

---

## API Issues

### Connection Refused

**Error:**
```
requests.exceptions.ConnectionError: Connection refused
```

**Solutions:**

1. **Check server is running:**
   ```python
   print(f"Server running: {server.is_running()}")
   ```

2. **Check correct host/port:**
   ```python
   from llamatelemetry.api.client import LlamaCppClient
   
   client = LlamaCppClient(
       host="127.0.0.1",  # Not "localhost"
       port=8080,
   )
   ```

3. **Wait for server ready:**
   ```python
   server.wait_until_ready(timeout=120)
   ```

---

### Request Timeout

**Error:**
```
requests.exceptions.ReadTimeout
```

**Solution:**
```python
from llamatelemetry.api.client import LlamaCppClient

client = LlamaCppClient(timeout=120)  # Increase timeout

# Or per-request
response = client.chat_completion(
    messages=[...],
    max_tokens=100,  # Limit output length
)
```

---

### Invalid Response Format

**Error:**
```
KeyError: 'choices'
```

**Solution:**
```python
# Check raw response
import requests

r = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    json={"messages": [{"role": "user", "content": "test"}]},
)
print(f"Status: {r.status_code}")
print(f"Response: {r.text}")
```

---

## Performance Issues

### Slow Inference

**Symptoms:**
- < 10 tokens/second
- Long time to first token

**Solutions:**

1. **Enable flash attention:**
   ```python
   config = ServerConfig(
       flash_attn=True,
   )
   ```

2. **Use full GPU offload:**
   ```python
   config = ServerConfig(
       n_gpu_layers=99,  # All layers on GPU
   )
   ```

3. **Increase threads:**
   ```python
   config = ServerConfig(
       n_threads=8,
   )
   ```

---

### Slow First Token

**Symptoms:**
- Long delay before first token
- Fast generation after first token

**Cause:** Context caching / prompt processing

**Solution:**
```python
# Pre-warm with short prompt
client.chat_completion(
    messages=[{"role": "user", "content": "Hi"}],
    max_tokens=1,
)
# Now subsequent prompts will be faster
```

---

## Kaggle-Specific Issues

### Session Expired

**Error:**
```
Session expired. Please restart.
```

**Prevention:**
- Save checkpoints regularly
- Use `/kaggle/working/` for persistence
- Monitor session time

```python
import time
SESSION_START = time.time()

def check_time():
    hours = (time.time() - SESSION_START) / 3600
    print(f"Session time: {hours:.1f}h / 12h")
    return hours < 11.5

if not check_time():
    print("⚠️ Save your work now!")
```

---

### RAPIDS Conflicts

**Error:**
```
ImportError after upgrading packages
```

**Solution:**
```python
# DON'T upgrade these on Kaggle
# pip install --upgrade cuda-python  ❌
# pip install --upgrade numba-cuda   ❌

# Use pre-installed versions
import cudf
print(cudf.__version__)  # Use as-is
```

---

### Internet Disconnected

**Error:**
```
Could not download model
```

**Solution:**
- Check Settings → Internet → On
- Use cached models when possible
- Download models at notebook start

---

## Getting Help

### Collect Debug Information

```python
# Collect system info
import sys
import platform

debug_info = {
    "python": sys.version,
    "platform": platform.platform(),
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
}

# llamatelemetry info
try:
    import llamatelemetry
    debug_info["llamatelemetry_version"] = llamatelemetry.__version__
except:
    debug_info["llamatelemetry_version"] = "not installed"

# Server logs
try:
    debug_info["server_logs"] = server.get_logs()[-500:]  # Last 500 chars
except:
    pass

print(debug_info)
```

### Where to Get Help

| Resource | URL |
|----------|-----|
| GitHub Issues | https://github.com/llamatelemetry/llamatelemetry/issues |
| Documentation | [`docs/`](.) |
| Kaggle Guide | [`KAGGLE_GUIDE.md`](KAGGLE_GUIDE.md) |

---

## Quick Reference

### Error → Solution Table

| Error | Quick Fix |
|-------|-----------|
| `No module named 'llamatelemetry'` | `pip install llamatelemetry` |
| `llama-server not found` | `server.ensure_binaries()` |
| `CUDA out of memory` | Reduce context_size |
| `Port already in use` | Change port number |
| `Server timeout` | Check `server.get_logs()` |
| `Model not found` | Verify file path |
| `Connection refused` | Wait for server ready |
| `Slow inference` | Enable `flash_attn=True` |
