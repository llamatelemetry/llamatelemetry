# llamatelemetry v2.2.0 - Configuration Guide

Complete reference for configuring llamatelemetry server and client components.

---

## Table of Contents

- [Server Configuration](#server-configuration)
- [Client Configuration](#client-configuration)
- [Multi-GPU Configuration](#multi-gpu-configuration)
- [Performance Tuning](#performance-tuning)
- [Environment Variables](#environment-variables)
- [Configuration Examples](#configuration-examples)

---

## Server Configuration

### ServerConfig Class

The `ServerConfig` class provides all server configuration options:

```python
from llamatelemetry.server import ServerConfig

config = ServerConfig(
    # Model
    model_path="/path/to/model.gguf",
    
    # Network
    host="127.0.0.1",
    port=8080,
    
    # GPU
    n_gpu_layers=99,
    main_gpu=0,
    tensor_split=None,
    
    # Context
    context_size=4096,
    n_batch=512,
    
    # Features
    flash_attn=True,
    embeddings=False,
    
    # Threading
    threads=4,
    threads_batch=4,
)
```

### Configuration Parameters

#### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | Required | Path to GGUF model file |
| `lora_path` | str | None | Path to LoRA adapter |
| `mmproj_path` | str | None | Multimodal projector path |

#### Network Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | "127.0.0.1" | Server bind address |
| `port` | int | 8080 | Server port |
| `api_key` | str | None | API key for authentication |

#### GPU Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_gpu_layers` | int | 0 | Layers to offload to GPU (99 = all) |
| `main_gpu` | int | 0 | Primary GPU index |
| `tensor_split` | str | None | VRAM split ratio (e.g., "0.5,0.5") |
| `split_mode` | str | "layer" | Split mode: "none", "layer", "row" |

#### Context Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context_size` | int | 2048 | Maximum context window |
| `n_batch` | int | 512 | Batch size for prompt processing |
| `n_ubatch` | int | 512 | Physical batch size |

#### Feature Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flash_attn` | bool | False | Enable flash attention |
| `embeddings` | bool | False | Enable embeddings endpoint |
| `mlock` | bool | False | Lock model in RAM |
| `no_mmap` | bool | False | Disable memory mapping |

#### Threading Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threads` | int | 4 | Threads for generation |
| `threads_batch` | int | 4 | Threads for batch processing |
| `parallel` | int | 1 | Number of parallel sequences |

### ServerManager Class

```python
from llamatelemetry.server import ServerManager, ServerConfig

# Create manager
server = ServerManager()

# Start with config
config = ServerConfig(
    model_path="model.gguf",
    n_gpu_layers=99,
    flash_attn=True,
)
server.start_with_config(config)

# Wait for ready
if server.wait_until_ready(timeout=120):
    print("Server ready!")

# Check status
print(f"Running: {server.is_running()}")
print(f"URL: {server.get_url()}")

# Stop server
server.stop()
```

### CLI Arguments

Convert configuration to CLI arguments:

```python
config = ServerConfig(
    model_path="model.gguf",
    n_gpu_layers=99,
    tensor_split="0.5,0.5",
    flash_attn=True,
)

print(config.to_cli_args())
# Output: -m model.gguf -ngl 99 --tensor-split 0.5,0.5 -fa
```

---

## Client Configuration

### LlamaCppClient

```python
from llamatelemetry.api.client import LlamaCppClient

client = LlamaCppClient(
    base_url="http://127.0.0.1:8080",
    api_key=None,  # Optional API key
    timeout=30,    # Request timeout
)
```

### Client Methods

#### Health Check
```python
health = client.health()
print(health)  # {"status": "ok"}
```

#### List Models
```python
models = client.list_models()
for model in models.data:
    print(model.id)
```

#### Chat Completion
```python
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
    stream=False,
)
print(response.choices[0].message.content)
```

#### Streaming
```python
for chunk in client.chat_completion_stream(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

#### Text Completion
```python
response = client.completion(
    prompt="The capital of France is",
    max_tokens=50,
    temperature=0.5,
)
print(response.choices[0].text)
```

#### Embeddings
```python
embeddings = client.embeddings(["Hello", "World"])
print(f"Dimension: {len(embeddings.data[0].embedding)}")
```

#### Tokenization
```python
tokens = client.tokenize("Hello, world!")
print(f"Tokens: {tokens.tokens}")

text = client.detokenize(tokens.tokens)
print(f"Text: {text.content}")
```

### OpenAI SDK Compatibility

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="any-model-name",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

---

## Multi-GPU Configuration

### Kaggle Dual T4 Configuration

```python
from llamatelemetry.api import kaggle_t4_dual_config

config = kaggle_t4_dual_config()
print(config.to_cli_args())
# Output: -ngl 99 --tensor-split 0.5,0.5 --split-mode layer -fa
```

### Manual Multi-GPU Configuration

```python
from llamatelemetry.server import ServerConfig

config = ServerConfig(
    model_path="model.gguf",
    
    # Multi-GPU settings
    n_gpu_layers=99,        # Offload all layers
    main_gpu=0,             # Primary GPU
    tensor_split="0.5,0.5", # 50% each GPU
    split_mode="layer",     # Layer-wise split
    
    # Performance
    flash_attn=True,
    context_size=4096,
)
```

### Tensor Split Ratios

| Configuration | GPU 0 | GPU 1 | Use Case |
|---------------|-------|-------|----------|
| `"0.5,0.5"` | 50% | 50% | Equal GPUs |
| `"0.6,0.4"` | 60% | 40% | Unequal VRAM |
| `"1,0"` | 100% | 0% | Single GPU |
| `"0.48,0.48"` | 48% | 48% | Leave headroom |

### Split Mode Options

| Mode | Description | Use Case |
|------|-------------|----------|
| `"layer"` | Split by layers | Default, works best |
| `"row"` | Split by tensor rows | Larger models |
| `"none"` | No split | Single GPU |

---

## Performance Tuning

### Context Size vs VRAM

| Context | KV Cache (7B) | KV Cache (70B) |
|---------|---------------|----------------|
| 512 | ~0.5 GB | ~2 GB |
| 2048 | ~1.5 GB | ~6 GB |
| 4096 | ~3 GB | ~12 GB |
| 8192 | ~6 GB | ~24 GB |

### Batch Size Tuning

```python
config = ServerConfig(
    model_path="model.gguf",
    n_gpu_layers=99,
    
    # Batch tuning
    n_batch=512,     # Prompt processing batch
    n_ubatch=256,    # Physical batch size
)
```

| n_batch | Memory | Speed | Recommendation |
|---------|--------|-------|----------------|
| 128 | Low | Slower | Memory-constrained |
| 256 | Medium | Medium | Balanced |
| 512 | High | Fast | Default |
| 1024 | Higher | Fastest | Large VRAM |

### Thread Configuration

```python
import os

# Get CPU count
cpu_count = os.cpu_count()

config = ServerConfig(
    model_path="model.gguf",
    threads=min(8, cpu_count),       # Generation threads
    threads_batch=min(8, cpu_count), # Batch threads
)
```

### Memory Optimization

```python
config = ServerConfig(
    model_path="model.gguf",
    
    # Memory options
    mlock=True,      # Lock model in RAM (prevents swapping)
    no_mmap=False,   # Use memory mapping (saves RAM)
    
    # For memory-constrained systems
    n_gpu_layers=30, # Partial GPU offload
    context_size=2048,
    n_batch=256,
)
```

---

## Environment Variables

### llamatelemetry Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLCUDA_BINARY_PATH` | Path to binaries | `~/.cache/llamatelemetry/bin` |
| `LLCUDA_MODEL_PATH` | Default model directory | `~/.cache/llamatelemetry/models` |
| `LLCUDA_LOG_LEVEL` | Logging level | `INFO` |

### CUDA Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU visibility | `"0,1"` or `"1"` |
| `CUDA_DEVICE_ORDER` | Device ordering | `"PCI_BUS_ID"` |

### Example Usage

```python
import os

# Use only GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Custom binary path
os.environ['LLCUDA_BINARY_PATH'] = '/custom/path/bin'

import llamatelemetry
```

---

## Configuration Examples

### Example: Dual GPU (Kaggle 2× T4)

```python
from llamatelemetry.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path="qwen2.5-7b-Q4_K_M.gguf",
    host="0.0.0.0",
    port=8080,
    n_gpu_layers=99,
    tensor_split="0.5,0.5",
    split_mode="layer",
    context_size=4096,
    flash_attn=True,
)

server = ServerManager()
server.start_with_config(config)
```

### Example 3: Large Model (70B on Dual T4)

```python
from llamatelemetry.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path="llama-70b-IQ3_XS.gguf",
    host="0.0.0.0",
    port=8080,
    n_gpu_layers=99,
    tensor_split="0.48,0.48",  # Leave headroom
    split_mode="layer",
    context_size=2048,         # Smaller context
    n_batch=128,               # Smaller batch
    flash_attn=True,
)

server = ServerManager()
server.start_with_config(config)
```

### Example 4: Split-GPU (LLM + RAPIDS)

```python
from llamatelemetry.server import ServerManager, ServerConfig
import os

# GPU 0 for LLM only
config = ServerConfig(
    model_path="gemma-7b-Q4_K_M.gguf",
    host="127.0.0.1",
    port=8080,
    n_gpu_layers=99,
    main_gpu=0,
    tensor_split="1,0",  # All on GPU 0
    flash_attn=True,
)

server = ServerManager()
server.start_with_config(config)

# GPU 1 for RAPIDS
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cudf  # Uses GPU 1
```

### Example 5: Production Server

```python
from llamatelemetry.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path="production-model.gguf",
    host="0.0.0.0",
    port=8080,
    api_key="your-secret-key",  # Enable authentication
    n_gpu_layers=99,
    tensor_split="0.5,0.5",
    context_size=8192,
    parallel=4,                  # Multiple concurrent users
    flash_attn=True,
    embeddings=True,             # Enable embeddings
    mlock=True,                  # Lock in RAM
)

server = ServerManager()
server.start_with_config(config)
```

---

## Next Steps

- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Kaggle Guide](KAGGLE_GUIDE.md)** - Kaggle-specific configuration
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
