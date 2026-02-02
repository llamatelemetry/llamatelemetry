# llamatelemetry v2.2.0 - API Reference

Complete Python API documentation for llamatelemetry.

---

## Table of Contents

- [Core Modules](#core-modules)
- [llamatelemetry.server](#llcudaserver)
- [llamatelemetry.api.client](#llcudaapiclient)
- [llamatelemetry.api.gguf](#llcudaapigguf)
- [llamatelemetry.api.multigpu](#llcudaapimultigpu)
- [Data Classes](#data-classes)
- [Exceptions](#exceptions)

---

## Core Modules

### Module Overview

| Module | Description |
|--------|-------------|
| `llamatelemetry` | Main package, version info |
| `llamatelemetry.server` | Server management (ServerManager, ServerConfig) |
| `llamatelemetry.api.client` | llama.cpp client (LlamaCppClient) |
| `llamatelemetry.api.gguf` | GGUF parsing and quantization |
| `llamatelemetry.api.multigpu` | Multi-GPU configuration |
| `llamatelemetry.api.nccl` | NCCL utilities (PyTorch distributed) |

### Import Examples

```python
# Main package
import llamatelemetry
print(llamatelemetry.__version__)

# Server management
from llamatelemetry.server import ServerManager, ServerConfig

# Client API
from llamatelemetry.api.client import LlamaCppClient

# GGUF utilities
from llamatelemetry.api.gguf import GGUFParser, QUANT_TYPE_INFO

# Multi-GPU
from llamatelemetry.api import kaggle_t4_dual_config
```

---

## llamatelemetry.server

### ServerConfig

Configuration class for llama-server.

```python
from llamatelemetry.server import ServerConfig

config = ServerConfig(
    model_path: str,              # Required: Path to GGUF model
    host: str = "127.0.0.1",      # Server bind address
    port: int = 8080,             # Server port
    n_gpu_layers: int = 0,        # Layers to offload (99 = all)
    main_gpu: int = 0,            # Primary GPU index
    tensor_split: str = None,     # VRAM split (e.g., "0.5,0.5")
    split_mode: str = "layer",    # Split mode
    context_size: int = 2048,     # Context window
    n_batch: int = 512,           # Batch size
    flash_attn: bool = False,     # Flash attention
    embeddings: bool = False,     # Enable embeddings
    threads: int = 4,             # CPU threads
    api_key: str = None,          # API authentication
)
```

#### Methods

```python
# Convert to CLI arguments string
config.to_cli_args() -> str

# Convert to dictionary
config.to_dict() -> dict

# Create from dictionary
ServerConfig.from_dict(data: dict) -> ServerConfig
```

### ServerManager

Manages llama-server lifecycle.

```python
from llamatelemetry.server import ServerManager, ServerConfig

server = ServerManager()
```

#### Methods

```python
# Start server with configuration
server.start_with_config(config: ServerConfig) -> bool

# Start with CLI arguments
server.start(args: str) -> bool

# Wait for server to be ready
server.wait_until_ready(timeout: int = 60) -> bool

# Check if server is running
server.is_running() -> bool

# Get server URL
server.get_url() -> str

# Stop server
server.stop() -> None

# Get server logs
server.get_logs() -> str
```

#### Example

```python
from llamatelemetry.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path="model.gguf",
    n_gpu_layers=99,
    flash_attn=True,
)

server = ServerManager()
server.start_with_config(config)

if server.wait_until_ready(timeout=120):
    print(f"Server ready at {server.get_url()}")
else:
    print("Server failed to start")
    print(server.get_logs())

# ... use server ...

server.stop()
```

---

## llamatelemetry.api.client

### LlamaCppClient

HTTP client for llama-server with OpenAI-compatible API.

```python
from llamatelemetry.api.client import LlamaCppClient

client = LlamaCppClient(
    base_url: str = "http://127.0.0.1:8080",
    api_key: str = None,
    timeout: int = 30,
)
```

#### Health & Status Methods

```python
# Health check
client.health() -> dict
# Returns: {"status": "ok"}

# List available models
client.list_models() -> ModelList
# Returns: ModelList with .data list of Model objects

# Get model properties
client.get_props() -> dict

# Get slot status
client.get_slots() -> list
```

#### Chat Completion

```python
client.chat_completion(
    messages: list[dict],         # Required: Chat messages
    max_tokens: int = 256,        # Max tokens to generate
    temperature: float = 0.8,     # Sampling temperature
    top_p: float = 0.95,          # Nucleus sampling
    top_k: int = 40,              # Top-k sampling
    repeat_penalty: float = 1.1,  # Repetition penalty
    stop: list[str] = None,       # Stop sequences
    seed: int = None,             # Random seed
    stream: bool = False,         # Enable streaming
) -> ChatCompletionResponse
```

**Example:**
```python
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100,
    temperature=0.7,
)

print(response.choices[0].message.content)
print(f"Tokens: {response.usage.total_tokens}")
```

#### Streaming Chat Completion

```python
client.chat_completion_stream(
    messages: list[dict],
    **kwargs
) -> Generator[ChatCompletionChunk, None, None]
```

**Example:**
```python
for chunk in client.chat_completion_stream(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### Text Completion

```python
client.completion(
    prompt: str,                  # Required: Text prompt
    max_tokens: int = 256,
    temperature: float = 0.8,
    stop: list[str] = None,
    **kwargs
) -> CompletionResponse
```

**Example:**
```python
response = client.completion(
    prompt="The capital of France is",
    max_tokens=50,
)
print(response.choices[0].text)
```

#### Embeddings

```python
client.embeddings(
    input: str | list[str],       # Text or list of texts
    model: str = None,
) -> EmbeddingResponse
```

**Example:**
```python
response = client.embeddings(["Hello", "World"])
print(f"Dimension: {len(response.data[0].embedding)}")
```

#### Tokenization

```python
# Tokenize text
client.tokenize(text: str) -> TokenizeResponse
# Returns: TokenizeResponse with .tokens list

# Detokenize tokens
client.detokenize(tokens: list[int]) -> DetokenizeResponse
# Returns: DetokenizeResponse with .content string
```

**Example:**
```python
tokens = client.tokenize("Hello, world!")
print(f"Tokens: {tokens.tokens}")

text = client.detokenize(tokens.tokens)
print(f"Text: {text.content}")
```

---

## llamatelemetry.api.gguf

### GGUFParser

Parse GGUF model files.

```python
from llamatelemetry.api.gguf import GGUFParser

parser = GGUFParser(file_path: str)
```

#### Methods

```python
# Get model metadata
parser.get_metadata() -> dict

# Get architecture info
parser.get_architecture() -> str

# Get parameter count
parser.get_parameter_count() -> int

# Get quantization type
parser.get_quantization_type() -> str

# Get context length
parser.get_context_length() -> int

# Get vocabulary size
parser.get_vocab_size() -> int
```

**Example:**
```python
from llamatelemetry.api.gguf import GGUFParser

parser = GGUFParser("model.gguf")
print(f"Architecture: {parser.get_architecture()}")
print(f"Parameters: {parser.get_parameter_count() / 1e9:.1f}B")
print(f"Quantization: {parser.get_quantization_type()}")
```

### QUANT_TYPE_INFO

Quantization type information dataclass.

```python
from llamatelemetry.api.gguf import QUANT_TYPE_INFO

@dataclass
class QUANT_TYPE_INFO:
    name: str           # e.g., "Q4_K_M"
    bits_per_weight: float
    description: str
    recommended_for: str
    vram_7b_gb: float   # Estimated VRAM for 7B model
```

### Quantization Reference

```python
from llamatelemetry.api.gguf import get_quantization_info

# Get info for specific type
info = get_quantization_info("Q4_K_M")
print(f"Bits: {info.bits_per_weight}")
print(f"VRAM (7B): {info.vram_7b_gb} GB")

# List all types
for quant in list_quantization_types():
    print(quant)
```

### Helper Functions

```python
from llamatelemetry.api.gguf import (
    estimate_vram,
    recommend_quantization,
    kaggle_t4_recommended_models,
)

# Estimate VRAM for model
vram = estimate_vram(
    parameter_count=7e9,
    quantization="Q4_K_M",
    context_size=4096
)
print(f"Estimated VRAM: {vram:.1f} GB")

# Get recommendation for VRAM budget
quant = recommend_quantization(
    parameter_count=7e9,
    available_vram=15  # GB
)
print(f"Recommended: {quant}")

# Kaggle-optimized recommendations
models = kaggle_t4_recommended_models()
for model in models:
    print(f"{model.name}: {model.quantization}")
```

---

## llamatelemetry.api.multigpu

### Multi-GPU Configuration Functions

```python
from llamatelemetry.api.multigpu import (
    kaggle_t4_dual_config,
    create_tensor_split_config,
    detect_gpus,
)
```

#### kaggle_t4_dual_config

Get optimal configuration for Kaggle's dual T4 GPUs.

```python
from llamatelemetry.api import kaggle_t4_dual_config

config = kaggle_t4_dual_config(
    model_path: str = None,       # Optional model path
    context_size: int = 4096,     # Context window
) -> ServerConfig

print(config.to_cli_args())
```

#### create_tensor_split_config

Create custom tensor split configuration.

```python
from llamatelemetry.api.multigpu import create_tensor_split_config

config = create_tensor_split_config(
    gpu_memory: list[float],      # VRAM per GPU in GB
    model_size_gb: float,         # Model size in GB
) -> str

# Example: Two GPUs with 15GB each, 10GB model
split = create_tensor_split_config([15, 15], 10)
print(split)  # "0.5,0.5"
```

#### detect_gpus

Detect available GPUs.

```python
from llamatelemetry.api.multigpu import detect_gpus

gpus = detect_gpus() -> list[GPUInfo]

for gpu in gpus:
    print(f"GPU {gpu.index}: {gpu.name}")
    print(f"  Memory: {gpu.total_memory_gb} GB")
    print(f"  Compute: {gpu.compute_capability}")
```

---

## Data Classes

### Response Objects

```python
from llamatelemetry.api.client import (
    ChatCompletionResponse,
    ChatCompletionChunk,
    CompletionResponse,
    EmbeddingResponse,
    ModelList,
)
```

#### ChatCompletionResponse

```python
@dataclass
class ChatCompletionResponse:
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: str

@dataclass
class Message:
    role: str
    content: str

@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

#### ChatCompletionChunk (Streaming)

```python
@dataclass
class ChatCompletionChunk:
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]

@dataclass
class ChunkChoice:
    index: int
    delta: Delta
    finish_reason: str | None

@dataclass
class Delta:
    role: str | None
    content: str | None
```

#### EmbeddingResponse

```python
@dataclass
class EmbeddingResponse:
    object: str = "list"
    data: list[Embedding]
    model: str
    usage: EmbeddingUsage

@dataclass
class Embedding:
    object: str = "embedding"
    embedding: list[float]
    index: int
```

### GPU Information

```python
from llamatelemetry.api.multigpu import GPUInfo

@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    compute_capability: str
```

---

## Exceptions

### Exception Hierarchy

```python
from llamatelemetry.exceptions import (
    LLCudaError,           # Base exception
    ServerError,           # Server-related errors
    ClientError,           # Client-related errors
    ModelError,            # Model loading errors
    ConfigurationError,    # Configuration errors
)
```

### Exception Handling

```python
from llamatelemetry.server import ServerManager, ServerConfig
from llamatelemetry.exceptions import ServerError, ModelError

try:
    server = ServerManager()
    server.start_with_config(config)
except ModelError as e:
    print(f"Model error: {e}")
except ServerError as e:
    print(f"Server error: {e}")
```

---

## Usage Patterns

### Pattern 1: Quick Inference

```python
from llamatelemetry.server import ServerManager, ServerConfig
from llamatelemetry.api.client import LlamaCppClient

# Start server
config = ServerConfig(model_path="model.gguf", n_gpu_layers=99)
server = ServerManager()
server.start_with_config(config)
server.wait_until_ready()

# Create client
client = LlamaCppClient()

# Chat
response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Cleanup
server.stop()
```

### Pattern 2: Multi-GPU Inference

```python
from llamatelemetry.api import kaggle_t4_dual_config
from llamatelemetry.server import ServerManager
from llamatelemetry.api.client import LlamaCppClient

# Get optimized config
config = kaggle_t4_dual_config(model_path="model.gguf")

# Start server
server = ServerManager()
server.start_with_config(config)
server.wait_until_ready()

# Use client
client = LlamaCppClient()
# ... inference ...

server.stop()
```

### Pattern 3: Streaming with OpenAI SDK

```python
from openai import OpenAI
from llamatelemetry.server import ServerManager, ServerConfig

# Start server
server = ServerManager()
server.start_with_config(ServerConfig(
    model_path="model.gguf",
    n_gpu_layers=99,
))
server.wait_until_ready()

# Use OpenAI SDK
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed"
)

stream = client.chat.completions.create(
    model="model",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

server.stop()
```

---

## Next Steps

- **[Configuration Guide](CONFIGURATION.md)** - Detailed configuration options
- **[Kaggle Guide](KAGGLE_GUIDE.md)** - Platform-specific API usage
- **[Tutorial Notebooks](../notebooks/README.md)** - Interactive examples
