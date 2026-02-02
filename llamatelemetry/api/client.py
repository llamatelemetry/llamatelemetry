"""
llamatelemetry.api.client - Main LlamaCppClient class

Unified client for all llama.cpp server endpoints with OpenAI-compatible interface.
"""

import json
import requests
from typing import Optional, Dict, Any, List, Iterator, Union
from dataclasses import dataclass, field
from enum import Enum
import time

# Try to import sseclient for streaming support
try:
    import sseclient
    _HAS_SSE = True
except ImportError:
    _HAS_SSE = False


class StopType(Enum):
    """Completion stop types."""
    NONE = "none"
    EOS = "eos"
    LIMIT = "limit"
    WORD = "word"


@dataclass
class Message:
    """Chat message."""
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    reasoning_content: Optional[str] = None


@dataclass
class Choice:
    """Completion choice."""
    index: int
    message: Optional[Message] = None
    text: Optional[str] = None
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


@dataclass
class Usage:
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Timings:
    """Performance timings."""
    prompt_n: int = 0
    prompt_ms: float = 0.0
    prompt_per_token_ms: float = 0.0
    prompt_per_second: float = 0.0
    predicted_n: int = 0
    predicted_ms: float = 0.0
    predicted_per_token_ms: float = 0.0
    predicted_per_second: float = 0.0
    cache_n: int = 0


@dataclass
class CompletionResponse:
    """Completion API response."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    timings: Optional[Timings] = None
    system_fingerprint: Optional[str] = None


@dataclass
class EmbeddingData:
    """Single embedding result."""
    index: int
    embedding: List[float]
    object: str = "embedding"


@dataclass
class EmbeddingsResponse:
    """Embeddings API response."""
    object: str
    data: List[EmbeddingData]
    model: str
    usage: Usage


@dataclass
class RerankResult:
    """Single rerank result."""
    index: int
    relevance_score: float
    document: Optional[str] = None


@dataclass
class RerankResponse:
    """Rerank API response."""
    model: str
    results: List[RerankResult]
    usage: Optional[Usage] = None


@dataclass
class TokenizeResponse:
    """Tokenize API response."""
    tokens: List[Union[int, Dict[str, Any]]]


@dataclass
class ModelInfo:
    """Model information."""
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "llamacpp"
    meta: Optional[Dict[str, Any]] = None
    status: Optional[Dict[str, Any]] = None
    multimodal: bool = False


@dataclass 
class SlotInfo:
    """Slot information."""
    id: int
    is_processing: bool
    n_ctx: int
    n_predict: int
    params: Dict[str, Any]
    prompt: str = ""
    next_token: Optional[Dict[str, Any]] = None


@dataclass
class HealthStatus:
    """Health check status."""
    status: str
    slots_idle: Optional[int] = None
    slots_processing: Optional[int] = None


@dataclass
class LoraAdapter:
    """LoRA adapter info."""
    id: int
    path: str
    scale: float


class LlamaCppClient:
    """
    Comprehensive Python client for llama.cpp server.
    
    Provides access to all server endpoints with type-safe responses.
    Supports both OpenAI-compatible and native llama.cpp APIs.
    
    Example:
        >>> client = LlamaCppClient("http://localhost:8080")
        >>> 
        >>> # Health check
        >>> health = client.health()
        >>> print(health.status)
        
        >>> # Chat completion (OpenAI-compatible)
        >>> response = client.chat.completions.create(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=100
        ... )
        
        >>> # Native completion
        >>> response = client.complete(
        ...     prompt="The meaning of life is",
        ...     n_predict=50,
        ...     temperature=0.8
        ... )
        
        >>> # Embeddings
        >>> embeddings = client.embeddings.create(
        ...     input=["Hello world", "Goodbye world"]
        ... )
        
        >>> # Tokenization
        >>> tokens = client.tokenize("Hello, world!")
        >>> text = client.detokenize(tokens.tokens)
    
    Attributes:
        base_url: Server base URL
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds
        chat: Chat completions API
        embeddings: Embeddings API
        models: Models management API
        slots: Slots management API
        lora: LoRA adapter management API
    """
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        verify_ssl: bool = True
    ):
        """
        Initialize the client.
        
        Args:
            base_url: Server URL (default: http://127.0.0.1:8080)
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds (default: 600)
            verify_ssl: Verify SSL certificates (default: True)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        # Initialize sub-APIs (lazy loading)
        self._chat = None
        self._embeddings = None
        self._models = None
        self._slots = None
        self._lora = None
        
    @property
    def chat(self) -> "ChatCompletionsAPI":
        """Chat completions API (OpenAI-compatible)."""
        if self._chat is None:
            self._chat = ChatCompletionsAPI(self)
        return self._chat
    
    @property
    def embeddings(self) -> "EmbeddingsClientAPI":
        """Embeddings API."""
        if self._embeddings is None:
            self._embeddings = EmbeddingsClientAPI(self)
        return self._embeddings
    
    @property
    def models(self) -> "ModelsClientAPI":
        """Models management API."""
        if self._models is None:
            self._models = ModelsClientAPI(self)
        return self._models
    
    @property
    def slots(self) -> "SlotsClientAPI":
        """Slots management API."""
        if self._slots is None:
            self._slots = SlotsClientAPI(self)
        return self._slots
    
    @property
    def lora(self) -> "LoraClientAPI":
        """LoRA adapter management API."""
        if self._lora is None:
            self._lora = LoraClientAPI(self)
        return self._lora
    
    def _headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Make HTTP request to server.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            json_data: JSON body data
            params: Query parameters
            stream: Enable streaming response
            
        Returns:
            Response data (dict or iterator for streaming)
            
        Raises:
            requests.HTTPError: On HTTP errors
        """
        url = f"{self.base_url}{endpoint}"
        
        response = requests.request(
            method=method,
            url=url,
            headers=self._headers(),
            json=json_data,
            params=params,
            timeout=self.timeout,
            verify=self.verify_ssl,
            stream=stream
        )
        
        response.raise_for_status()
        
        if stream:
            return self._stream_response(response)
        
        return response.json()
    
    def _stream_response(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        """Parse SSE stream response."""
        if _HAS_SSE:
            client = sseclient.SSEClient(response)
            for event in client.events():
                if event.data == "[DONE]":
                    break
                try:
                    yield json.loads(event.data)
                except json.JSONDecodeError:
                    continue
        else:
            # Fallback: simple line-based SSE parsing
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line_str.startswith("data: "):
                        data = line_str[6:]
                        if data == "[DONE]":
                            break
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue
    
    # =========================================================================
    # Health Endpoints
    # =========================================================================
    
    def health(self) -> HealthStatus:
        """
        Check server health.
        
        Returns:
            HealthStatus with server status
            
        Example:
            >>> health = client.health()
            >>> print(health.status)  # "ok"
        """
        try:
            data = self._request("GET", "/health")
            return HealthStatus(
                status=data.get("status", "ok"),
                slots_idle=data.get("slots_idle"),
                slots_processing=data.get("slots_processing")
            )
        except requests.HTTPError as e:
            if e.response.status_code == 503:
                return HealthStatus(status="loading")
            raise
    
    def is_ready(self) -> bool:
        """Check if server is ready to accept requests."""
        try:
            return self.health().status == "ok"
        except Exception:
            return False
    
    def wait_until_ready(self, timeout: float = 60.0, poll_interval: float = 1.0) -> bool:
        """
        Wait until server is ready.
        
        Args:
            timeout: Maximum wait time in seconds
            poll_interval: Time between health checks
            
        Returns:
            True if server became ready, False on timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.is_ready():
                return True
            time.sleep(poll_interval)
        return False
    
    # =========================================================================
    # Props Endpoints
    # =========================================================================
    
    def props(self) -> Dict[str, Any]:
        """
        Get server global properties.
        
        Returns:
            Dictionary with server properties including:
            - default_generation_settings
            - total_slots
            - model_path
            - chat_template
            - modalities
            - is_sleeping
        """
        return self._request("GET", "/props")
    
    def set_props(self, **kwargs) -> Dict[str, Any]:
        """
        Set server global properties (requires --props flag).
        
        Args:
            **kwargs: Properties to set
            
        Returns:
            Updated properties
        """
        return self._request("POST", "/props", json_data=kwargs)
    
    # =========================================================================
    # Native Completion Endpoint
    # =========================================================================
    
    def complete(
        self,
        prompt: Union[str, List[Union[str, int]]],
        n_predict: int = -1,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 64,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        grammar: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        seed: int = -1,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        cache_prompt: bool = True,
        n_probs: int = 0,
        samplers: Optional[List[str]] = None,
        # DRY sampling
        dry_multiplier: float = 0.0,
        dry_base: float = 1.75,
        dry_allowed_length: int = 2,
        dry_penalty_last_n: int = -1,
        # XTC sampling
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.1,
        # Dynamic temperature
        dynatemp_range: float = 0.0,
        dynatemp_exponent: float = 1.0,
        # Penalties
        typical_p: float = 1.0,
        # Advanced
        id_slot: int = -1,
        return_tokens: bool = False,
        **kwargs
    ) -> Union[CompletionResponse, Iterator[Dict[str, Any]]]:
        """
        Generate text completion (native llama.cpp API).
        
        This is the non-OpenAI-compatible completion endpoint with full
        access to all llama.cpp sampling parameters.
        
        Args:
            prompt: Input prompt (string or token array)
            n_predict: Max tokens to generate (-1 = unlimited)
            temperature: Sampling temperature (0.0-2.0)
            top_k: Top-k sampling (0 = disabled)
            top_p: Top-p/nucleus sampling (1.0 = disabled)
            min_p: Min-p sampling (0.0 = disabled)
            repeat_penalty: Repetition penalty
            repeat_last_n: Tokens to consider for repeat penalty
            presence_penalty: Presence penalty (0.0 = disabled)
            frequency_penalty: Frequency penalty (0.0 = disabled)
            mirostat: Mirostat mode (0=off, 1=v1, 2=v2)
            mirostat_tau: Mirostat target entropy
            mirostat_eta: Mirostat learning rate
            grammar: BNF grammar for constrained generation
            json_schema: JSON schema for structured output
            seed: RNG seed (-1 = random)
            stop: Stop sequences
            stream: Enable streaming
            cache_prompt: Reuse KV cache from previous request
            n_probs: Return top N token probabilities
            samplers: Sampler order
            dry_multiplier: DRY sampling multiplier
            dry_base: DRY base value
            dry_allowed_length: DRY allowed length
            dry_penalty_last_n: DRY penalty window
            xtc_probability: XTC probability
            xtc_threshold: XTC threshold
            dynatemp_range: Dynamic temperature range
            dynatemp_exponent: Dynamic temperature exponent
            typical_p: Locally typical sampling
            id_slot: Specific slot ID (-1 = auto)
            return_tokens: Return raw token IDs
            **kwargs: Additional parameters
            
        Returns:
            CompletionResponse or stream iterator
            
        Example:
            >>> response = client.complete(
            ...     prompt="Once upon a time",
            ...     n_predict=100,
            ...     temperature=0.7
            ... )
            >>> print(response.choices[0].text)
        """
        data = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "mirostat": mirostat,
            "mirostat_tau": mirostat_tau,
            "mirostat_eta": mirostat_eta,
            "seed": seed,
            "stream": stream,
            "cache_prompt": cache_prompt,
            "n_probs": n_probs,
            "dry_multiplier": dry_multiplier,
            "dry_base": dry_base,
            "dry_allowed_length": dry_allowed_length,
            "dry_penalty_last_n": dry_penalty_last_n,
            "xtc_probability": xtc_probability,
            "xtc_threshold": xtc_threshold,
            "dynatemp_range": dynatemp_range,
            "dynatemp_exponent": dynatemp_exponent,
            "typical_p": typical_p,
            "id_slot": id_slot,
            "return_tokens": return_tokens,
        }
        
        if grammar:
            data["grammar"] = grammar
        if json_schema:
            data["json_schema"] = json_schema
        if stop:
            data["stop"] = stop
        if samplers:
            data["samplers"] = samplers
            
        data.update(kwargs)
        
        response = self._request("POST", "/completion", json_data=data, stream=stream)
        
        if stream:
            return response
        
        return self._parse_completion_response(response)
    
    def _parse_completion_response(self, data: Dict[str, Any]) -> CompletionResponse:
        """Parse raw completion response into dataclass."""
        timings = None
        if "timings" in data:
            t = data["timings"]
            timings = Timings(
                prompt_n=t.get("prompt_n", 0),
                prompt_ms=t.get("prompt_ms", 0.0),
                prompt_per_token_ms=t.get("prompt_per_token_ms", 0.0),
                prompt_per_second=t.get("prompt_per_second", 0.0),
                predicted_n=t.get("predicted_n", 0),
                predicted_ms=t.get("predicted_ms", 0.0),
                predicted_per_token_ms=t.get("predicted_per_token_ms", 0.0),
                predicted_per_second=t.get("predicted_per_second", 0.0),
                cache_n=t.get("cache_n", 0)
            )
        
        return CompletionResponse(
            id=data.get("id", ""),
            object="text_completion",
            created=data.get("created", int(time.time())),
            model=data.get("model", ""),
            choices=[Choice(
                index=0,
                text=data.get("content", ""),
                finish_reason=data.get("stop_type", "stop")
            )],
            timings=timings
        )
    
    # =========================================================================
    # Tokenization Endpoints
    # =========================================================================
    
    def tokenize(
        self,
        content: str,
        add_special: bool = False,
        parse_special: bool = True,
        with_pieces: bool = False
    ) -> TokenizeResponse:
        """
        Tokenize text into tokens.
        
        Args:
            content: Text to tokenize
            add_special: Add BOS/EOS tokens
            parse_special: Parse special token syntax
            with_pieces: Return token pieces along with IDs
            
        Returns:
            TokenizeResponse with token IDs
            
        Example:
            >>> tokens = client.tokenize("Hello, world!")
            >>> print(tokens.tokens)  # [15496, 11, 995, 0]
        """
        data = self._request("POST", "/tokenize", json_data={
            "content": content,
            "add_special": add_special,
            "parse_special": parse_special,
            "with_pieces": with_pieces
        })
        return TokenizeResponse(tokens=data.get("tokens", []))
    
    def detokenize(self, tokens: List[int]) -> str:
        """
        Convert tokens back to text.
        
        Args:
            tokens: Token IDs to detokenize
            
        Returns:
            Decoded text string
            
        Example:
            >>> text = client.detokenize([15496, 11, 995, 0])
            >>> print(text)  # "Hello, world!"
        """
        data = self._request("POST", "/detokenize", json_data={"tokens": tokens})
        return data.get("content", "")
    
    # =========================================================================
    # Template Endpoint
    # =========================================================================
    
    def apply_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Apply chat template to messages without generating.
        
        Useful for inspecting the formatted prompt before inference.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Formatted prompt string
            
        Example:
            >>> prompt = client.apply_template([
            ...     {"role": "user", "content": "Hello!"}
            ... ])
        """
        data = self._request("POST", "/apply-template", json_data={"messages": messages})
        return data.get("prompt", "")
    
    # =========================================================================
    # Embedding Endpoint (Native)
    # =========================================================================
    
    def embed(
        self,
        content: Union[str, List[str]],
        embd_normalize: int = 2
    ) -> List[List[float]]:
        """
        Generate embeddings (native API).
        
        Args:
            content: Text or list of texts to embed
            embd_normalize: Normalization type (-1=none, 0=max, 1=taxicab, 2=L2)
            
        Returns:
            List of embedding vectors
        """
        if isinstance(content, str):
            content = [content]
        
        data = self._request("POST", "/embedding", json_data={
            "content": content,
            "embd_normalize": embd_normalize
        })
        
        return data.get("embedding", [])
    
    # =========================================================================
    # Reranking Endpoint
    # =========================================================================
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None
    ) -> RerankResponse:
        """
        Rerank documents by relevance to query.
        
        Requires a reranker model and --embedding --pooling rank options.
        
        Args:
            query: Query string
            documents: List of documents to rank
            top_n: Return only top N results
            
        Returns:
            RerankResponse with ranked results
            
        Example:
            >>> results = client.rerank(
            ...     query="What is a panda?",
            ...     documents=[
            ...         "A panda is a bear",
            ...         "Hello world",
            ...         "Pandas eat bamboo"
            ...     ],
            ...     top_n=2
            ... )
        """
        request_data = {
            "query": query,
            "documents": documents
        }
        if top_n:
            request_data["top_n"] = top_n
        
        data = self._request("POST", "/v1/rerank", json_data=request_data)
        
        results = [
            RerankResult(
                index=r.get("index", i),
                relevance_score=r.get("relevance_score", 0.0),
                document=r.get("document")
            )
            for i, r in enumerate(data.get("results", []))
        ]
        
        return RerankResponse(
            model=data.get("model", ""),
            results=results
        )
    
    # =========================================================================
    # Infill Endpoint (Code Completion)
    # =========================================================================
    
    def infill(
        self,
        input_prefix: str,
        input_suffix: str,
        input_extra: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[CompletionResponse, Iterator[Dict[str, Any]]]:
        """
        Fill-in-the-middle code completion.
        
        Args:
            input_prefix: Code before cursor
            input_suffix: Code after cursor
            input_extra: Additional context files
            prompt: Text added after FIM_MID token
            stream: Enable streaming
            **kwargs: Additional completion parameters
            
        Returns:
            Completion response or stream
            
        Example:
            >>> result = client.infill(
            ...     input_prefix="def hello():\\n    ",
            ...     input_suffix="\\n    return greeting",
            ...     n_predict=50
            ... )
        """
        data = {
            "input_prefix": input_prefix,
            "input_suffix": input_suffix,
            "stream": stream,
            **kwargs
        }
        
        if input_extra:
            data["input_extra"] = input_extra
        if prompt:
            data["prompt"] = prompt
        
        response = self._request("POST", "/infill", json_data=data, stream=stream)
        
        if stream:
            return response
        
        return self._parse_completion_response(response)
    
    # =========================================================================
    # Metrics Endpoint
    # =========================================================================
    
    def metrics(self) -> str:
        """
        Get Prometheus-compatible metrics.
        
        Requires --metrics flag on server.
        
        Returns:
            Prometheus metrics text
        """
        url = f"{self.base_url}/metrics"
        response = requests.get(url, headers=self._headers(), timeout=self.timeout)
        response.raise_for_status()
        return response.text


# =============================================================================
# Sub-API Classes
# =============================================================================

class ChatCompletionsAPI:
    """OpenAI-compatible chat completions API."""
    
    def __init__(self, client: LlamaCppClient):
        self.client = client
        self.completions = self
    
    def create(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-3.5-turbo",
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        # llama.cpp specific
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        grammar: Optional[str] = None,
        min_p: float = 0.05,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        **kwargs
    ) -> Union[CompletionResponse, Iterator[Dict[str, Any]]]:
        """
        Create chat completion (OpenAI-compatible).
        
        Args:
            messages: Chat messages
            model: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            n: Number of completions
            stream: Enable streaming
            stop: Stop sequences
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            logit_bias: Token logit biases
            user: User identifier
            response_format: Response format (json_object, json_schema)
            tools: Tool definitions for function calling
            tool_choice: Tool selection mode
            seed: RNG seed
            mirostat: Mirostat mode (llama.cpp)
            mirostat_tau: Mirostat tau (llama.cpp)
            mirostat_eta: Mirostat eta (llama.cpp)
            grammar: BNF grammar (llama.cpp)
            min_p: Min-p sampling (llama.cpp)
            top_k: Top-k sampling (llama.cpp)
            repeat_penalty: Repeat penalty (llama.cpp)
            
        Returns:
            CompletionResponse or stream iterator
        """
        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            # llama.cpp specific
            "mirostat": mirostat,
            "mirostat_tau": mirostat_tau,
            "mirostat_eta": mirostat_eta,
            "min_p": min_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
        }
        
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if stop:
            data["stop"] = stop if isinstance(stop, list) else [stop]
        if logit_bias:
            data["logit_bias"] = logit_bias
        if user:
            data["user"] = user
        if response_format:
            data["response_format"] = response_format
        if tools:
            data["tools"] = tools
        if tool_choice:
            data["tool_choice"] = tool_choice
        if seed is not None:
            data["seed"] = seed
        if grammar:
            data["grammar"] = grammar
        
        data.update(kwargs)
        
        response = self.client._request(
            "POST", "/v1/chat/completions",
            json_data=data, stream=stream
        )
        
        if stream:
            return response
        
        return self._parse_response(response)
    
    def _parse_response(self, data: Dict[str, Any]) -> CompletionResponse:
        """Parse OpenAI-format response."""
        choices = []
        for i, choice in enumerate(data.get("choices", [])):
            message_data = choice.get("message", {})
            message = Message(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content", ""),
                tool_calls=message_data.get("tool_calls"),
                reasoning_content=message_data.get("reasoning_content")
            )
            choices.append(Choice(
                index=choice.get("index", i),
                message=message,
                finish_reason=choice.get("finish_reason")
            ))
        
        usage = None
        if "usage" in data:
            u = data["usage"]
            usage = Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0)
            )
        
        timings = None
        if "timings" in data:
            t = data["timings"]
            timings = Timings(
                prompt_n=t.get("prompt_n", 0),
                prompt_ms=t.get("prompt_ms", 0.0),
                predicted_n=t.get("predicted_n", 0),
                predicted_ms=t.get("predicted_ms", 0.0),
                predicted_per_second=t.get("predicted_per_second", 0.0),
                cache_n=t.get("cache_n", 0)
            )
        
        return CompletionResponse(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", int(time.time())),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            timings=timings,
            system_fingerprint=data.get("system_fingerprint")
        )


class EmbeddingsClientAPI:
    """Embeddings API (OpenAI-compatible)."""
    
    def __init__(self, client: LlamaCppClient):
        self.client = client
    
    def create(
        self,
        input: Union[str, List[str]],
        model: str = "text-embedding-ada-002",
        encoding_format: str = "float",
        dimensions: Optional[int] = None
    ) -> EmbeddingsResponse:
        """
        Create embeddings (OpenAI-compatible).
        
        Args:
            input: Text or list of texts
            model: Model identifier
            encoding_format: Output format (float, base64)
            dimensions: Embedding dimensions
            
        Returns:
            EmbeddingsResponse with embedding vectors
        """
        if isinstance(input, str):
            input = [input]
        
        data = {
            "input": input,
            "model": model,
            "encoding_format": encoding_format
        }
        if dimensions:
            data["dimensions"] = dimensions
        
        response = self.client._request("POST", "/v1/embeddings", json_data=data)
        
        embeddings = [
            EmbeddingData(
                index=e.get("index", i),
                embedding=e.get("embedding", [])
            )
            for i, e in enumerate(response.get("data", []))
        ]
        
        usage_data = response.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=0,
            total_tokens=usage_data.get("total_tokens", 0)
        )
        
        return EmbeddingsResponse(
            object="list",
            data=embeddings,
            model=response.get("model", model),
            usage=usage
        )


class ModelsClientAPI:
    """Models management API."""
    
    def __init__(self, client: LlamaCppClient):
        self.client = client
    
    def list(self) -> List[ModelInfo]:
        """
        List available models.
        
        Returns:
            List of ModelInfo objects
        """
        data = self.client._request("GET", "/v1/models")
        
        return [
            ModelInfo(
                id=m.get("id", ""),
                object=m.get("object", "model"),
                created=m.get("created", 0),
                owned_by=m.get("owned_by", "llamacpp"),
                meta=m.get("meta"),
                status=m.get("status"),
                multimodal=m.get("meta", {}).get("multimodal", False) if m.get("meta") else False
            )
            for m in data.get("data", [])
        ]
    
    def load(self, model: str) -> bool:
        """
        Load a model (router mode).
        
        Args:
            model: Model name or path
            
        Returns:
            True if successful
        """
        data = self.client._request("POST", "/models/load", json_data={"model": model})
        return data.get("success", False)
    
    def unload(self, model: str) -> bool:
        """
        Unload a model (router mode).
        
        Args:
            model: Model name
            
        Returns:
            True if successful
        """
        data = self.client._request("POST", "/models/unload", json_data={"model": model})
        return data.get("success", False)


class SlotsClientAPI:
    """Slots management API."""
    
    def __init__(self, client: LlamaCppClient):
        self.client = client
    
    def list(self, fail_on_no_slot: bool = False) -> List[SlotInfo]:
        """
        List server slots.
        
        Args:
            fail_on_no_slot: Return 503 if no slots available
            
        Returns:
            List of SlotInfo objects
        """
        params = {}
        if fail_on_no_slot:
            params["fail_on_no_slot"] = 1
        
        data = self.client._request("GET", "/slots", params=params)
        
        return [
            SlotInfo(
                id=s.get("id", i),
                is_processing=s.get("is_processing", False),
                n_ctx=s.get("n_ctx", 0),
                n_predict=s.get("n_predict", 0),
                params=s.get("params", {}),
                prompt=s.get("prompt", "")
            )
            for i, s in enumerate(data if isinstance(data, list) else [])
        ]
    
    def save(self, slot_id: int, filename: str) -> Dict[str, Any]:
        """
        Save slot KV cache to file.
        
        Args:
            slot_id: Slot ID
            filename: Output filename
            
        Returns:
            Save result with timings
        """
        return self.client._request(
            "POST", f"/slots/{slot_id}",
            params={"action": "save"},
            json_data={"filename": filename}
        )
    
    def restore(self, slot_id: int, filename: str) -> Dict[str, Any]:
        """
        Restore slot KV cache from file.
        
        Args:
            slot_id: Slot ID
            filename: Input filename
            
        Returns:
            Restore result with timings
        """
        return self.client._request(
            "POST", f"/slots/{slot_id}",
            params={"action": "restore"},
            json_data={"filename": filename}
        )
    
    def erase(self, slot_id: int) -> Dict[str, Any]:
        """
        Erase slot KV cache.
        
        Args:
            slot_id: Slot ID
            
        Returns:
            Erase result
        """
        return self.client._request(
            "POST", f"/slots/{slot_id}",
            params={"action": "erase"}
        )


class LoraClientAPI:
    """LoRA adapter management API."""
    
    def __init__(self, client: LlamaCppClient):
        self.client = client
    
    def list(self) -> List[LoraAdapter]:
        """
        List loaded LoRA adapters.
        
        Returns:
            List of LoraAdapter objects
        """
        data = self.client._request("GET", "/lora-adapters")
        
        return [
            LoraAdapter(
                id=a.get("id", i),
                path=a.get("path", ""),
                scale=a.get("scale", 0.0)
            )
            for i, a in enumerate(data if isinstance(data, list) else [])
        ]
    
    def set_scales(self, adapters: List[Dict[str, Any]]) -> bool:
        """
        Set LoRA adapter scales.
        
        Args:
            adapters: List of {"id": int, "scale": float}
            
        Returns:
            True if successful
            
        Example:
            >>> client.lora.set_scales([
            ...     {"id": 0, "scale": 0.5},
            ...     {"id": 1, "scale": 0.8}
            ... ])
        """
        self.client._request("POST", "/lora-adapters", json_data=adapters)
        return True
