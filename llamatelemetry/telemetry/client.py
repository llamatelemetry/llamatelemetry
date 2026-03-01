"""llamatelemetry.telemetry.client

Notebook-friendly client wrapper that:
- wraps llamatelemetry.api.client.LlamaCppClient
- attaches LlamaCppClientInstrumentor for spans
- optionally records inference metrics via GpuMetricsCollector
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Iterator, Union, List
import time

from ..api.client import LlamaCppClient, CompletionResponse
from .instrumentor import LlamaCppClientInstrumentor
from .metrics import GpuMetricsCollector


@dataclass
class InstrumentationConfig:
    """Configuration for instrumentation behavior."""
    provider_name: str = "llama_cpp"
    include_payload_sizes: bool = True
    estimate_prompt_tokens: bool = True


class InstrumentedLlamaCppClient:
    """Thin wrapper around LlamaCppClient with automatic instrumentation."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        api_key: Optional[str] = None,
        *,
        tracer_provider: Optional[Any] = None,
        meter_provider: Optional[Any] = None,
        gpu_metrics: Optional[GpuMetricsCollector] = None,
        instrumentation: Optional[InstrumentationConfig] = None,
        session_attributes: Optional[Dict[str, Any]] = None,
        request_timeout_s: float = 600.0,
    ) -> None:
        self.client = LlamaCppClient(base_url=base_url, api_key=api_key, timeout=request_timeout_s)
        self.gpu_metrics = gpu_metrics
        self.instrumentation = instrumentation or InstrumentationConfig()
        self.session_attributes = session_attributes or {}

        # Attach automatic instrumentation by wrapping the client's internal request method.
        self._instrumentor = LlamaCppClientInstrumentor()
        self._instrumentor.instrument(tracer_provider=tracer_provider, meter_provider=meter_provider)

    # ----------------------------
    # Convenience endpoints
    # ----------------------------

    def chat_completions(self, payload: Dict[str, Any]) -> Union[CompletionResponse, Iterator[Dict[str, Any]]]:
        """Call /v1/chat/completions and record metrics if possible."""
        t0 = time.perf_counter()
        result = self.client.chat.completions.create(**payload)
        if isinstance(result, Iterator):
            return result
        self._maybe_record_inference(t0, result, model=_safe_get(payload, "model"))
        return result

    def completions(self, payload: Dict[str, Any]) -> Union[CompletionResponse, Iterator[Dict[str, Any]]]:
        """Call native completion API and record metrics if possible."""
        t0 = time.perf_counter()
        args = _map_completion_payload(payload)
        result = self.client.complete(**args)
        if isinstance(result, Iterator):
            return result
        self._maybe_record_inference(t0, result, model=_safe_get(payload, "model"))
        return result

    def embeddings(self, payload: Dict[str, Any]) -> Any:
        """Call /v1/embeddings and record latency if possible."""
        t0 = time.perf_counter()
        result = self.client.embeddings.create(**payload)
        self._maybe_record_inference(t0, result, model=_safe_get(payload, "model"), tokens_override=0)
        return result

    # ----------------------------
    # Streaming helpers
    # ----------------------------

    def chat_completions_stream(self, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Stream /v1/chat/completions if server supports SSE streaming."""
        t0 = time.perf_counter()
        tokens = 0
        payload = dict(payload)
        payload["stream"] = True
        for chunk in self.client.chat.completions.create(**payload):
            tokens += _infer_chunk_tokens(chunk)
            yield chunk
        self._maybe_record_inference(t0, {"_stream_tokens": tokens}, model=_safe_get(payload, "model"), tokens_override=tokens)

    # ----------------------------
    # Internals
    # ----------------------------

    def _maybe_record_inference(
        self,
        t0: float,
        response: Any,
        *,
        model: str = "",
        tokens_override: Optional[int] = None,
    ) -> None:
        if not self.gpu_metrics:
            return

        latency_ms = (time.perf_counter() - t0) * 1000.0
        tokens = tokens_override
        if tokens is None:
            tokens = _infer_tokens_from_response(response)

        try:
            self.gpu_metrics.record_inference(latency_ms=float(latency_ms), tokens=int(tokens or 0), model=model or "")
        except Exception:
            return


# ----------------------------
# Helpers
# ----------------------------


def _safe_get(d: Dict[str, Any], key: str, default: str = "") -> str:
    v = d.get(key, default)
    return str(v) if v is not None else default


def _infer_tokens_from_response(resp: Any) -> int:
    """Best-effort token estimate when the server doesn't return usage."""
    try:
        if isinstance(resp, CompletionResponse):
            usage = resp.usage
            if usage:
                return int(usage.completion_tokens or 0)
            if resp.choices:
                text = resp.choices[0].text or ""
                if resp.choices[0].message and resp.choices[0].message.content:
                    text = resp.choices[0].message.content
                return max(0, len(text.split()))
        if isinstance(resp, dict):
            usage = resp.get("usage")
            if isinstance(usage, dict):
                return int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            return int(resp.get("_stream_tokens") or 0)
    except Exception:
        pass
    return 0


def _infer_chunk_tokens(chunk: Dict[str, Any]) -> int:
    """Attempt to estimate emitted tokens for a streaming chunk."""
    try:
        if "choices" not in chunk:
            return 0
        delta = chunk["choices"][0].get("delta") or {}
        content = delta.get("content")
        if not content:
            return 0
        return max(0, len(str(content).split()))
    except Exception:
        return 0


def _map_completion_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map OpenAI-style completion payload to llama.cpp native complete() args."""
    args: Dict[str, Any] = {}
    if "prompt" in payload:
        args["prompt"] = payload["prompt"]
    if "max_tokens" in payload and "n_predict" not in payload:
        args["n_predict"] = payload["max_tokens"]

    passthrough = {
        "n_predict",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "repeat_penalty",
        "repeat_last_n",
        "presence_penalty",
        "frequency_penalty",
        "mirostat",
        "mirostat_tau",
        "mirostat_eta",
        "grammar",
        "seed",
        "stop",
        "stream",
        "cache_prompt",
        "n_probs",
        "samplers",
        "penalize_nl",
        "ignore_eos",
    }
    for key in passthrough:
        if key in payload:
            args[key] = payload[key]
    return args


__all__ = ["InstrumentationConfig", "InstrumentedLlamaCppClient"]
