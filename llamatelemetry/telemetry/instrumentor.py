"""
llamatelemetry.telemetry.instrumentor - OpenTelemetry instrumentor for LlamaCppClient.

Follows opentelemetry-python-contrib patterns for auto-instrumentation.

Example:
    >>> from llamatelemetry.telemetry import LlamaCppClientInstrumentor
    >>>
    >>> # Auto-instrument all LlamaCppClient instances
    >>> LlamaCppClientInstrumentor().instrument()
    >>>
    >>> # Now all client calls are traced
    >>> client = LlamaCppClient()
    >>> response = client.complete("Hello")  # Automatically traced
"""

from typing import Collection, Optional, Any
import time


class LlamaCppClientInstrumentor:
    """
    Auto-instrumentor for LlamaCppClient.

    Wraps LlamaCppClient methods to automatically create OpenTelemetry
    spans with LLM-specific attributes.

    Example:
        >>> # Instrument at application startup
        >>> instrumentor = LlamaCppClientInstrumentor()
        >>> instrumentor.instrument()
        >>>
        >>> # All client calls are now traced
        >>> client = LlamaCppClient(base_url="http://localhost:8080")
        >>> response = client.chat.create(messages=[...])
        >>>
        >>> # Uninstrument when done
        >>> instrumentor.uninstrument()
    """

    _original_request = None
    _original_chat_create = None
    _original_complete = None
    _instrumented = False
    _tracer = None

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return required packages for instrumentation."""
        return ["llamatelemetry >= 0.1.0"]

    def instrument(
        self,
        tracer_provider: Optional[Any] = None,
        meter_provider: Optional[Any] = None,
    ) -> None:
        """
        Instrument LlamaCppClient.

        Args:
            tracer_provider: OpenTelemetry TracerProvider (uses global if None)
            meter_provider: OpenTelemetry MeterProvider (uses global if None)
        """
        if self._instrumented:
            return

        try:
            from ..api.client import LlamaCppClient
        except ImportError:
            return

        # Get tracer
        if tracer_provider is None:
            try:
                from opentelemetry import trace
                tracer_provider = trace.get_tracer_provider()
            except ImportError:
                return

        self._tracer = tracer_provider.get_tracer(
            "llamatelemetry.instrumentor",
            "0.1.0"
        )

        # Store original methods
        self._original_request = LlamaCppClient._request

        # Wrap _request method (lowest level, catches all API calls)
        def instrumented_request(
            self_client,
            method: str,
            endpoint: str,
            json_data=None,
            params=None,
            stream=False
        ):
            tracer = LlamaCppClientInstrumentor._tracer
            if tracer is None:
                return LlamaCppClientInstrumentor._original_request(
                    self_client, method, endpoint, json_data, params, stream
                )

            # Determine operation name from endpoint
            operation = _endpoint_to_operation(endpoint, method)

            with tracer.start_as_current_span(operation) as span:
                span.set_attribute("http.method", method)
                span.set_attribute("http.url", f"{self_client.base_url}{endpoint}")
                span.set_attribute("llamacpp.endpoint", endpoint)
                span.set_attribute("llm.system", "llamatelemetry")

                # Add request attributes
                if json_data:
                    _add_request_attributes(span, json_data)

                start_time = time.time()

                try:
                    result = LlamaCppClientInstrumentor._original_request(
                        self_client, method, endpoint, json_data, params, stream
                    )

                    latency_ms = (time.time() - start_time) * 1000
                    span.set_attribute("llm.latency_ms", latency_ms)

                    # Add response attributes
                    if isinstance(result, dict):
                        _add_response_attributes(span, result, latency_ms)

                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("llm.error", str(e))
                    raise

        LlamaCppClient._request = instrumented_request
        self._instrumented = True

    def uninstrument(self) -> None:
        """Remove instrumentation."""
        if not self._instrumented:
            return

        try:
            from ..api.client import LlamaCppClient

            if self._original_request:
                LlamaCppClient._request = self._original_request

        except ImportError:
            pass

        self._instrumented = False
        self._tracer = None

    def is_instrumented(self) -> bool:
        """Check if instrumentation is active."""
        return self._instrumented


def _endpoint_to_operation(endpoint: str, method: str) -> str:
    """Convert API endpoint to span operation name."""
    endpoint = endpoint.strip("/")

    # Map common endpoints to semantic operation names
    endpoint_map = {
        "completion": "llm.completion",
        "v1/completions": "llm.completion",
        "chat/completions": "llm.chat",
        "v1/chat/completions": "llm.chat",
        "embeddings": "llm.embed",
        "v1/embeddings": "llm.embed",
        "tokenize": "llm.tokenize",
        "detokenize": "llm.detokenize",
        "health": "llamacpp.health",
        "metrics": "llamacpp.metrics",
        "slots": "llamacpp.slots",
        "props": "llamacpp.props",
        "models": "llamacpp.models",
    }

    return endpoint_map.get(endpoint, f"llamacpp.{method.lower()}.{endpoint}")


def _add_request_attributes(span: Any, json_data: dict) -> None:
    """Add request-specific attributes to span."""
    try:
        # Prompt/message info
        if "prompt" in json_data:
            prompt = json_data["prompt"]
            span.set_attribute("llm.prompt_length", len(str(prompt)))
            # Approximate token count
            span.set_attribute("llm.input.tokens_approx", len(str(prompt).split()))

        if "messages" in json_data:
            messages = json_data["messages"]
            span.set_attribute("llm.message_count", len(messages))
            total_len = sum(len(str(m.get("content", ""))) for m in messages)
            span.set_attribute("llm.prompt_length", total_len)

        # Generation parameters
        if "n_predict" in json_data:
            span.set_attribute("llm.max_tokens", json_data["n_predict"])
        if "max_tokens" in json_data:
            span.set_attribute("llm.max_tokens", json_data["max_tokens"])

        if "temperature" in json_data:
            span.set_attribute("llm.temperature", json_data["temperature"])

        if "top_p" in json_data:
            span.set_attribute("llm.top_p", json_data["top_p"])

        if "top_k" in json_data:
            span.set_attribute("llm.top_k", json_data["top_k"])

        if "stream" in json_data:
            span.set_attribute("llm.stream", json_data["stream"])

    except Exception:
        pass


def _add_response_attributes(span: Any, result: dict, latency_ms: float) -> None:
    """Add response-specific attributes to span."""
    try:
        # Token counts
        if "tokens_predicted" in result:
            tokens = result["tokens_predicted"]
            span.set_attribute("llm.output.tokens", tokens)
            if latency_ms > 0:
                span.set_attribute("llm.tokens_per_sec", tokens / (latency_ms / 1000))

        if "tokens_evaluated" in result:
            span.set_attribute("llm.input.tokens", result["tokens_evaluated"])

        # Usage object (OpenAI format)
        if "usage" in result:
            usage = result["usage"]
            if "prompt_tokens" in usage:
                span.set_attribute("llm.input.tokens", usage["prompt_tokens"])
            if "completion_tokens" in usage:
                tokens = usage["completion_tokens"]
                span.set_attribute("llm.output.tokens", tokens)
                if latency_ms > 0:
                    span.set_attribute("llm.tokens_per_sec", tokens / (latency_ms / 1000))
            if "total_tokens" in usage:
                span.set_attribute("llm.total_tokens", usage["total_tokens"])

        # Timing information
        if "timings" in result:
            t = result["timings"]
            if "predicted_per_second" in t:
                span.set_attribute("llm.tokens_per_sec", t["predicted_per_second"])
            if "prompt_ms" in t:
                span.set_attribute("llm.prompt_ms", t["prompt_ms"])
            if "predicted_ms" in t:
                span.set_attribute("llm.generation_ms", t["predicted_ms"])

        # Model info
        if "model" in result:
            span.set_attribute("llm.model", result["model"])

        # Stop reason
        if "stop_reason" in result:
            span.set_attribute("llm.finish_reason", result["stop_reason"])
        if "finish_reason" in result:
            span.set_attribute("llm.finish_reason", result["finish_reason"])

    except Exception:
        pass


# Global instrumentor instance for convenience
_global_instrumentor: Optional[LlamaCppClientInstrumentor] = None


def instrument_llamacpp_client(
    tracer_provider: Optional[Any] = None,
    meter_provider: Optional[Any] = None,
) -> LlamaCppClientInstrumentor:
    """
    Convenience function to instrument LlamaCppClient.

    Args:
        tracer_provider: OpenTelemetry TracerProvider
        meter_provider: OpenTelemetry MeterProvider

    Returns:
        Instrumentor instance

    Example:
        >>> from llamatelemetry.telemetry import instrument_llamacpp_client
        >>> instrumentor = instrument_llamacpp_client()
    """
    global _global_instrumentor

    if _global_instrumentor is None:
        _global_instrumentor = LlamaCppClientInstrumentor()

    _global_instrumentor.instrument(tracer_provider, meter_provider)
    return _global_instrumentor


def uninstrument_llamacpp_client() -> None:
    """
    Convenience function to remove LlamaCppClient instrumentation.
    """
    global _global_instrumentor

    if _global_instrumentor is not None:
        _global_instrumentor.uninstrument()


__all__ = [
    "LlamaCppClientInstrumentor",
    "instrument_llamacpp_client",
    "uninstrument_llamacpp_client",
]
