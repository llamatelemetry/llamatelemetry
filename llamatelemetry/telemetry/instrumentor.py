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
import os

from .semconv import set_gen_ai_attr, set_gen_ai_provider

try:
    from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
except Exception:
    class BaseInstrumentor:  # type: ignore
        def instrumentation_dependencies(self):
            return []
        def instrument(self, **kwargs):
            return self._instrument(**kwargs)
        def uninstrument(self, **kwargs):
            return self._uninstrument(**kwargs)
        def _instrument(self, **kwargs):
            return None
        def _uninstrument(self, **kwargs):
            return None


class LlamaCppClientInstrumentor(BaseInstrumentor):
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
        return ["llamatelemetry >= 0.1.1"]

    def instrument(
        self,
        tracer_provider: Optional[Any] = None,
        meter_provider: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Instrument LlamaCppClient.

        Args:
            tracer_provider: OpenTelemetry TracerProvider (uses global if None)
            meter_provider: OpenTelemetry MeterProvider (uses global if None)
        """
        return super().instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            **kwargs,
        )

    def _instrument(
        self,
        tracer_provider: Optional[Any] = None,
        meter_provider: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
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
            "0.1.1"
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

            if stream:
                def _streaming():
                    with tracer.start_as_current_span(operation) as span:
                        span.set_attribute("http.method", method)
                        span.set_attribute("http.url", f"{self_client.base_url}{endpoint}")
                        span.set_attribute("llamacpp.endpoint", endpoint)
                        span.set_attribute("llm.system", "llamatelemetry")
                        set_gen_ai_provider(span, "llama.cpp")

                        gen_ai_operation = _endpoint_to_gen_ai_operation(endpoint, method)
                        if gen_ai_operation:
                            set_gen_ai_attr(span, "operation_name", gen_ai_operation)

                        if json_data:
                            _add_request_attributes(span, json_data)

                        start_time = time.time()
                        first_chunk = True

                        try:
                            result_iter = LlamaCppClientInstrumentor._original_request(
                                self_client, method, endpoint, json_data, params, stream
                            )
                            for item in result_iter:
                                if first_chunk:
                                    ttft_ms = (time.time() - start_time) * 1000
                                    span.set_attribute("llm.ttft_ms", ttft_ms)
                                    first_chunk = False
                                yield item

                            latency_ms = (time.time() - start_time) * 1000
                            span.set_attribute("llm.latency_ms", latency_ms)
                        except Exception as e:
                            span.record_exception(e)
                            span.set_attribute("llm.error", str(e))
                            raise

                return _streaming()

            with tracer.start_as_current_span(operation) as span:
                span.set_attribute("http.method", method)
                span.set_attribute("http.url", f"{self_client.base_url}{endpoint}")
                span.set_attribute("llamacpp.endpoint", endpoint)
                span.set_attribute("llm.system", "llamatelemetry")
                set_gen_ai_provider(span, "llama.cpp")

                gen_ai_operation = _endpoint_to_gen_ai_operation(endpoint, method)
                if gen_ai_operation:
                    set_gen_ai_attr(span, "operation_name", gen_ai_operation)

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

    def uninstrument(self, **kwargs: Any) -> None:
        """Remove instrumentation."""
        return super().uninstrument(**kwargs)

    def _uninstrument(self, **kwargs: Any) -> None:
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


def _endpoint_to_gen_ai_operation(endpoint: str, method: str) -> Optional[str]:
    endpoint = endpoint.strip("/")
    if endpoint in ("chat/completions", "v1/chat/completions"):
        return "chat"
    if endpoint in ("completion", "v1/completions"):
        return "completion"
    if endpoint in ("embeddings", "v1/embeddings"):
        return "embeddings"
    if endpoint in ("tokenize", "detokenize"):
        return "tokenize"
    return None


def _content_enabled() -> bool:
    return os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false").lower() == "true"


def _messages_to_gen_ai(messages: list) -> list:
    if not messages:
        return []
    out = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if not role:
            continue
        entry = {"role": role}
        if _content_enabled():
            content = msg.get("content")
            if content is not None:
                entry["content"] = content
        if msg.get("tool_calls"):
            entry["tool_calls"] = msg.get("tool_calls")
        out.append(entry)
    return out


def _emit_gen_ai_message_events(span: Any, messages: list) -> None:
    if span is None or not messages:
        return
    capture = _content_enabled()
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if not role:
            continue
        attributes = {"gen_ai.system": "llama.cpp", "gen_ai.role": role}
        if capture:
            content = msg.get("content")
            if content is not None:
                attributes["gen_ai.message"] = content
            if msg.get("tool_calls") is not None:
                attributes["gen_ai.tool_calls"] = str(msg.get("tool_calls"))
        span.add_event(f"gen_ai.{role}.message", attributes=attributes)


def _add_request_attributes(span: Any, json_data: dict) -> None:
    """Add request-specific attributes to span."""
    try:
        # Prompt/message info
        if "prompt" in json_data:
            prompt = json_data["prompt"]
            span.set_attribute("llm.prompt_length", len(str(prompt)))
            # Approximate token count
            span.set_attribute("llm.input.tokens_approx", len(str(prompt).split()))
            if _content_enabled():
                set_gen_ai_attr(span, "prompt", prompt)
            set_gen_ai_attr(span, "request_model", json_data.get("model"))

        if "messages" in json_data:
            messages = json_data["messages"]
            span.set_attribute("llm.message_count", len(messages))
            total_len = sum(len(str(m.get("content", ""))) for m in messages)
            span.set_attribute("llm.prompt_length", total_len)
            set_gen_ai_attr(span, "request_model", json_data.get("model"))
            if _content_enabled():
                set_gen_ai_attr(span, "input_messages", _messages_to_gen_ai(messages))
            _emit_gen_ai_message_events(span, messages)

        # Generation parameters
        if "n_predict" in json_data:
            span.set_attribute("llm.max_tokens", json_data["n_predict"])
            set_gen_ai_attr(span, "request_max_tokens", json_data.get("n_predict"))
        if "max_tokens" in json_data:
            span.set_attribute("llm.max_tokens", json_data["max_tokens"])
            set_gen_ai_attr(span, "request_max_tokens", json_data.get("max_tokens"))

        if "temperature" in json_data:
            span.set_attribute("llm.temperature", json_data["temperature"])
            set_gen_ai_attr(span, "request_temperature", json_data.get("temperature"))

        if "top_p" in json_data:
            span.set_attribute("llm.top_p", json_data["top_p"])
            set_gen_ai_attr(span, "request_top_p", json_data.get("top_p"))

        if "top_k" in json_data:
            span.set_attribute("llm.top_k", json_data["top_k"])
            set_gen_ai_attr(span, "request_top_k", json_data.get("top_k"))

        if "stream" in json_data:
            span.set_attribute("llm.stream", json_data["stream"])

        if "stop" in json_data:
            set_gen_ai_attr(span, "request_stop_sequences", json_data.get("stop"))

        if "seed" in json_data:
            set_gen_ai_attr(span, "request_seed", json_data.get("seed"))

        if "presence_penalty" in json_data:
            set_gen_ai_attr(span, "request_presence_penalty", json_data.get("presence_penalty"))

        if "frequency_penalty" in json_data:
            set_gen_ai_attr(span, "request_frequency_penalty", json_data.get("frequency_penalty"))

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
            set_gen_ai_attr(span, "usage_output_tokens", tokens)

        if "tokens_evaluated" in result:
            span.set_attribute("llm.input.tokens", result["tokens_evaluated"])
            set_gen_ai_attr(span, "usage_input_tokens", result.get("tokens_evaluated"))

        # Usage object (OpenAI format)
        if "usage" in result:
            usage = result["usage"]
            if "prompt_tokens" in usage:
                span.set_attribute("llm.input.tokens", usage["prompt_tokens"])
                set_gen_ai_attr(span, "usage_input_tokens", usage.get("prompt_tokens"))
            if "completion_tokens" in usage:
                tokens = usage["completion_tokens"]
                span.set_attribute("llm.output.tokens", tokens)
                if latency_ms > 0:
                    span.set_attribute("llm.tokens_per_sec", tokens / (latency_ms / 1000))
                set_gen_ai_attr(span, "usage_output_tokens", usage.get("completion_tokens"))
            if "total_tokens" in usage:
                span.set_attribute("llm.total_tokens", usage["total_tokens"])
                set_gen_ai_attr(span, "usage_total_tokens", usage.get("total_tokens"))

        # Timing information
        if "timings" in result:
            t = result["timings"]
            if "predicted_per_second" in t:
                span.set_attribute("llm.tokens_per_sec", t["predicted_per_second"])
            if "prompt_ms" in t:
                span.set_attribute("llm.prompt_ms", t["prompt_ms"])
            if "predicted_ms" in t:
                span.set_attribute("llm.generation_ms", t["predicted_ms"])
            ttft_ms = _estimate_ttft_ms(t)
            if ttft_ms is not None:
                span.set_attribute("llm.ttft_ms", ttft_ms)

        # Model info
        if "model" in result:
            span.set_attribute("llm.model", result["model"])
            set_gen_ai_attr(span, "response_model", result.get("model"))

        if "id" in result:
            set_gen_ai_attr(span, "response_id", result.get("id"))

        # Stop reason
        if "stop_reason" in result:
            span.set_attribute("llm.finish_reason", result["stop_reason"])
        if "finish_reason" in result:
            span.set_attribute("llm.finish_reason", result["finish_reason"])

        if "choices" in result and isinstance(result["choices"], list):
            finish_reasons = [c.get("finish_reason") for c in result["choices"] if isinstance(c, dict)]
            finish_reasons = [r for r in finish_reasons if r is not None]
            if finish_reasons:
                set_gen_ai_attr(span, "response_finish_reasons", finish_reasons)
            _emit_gen_ai_choice_events(span, result["choices"])

    except Exception:
        pass


def _emit_gen_ai_choice_events(span: Any, choices: list) -> None:
    if span is None or not choices:
        return
    capture = _content_enabled()
    for idx, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        attributes = {"gen_ai.system": "llama.cpp", "index": idx}
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            attributes["finish_reason"] = finish_reason
        message = choice.get("message")
        if capture and isinstance(message, dict):
            content = message.get("content")
            if content is not None:
                attributes["gen_ai.message"] = content
            role = message.get("role")
            if role is not None:
                attributes["gen_ai.role"] = role
        span.add_event("gen_ai.choice", attributes=attributes)


def _estimate_ttft_ms(timings: Any) -> Optional[float]:
    try:
        prompt_ms = float(timings.get("prompt_ms", 0.0))
        pred_per_token = float(timings.get("predicted_per_token_ms", 0.0))
        predicted_ms = float(timings.get("predicted_ms", 0.0))
        predicted_n = float(timings.get("predicted_n", 0.0))
        if pred_per_token > 0:
            return prompt_ms + pred_per_token
        if predicted_ms > 0 and predicted_n > 0:
            return prompt_ms + (predicted_ms / predicted_n)
        if prompt_ms > 0:
            return prompt_ms
    except Exception:
        return None
    return None


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
