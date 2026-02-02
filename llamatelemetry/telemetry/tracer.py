"""
llamatelemetry.telemetry.tracer - Inference-aware TracerProvider

Wraps the OpenTelemetry TracerProvider with LLM-specific span attributes
and semantic conventions for inference tracing.

Span semantic conventions used:
    llm.system          - Inference backend (llamatelemetry/llama.cpp)
    llm.model           - GGUF model name
    llm.input.tokens    - Prompt token count
    llm.output.tokens   - Generated token count
    llm.latency_ms      - End-to-end inference latency
    llm.tokens_per_sec  - Generation throughput
    gpu.device_id       - GPU device used
    nccl.split_mode     - Multi-GPU split strategy
"""

from typing import Any, List, Optional


class InferenceTracerProvider:
    """
    TracerProvider wrapper with LLM inference semantic conventions.

    Delegates to the real OTel TracerProvider but adds helpers for
    attaching GPU and NCCL metadata to spans.
    """

    def __init__(self, resource: Any = None, span_exporters: Optional[List[Any]] = None):
        """
        Args:
            resource: OTel Resource (from resource.build_gpu_resource)
            span_exporters: List of SpanExporter instances
        """
        self._graphistry_exporters = []
        self._provider = None
        self._resource = resource
        self._exporters = span_exporters or []

        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            self._provider = TracerProvider(resource=resource)
            for exporter in self._exporters:
                self._provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            pass

    def get_tracer(self, name: str, version: str = "") -> Any:
        """Get a Tracer from the underlying provider."""
        if self._provider:
            return self._provider.get_tracer(name, version)

        # Fallback: return a no-op tracer-like object
        return _NoopTracer()

    def add_graphistry_exporter(self, exporter: Any) -> None:
        """Register a pygraphistry trace exporter."""
        self._graphistry_exporters.append(exporter)

    def shutdown(self) -> None:
        """Shutdown the underlying provider."""
        if self._provider:
            self._provider.shutdown()


class _NoopTracer:
    """No-op tracer when OTel is unavailable."""

    def start_as_current_span(self, name: str, **kwargs):
        return _NoopSpanContext()


class _NoopSpanContext:
    """No-op context manager for spans."""

    def __enter__(self):
        return _NoopSpan()

    def __exit__(self, *args):
        pass


class _NoopSpan:
    """No-op span."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass


def annotate_inference_span(span: Any, model: str, prompt_tokens: int,
                            output_tokens: int, latency_ms: float,
                            gpu_id: int = 0, split_mode: str = "none") -> None:
    """
    Attach standard LLM inference attributes to an active span.

    Args:
        span: Active OTel Span
        model: Model name/path
        prompt_tokens: Number of prompt tokens
        output_tokens: Number of generated tokens
        latency_ms: Total inference latency in milliseconds
        gpu_id: Primary GPU device ID
        split_mode: NCCL split mode (none, layer, row)
    """
    tokens_per_sec = (output_tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

    span.set_attribute("llm.system", "llamatelemetry")
    span.set_attribute("llm.model", model)
    span.set_attribute("llm.input.tokens", prompt_tokens)
    span.set_attribute("llm.output.tokens", output_tokens)
    span.set_attribute("llm.latency_ms", latency_ms)
    span.set_attribute("llm.tokens_per_sec", tokens_per_sec)
    span.set_attribute("gpu.device_id", gpu_id)
    span.set_attribute("nccl.split_mode", split_mode)
