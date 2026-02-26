"""
llamatelemetry.telemetry.auto_instrument - Auto-instrumentation utilities.

Provides decorators and context managers for automatic span creation
in LLM inference operations.

Example:
    >>> from llamatelemetry.telemetry.auto_instrument import inference_span
    >>>
    >>> with inference_span(tracer, model="gemma-3-4b") as span:
    ...     result = engine.infer(prompt)
    ...     span.set_attribute("llm.output.tokens", result.tokens_generated)
"""

from functools import wraps
from typing import Callable, Any, Optional, Dict
from contextlib import contextmanager
import time


def instrument_inference(
    tracer: Any,
    model_name: str = "",
    gpu_id: int = 0,
    split_mode: str = "none",
    operation: str = "llm.inference"
) -> Callable:
    """
    Decorator to auto-instrument inference methods.

    Wraps a function to automatically create OpenTelemetry spans
    with LLM-specific attributes.

    Args:
        tracer: OpenTelemetry tracer instance
        model_name: Model name for span attributes
        gpu_id: GPU device ID
        split_mode: Multi-GPU split mode ("none", "layer", "row")
        operation: Span operation name

    Returns:
        Decorated function with automatic span creation

    Example:
        >>> @instrument_inference(tracer, model_name="gemma-3-4b")
        ... def generate(prompt: str) -> str:
        ...     return engine.infer(prompt).text
        >>>
        >>> text = generate("Hello!")  # Automatically traced
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if tracer is None:
                return func(*args, **kwargs)

            with tracer.start_as_current_span(operation) as span:
                span.set_attribute("llm.system", "llamatelemetry")
                span.set_attribute("llm.model", model_name)
                span.set_attribute("gpu.device_id", gpu_id)
                span.set_attribute("nccl.split_mode", split_mode)

                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    latency_ms = (time.time() - start_time) * 1000
                    span.set_attribute("llm.latency_ms", latency_ms)

                    # Try to extract metrics from result
                    _annotate_from_result(span, result, latency_ms)

                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("llm.error", str(e))
                    raise

        return wrapper
    return decorator


def _annotate_from_result(span: Any, result: Any, latency_ms: float) -> None:
    """Extract metrics from result and annotate span."""
    try:
        # Handle InferResult objects
        if hasattr(result, 'tokens_generated'):
            tokens = result.tokens_generated
            span.set_attribute("llm.output.tokens", tokens)
            if latency_ms > 0:
                span.set_attribute("llm.tokens_per_sec", tokens / (latency_ms / 1000))

        if hasattr(result, 'success'):
            span.set_attribute("llm.success", result.success)

        if hasattr(result, 'error_message') and result.error_message:
            span.set_attribute("llm.error", result.error_message)

        # Handle dict results
        if isinstance(result, dict):
            if "tokens_predicted" in result:
                tokens = result["tokens_predicted"]
                span.set_attribute("llm.output.tokens", tokens)
                if latency_ms > 0:
                    span.set_attribute("llm.tokens_per_sec", tokens / (latency_ms / 1000))

            if "timings" in result:
                t = result["timings"]
                if "predicted_per_second" in t:
                    span.set_attribute("llm.tokens_per_sec", t["predicted_per_second"])

    except Exception:
        pass  # Don't fail if annotation fails


@contextmanager
def inference_span(
    tracer: Any,
    operation: str = "llm.inference",
    model: str = "",
    **attributes
):
    """
    Context manager for inference spans.

    Creates an OpenTelemetry span with LLM-specific attributes.
    Automatically records timing and allows adding custom attributes.

    Args:
        tracer: OpenTelemetry tracer instance
        operation: Span operation name
        model: Model name for span attributes
        **attributes: Additional span attributes

    Yields:
        OpenTelemetry span instance (or None if tracer is None)

    Example:
        >>> with inference_span(tracer, model="gemma-3-4b") as span:
        ...     result = engine.infer(prompt)
        ...     if span:
        ...         span.set_attribute("llm.output.tokens", result.tokens_generated)
    """
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(operation) as span:
        span.set_attribute("llm.system", "llamatelemetry")
        span.set_attribute("llm.model", model)

        for key, value in attributes.items():
            span.set_attribute(key, value)

        start_time = time.time()
        try:
            yield span
        finally:
            latency_ms = (time.time() - start_time) * 1000
            span.set_attribute("llm.latency_ms", latency_ms)


@contextmanager
def batch_inference_span(
    tracer: Any,
    batch_size: int,
    model: str = "",
    **attributes
):
    """
    Context manager for batch inference spans.

    Creates a parent span for batch operations with child spans
    for individual requests.

    Args:
        tracer: OpenTelemetry tracer instance
        batch_size: Number of items in batch
        model: Model name
        **attributes: Additional span attributes

    Yields:
        Tuple of (parent_span, create_child_span function)

    Example:
        >>> with batch_inference_span(tracer, batch_size=10, model="gemma") as (parent, child_fn):
        ...     for i, prompt in enumerate(prompts):
        ...         with child_fn(i):
        ...             result = engine.infer(prompt)
    """
    if tracer is None:
        yield None, lambda idx: _null_context()
        return

    with tracer.start_as_current_span("llm.batch_inference") as parent:
        parent.set_attribute("llm.system", "llamatelemetry")
        parent.set_attribute("llm.model", model)
        parent.set_attribute("llm.batch_size", batch_size)

        for key, value in attributes.items():
            parent.set_attribute(key, value)

        def create_child(index: int):
            return tracer.start_as_current_span(f"llm.inference.{index}")

        yield parent, create_child


@contextmanager
def _null_context():
    """Null context manager for when tracer is None."""
    yield None


def create_llm_attributes(
    model: str = "",
    prompt_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: float = 0.0,
    gpu_id: int = 0,
    split_mode: str = "none"
) -> Dict[str, Any]:
    """
    Create standard LLM span attributes.

    Args:
        model: Model name
        prompt_tokens: Number of prompt tokens
        output_tokens: Number of generated tokens
        latency_ms: Inference latency in milliseconds
        gpu_id: GPU device ID
        split_mode: Multi-GPU split mode

    Returns:
        Dictionary of span attributes
    """
    attrs = {
        "llm.system": "llamatelemetry",
        "llm.model": model,
        "llm.input.tokens": prompt_tokens,
        "llm.output.tokens": output_tokens,
        "llm.latency_ms": latency_ms,
        "gpu.device_id": gpu_id,
        "nccl.split_mode": split_mode,
    }

    if latency_ms > 0 and output_tokens > 0:
        attrs["llm.tokens_per_sec"] = output_tokens / (latency_ms / 1000)

    return attrs


def annotate_span_from_result(
    span: Any,
    result: Any,
    model: str = "",
    gpu_id: int = 0,
    split_mode: str = "none"
) -> None:
    """
    Annotate span with attributes from inference result.

    Args:
        span: OpenTelemetry span
        result: InferResult or dict with inference results
        model: Model name
        gpu_id: GPU device ID
        split_mode: Multi-GPU split mode
    """
    if span is None:
        return

    try:
        span.set_attribute("llm.system", "llamatelemetry")
        span.set_attribute("llm.model", model)
        span.set_attribute("gpu.device_id", gpu_id)
        span.set_attribute("nccl.split_mode", split_mode)

        if hasattr(result, 'tokens_generated'):
            span.set_attribute("llm.output.tokens", result.tokens_generated)

        if hasattr(result, 'latency_ms'):
            span.set_attribute("llm.latency_ms", result.latency_ms)

        if hasattr(result, 'tokens_per_sec'):
            span.set_attribute("llm.tokens_per_sec", result.tokens_per_sec)

        if hasattr(result, 'success'):
            span.set_attribute("llm.success", result.success)

        if hasattr(result, 'error_message') and result.error_message:
            span.set_attribute("llm.error", result.error_message)

    except Exception:
        pass


__all__ = [
    "instrument_inference",
    "inference_span",
    "batch_inference_span",
    "create_llm_attributes",
    "annotate_span_from_result",
]
