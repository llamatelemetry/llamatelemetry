"""
llamatelemetry.telemetry - OpenTelemetry integration layer

Provides GPU-native tracing, metrics, and logging for LLM inference pipelines.
Integrates opentelemetry-python (API + SDK) and opentelemetry-python-contrib
instrumentation into the llamatelemetry runtime.

Components:
    - TracerProvider: Wraps OTel TracerProvider with GPU-aware resource detection
    - GpuMetricsCollector: Exports fine-grained GPU metrics (latency, tokens/sec, VRAM, NCCL)
    - InferenceTracer: End-to-end span tracing for LLM inference requests
    - OTLPExporter: Vendor-neutral export via OTLP (gRPC/HTTP)
    - GraphistryExporter: Real-time trace graph export to pygraphistry

Usage:
    >>> from llamatelemetry.telemetry import setup_telemetry
    >>> tracer, meter = setup_telemetry(
    ...     service_name="llamatelemetry-inference",
    ...     otlp_endpoint="http://localhost:4317"
    ... )
    >>> with tracer.start_as_current_span("llm.inference") as span:
    ...     span.set_attribute("llm.model", "gemma-3-1b")
    ...     result = engine.infer("Hello")
"""

from typing import Optional, Tuple, Any

# Lazy imports to avoid hard dependency on opentelemetry packages
_OTEL_AVAILABLE = False
_GRAPHISTRY_AVAILABLE = False
_GPU_COLLECTOR = None

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    _OTEL_AVAILABLE = True
except ImportError:
    pass

try:
    import graphistry
    _GRAPHISTRY_AVAILABLE = True
except ImportError:
    pass


def is_otel_available() -> bool:
    """Check if OpenTelemetry SDK is installed."""
    return _OTEL_AVAILABLE


def is_graphistry_available() -> bool:
    """Check if pygraphistry is installed."""
    return _GRAPHISTRY_AVAILABLE


def get_metrics_collector() -> Any:
    """Return the active GPU metrics collector if initialized."""
    return _GPU_COLLECTOR


def setup_telemetry(
    service_name: str = "llamatelemetry",
    service_version: str = "0.1.0",
    otlp_endpoint: Optional[str] = None,
    enable_graphistry: bool = False,
    graphistry_server: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Initialize OpenTelemetry tracing and metrics for llamatelemetry.

    Sets up a TracerProvider and MeterProvider with GPU-aware resource
    attributes. Optionally configures OTLP export and pygraphistry
    real-time graph export.

    Args:
        service_name: OpenTelemetry service name
        service_version: Service version string
        otlp_endpoint: OTLP collector endpoint (gRPC, e.g. http://localhost:4317)
        enable_graphistry: Enable real-time graph export to pygraphistry
        graphistry_server: Graphistry server URL (uses cloud if None)

    Returns:
        Tuple of (tracer, meter) — OpenTelemetry Tracer and Meter instances.
        Returns (None, None) if opentelemetry-sdk is not installed.

    Example:
        >>> tracer, meter = setup_telemetry(
        ...     service_name="my-llm-app",
        ...     otlp_endpoint="http://localhost:4317"
        ... )
        >>> if tracer:
        ...     with tracer.start_as_current_span("inference"):
        ...         pass
    """
    if not _OTEL_AVAILABLE:
        import warnings
        warnings.warn(
            "OpenTelemetry SDK not installed. Install with: "
            "pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp",
            ImportWarning,
        )
        return None, None

    from .tracer import InferenceTracerProvider
    from .metrics import GpuMetricsCollector
    from .exporter import build_exporters

    # Build resource with GPU info
    from .resource import build_gpu_resource
    resource = build_gpu_resource(service_name, service_version)

    # Create TracerProvider
    span_exporters = build_exporters(otlp_endpoint)
    tracer_provider = InferenceTracerProvider(
        resource=resource,
        span_exporters=span_exporters,
    )
    trace.set_tracer_provider(tracer_provider)

    # Create MeterProvider
    meter_provider = MeterProvider(resource=resource)
    metrics.set_meter_provider(meter_provider)

    # Start GPU metrics collector
    global _GPU_COLLECTOR
    gpu_collector = GpuMetricsCollector(meter_provider)
    gpu_collector.start()
    _GPU_COLLECTOR = gpu_collector

    # Optionally set up pygraphistry export
    if enable_graphistry:
        from .graphistry_export import GraphistryTraceExporter
        g_exporter = GraphistryTraceExporter(server=graphistry_server)
        tracer_provider.add_graphistry_exporter(g_exporter)

    tracer = tracer_provider.get_tracer(service_name, version=service_version)
    meter = meter_provider.get_meter(service_name, version=service_version)

    return tracer, meter


# New v0.2.0+ modules
from .auto_instrument import (
    instrument_inference,
    inference_span,
    batch_inference_span,
    create_llm_attributes,
    annotate_span_from_result,
)

from .instrumentor import (
    LlamaCppClientInstrumentor,
    instrument_llamacpp_client,
    uninstrument_llamacpp_client,
)

from .monitor import (
    PerformanceSnapshot,
    InferenceRecord,
    PerformanceMonitor,
)

__all__ = [
    # Core setup
    "setup_telemetry",
    "is_otel_available",
    "is_graphistry_available",
    "get_metrics_collector",

    # Auto-instrumentation (v0.2.0+)
    "instrument_inference",
    "inference_span",
    "batch_inference_span",
    "create_llm_attributes",
    "annotate_span_from_result",

    # Client instrumentor (v0.2.0+)
    "LlamaCppClientInstrumentor",
    "instrument_llamacpp_client",
    "uninstrument_llamacpp_client",

    # Performance monitor (v0.2.0+)
    "PerformanceSnapshot",
    "InferenceRecord",
    "PerformanceMonitor",
]
