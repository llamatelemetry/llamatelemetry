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
    llama_server_url: Optional[str] = None,
    llama_server_path: Optional[str] = None,
    enable_llama_metrics: bool = False,
    llama_metrics_interval: float = 5.0,
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
        llama_server_url: Optional llama-server URL for /props metadata
        llama_server_path: Optional llama-server path for --version metadata
        enable_llama_metrics: Enable llama.cpp /metrics collection via OTel gauges
        llama_metrics_interval: Cache interval for llama.cpp /metrics polling

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
    resource = build_gpu_resource(
        service_name,
        service_version,
        llama_server_url=llama_server_url,
        llama_server_path=llama_server_path,
    )

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
    gpu_collector = GpuMetricsCollector(
        meter_provider,
        server_url=llama_server_url if enable_llama_metrics else None,
        llama_cpp_poll_interval=llama_metrics_interval,
    )
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


from .semconv import (
    has_gen_ai_semconv,
    set_gen_ai_attr,
    set_gen_ai_provider,
    metric_name as gen_ai_metric_name,
    attr_name as gen_ai_attr_name,
)

def setup_otlp_env_from_kaggle_secrets(
    endpoint_key: str = "OTLP_ENDPOINT",
    token_key: str = "OTLP_TOKEN",
    header_key: str = "authorization",
    token_prefix: str = "Bearer",
):
    """
    Load OTLP endpoint/token from Kaggle secrets into OTEL_* env vars.

    Returns:
        Dict with endpoint and header values
    """
    import os
    try:
        from ..kaggle.secrets import KaggleSecrets
    except Exception:
        return {"endpoint": None, "headers": None}

    secrets = KaggleSecrets()
    endpoint = secrets.get(endpoint_key)
    token = secrets.get(token_key)

    if endpoint:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint

    headers = None
    if token:
        headers = f"{header_key}={token_prefix} {token}"
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = headers

    return {"endpoint": endpoint, "headers": headers}


def setup_grafana_otlp(
    service_name: str = "llamatelemetry",
    service_version: str = "0.1.0",
    otlp_endpoint: Optional[str] = None,
    enable_graphistry: bool = False,
    graphistry_server: Optional[str] = None,
    llama_server_url: Optional[str] = None,
    llama_server_path: Optional[str] = None,
    enable_llama_metrics: bool = False,
    llama_metrics_interval: float = 5.0,
):
    """
    Configure OTLP env vars from Kaggle secrets and initialize telemetry.

    Returns:
        (tracer, meter) or (None, None) if OTel is unavailable
    """
    env = setup_otlp_env_from_kaggle_secrets()
    endpoint = otlp_endpoint or env.get("endpoint")
    return setup_telemetry(
        service_name=service_name,
        service_version=service_version,
        otlp_endpoint=endpoint,
        enable_graphistry=enable_graphistry,
        graphistry_server=graphistry_server,
        llama_server_url=llama_server_url,
        llama_server_path=llama_server_path,
        enable_llama_metrics=enable_llama_metrics,
        llama_metrics_interval=llama_metrics_interval,
    )


# Additional telemetry modules
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

from .client import (
    InstrumentedLlamaCppClient,
    InstrumentationConfig,
)


class InstrumentedLLMClient:
    """
    Convenience wrapper that returns an instrumented LlamaCppClient.
    """

    def __init__(self, *args, **kwargs):
        from ..api import LlamaCppClient
        self._instrumentor = LlamaCppClientInstrumentor()
        self._instrumentor.instrument()
        self._client = LlamaCppClient(*args, **kwargs)

    def uninstrument(self):
        """Disable instrumentation globally."""
        try:
            self._instrumentor.uninstrument()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._client, name)

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
    "setup_otlp_env_from_kaggle_secrets",
    "setup_grafana_otlp",
    "has_gen_ai_semconv",
    "set_gen_ai_attr",
    "set_gen_ai_provider",
    "gen_ai_metric_name",
    "gen_ai_attr_name",

    # Auto-instrumentation
    "instrument_inference",
    "inference_span",
    "batch_inference_span",
    "create_llm_attributes",
    "annotate_span_from_result",

    # Client instrumentor
    "LlamaCppClientInstrumentor",
    "instrument_llamacpp_client",
    "uninstrument_llamacpp_client",
    "InstrumentedLLMClient",
    "InstrumentedLlamaCppClient",
    "InstrumentationConfig",

    # Performance monitor
    "PerformanceSnapshot",
    "InferenceRecord",
    "PerformanceMonitor",
]
