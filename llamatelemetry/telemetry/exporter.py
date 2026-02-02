"""
llamatelemetry.telemetry.exporter - Span/Metric Exporters

Builds and configures OpenTelemetry exporters:
    - OTLP gRPC (default, vendor-neutral)
    - OTLP HTTP (alternative transport)
    - Console (debug/local)

These exporters are plugged into the TracerProvider via BatchSpanProcessor.
"""

from typing import Any, List, Optional


def build_exporters(otlp_endpoint: Optional[str] = None) -> List[Any]:
    """
    Build a list of SpanExporter instances based on configuration.

    Args:
        otlp_endpoint: OTLP collector endpoint URL.
            - If starts with "http://" or "https://", uses HTTP exporter
            - Otherwise assumes gRPC transport
            - If None, uses ConsoleSpanExporter only (debug mode)

    Returns:
        List of configured SpanExporter instances
    """
    exporters = []

    if otlp_endpoint is None:
        # Debug mode: console only
        try:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            exporters.append(ConsoleSpanExporter())
        except ImportError:
            pass
        return exporters

    # OTLP export
    if otlp_endpoint.startswith("http://") or otlp_endpoint.startswith("https://"):
        # HTTP transport
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            exporters.append(OTLPSpanExporter(endpoint=otlp_endpoint))
        except ImportError:
            import warnings
            warnings.warn(
                "OTLP HTTP exporter not available. Install: "
                "pip install opentelemetry-exporter-otlp-proto-http",
                ImportWarning,
            )
    else:
        # gRPC transport (default)
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporters.append(OTLPSpanExporter(endpoint=otlp_endpoint))
        except ImportError:
            import warnings
            warnings.warn(
                "OTLP gRPC exporter not available. Install: "
                "pip install opentelemetry-exporter-otlp-proto-grpc",
                ImportWarning,
            )

    # Always add console exporter for local visibility
    try:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        exporters.append(ConsoleSpanExporter())
    except ImportError:
        pass

    return exporters
