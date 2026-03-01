"""
llamatelemetry.telemetry.graphistry_export - Real-time trace graph visualization

Exports OpenTelemetry spans into a live graph structure consumable by
pygraphistry for GPU-accelerated real-time visualization of inference
activity, model internals, and NCCL communication patterns.

Graph schema:
    Nodes: Each span becomes a node with attributes (model, latency, tokens, ...)
    Edges: Parent-child span relationships become directed edges
    Node color: Mapped to inference latency (cool=fast, warm=slow)
    Edge weight: Token throughput on the edge

Usage:
    >>> from llamatelemetry.telemetry.graphistry_export import GraphistryTraceExporter
    >>> exporter = GraphistryTraceExporter()
    >>> exporter.export_spans(spans)
    >>> df_nodes, df_edges = exporter.get_dataframes()
"""

import time
from typing import Any, List, Optional, Tuple

try:
    import pandas as pd
    _PD_AVAILABLE = True
except ImportError:
    _PD_AVAILABLE = False


class GraphistryTraceExporter:
    """
    Collects OTel spans and converts them to pygraphistry node/edge DataFrames.

    Supports incremental export — new spans are appended each time
    export_spans() is called. Call get_dataframes() to retrieve the
    current graph state.
    """

    SCHEMA_VERSION = "llamatelemetry.graph.v1"
    NODE_TYPES = {"span", "model", "gpu", "unknown"}
    EDGE_TYPES = {"child_of", "uses_model", "runs_on", "unknown"}

    def __init__(self, server: Optional[str] = None, max_spans: int = 10000):
        """
        Args:
            server: Graphistry server URL (None = cloud)
            max_spans: Maximum spans to retain before oldest are dropped
        """
        self._server = server
        self._max_spans = max_spans
        self._spans_raw: List[dict] = []

    def export_spans(self, spans: List[Any]) -> None:
        """
        Ingest a batch of OTel spans.

        Args:
            spans: List of ReadableSpan objects from OTel SDK
        """
        for span in spans:
            self._spans_raw.append(self._span_to_dict(span))

        # Trim to max
        if len(self._spans_raw) > self._max_spans:
            self._spans_raw = self._spans_raw[-self._max_spans:]

    def _span_to_dict(self, span: Any) -> dict:
        """Convert a ReadableSpan to a flat dictionary."""
        ctx = span.get_span_context() if hasattr(span, "get_span_context") else None
        parent = span.parent if hasattr(span, "parent") else None

        span_id = hex(ctx.span_id)[2:] if ctx else str(id(span))
        parent_id = hex(parent.span_id)[2:] if parent and hasattr(parent, "span_id") else None
        trace_id = hex(ctx.trace_id)[2:] if ctx else ""

        attrs = dict(span.attributes) if hasattr(span, "attributes") and span.attributes else {}

        start_time = span.start_time if hasattr(span, "start_time") else 0
        end_time = span.end_time if hasattr(span, "end_time") else 0
        duration_ms = (end_time - start_time) / 1e6 if end_time and start_time else 0.0

        return {
            "span_id": span_id,
            "parent_span_id": parent_id,
            "trace_id": trace_id,
            "name": span.name if hasattr(span, "name") else "unknown",
            "start_time": start_time / 1e9 if start_time else 0,
            "end_time": end_time / 1e9 if end_time else 0,
            "duration_ms": duration_ms,
            "llm.model": attrs.get("llm.model", ""),
            "llm.input.tokens": attrs.get("llm.input.tokens", 0),
            "llm.output.tokens": attrs.get("llm.output.tokens", 0),
            "llm.latency_ms": attrs.get("llm.latency_ms", duration_ms),
            "llm.tokens_per_sec": attrs.get("llm.tokens_per_sec", 0.0),
            "llm.ttft_ms": attrs.get("llm.ttft_ms", 0.0),
            "gen_ai.operation.name": attrs.get("gen_ai.operation.name", ""),
            "gen_ai.request.model": attrs.get("gen_ai.request.model", ""),
            "gen_ai.response.model": attrs.get("gen_ai.response.model", ""),
            "gen_ai.usage.input_tokens": attrs.get("gen_ai.usage.input_tokens", 0),
            "gen_ai.usage.output_tokens": attrs.get("gen_ai.usage.output_tokens", 0),
            "gpu.device_id": attrs.get("gpu.device_id", 0),
            "nccl.split_mode": attrs.get("nccl.split_mode", "none"),
            "status": str(span.status) if hasattr(span, "status") else "OK",
        }

    def get_dataframes(self) -> Tuple[Any, Any]:
        """
        Build node and edge DataFrames from collected spans.

        Returns:
            Tuple of (nodes_df, edges_df) as pandas DataFrames.
            Returns (None, None) if pandas is not installed or no spans collected.

        Node columns:
            span_id, name, duration_ms, llm.model, llm.latency_ms,
            llm.tokens_per_sec, gpu.device_id, nccl.split_mode

        Edge columns:
            src (parent span_id), dst (child span_id), weight (tokens/sec)
        """
        if not _PD_AVAILABLE or not self._spans_raw:
            return None, None

        nodes = []
        edges = []
        extra_nodes = {}

        for s in self._spans_raw:
            node_id = s["span_id"]
            nodes.append({
                "node_id": node_id,
                "span_id": s["span_id"],
                "node_type": "span",
                "schema_version": self.SCHEMA_VERSION,
                "name": s["name"],
                "duration_ms": s["duration_ms"],
                "llm_model": s["llm.model"],
                "llm_latency_ms": s["llm.latency_ms"],
                "llm_tokens_per_sec": s["llm.tokens_per_sec"],
                "llm_input_tokens": s["llm.input.tokens"],
                "llm_output_tokens": s["llm.output.tokens"],
                "llm_ttft_ms": s["llm.ttft_ms"],
                "gen_ai_operation": s["gen_ai.operation.name"],
                "gen_ai_request_model": s["gen_ai.request.model"],
                "gen_ai_response_model": s["gen_ai.response.model"],
                "gen_ai_input_tokens": s["gen_ai.usage.input_tokens"],
                "gen_ai_output_tokens": s["gen_ai.usage.output_tokens"],
                "gpu_device_id": s["gpu.device_id"],
                "nccl_split_mode": s["nccl.split_mode"],
                "trace_id": s["trace_id"],
            })

            if s["parent_span_id"]:
                edges.append({
                    "src": s["parent_span_id"],
                    "dst": node_id,
                    "edge_type": "child_of",
                    "schema_version": self.SCHEMA_VERSION,
                    "weight": s["llm.tokens_per_sec"],
                    "latency_ms": s["llm.latency_ms"],
                })

            # Model node + edge
            if s["llm.model"]:
                model_id = f"model:{s['llm.model']}"
                if model_id not in extra_nodes:
                    extra_nodes[model_id] = {
                        "node_id": model_id,
                        "node_type": "model",
                        "name": s["llm.model"],
                        "schema_version": self.SCHEMA_VERSION,
                    }
                edges.append({
                    "src": node_id,
                    "dst": model_id,
                    "edge_type": "uses_model",
                    "schema_version": self.SCHEMA_VERSION,
                    "weight": 1,
                })

            # GPU node + edge
            gpu_id = s["gpu.device_id"]
            gpu_node_id = f"gpu:{gpu_id}"
            if gpu_node_id not in extra_nodes:
                extra_nodes[gpu_node_id] = {
                    "node_id": gpu_node_id,
                    "node_type": "gpu",
                    "name": f"GPU {gpu_id}",
                    "schema_version": self.SCHEMA_VERSION,
                }
            edges.append({
                "src": node_id,
                "dst": gpu_node_id,
                "edge_type": "runs_on",
                "schema_version": self.SCHEMA_VERSION,
                "weight": 1,
            })

        if extra_nodes:
            nodes.extend(extra_nodes.values())

        nodes, edges = self._validate_schema(nodes, edges)

        nodes_df = pd.DataFrame(nodes)
        edges_df = pd.DataFrame(edges) if edges else pd.DataFrame(columns=["src", "dst", "edge_type", "weight", "latency_ms"])

        return nodes_df, edges_df

    def visualize(self) -> Any:
        """
        Push the current trace graph to pygraphistry and return a Graphistry plotter.

        Returns:
            Graphistry Plotter object, or None if pygraphistry is unavailable.
        """
        try:
            import graphistry
        except ImportError:
            import warnings
            warnings.warn("pygraphistry not installed. Install with: pip install pygraphistry")
            return None

        nodes_df, edges_df = self.get_dataframes()
        if nodes_df is None or edges_df is None or len(nodes_df) == 0:
            return None

        if self._server:
            graphistry.register(server=self._server)

        return (
            graphistry.edges(edges_df, "src", "dst")
            .nodes(nodes_df, "node_id")
            .bind(edge_color="latency_ms", node_color="llm_latency_ms")
        )

    def _validate_schema(self, nodes: List[dict], edges: List[dict]) -> Tuple[List[dict], List[dict]]:
        import warnings
        cleaned_nodes = []
        for node in nodes:
            node_type = node.get("node_type") or "unknown"
            if node_type not in self.NODE_TYPES:
                warnings.warn(f"Unknown node_type '{node_type}', coercing to 'unknown'")
                node_type = "unknown"
            node["node_type"] = node_type
            node.setdefault("schema_version", self.SCHEMA_VERSION)
            if not node.get("node_id"):
                warnings.warn("Skipping node with missing node_id")
                continue
            cleaned_nodes.append(node)

        cleaned_edges = []
        for edge in edges:
            edge_type = edge.get("edge_type") or "unknown"
            if edge_type not in self.EDGE_TYPES:
                warnings.warn(f"Unknown edge_type '{edge_type}', coercing to 'unknown'")
                edge_type = "unknown"
            edge["edge_type"] = edge_type
            edge.setdefault("schema_version", self.SCHEMA_VERSION)
            if not edge.get("src") or not edge.get("dst"):
                warnings.warn("Skipping edge with missing src/dst")
                continue
            cleaned_edges.append(edge)

        return cleaned_nodes, cleaned_edges

    def clear(self) -> None:
        """Clear all collected spans."""
        self._spans_raw.clear()

    def __len__(self) -> int:
        return len(self._spans_raw)
