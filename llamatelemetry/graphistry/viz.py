"""
llamatelemetry.graphistry.viz - High-level Graphistry visualization builder.

Simplifies common visualization patterns for LLM inference analysis.

Example:
    >>> from llamatelemetry.graphistry import GraphistryViz
    >>>
    >>> viz = GraphistryViz(auto_register=True)
    >>>
    >>> # Plot inference results
    >>> viz.plot_inference_results(results)
    >>>
    >>> # Plot trace graph
    >>> viz.plot_trace_graph(spans)
"""

from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import pandas as pd
    import graphistry


@dataclass
class TraceVisualization:
    """Configuration for trace visualization."""
    color_by: str = "latency_ms"
    size_by: str = "tokens"
    layout: str = "force"
    title: str = "LLM Inference Traces"
    palette: str = "viridis"


@dataclass
class MetricsVisualization:
    """Configuration for metrics visualization."""
    metric: str = "latency_ms"
    aggregation: str = "mean"
    time_window: str = "1min"


class GraphistryViz:
    """
    High-level visualization builder for LLM telemetry data.

    Provides simplified interfaces for common visualization patterns
    including inference results, trace graphs, and GPU metrics.

    Example:
        >>> viz = GraphistryViz()
        >>>
        >>> # From inference results
        >>> viz.plot_latency_distribution(results)
        >>>
        >>> # From spans
        >>> viz.plot_trace_graph(spans)
        >>>
        >>> # From metrics
        >>> viz.plot_token_throughput(metrics)
    """

    def __init__(self, auto_register: bool = True):
        """
        Initialize GraphistryViz.

        Args:
            auto_register: Try to register using environment variables or secrets

        Raises:
            ImportError: If pygraphistry is not installed
        """
        self._graphistry = None
        self._registered = False
        self._pd = None

        try:
            import graphistry
            import pandas as pd
            self._graphistry = graphistry
            self._pd = pd
        except ImportError:
            raise ImportError(
                "pygraphistry and pandas required. Install with: "
                "pip install graphistry[ai] pandas"
            )

        if auto_register:
            self._registered = self._try_register()

    def _try_register(self) -> bool:
        """Try to register with Graphistry."""
        try:
            from .connector import register_graphistry
            return register_graphistry()
        except Exception:
            pass

        # Try from kaggle secrets
        try:
            from ..kaggle.secrets import setup_graphistry_auth
            return setup_graphistry_auth()
        except Exception:
            pass

        return False

    @property
    def is_registered(self) -> bool:
        """Check if registered with Graphistry."""
        return self._registered

    def plot_inference_results(
        self,
        results: List[Any],
        color_by: str = "latency_ms",
        size_by: str = "tokens_generated",
        title: str = "Inference Results",
        **kwargs
    ):
        """
        Plot inference results as a graph.

        Creates a visualization where each inference result is a node,
        connected sequentially to show the progression of requests.

        Args:
            results: List of InferResult objects or dicts
            color_by: Attribute to use for node color
            size_by: Attribute to use for node size
            title: Visualization title
            **kwargs: Additional graphistry plot arguments

        Returns:
            Graphistry plotter object
        """
        pd = self._pd
        g = self._graphistry

        # Convert results to DataFrame
        data = []
        for i, r in enumerate(results):
            record = {"id": i}

            # Extract attributes
            if hasattr(r, "latency_ms"):
                record["latency_ms"] = r.latency_ms
            elif isinstance(r, dict):
                record["latency_ms"] = r.get("latency_ms", 0)
            else:
                record["latency_ms"] = 0

            if hasattr(r, "tokens_generated"):
                record["tokens_generated"] = r.tokens_generated
            elif isinstance(r, dict):
                record["tokens_generated"] = r.get("tokens_generated", 0)
            else:
                record["tokens_generated"] = 0

            if hasattr(r, "tokens_per_sec"):
                record["tokens_per_sec"] = r.tokens_per_sec
            elif isinstance(r, dict):
                record["tokens_per_sec"] = r.get("tokens_per_sec", 0)
            else:
                record["tokens_per_sec"] = 0

            if hasattr(r, "success"):
                record["success"] = r.success
            elif isinstance(r, dict):
                record["success"] = r.get("success", True)
            else:
                record["success"] = True

            data.append(record)

        nodes_df = pd.DataFrame(data)

        # Create edges (sequential requests)
        if len(results) > 1:
            edges_df = pd.DataFrame({
                "src": list(range(len(results) - 1)),
                "dst": list(range(1, len(results))),
            })
        else:
            edges_df = pd.DataFrame({"src": [], "dst": []})

        # Build graph
        plotter = g.edges(edges_df, "src", "dst")
        plotter = plotter.nodes(nodes_df, "id")

        # Apply encodings
        if color_by in nodes_df.columns:
            plotter = plotter.encode_point_color(color_by)

        if size_by in nodes_df.columns:
            plotter = plotter.encode_point_size(size_by)

        plotter = plotter.settings(url_params={"title": title})

        return plotter.plot(**kwargs)

    def plot_trace_graph(
        self,
        spans: List[Dict[str, Any]],
        color_by: str = "duration_ms",
        size_by: str = "tokens",
        title: str = "Trace Graph",
        **kwargs
    ):
        """
        Plot OpenTelemetry spans as a directed graph.

        Creates a visualization showing the parent-child relationships
        between spans, useful for understanding inference pipelines.

        Args:
            spans: List of span dictionaries with trace_id, span_id, parent_span_id
            color_by: Attribute for node color
            size_by: Attribute for node size
            title: Visualization title
            **kwargs: Additional graphistry plot arguments

        Returns:
            Graphistry plotter object or None if no relationships
        """
        pd = self._pd
        g = self._graphistry

        # Build edges from parent relationships
        edges = []
        for span in spans:
            parent_id = span.get("parent_span_id")
            if parent_id:
                edges.append({
                    "src": parent_id,
                    "dst": span.get("span_id", ""),
                    "trace_id": span.get("trace_id", ""),
                })

        if not edges:
            print("No parent-child relationships found in spans")
            return None

        edges_df = pd.DataFrame(edges)
        nodes_df = pd.DataFrame(spans)

        # Build graph
        plotter = g.edges(edges_df, "src", "dst")

        if "span_id" in nodes_df.columns:
            plotter = plotter.nodes(nodes_df, "span_id")

        # Apply encodings
        if color_by in nodes_df.columns:
            plotter = plotter.encode_point_color(color_by)

        if size_by in nodes_df.columns:
            plotter = plotter.encode_point_size(size_by)

        plotter = plotter.settings(url_params={"title": title})

        return plotter.plot(**kwargs)

    def plot_gpu_metrics(
        self,
        metrics: List[Dict[str, Any]],
        time_column: str = "timestamp",
        color_by: str = "gpu_utilization",
        size_by: str = "memory_used",
        title: str = "GPU Metrics Timeline",
        **kwargs
    ):
        """
        Plot GPU metrics over time as a timeline graph.

        Args:
            metrics: List of metric dictionaries
            time_column: Column containing timestamps
            color_by: Attribute for node color
            size_by: Attribute for node size
            title: Visualization title
            **kwargs: Additional graphistry plot arguments

        Returns:
            Graphistry plotter object
        """
        pd = self._pd
        g = self._graphistry

        df = pd.DataFrame(metrics)

        if df.empty:
            print("No metrics to plot")
            return None

        df = df.reset_index()

        # Create time-series edges
        if len(df) > 1:
            edges = pd.DataFrame({
                "src": list(range(len(df) - 1)),
                "dst": list(range(1, len(df))),
            })
        else:
            edges = pd.DataFrame({"src": [], "dst": []})

        # Build graph
        plotter = g.edges(edges, "src", "dst")
        plotter = plotter.nodes(df, "index")

        # Apply encodings
        if color_by in df.columns:
            plotter = plotter.encode_point_color(color_by)

        if size_by in df.columns:
            plotter = plotter.encode_point_size(size_by)

        plotter = plotter.settings(url_params={"title": title})

        return plotter.plot(**kwargs)

    def plot_latency_distribution(
        self,
        results: List[Any],
        bins: int = 20,
        title: str = "Latency Distribution",
        **kwargs
    ):
        """
        Plot latency distribution as a histogram-style graph.

        Args:
            results: List of InferResult objects or dicts
            bins: Number of histogram bins
            title: Visualization title
            **kwargs: Additional graphistry plot arguments

        Returns:
            Graphistry plotter object
        """
        pd = self._pd
        g = self._graphistry

        # Extract latencies
        latencies = []
        for r in results:
            if hasattr(r, "latency_ms"):
                latencies.append(r.latency_ms)
            elif isinstance(r, dict):
                latencies.append(r.get("latency_ms", 0))

        if not latencies:
            print("No latency data to plot")
            return None

        # Create histogram data
        import numpy as np
        hist, bin_edges = np.histogram(latencies, bins=bins)

        nodes = []
        for i, count in enumerate(hist):
            nodes.append({
                "id": i,
                "bin_start": bin_edges[i],
                "bin_end": bin_edges[i + 1],
                "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
                "count": count,
            })

        nodes_df = pd.DataFrame(nodes)

        # Create edges between adjacent bins
        if len(nodes) > 1:
            edges = pd.DataFrame({
                "src": list(range(len(nodes) - 1)),
                "dst": list(range(1, len(nodes))),
            })
        else:
            edges = pd.DataFrame({"src": [], "dst": []})

        # Build graph
        plotter = g.edges(edges, "src", "dst")
        plotter = plotter.nodes(nodes_df, "id")
        plotter = plotter.encode_point_color("bin_center")
        plotter = plotter.encode_point_size("count")
        plotter = plotter.settings(url_params={"title": title})

        return plotter.plot(**kwargs)

    def plot_knowledge_graph(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        entity_id_col: str = "id",
        entity_label_col: str = "name",
        rel_source_col: str = "source",
        rel_target_col: str = "target",
        rel_type_col: str = "type",
        title: str = "Knowledge Graph",
        **kwargs
    ):
        """
        Plot a knowledge graph from entities and relationships.

        Args:
            entities: List of entity dictionaries
            relationships: List of relationship dictionaries
            entity_id_col: Entity ID column name
            entity_label_col: Entity label column name
            rel_source_col: Relationship source column name
            rel_target_col: Relationship target column name
            rel_type_col: Relationship type column name
            title: Visualization title
            **kwargs: Additional graphistry plot arguments

        Returns:
            Graphistry plotter object
        """
        pd = self._pd
        g = self._graphistry

        nodes_df = pd.DataFrame(entities)
        edges_df = pd.DataFrame(relationships)

        if nodes_df.empty or edges_df.empty:
            print("No entities or relationships to plot")
            return None

        # Build graph
        plotter = g.edges(edges_df, rel_source_col, rel_target_col)

        if entity_id_col in nodes_df.columns:
            plotter = plotter.nodes(nodes_df, entity_id_col)

        # Apply label
        if entity_label_col in nodes_df.columns:
            plotter = plotter.bind(point_title=entity_label_col)

        # Color edges by type
        if rel_type_col in edges_df.columns:
            plotter = plotter.encode_edge_color(rel_type_col)

        plotter = plotter.settings(url_params={"title": title})

        return plotter.plot(**kwargs)


def create_graph_viz(
    edges: "pd.DataFrame",
    nodes: Optional["pd.DataFrame"] = None,
    source: str = "source",
    target: str = "target",
    node_id: str = "id",
    color_by: Optional[str] = None,
    size_by: Optional[str] = None,
    title: str = "Graph",
    auto_register: bool = True,
    **kwargs
):
    """
    Convenience function to create a graph visualization.

    Args:
        edges: DataFrame with edge data
        nodes: DataFrame with node data (optional)
        source: Source column name in edges
        target: Target column name in edges
        node_id: Node ID column name in nodes
        color_by: Attribute for node color
        size_by: Attribute for node size
        title: Visualization title
        auto_register: Try to auto-register with Graphistry
        **kwargs: Additional graphistry plot arguments

    Returns:
        Graphistry plotter object
    """
    import graphistry as g

    if auto_register:
        try:
            from .connector import register_graphistry
            register_graphistry()
        except Exception:
            pass

    plotter = g.edges(edges, source, target)

    if nodes is not None:
        plotter = plotter.nodes(nodes, node_id)

    if color_by:
        plotter = plotter.encode_point_color(color_by)

    if size_by:
        plotter = plotter.encode_point_size(size_by)

    plotter = plotter.settings(url_params={"title": title})

    return plotter.plot(**kwargs)


__all__ = [
    "GraphistryViz",
    "TraceVisualization",
    "MetricsVisualization",
    "create_graph_viz",
]
