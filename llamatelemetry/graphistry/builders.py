"""
Graphistry builders for common LLM graph patterns.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


class GraphistryBuilders:
    """
    Helpers to build (nodes_df, edges_df) pairs for Graphistry.
    """

    @staticmethod
    def knowledge_graph(
        entities: Optional[List[Any]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Build dataframes for knowledge graphs.

        entities: list of strings or dicts with id/name/label fields
        relationships: list of dicts with source/target/type
        """
        try:
            import pandas as pd
        except Exception as exc:
            raise ImportError("pandas is required for knowledge_graph") from exc

        entities = entities or []
        relationships = relationships or []

        nodes: List[Dict[str, Any]] = []
        if entities:
            for ent in entities:
                if isinstance(ent, str):
                    nodes.append({"id": ent, "label": ent})
                elif isinstance(ent, dict):
                    node_id = ent.get("id") or ent.get("name") or ent.get("label")
                    node = {"id": node_id, **ent}
                    nodes.append(node)

        # Derive nodes from relationships if none provided
        if not nodes and relationships:
            seen = set()
            for rel in relationships:
                for key in ("source", "target"):
                    val = rel.get(key)
                    if val is not None and val not in seen:
                        nodes.append({"id": val, "label": val})
                        seen.add(val)

        edges = []
        for rel in relationships:
            src = rel.get("source")
            dst = rel.get("target")
            if src is None or dst is None:
                continue
            edge = {"src": src, "dst": dst}
            if "type" in rel:
                edge["type"] = rel["type"]
            edge.update({k: v for k, v in rel.items() if k not in {"source", "target"}})
            edges.append(edge)

        return pd.DataFrame(nodes), pd.DataFrame(edges)

    @staticmethod
    def document_similarity(
        documents: List[Any],
        similarities: List[Dict[str, Any]],
        doc_id_key: str = "id",
    ):
        """
        Build graph frames for document similarity networks.
        """
        try:
            import pandas as pd
        except Exception as exc:
            raise ImportError("pandas is required for document_similarity") from exc

        nodes = []
        for i, doc in enumerate(documents):
            if isinstance(doc, dict):
                node_id = doc.get(doc_id_key, i)
                nodes.append({"id": node_id, **doc})
            else:
                nodes.append({"id": i, "text": str(doc)})

        edges = []
        for sim in similarities:
            src = sim.get("source")
            dst = sim.get("target")
            if src is None or dst is None:
                continue
            edge = {"src": src, "dst": dst}
            edge.update({k: v for k, v in sim.items() if k not in {"source", "target"}})
            edges.append(edge)

        return pd.DataFrame(nodes), pd.DataFrame(edges)

    @staticmethod
    def embedding_knn(
        embeddings: List[List[float]],
        labels: Optional[List[str]] = None,
        k: int = 5,
        metric: str = "cosine",
    ):
        """
        Build a kNN graph from embeddings.
        """
        try:
            import numpy as np
            import pandas as pd
        except Exception as exc:
            raise ImportError("numpy and pandas are required for embedding_knn") from exc

        X = np.asarray(embeddings, dtype=float)
        if metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            Xn = X / norms
            sims = Xn @ Xn.T
        else:
            # Euclidean similarity (negative distance)
            dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
            sims = -dists

        n = X.shape[0]
        labels = labels or [str(i) for i in range(n)]
        nodes = pd.DataFrame({"id": list(range(n)), "label": labels})

        edges = []
        for i in range(n):
            idx = np.argsort(-sims[i])
            for j in idx[1 : k + 1]:
                edges.append({"src": i, "dst": int(j), "score": float(sims[i, j])})

        return nodes, pd.DataFrame(edges)

    @staticmethod
    def attention_graph(
        attention: List[List[float]],
        tokens: Optional[List[str]] = None,
        threshold: float = 0.0,
    ):
        """
        Build a token attention graph from an attention matrix.
        """
        try:
            import numpy as np
            import pandas as pd
        except Exception as exc:
            raise ImportError("numpy and pandas are required for attention_graph") from exc

        A = np.asarray(attention, dtype=float)
        n = A.shape[0]
        tokens = tokens or [str(i) for i in range(n)]

        nodes = pd.DataFrame({"id": list(range(n)), "token": tokens})
        edges = []
        for i in range(n):
            for j in range(n):
                w = float(A[i, j])
                if w >= threshold:
                    edges.append({"src": i, "dst": j, "weight": w})

        return nodes, pd.DataFrame(edges)


@dataclass
class InferenceRecord:
    """Normalized inference record (one row per request)."""
    ts: float
    operation: str
    model: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    ttfb_ms: Optional[float] = None
    prompt_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    gpu_id: Optional[int] = None
    split_mode: Optional[str] = None
    success: Optional[bool] = None
    error_type: Optional[str] = None


def records_to_dataframe(records: Iterable[InferenceRecord]):
    """Convert normalized records into a dataframe."""
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("pandas is required for records_to_dataframe") from exc
    rows = [r.__dict__ for r in records]
    return pd.DataFrame(rows)


def traces_to_records(spans: List[Dict[str, Any]]) -> List[InferenceRecord]:
    """Best-effort conversion from exported span JSON to InferenceRecord."""
    out: List[InferenceRecord] = []
    for s in spans:
        attrs = _attrs_map(s)
        ts = float(s.get("start_time_unix_nano", 0)) / 1e9 if "start_time_unix_nano" in s else float(s.get("ts", 0))
        name = str(s.get("name") or attrs.get("gen_ai.operation.name") or attrs.get("llm.operation") or "")
        model = str(attrs.get("gen_ai.request.model") or attrs.get("llm.model") or "")
        latency_ms = _duration_ms(s)

        rec = InferenceRecord(
            ts=ts,
            operation=name,
            model=model,
            latency_ms=latency_ms,
            input_tokens=_int_or_none(attrs.get("gen_ai.usage.input_tokens") or attrs.get("llm.input.tokens")),
            output_tokens=_int_or_none(attrs.get("gen_ai.usage.output_tokens") or attrs.get("llm.output.tokens")),
            ttfb_ms=_float_or_none(attrs.get("llm.ttft_ms") or attrs.get("llm.ttfb_ms") or attrs.get("llama.ttfb_ms")),
            prompt_ms=_float_or_none(attrs.get("llm.prompt_ms")),
            generation_ms=_float_or_none(attrs.get("llm.generation_ms")),
            gpu_id=_int_or_none(attrs.get("gpu.device_id")),
            split_mode=str(attrs.get("nccl.split_mode") or "") or None,
            success=_bool_or_none(attrs.get("llm.success")),
            error_type=str(attrs.get("error.type") or attrs.get("llm.error") or "") or None,
        )
        out.append(rec)
    return out


def build_graph_nodes_edges(
    df,
    *,
    node_id_col: str = "operation",
    group_col: str = "model",
) -> Tuple[Any, Any]:
    """Build simple nodes/edges for a directed sequence graph."""
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("pandas is required for build_graph_nodes_edges") from exc

    if df is None or df.empty:
        return pd.DataFrame(columns=["id", "label", "group", "count"]), pd.DataFrame(columns=["src", "dst", "weight"])

    dfo = df.sort_values("ts").reset_index(drop=True)
    node_counts = dfo.groupby([node_id_col, group_col]).size().reset_index(name="count")
    node_counts["id"] = node_counts[node_id_col].astype(str) + "::" + node_counts[group_col].astype(str)
    node_counts["label"] = node_counts[node_id_col].astype(str)
    node_counts["group"] = node_counts[group_col].astype(str)

    nodes_df = node_counts[["id", "label", "group", "count"]]

    src = dfo[node_id_col].astype(str) + "::" + dfo[group_col].astype(str)
    dst = src.shift(-1)
    edges = pd.DataFrame({"src": src[:-1], "dst": dst[:-1]})
    edges["weight"] = 1
    edges_df = edges.groupby(["src", "dst"], as_index=False)["weight"].sum()
    return nodes_df, edges_df


def build_latency_time_series(df, *, bucket: str = "1min"):
    """Aggregate latency into a time series suitable for dashboards."""
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("pandas is required for build_latency_time_series") from exc

    if df is None or df.empty:
        return pd.DataFrame(columns=["time", "latency_ms_p50", "latency_ms_p95", "count"])
    dfo = df.copy()
    dfo["time"] = pd.to_datetime(dfo["ts"], unit="s")
    g = dfo.set_index("time").resample(bucket)["latency_ms"]
    out = pd.DataFrame({
        "latency_ms_p50": g.quantile(0.50),
        "latency_ms_p95": g.quantile(0.95),
        "count": g.count(),
    }).reset_index()
    return out


def _attrs_map(span: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize span attributes for multiple exporters."""
    if "attributes" in span and isinstance(span["attributes"], dict):
        return span["attributes"]
    attrs = span.get("attributes")
    if isinstance(attrs, list):
        out: Dict[str, Any] = {}
        for a in attrs:
            k = a.get("key")
            v = a.get("value")
            out[k] = _unwrap_otlp_value(v)
        return out
    return {}


def _unwrap_otlp_value(v: Any) -> Any:
    if not isinstance(v, dict):
        return v
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in v:
            return v[key]
    if "arrayValue" in v and isinstance(v["arrayValue"], dict):
        vals = v["arrayValue"].get("values") or []
        return [_unwrap_otlp_value(x) for x in vals]
    return v


def _duration_ms(span: Dict[str, Any]) -> float:
    try:
        if "start_time_unix_nano" in span and "end_time_unix_nano" in span:
            return (float(span["end_time_unix_nano"]) - float(span["start_time_unix_nano"])) / 1e6
        if "duration_ms" in span:
            return float(span["duration_ms"])
    except Exception:
        pass
    return 0.0


def _int_or_none(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except Exception:
        return None


def _float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _bool_or_none(v: Any) -> Optional[bool]:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


__all__ = [
    "GraphistryBuilders",
    "InferenceRecord",
    "records_to_dataframe",
    "traces_to_records",
    "build_graph_nodes_edges",
    "build_latency_time_series",
]
