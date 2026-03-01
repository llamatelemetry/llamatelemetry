"""
llamatelemetry.telemetry.monitor - Performance monitor for aggregating inference metrics.

Provides PerformanceMonitor for tracking latency, throughput, and GPU metrics
across multiple inference requests.

Example:
    >>> from llamatelemetry.telemetry import PerformanceMonitor
    >>>
    >>> monitor = PerformanceMonitor()
    >>> monitor.start()
    >>>
    >>> for prompt in prompts:
    ...     result = engine.infer(prompt)
    ...     monitor.record(result)
    >>>
    >>> summary = monitor.get_summary()
    >>> print(f"P95 latency: {summary.latency_p95_ms:.1f}ms")
    >>> print(f"Throughput: {summary.tokens_per_sec:.1f} tok/s")
    >>>
    >>> monitor.stop()
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import threading
from collections import deque


@dataclass
class PerformanceSnapshot:
    """
    Point-in-time performance metrics.

    Attributes:
        timestamp: Unix timestamp of snapshot
        requests_total: Total requests processed
        tokens_total: Total tokens generated
        latency_p50_ms: 50th percentile latency
        latency_p95_ms: 95th percentile latency
        latency_p99_ms: 99th percentile latency
        latency_mean_ms: Mean latency
        latency_min_ms: Minimum latency
        latency_max_ms: Maximum latency
        tokens_per_sec: Generation throughput
        requests_per_sec: Request throughput
        gpu_memory_used_mb: GPU memory usage
        gpu_utilization_pct: GPU utilization percentage
    """
    timestamp: float
    requests_total: int
    tokens_total: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    tokens_per_sec: float = 0.0
    requests_per_sec: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_utilization_pct: float = 0.0


@dataclass
class InferenceRecord:
    """Single inference record for tracking."""
    timestamp: float
    latency_ms: float
    tokens_generated: int
    success: bool = True
    model: str = ""


def parse_llama_cpp_metrics(text: str) -> Dict[str, Any]:
    """
    Parse llama.cpp Prometheus metrics into a structured summary.

    Returns:
        Dict with normalized keys and raw aggregates.
    """
    import re

    pattern = re.compile(
        r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)$"
    )
    agg: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        name = match.group(1)
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        if name.startswith("llama_cpp_") or name.startswith("llamacpp:"):
            agg[name] = agg.get(name, 0.0) + value

    mapping = {
        # Older llama.cpp metric names (underscored)
        "llama_cpp_tokens_total": "tokens_total",
        "llama_cpp_tokens_prompt": "tokens_prompt",
        "llama_cpp_tokens_predicted": "tokens_predicted",
        "llama_cpp_requests_total": "requests_total",
        "llama_cpp_requests_processing": "requests_processing",
        "llama_cpp_requests_done": "requests_done",
        "llama_cpp_requests_err": "requests_error",
        "llama_cpp_requests_error": "requests_error",
        "llama_cpp_slots_total": "slots_total",
        "llama_cpp_slots_idle": "slots_idle",
        "llama_cpp_slots_processing": "slots_processing",
        "llama_cpp_kv_cache_used": "kv_cache_used",
        "llama_cpp_kv_cache_used_bytes": "kv_cache_used_bytes",
        "llama_cpp_cache_used": "cache_used",
        "llama_cpp_cache_used_bytes": "cache_used_bytes",
        "llama_cpp_embeddings_total": "embeddings_total",
        "llama_cpp_eval_count": "eval_count",
        "llama_cpp_eval_duration": "eval_duration",
        "llama_cpp_prompt_eval_count": "prompt_eval_count",
        "llama_cpp_prompt_eval_duration": "prompt_eval_duration",
        # Newer llama.cpp metric names (namespaced)
        "llamacpp:prompt_tokens_total": "tokens_prompt",
        "llamacpp:tokens_predicted_total": "tokens_predicted",
        "llamacpp:prompt_tokens_seconds": "prompt_tokens_per_sec",
        "llamacpp:predicted_tokens_seconds": "predicted_tokens_per_sec",
        "llamacpp:kv_cache_usage_ratio": "kv_cache_usage_ratio",
        "llamacpp:kv_cache_tokens": "kv_cache_tokens",
        "llamacpp:requests_processing": "requests_processing",
        "llamacpp:requests_deferred": "requests_deferred",
        "llamacpp:n_tokens_max": "n_tokens_max",
    }

    normalized: Dict[str, float] = {}
    for key, norm in mapping.items():
        if key in agg:
            normalized[norm] = agg[key]

    return {
        "normalized": normalized,
        "raw_aggregate": agg,
    }


class PerformanceMonitor:
    """
    Aggregate and monitor inference performance metrics.

    Tracks latency percentiles, throughput, and optionally GPU metrics
    across multiple inference requests. Supports real-time monitoring
    with background snapshot collection.

    Example:
        >>> monitor = PerformanceMonitor()
        >>> monitor.start()
        >>>
        >>> for _ in range(100):
        ...     result = engine.infer("Hello")
        ...     monitor.record(result)
        >>>
        >>> summary = monitor.get_summary()
        >>> print(f"P95 latency: {summary.latency_p95_ms:.1f}ms")
        >>>
        >>> monitor.stop()
    """

    def __init__(
        self,
        window_size: int = 1000,
        snapshot_interval: float = 5.0,
        collect_gpu_metrics: bool = True
    ):
        """
        Initialize performance monitor.

        Args:
            window_size: Number of recent results to keep for percentile calculation
            snapshot_interval: Seconds between automatic snapshots
            collect_gpu_metrics: Whether to collect GPU memory/utilization metrics
        """
        self.window_size = window_size
        self.snapshot_interval = snapshot_interval
        self.collect_gpu_metrics = collect_gpu_metrics

        self._records: deque = deque(maxlen=window_size)
        self._snapshots: List[PerformanceSnapshot] = []
        self._last_server_metrics: Dict[str, float] = {}
        self._last_llama_cpp_metrics: Dict[str, Any] = {}

        self._total_requests = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0
        self._start_time: Optional[float] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start monitoring with background snapshot collection."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()

        if self.snapshot_interval > 0:
            self._thread = threading.Thread(target=self._snapshot_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop monitoring and background threads."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def record(self, result: Any, model: str = "") -> None:
        """
        Record an inference result.

        Args:
            result: InferResult object or dict with latency_ms and tokens_generated
            model: Model name (optional)
        """
        # Extract metrics from result
        latency_ms = 0.0
        tokens = 0
        success = True

        if hasattr(result, "latency_ms"):
            latency_ms = result.latency_ms
        elif isinstance(result, dict) and "latency_ms" in result:
            latency_ms = result["latency_ms"]

        if hasattr(result, "tokens_generated"):
            tokens = result.tokens_generated
        elif isinstance(result, dict) and "tokens_generated" in result:
            tokens = result["tokens_generated"]
        elif isinstance(result, dict) and "tokens_predicted" in result:
            tokens = result["tokens_predicted"]

        if hasattr(result, "success"):
            success = result.success
        elif isinstance(result, dict) and "success" in result:
            success = result["success"]

        record = InferenceRecord(
            timestamp=time.time(),
            latency_ms=latency_ms,
            tokens_generated=tokens,
            success=success,
            model=model
        )

        with self._lock:
            self._records.append(record)
            self._total_requests += 1
            self._total_tokens += tokens
            self._total_latency_ms += latency_ms

    def record_manual(
        self,
        latency_ms: float,
        tokens_generated: int,
        success: bool = True,
        model: str = ""
    ) -> None:
        """
        Manually record inference metrics.

        Args:
            latency_ms: Inference latency in milliseconds
            tokens_generated: Number of tokens generated
            success: Whether inference was successful
            model: Model name
        """
        record = InferenceRecord(
            timestamp=time.time(),
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
            success=success,
            model=model
        )

        with self._lock:
            self._records.append(record)
            self._total_requests += 1
            self._total_tokens += tokens_generated
            self._total_latency_ms += latency_ms

    def get_summary(self) -> PerformanceSnapshot:
        """
        Get current performance summary.

        Returns:
            PerformanceSnapshot with current metrics
        """
        with self._lock:
            return self._calculate_snapshot()

    def get_snapshots(self) -> List[PerformanceSnapshot]:
        """
        Get all collected snapshots.

        Returns:
            List of PerformanceSnapshot objects
        """
        return self._snapshots.copy()

    def snapshots_to_dataframe(self):
        """
        Convert snapshots to pandas DataFrame.
        """
        try:
            import pandas as pd
        except Exception as exc:
            raise ImportError("pandas is required for snapshots_to_dataframe") from exc
        return pd.DataFrame([s.__dict__ for s in self._snapshots])

    def snapshot_to_dataframe(self):
        """
        Backward-compatible alias for snapshots_to_dataframe().
        """
        return self.snapshots_to_dataframe()

    def records_to_dataframe(self):
        """
        Convert inference records to pandas DataFrame.
        """
        try:
            import pandas as pd
        except Exception as exc:
            raise ImportError("pandas is required for records_to_dataframe") from exc
        return pd.DataFrame([r.__dict__ for r in self._records])

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self._records.clear()
            self._snapshots.clear()
            self._last_server_metrics = {}
            self._last_llama_cpp_metrics = {}
            self._total_requests = 0
            self._total_tokens = 0
            self._total_latency_ms = 0.0
            self._start_time = time.time()

    def _parse_prometheus_metrics(self, text: str) -> Dict[str, float]:
        """
        Parse Prometheus metrics text into a flat dict.
        """
        metrics: Dict[str, float] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            try:
                value = float(parts[1])
            except ValueError:
                continue
            metrics[key] = value
        return metrics

    def parse_llama_cpp_metrics(self, text: str) -> Dict[str, Any]:
        """
        Parse llama.cpp Prometheus metrics into a structured summary.

        Returns:
            Dict with normalized keys and raw aggregates.
        """
        return parse_llama_cpp_metrics(text)

    def record_metrics_from_llama_server(
        self,
        server_url: str = "http://127.0.0.1:8090",
        metrics_text: Optional[str] = None,
        timeout: float = 5.0,
    ) -> Dict[str, float]:
        """
        Fetch and parse llama.cpp /metrics for observability.

        Args:
            server_url: Base URL of llama-server
            metrics_text: Optional metrics text (skips HTTP fetch)
            timeout: Request timeout

        Returns:
            Dict of parsed Prometheus metrics
        """
        if metrics_text is None:
            try:
                import requests
                response = requests.get(f"{server_url}/metrics", timeout=timeout)
                response.raise_for_status()
                metrics_text = response.text
            except Exception:
                return {}

        metrics = self._parse_prometheus_metrics(metrics_text)
        self._last_server_metrics = metrics
        self._last_llama_cpp_metrics = self.parse_llama_cpp_metrics(metrics_text)
        return metrics

    def get_last_server_metrics(self) -> Dict[str, float]:
        """Return last parsed /metrics snapshot."""
        return self._last_server_metrics.copy()

    def get_last_llama_cpp_metrics(self) -> Dict[str, Any]:
        """Return last structured llama.cpp metrics snapshot."""
        return self._last_llama_cpp_metrics.copy()

    def _calculate_snapshot(self) -> PerformanceSnapshot:
        """Calculate current snapshot (called with lock held)."""
        now = time.time()
        elapsed = now - (self._start_time or now)

        # Get latencies from records
        latencies = sorted(r.latency_ms for r in self._records) if self._records else [0.0]

        # Calculate percentiles
        p50 = self._percentile(latencies, 50)
        p95 = self._percentile(latencies, 95)
        p99 = self._percentile(latencies, 99)

        # Calculate throughput
        tokens_per_sec = self._total_tokens / elapsed if elapsed > 0 else 0.0
        requests_per_sec = self._total_requests / elapsed if elapsed > 0 else 0.0

        # Get GPU metrics
        gpu_memory = 0.0
        gpu_util = 0.0
        if self.collect_gpu_metrics:
            gpu_memory = self._get_gpu_memory()
            gpu_util = self._get_gpu_utilization()

        return PerformanceSnapshot(
            timestamp=now,
            requests_total=self._total_requests,
            tokens_total=self._total_tokens,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            latency_mean_ms=self._total_latency_ms / self._total_requests if self._total_requests > 0 else 0.0,
            latency_min_ms=min(latencies) if latencies else 0.0,
            latency_max_ms=max(latencies) if latencies else 0.0,
            tokens_per_sec=tokens_per_sec,
            requests_per_sec=requests_per_sec,
            gpu_memory_used_mb=gpu_memory,
            gpu_utilization_pct=gpu_util,
        )

    def _percentile(self, data: List[float], p: int) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        idx = int(len(data) * p / 100)
        idx = min(idx, len(data) - 1)
        return data[idx]

    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            from ..api.multigpu import detect_gpus
            gpus = detect_gpus()
            if gpus:
                return gpus[0].memory_used / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        try:
            from ..api.multigpu import detect_gpus
            gpus = detect_gpus()
            if gpus:
                return float(gpus[0].utilization)
        except Exception:
            pass
        return 0.0

    def _snapshot_loop(self) -> None:
        """Background thread for periodic snapshots."""
        while self._running:
            time.sleep(self.snapshot_interval)
            if self._running:
                with self._lock:
                    snapshot = self._calculate_snapshot()
                    self._snapshots.append(snapshot)

    def print_summary(self) -> None:
        """Print formatted summary to stdout."""
        summary = self.get_summary()
        print("=" * 50)
        print("PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total Requests:     {summary.requests_total}")
        print(f"Total Tokens:       {summary.tokens_total}")
        print(f"Throughput:         {summary.tokens_per_sec:.1f} tok/s")
        print(f"Request Rate:       {summary.requests_per_sec:.2f} req/s")
        print("-" * 50)
        print("Latency (ms):")
        print(f"  Mean:             {summary.latency_mean_ms:.1f}")
        print(f"  P50:              {summary.latency_p50_ms:.1f}")
        print(f"  P95:              {summary.latency_p95_ms:.1f}")
        print(f"  P99:              {summary.latency_p99_ms:.1f}")
        print(f"  Min:              {summary.latency_min_ms:.1f}")
        print(f"  Max:              {summary.latency_max_ms:.1f}")
        if self.collect_gpu_metrics:
            print("-" * 50)
            print("GPU:")
            print(f"  Memory Used:      {summary.gpu_memory_used_mb:.1f} MB")
            print(f"  Utilization:      {summary.gpu_utilization_pct:.1f}%")
        print("=" * 50)

    def to_dict(self) -> Dict[str, Any]:
        """Convert current summary to dictionary."""
        summary = self.get_summary()
        return {
            "requests_total": summary.requests_total,
            "tokens_total": summary.tokens_total,
            "latency": {
                "mean_ms": summary.latency_mean_ms,
                "p50_ms": summary.latency_p50_ms,
                "p95_ms": summary.latency_p95_ms,
                "p99_ms": summary.latency_p99_ms,
                "min_ms": summary.latency_min_ms,
                "max_ms": summary.latency_max_ms,
            },
            "throughput": {
                "tokens_per_sec": summary.tokens_per_sec,
                "requests_per_sec": summary.requests_per_sec,
            },
            "gpu": {
                "memory_used_mb": summary.gpu_memory_used_mb,
                "utilization_pct": summary.gpu_utilization_pct,
            },
        }

    def __enter__(self) -> "PerformanceMonitor":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


__all__ = [
    "PerformanceSnapshot",
    "InferenceRecord",
    "PerformanceMonitor",
]
