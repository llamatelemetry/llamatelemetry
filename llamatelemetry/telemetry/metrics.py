"""
llamatelemetry.telemetry.metrics - GPU Metrics Collector

Continuously collects and exports fine-grained GPU metrics via OTel instruments:

Instruments:
    llamatelemetry.gpu.memory.used          (Gauge, bytes)
    llamatelemetry.gpu.memory.total         (Gauge, bytes)
    llamatelemetry.gpu.utilization          (Gauge, percent)
    llamatelemetry.gpu.temperature          (Gauge, celsius)
    llamatelemetry.llm.inference.latency    (Histogram, ms)
    llamatelemetry.llm.inference.tokens     (Counter, tokens)
    llamatelemetry.llm.inference.requests   (Counter, requests)
    llamatelemetry.nccl.bytes_transferred   (Counter, bytes)
"""

import subprocess
import threading
import time
from typing import Any, Optional


class GpuMetricsCollector:
    """
    Background thread that polls nvidia-smi and exports GPU metrics via OTel.

    Instruments are created on the provided MeterProvider and updated
    every `poll_interval` seconds.
    """

    def __init__(self, meter_provider: Any, poll_interval: float = 5.0):
        """
        Args:
            meter_provider: OTel MeterProvider
            poll_interval: Seconds between nvidia-smi polls (default: 5)
        """
        self._meter_provider = meter_provider
        self._poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Cumulative counters (updated by inference engine)
        self._total_tokens = 0
        self._total_requests = 0
        self._nccl_bytes = 0

        self._setup_instruments()

    def _setup_instruments(self) -> None:
        """Create OTel meter instruments."""
        try:
            meter = self._meter_provider.get_meter("llamatelemetry.gpu")

            # Gauges (observed via callback)
            self._gpu_memory_used = meter.create_observable_gauge(
                name="llamatelemetry.gpu.memory.used",
                description="GPU memory currently used",
                unit="By",
                callbacks=[self._observe_gpu_memory_used],
            )
            self._gpu_memory_total = meter.create_observable_gauge(
                name="llamatelemetry.gpu.memory.total",
                description="GPU total memory",
                unit="By",
                callbacks=[self._observe_gpu_memory_total],
            )
            self._gpu_utilization = meter.create_observable_gauge(
                name="llamatelemetry.gpu.utilization",
                description="GPU utilization percentage",
                unit="%",
                callbacks=[self._observe_gpu_utilization],
            )

            # Counters (incremented by inference engine)
            self._inference_tokens = meter.create_counter(
                name="llamatelemetry.llm.inference.tokens",
                description="Total tokens generated",
                unit="tokens",
            )
            self._inference_requests = meter.create_counter(
                name="llamatelemetry.llm.inference.requests",
                description="Total inference requests",
                unit="requests",
            )

            # Histogram
            self._inference_latency = meter.create_histogram(
                name="llamatelemetry.llm.inference.latency",
                description="Inference request latency",
                unit="ms",
            )

            self._instruments_ready = True
        except Exception:
            self._instruments_ready = False

    def _query_nvidia_smi(self) -> list:
        """Query nvidia-smi for memory and utilization."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append({
                            "memory_used_mib": int(parts[0]),
                            "memory_total_mib": int(parts[1]),
                            "utilization_pct": int(parts[2]),
                        })
                return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return []

    def _observe_gpu_memory_used(self, options=None):
        """OTel observable callback for GPU memory used."""
        try:
            from opentelemetry.metrics import Observation
            gpus = self._query_nvidia_smi()
            observations = []
            for i, gpu in enumerate(gpus):
                observations.append(
                    Observation(gpu["memory_used_mib"] * 1024 * 1024, {"gpu.id": str(i)})
                )
            return observations
        except Exception:
            return []

    def _observe_gpu_memory_total(self, options=None):
        """OTel observable callback for GPU total memory."""
        try:
            from opentelemetry.metrics import Observation
            gpus = self._query_nvidia_smi()
            observations = []
            for i, gpu in enumerate(gpus):
                observations.append(
                    Observation(gpu["memory_total_mib"] * 1024 * 1024, {"gpu.id": str(i)})
                )
            return observations
        except Exception:
            return []

    def _observe_gpu_utilization(self, options=None):
        """OTel observable callback for GPU utilization."""
        try:
            from opentelemetry.metrics import Observation
            gpus = self._query_nvidia_smi()
            observations = []
            for i, gpu in enumerate(gpus):
                observations.append(
                    Observation(gpu["utilization_pct"], {"gpu.id": str(i)})
                )
            return observations
        except Exception:
            return []

    def record_inference(self, latency_ms: float, tokens: int, model: str = "") -> None:
        """
        Record an inference event. Called by InferenceEngine after each request.

        Args:
            latency_ms: Request latency in milliseconds
            tokens: Tokens generated
            model: Model name for attributes
        """
        if not self._instruments_ready:
            return

        attrs = {"llm.model": model} if model else {}

        self._inference_latency.record(latency_ms, attrs)
        self._inference_tokens.add(tokens, attrs)
        self._inference_requests.add(1, attrs)

        self._total_tokens += tokens
        self._total_requests += 1

    def record_nccl_transfer(self, bytes_transferred: int) -> None:
        """Record NCCL data transfer volume."""
        self._nccl_bytes += bytes_transferred

    def start(self) -> None:
        """Start background GPU polling (no-op if already running)."""
        if self._running:
            return
        self._running = True
        # OTel observable gauges use callbacks — no background thread needed
        # for gauge-based metrics. Start is a no-op for current architecture.

    def stop(self) -> None:
        """Stop background polling."""
        self._running = False

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_requests(self) -> int:
        return self._total_requests
