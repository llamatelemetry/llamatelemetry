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
from typing import Any, Optional, Dict

from .semconv import metric_name, attr_name


class GpuMetricsCollector:
    """
    Background thread that polls nvidia-smi and exports GPU metrics via OTel.

    Instruments are created on the provided MeterProvider and updated
    every `poll_interval` seconds.
    """

    def __init__(
        self,
        meter_provider: Any,
        poll_interval: float = 5.0,
        server_url: Optional[str] = None,
        llama_cpp_poll_interval: float = 5.0,
    ):
        """
        Args:
            meter_provider: OTel MeterProvider
            poll_interval: Seconds between nvidia-smi polls (default: 5)
            server_url: Optional llama-server URL for /metrics observability
            llama_cpp_poll_interval: Cache interval for llama.cpp metrics fetch
        """
        self._meter_provider = meter_provider
        self._poll_interval = poll_interval
        self._server_url = server_url
        self._llama_cpp_poll_interval = max(0.5, float(llama_cpp_poll_interval))
        self._llama_cpp_last_fetch = 0.0
        self._llama_cpp_cache: Dict[str, float] = {}
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

            # GenAI semantic convention metrics (mirrors)
            self._gen_ai_request_duration = meter.create_histogram(
                name=metric_name("server_request_duration"),
                description="GenAI server request duration",
                unit="ms",
            )
            self._gen_ai_time_to_first_token = meter.create_histogram(
                name=metric_name("server_time_to_first_token"),
                description="GenAI server time to first token",
                unit="ms",
            )
            self._gen_ai_time_per_output_token = meter.create_histogram(
                name=metric_name("server_time_per_output_token"),
                description="GenAI server time per output token",
                unit="ms",
            )
            self._gen_ai_token_usage = meter.create_histogram(
                name=metric_name("client_token_usage"),
                description="GenAI token usage",
                unit="tokens",
            )

            # NCCL bytes transferred
            self._nccl_bytes_counter = meter.create_counter(
                name="llamatelemetry.nccl.bytes_transferred",
                description="Total NCCL bytes transferred",
                unit="By",
            )

            # Optional llama.cpp /metrics observability
            if self._server_url:
                self._llama_cpp_requests_processing = meter.create_observable_gauge(
                    name="llamatelemetry.llama_cpp.requests_processing",
                    description="llama.cpp requests currently processing",
                    unit="1",
                    callbacks=[self._observe_llama_cpp_requests_processing],
                )
                self._llama_cpp_requests_deferred = meter.create_observable_gauge(
                    name="llamatelemetry.llama_cpp.requests_deferred",
                    description="llama.cpp requests deferred",
                    unit="1",
                    callbacks=[self._observe_llama_cpp_requests_deferred],
                )
                self._llama_cpp_prompt_tokens_total = meter.create_observable_gauge(
                    name="llamatelemetry.llama_cpp.prompt_tokens_total",
                    description="llama.cpp prompt tokens total",
                    unit="tokens",
                    callbacks=[self._observe_llama_cpp_prompt_tokens_total],
                )
                self._llama_cpp_predicted_tokens_total = meter.create_observable_gauge(
                    name="llamatelemetry.llama_cpp.predicted_tokens_total",
                    description="llama.cpp predicted tokens total",
                    unit="tokens",
                    callbacks=[self._observe_llama_cpp_predicted_tokens_total],
                )
                self._llama_cpp_prompt_tokens_per_sec = meter.create_observable_gauge(
                    name="llamatelemetry.llama_cpp.prompt_tokens_per_sec",
                    description="llama.cpp prompt throughput tokens/sec",
                    unit="tokens/s",
                    callbacks=[self._observe_llama_cpp_prompt_tokens_per_sec],
                )
                self._llama_cpp_predicted_tokens_per_sec = meter.create_observable_gauge(
                    name="llamatelemetry.llama_cpp.predicted_tokens_per_sec",
                    description="llama.cpp predicted throughput tokens/sec",
                    unit="tokens/s",
                    callbacks=[self._observe_llama_cpp_predicted_tokens_per_sec],
                )
                self._llama_cpp_kv_cache_usage_ratio = meter.create_observable_gauge(
                    name="llamatelemetry.llama_cpp.kv_cache_usage_ratio",
                    description="llama.cpp KV-cache usage ratio",
                    unit="1",
                    callbacks=[self._observe_llama_cpp_kv_cache_usage_ratio],
                )
                self._llama_cpp_kv_cache_tokens = meter.create_observable_gauge(
                    name="llamatelemetry.llama_cpp.kv_cache_tokens",
                    description="llama.cpp KV-cache tokens",
                    unit="tokens",
                    callbacks=[self._observe_llama_cpp_kv_cache_tokens],
                )
                self._llama_cpp_n_tokens_max = meter.create_observable_gauge(
                    name="llamatelemetry.llama_cpp.n_tokens_max",
                    description="llama.cpp max observed tokens",
                    unit="tokens",
                    callbacks=[self._observe_llama_cpp_n_tokens_max],
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

    def record_inference(self, latency_ms: float, tokens: int, model: str = "", ttft_ms: Optional[float] = None) -> None:
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

        gen_ai_attrs: Dict[str, Any] = {}
        if model:
            gen_ai_attrs[attr_name("request_model")] = model
            gen_ai_attrs[attr_name("response_model")] = model
        gen_ai_attrs[attr_name("system")] = "llama.cpp"

        try:
            self._gen_ai_request_duration.record(latency_ms, gen_ai_attrs)
            if tokens > 0 and latency_ms > 0:
                self._gen_ai_time_per_output_token.record(latency_ms / tokens, gen_ai_attrs)
            if ttft_ms is not None:
                self._gen_ai_time_to_first_token.record(ttft_ms, gen_ai_attrs)
            self._gen_ai_token_usage.record(tokens, gen_ai_attrs)
        except Exception:
            pass

    def record_nccl_transfer(self, bytes_transferred: int) -> None:
        """Record NCCL data transfer volume."""
        self._nccl_bytes += bytes_transferred
        try:
            if self._instruments_ready:
                self._nccl_bytes_counter.add(int(bytes_transferred))
        except Exception:
            pass

    def _fetch_llama_cpp_metrics(self) -> Dict[str, float]:
        if not self._server_url:
            return {}
        now = time.time()
        if now - self._llama_cpp_last_fetch < self._llama_cpp_poll_interval:
            return self._llama_cpp_cache

        try:
            import requests
            response = requests.get(f"{self._server_url}/metrics", timeout=3.0)
            response.raise_for_status()
            from .monitor import parse_llama_cpp_metrics
            parsed = parse_llama_cpp_metrics(response.text)
            normalized = parsed.get("normalized", {})
            if isinstance(normalized, dict):
                self._llama_cpp_cache = {k: float(v) for k, v in normalized.items()}
        except Exception:
            pass

        self._llama_cpp_last_fetch = now
        return self._llama_cpp_cache

    def _observe_llama_cpp_metric(self, key: str):
        try:
            from opentelemetry.metrics import Observation
            metrics = self._fetch_llama_cpp_metrics()
            value = metrics.get(key, 0.0)
            return [Observation(value)]
        except Exception:
            return []

    def _observe_llama_cpp_requests_processing(self, options=None):
        return self._observe_llama_cpp_metric("requests_processing")

    def _observe_llama_cpp_requests_deferred(self, options=None):
        return self._observe_llama_cpp_metric("requests_deferred")

    def _observe_llama_cpp_prompt_tokens_total(self, options=None):
        return self._observe_llama_cpp_metric("tokens_prompt")

    def _observe_llama_cpp_predicted_tokens_total(self, options=None):
        return self._observe_llama_cpp_metric("tokens_predicted")

    def _observe_llama_cpp_prompt_tokens_per_sec(self, options=None):
        return self._observe_llama_cpp_metric("prompt_tokens_per_sec")

    def _observe_llama_cpp_predicted_tokens_per_sec(self, options=None):
        return self._observe_llama_cpp_metric("predicted_tokens_per_sec")

    def _observe_llama_cpp_kv_cache_usage_ratio(self, options=None):
        return self._observe_llama_cpp_metric("kv_cache_usage_ratio")

    def _observe_llama_cpp_kv_cache_tokens(self, options=None):
        return self._observe_llama_cpp_metric("kv_cache_tokens")

    def _observe_llama_cpp_n_tokens_max(self, options=None):
        return self._observe_llama_cpp_metric("n_tokens_max")

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
