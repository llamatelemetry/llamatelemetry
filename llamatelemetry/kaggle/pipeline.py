"""llamatelemetry.kaggle.pipeline

End-to-end helpers for Kaggle notebooks to reduce repeated glue code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import os

from .secrets import KaggleSecrets
from .presets import ServerPreset, get_preset_config
from ..server import ServerManager
from ..telemetry import setup_telemetry, get_metrics_collector, setup_otlp_env_from_kaggle_secrets
from ..telemetry.client import InstrumentedLlamaCppClient


@dataclass
class KagglePipelineConfig:
    service_name: str = "llamatelemetry"
    service_version: str = "0.1.0"
    otlp_endpoint: Optional[str] = None
    enable_graphistry: bool = False
    graphistry_server: Optional[str] = None
    enable_llama_metrics: bool = True
    llama_metrics_interval: float = 5.0


def load_grafana_otlp_env_from_kaggle(
    *,
    endpoint_secret: str = "GRAFANA_OTLP_ENDPOINT",
    headers_secret: str = "GRAFANA_OTLP_HEADERS",
    token_secret: str = "GRAFANA_OTLP_TOKEN",
) -> Dict[str, str]:
    """Load OTLP endpoint + headers from Kaggle secrets into OTEL env vars.

    Falls back to OTLP_ENDPOINT / OTLP_TOKEN if Grafana-specific secrets are not set.
    """
    secrets = KaggleSecrets()
    endpoint = secrets.get(endpoint_secret) or ""
    headers = secrets.get(headers_secret) or ""
    token = secrets.get(token_secret) or ""

    if not headers and token:
        headers = f"Authorization=Bearer {token}"

    if endpoint:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    if headers:
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = headers

    if not endpoint:
        # Fall back to standard OTLP secrets
        fallback = setup_otlp_env_from_kaggle_secrets()
        endpoint = fallback.get("endpoint") or ""
        headers = fallback.get("headers") or headers

    return {"endpoint": endpoint, "headers": headers}


def start_server_from_preset(
    model_path: str,
    preset: ServerPreset,
    *,
    extra_args: Optional[Dict[str, Any]] = None,
) -> ServerManager:
    """Start llama-server using a preset config and wait for readiness."""
    cfg = get_preset_config(preset)
    mgr = ServerManager()
    kwargs: Dict[str, Any] = cfg.to_server_kwargs()
    kwargs["model_path"] = model_path
    if extra_args:
        kwargs.update(extra_args)
    mgr.start_server(**kwargs)
    mgr.wait_ready(timeout=180)
    return mgr


def setup_otel_and_client(
    base_url: str,
    cfg: KagglePipelineConfig,
) -> Dict[str, Any]:
    """One-stop telemetry + instrumented client setup."""
    tracer, meter = setup_telemetry(
        service_name=cfg.service_name,
        service_version=cfg.service_version,
        otlp_endpoint=cfg.otlp_endpoint,
        enable_graphistry=cfg.enable_graphistry,
        graphistry_server=cfg.graphistry_server,
        llama_server_url=base_url,
        llama_server_path=os.environ.get("LLAMA_SERVER_PATH"),
        enable_llama_metrics=cfg.enable_llama_metrics,
        llama_metrics_interval=cfg.llama_metrics_interval,
    )

    gpu_metrics = get_metrics_collector()
    client = InstrumentedLlamaCppClient(
        base_url=base_url,
        gpu_metrics=gpu_metrics,
    )

    return {"tracer": tracer, "meter": meter, "client": client, "gpu_metrics": gpu_metrics}


__all__ = [
    "KagglePipelineConfig",
    "load_grafana_otlp_env_from_kaggle",
    "start_server_from_preset",
    "setup_otel_and_client",
]
