# Integration Guide

This guide summarizes common integrations for `llamatelemetry`.

## OpenTelemetry

```python
from llamatelemetry.telemetry import setup_telemetry

tracer, meter = setup_telemetry(
    service_name="llamatelemetry-demo",
    otlp_endpoint="http://localhost:4317",
    enable_llama_metrics=True,
)
```

## Graphistry

```python
from llamatelemetry.graphistry.connector import GraphistryConnector

connector = GraphistryConnector()
connector.login()
```

## Kaggle

```python
from llamatelemetry.api import kaggle_t4_dual_config

cfg = kaggle_t4_dual_config()
print(cfg)
```

## Unsloth

```python
from llamatelemetry.unsloth import UnslothExporter

exporter = UnslothExporter()
```
