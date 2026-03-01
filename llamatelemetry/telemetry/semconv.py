"""
OpenTelemetry GenAI semantic convention helpers.

Provides safe access to gen_ai.* semconv constants with fallbacks.
"""

from typing import Any, Optional

try:
    from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
    from opentelemetry.semconv._incubating.metrics import gen_ai_metrics as GenAIMetrics
    _HAS_GEN_AI = True
except Exception:
    GenAIAttributes = None
    GenAIMetrics = None
    _HAS_GEN_AI = False


GEN_AI_ATTR = {
    "operation_name": "gen_ai.operation.name",
    "system": "gen_ai.system",
    "provider_name": "gen_ai.provider.name",
    "request_model": "gen_ai.request.model",
    "response_model": "gen_ai.response.model",
    "response_id": "gen_ai.response.id",
    "request_max_tokens": "gen_ai.request.max_tokens",
    "request_temperature": "gen_ai.request.temperature",
    "request_top_p": "gen_ai.request.top_p",
    "request_top_k": "gen_ai.request.top_k",
    "request_seed": "gen_ai.request.seed",
    "request_stop_sequences": "gen_ai.request.stop_sequences",
    "request_presence_penalty": "gen_ai.request.presence_penalty",
    "request_frequency_penalty": "gen_ai.request.frequency_penalty",
    "request_choice_count": "gen_ai.request.choice.count",
    "output_type": "gen_ai.output.type",
    "prompt": "gen_ai.prompt",
    "input_messages": "gen_ai.input.messages",
    "response_finish_reasons": "gen_ai.response.finish_reasons",
    "usage_input_tokens": "gen_ai.usage.input_tokens",
    "usage_output_tokens": "gen_ai.usage.output_tokens",
    "usage_total_tokens": "gen_ai.usage.total_tokens",
    "token_type": "gen_ai.token.type",
    "embeddings_dimension_count": "gen_ai.embeddings.dimension.count",
}


GEN_AI_METRIC = {
    "server_request_duration": "gen_ai.server.request.duration",
    "server_time_to_first_token": "gen_ai.server.time_to_first_token",
    "server_time_per_output_token": "gen_ai.server.time_per_output_token",
    "client_operation_duration": "gen_ai.client.operation.duration",
    "client_token_usage": "gen_ai.client.token.usage",
}


def _attr_key(name: str) -> str:
    if GenAIAttributes is None:
        return GEN_AI_ATTR[name]
    return getattr(GenAIAttributes, f"GEN_AI_{name.upper()}", GEN_AI_ATTR[name])


def _metric_key(name: str) -> str:
    if GenAIMetrics is None:
        return GEN_AI_METRIC[name]
    return getattr(GenAIMetrics, f"GEN_AI_{name.upper()}", GEN_AI_METRIC[name])


def set_gen_ai_attr(span: Any, name: str, value: Any) -> None:
    if span is None:
        return
    if value is None:
        return
    try:
        span.set_attribute(_attr_key(name), value)
    except Exception:
        pass


def set_gen_ai_provider(span: Any, provider: Optional[str] = None) -> None:
    provider = provider or "llama.cpp"
    set_gen_ai_attr(span, "provider_name", provider)
    # Deprecated but still used in some instrumentations
    set_gen_ai_attr(span, "system", provider)


def metric_name(name: str) -> str:
    return _metric_key(name)


def attr_name(name: str) -> str:
    return _attr_key(name)


def has_gen_ai_semconv() -> bool:
    return _HAS_GEN_AI
