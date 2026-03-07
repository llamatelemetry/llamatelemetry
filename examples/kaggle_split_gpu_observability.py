"""
Kaggle split-GPU quickstart with observability.

Targets dual T4: GPU 0 for llama-server, GPU 1 for RAPIDS/Graphistry.
"""

from llamatelemetry.kaggle import KaggleEnvironment, split_gpu_session
from llamatelemetry.telemetry import setup_grafana_otlp, InstrumentedLLMClient


def main():
    env = KaggleEnvironment.setup(split_gpu_mode=True, enable_graphistry=False)

    # OTLP endpoint and token are loaded from Kaggle secrets if present
    setup_grafana_otlp(service_name="llama-gguf")

    with split_gpu_session(llm_gpu=0, graph_gpu=1) as ctx:
        engine = env.create_engine("gemma-3-1b-Q4_K_M", **ctx["llm_server_kwargs"])
        result = engine.generate("Hello from split GPUs", max_tokens=64)
        print(result.text)

    # Instrumented client for llama-server OpenAI-compatible endpoints
    client = InstrumentedLLMClient("http://127.0.0.1:8080")
    resp = client.chat_completion(
        messages=[{"role": "user", "content": "Explain GGUF in one paragraph."}],
        max_tokens=128,
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
