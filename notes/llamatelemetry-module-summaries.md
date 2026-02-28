# llamatelemetry Module Summaries

Generated: 2026-02-28T10:08:14
Package root: /media/waqasm86/External1/Project-Nvidia-Office/Project-Llamatelemetry/llamatelemetry/llamatelemetry

## llamatelemetry/__init__.py
llamatelemetry - Clean, Ultra-Lightweight CUDA 12 LLM Inference for Python 3.11+
Top-level definitions:
- class InferenceEngine
- class InferResult
- func check_cuda_available
- func get_cuda_device_info
- func quick_infer

## llamatelemetry/_internal/__init__.py
(no module docstring)

## llamatelemetry/_internal/bootstrap.py
llamatelemetry v0.1.0 Bootstrap Module - Kaggle Dual T4 GPUs
Top-level definitions:
- func detect_gpu_compute_capability
- func detect_platform
- func verify_gpu_compatibility
- func download_file
- func extract_tarball
- func locate_bin_and_lib_dirs
- func verify_sha256
- func download_from_huggingface
- func download_t4_binaries
- func download_default_model
- func bootstrap

## llamatelemetry/_internal/registry.py
Model Registry - Curated list of GGUF models from HuggingFace
Top-level definitions:
- func list_registry_models
- func get_model_info
- func find_models_by_vram

## llamatelemetry/api/__init__.py
llamatelemetry.api - Comprehensive llama.cpp Server API Client Module

## llamatelemetry/api/client.py
llamatelemetry.api.client - Main LlamaCppClient class
Top-level definitions:
- class StopType
- class Message
- class Choice
- class Usage
- class Timings
- class CompletionResponse
- class EmbeddingData
- class EmbeddingsResponse
- class RerankResult
- class RerankResponse
- class TokenizeResponse
- class ModelInfo
- class SlotInfo
- class HealthStatus
- class LoraAdapter
- class LlamaCppClient
- class ChatCompletionsAPI
- class EmbeddingsClientAPI
- class ModelsClientAPI
- class SlotsClientAPI
- class LoraClientAPI

## llamatelemetry/api/gguf.py
llamatelemetry.api.gguf - GGUF Model Utilities
Top-level definitions:
- class GGUFValueType
- class GGMLType
- class QuantTypeInfo
- class GGUFMetadata
- class GGUFTensorInfo
- class GGUFModelInfo
- func read_string
- func read_value
- func parse_gguf_header
- func find_llama_tool
- func quantize
- func convert_hf_to_gguf
- func merge_lora
- func generate_imatrix
- func find_gguf_models
- func get_model_summary
- func compare_models
- func validate_gguf
- func get_recommended_quant
- func estimate_gguf_size
- func recommend_quant_for_kaggle
- func print_quant_guide

## llamatelemetry/api/multigpu.py
llamatelemetry.api.multigpu - Multi-GPU Configuration Module
Top-level definitions:
- class SplitMode
- class GPUInfo
- class MultiGPUConfig
- func run_nvidia_smi
- func detect_gpus
- func get_cuda_version
- func get_total_vram
- func get_free_vram
- func is_multi_gpu
- func gpu_count
- func kaggle_t4_dual_config
- func colab_t4_single_config
- func auto_config
- func estimate_model_vram
- func can_fit_model
- func recommend_quantization
- func set_cuda_visible_devices
- func get_cuda_visible_devices
- func print_gpu_info

## llamatelemetry/api/nccl.py
llamatelemetry.api.nccl - NVIDIA Collective Communications Library (NCCL) Integration
Top-level definitions:
- class NCCLResult
- class NCCLDataType
- class NCCLRedOp
- func _find_nccl_library
- func _load_nccl
- func is_nccl_available
- func get_nccl_version
- class NCCLConfig
- class NCCLInfo
- class NCCLCommunicator
- func get_nccl_info
- func setup_nccl_environment
- func kaggle_nccl_config
- func print_nccl_info
- func get_llama_cpp_nccl_args

## llamatelemetry/chat.py
llamatelemetry.chat - Chat and Conversation Management
Top-level definitions:
- class Message
- class ChatEngine
- class ConversationManager

## llamatelemetry/cuda/__init__.py
llamatelemetry CUDA Optimization APIs

## llamatelemetry/cuda/graphs.py
CUDA Graphs for Inference Optimization
Top-level definitions:
- class GraphCaptureConfig
- class CUDAGraph
- class GraphPool
- func capture_graph
- func replay_graph
- func enable_cuda_graphs

## llamatelemetry/cuda/tensor_core.py
Tensor Core Utilities
Top-level definitions:
- class TensorCoreConfig
- func check_tensor_core_support
- func enable_tensor_cores
- func matmul_tensor_core
- func enable_amp
- class TensorCoreMatMul
- func optimize_for_tensor_cores
- func get_tensor_core_info

## llamatelemetry/cuda/triton_kernels.py
Triton Kernel Integration
Top-level definitions:
- class KernelConfig
- class TritonKernel
- func register_kernel
- func get_kernel
- func list_kernels
- func triton_add
- func triton_layernorm
- func triton_softmax

## llamatelemetry/embeddings.py
llamatelemetry.embeddings - Text Embedding Support
Top-level definitions:
- class EmbeddingEngine
- func cosine_similarity
- func euclidean_distance
- func dot_product_similarity
- class SemanticSearch
- class TextClustering

## llamatelemetry/gguf_parser.py
GGUF Parser for llamatelemetry v0.1.0
Top-level definitions:
- class GGUFValueType
- class GGMLType
- class GGUFTensorInfo
- class GGUFReader
- func inspect_gguf

## llamatelemetry/graphistry/__init__.py
llamatelemetry Graphistry Integration

## llamatelemetry/graphistry/connector.py
Graphistry Connector
Top-level definitions:
- func register_graphistry
- class GraphistryConnector
- func plot_graph

## llamatelemetry/graphistry/rapids.py
RAPIDS Backend for GPU-Accelerated Graph Operations
Top-level definitions:
- func check_rapids_available
- class RAPIDSBackend
- func create_cudf_dataframe
- func run_cugraph_algorithm

## llamatelemetry/graphistry/viz.py
llamatelemetry.graphistry.viz - High-level Graphistry visualization builder.
Top-level definitions:
- class TraceVisualization
- class MetricsVisualization
- class GraphistryViz
- func create_graph_viz

## llamatelemetry/graphistry/workload.py
Split-GPU Workload Management
Top-level definitions:
- class GPUAssignment
- class SplitGPUManager
- class GraphWorkload
- func create_graph_from_llm_output
- func visualize_knowledge_graph

## llamatelemetry/inference/__init__.py
llamatelemetry Advanced Inference APIs

## llamatelemetry/inference/batch.py
Batch Inference Optimization
Top-level definitions:
- class BatchConfig
- class BatchInferenceOptimizer
- class ContinuousBatching
- func batch_inference_optimized

## llamatelemetry/inference/flash_attn.py
FlashAttention Integration
Top-level definitions:
- class FlashAttentionConfig
- func enable_flash_attention
- func flash_attention_forward
- func _standard_attention
- func check_flash_attention_available
- func get_optimal_context_length

## llamatelemetry/inference/kv_cache.py
KV-Cache Optimization
Top-level definitions:
- class KVCacheConfig
- class KVCache
- class PagedKVCache
- func optimize_kv_cache

## llamatelemetry/jupyter.py
llamatelemetry.jupyter - JupyterLab-Specific Features
Top-level definitions:
- func is_jupyter_available
- func check_dependencies
- func stream_generate
- func progress_generate
- func display_metrics
- class ChatWidget
- func compare_temperatures
- func visualize_tokens

## llamatelemetry/kaggle/__init__.py
llamatelemetry.kaggle - Kaggle-specific utilities for zero-boilerplate setup.

## llamatelemetry/kaggle/environment.py
llamatelemetry.kaggle.environment - Zero-boilerplate Kaggle setup.
Top-level definitions:
- class KaggleEnvironment
- func quick_setup

## llamatelemetry/kaggle/gpu_context.py
llamatelemetry.kaggle.gpu_context - GPU context manager for clean RAPIDS/CUDA isolation.
Top-level definitions:
- class GPUContext
- func rapids_gpu
- func llm_gpu
- func single_gpu
- func get_current_gpu_context
- func set_gpu_for_rapids
- func reset_gpu_context

## llamatelemetry/kaggle/presets.py
llamatelemetry.kaggle.presets - Server presets for common Kaggle configurations.
Top-level definitions:
- class TensorSplitMode
- class ServerPreset
- class PresetConfig
- func get_preset_config
- func _auto_detect_preset

## llamatelemetry/kaggle/secrets.py
llamatelemetry.kaggle.secrets - Auto-load Kaggle secrets.
Top-level definitions:
- class KaggleSecrets
- func auto_load_secrets
- func setup_huggingface_auth
- func setup_graphistry_auth

## llamatelemetry/louie/__init__.py
llamatelemetry Louie.AI Integration

## llamatelemetry/louie/client.py
Louie.AI Client Integration
Top-level definitions:
- class QueryResult
- class LouieClient
- func natural_query
- func extract_entities
- func extract_relationships

## llamatelemetry/louie/knowledge.py
Knowledge Extraction and Graph Building
Top-level definitions:
- class EntityType
- class RelationType
- class Entity
- class Relationship
- class KnowledgeGraph
- class KnowledgeExtractor
- func build_knowledge_graph

## llamatelemetry/models.py
llamatelemetry.models - Model Management and Discovery
Top-level definitions:
- class ModelInfo
- func list_models
- func download_model
- func get_model_recommendations
- func print_model_catalog
- class ModelManager
- func load_model_smart
- func list_registry_models
- func print_registry_models
- class SmartModelDownloader

## llamatelemetry/quantization/__init__.py
llamatelemetry Quantization API

## llamatelemetry/quantization/dynamic.py
Dynamic Quantization API
Top-level definitions:
- class QuantStrategy
- class AutoQuantConfig
- class DynamicQuantizer
- func quantize_dynamic

## llamatelemetry/quantization/gguf.py
GGUF Conversion API
Top-level definitions:
- class GGUFQuantType
- class GGUFValueType
- class GGUFTensor
- class GGUFConverter
- func convert_to_gguf
- func save_gguf
- func load_gguf_metadata

## llamatelemetry/quantization/nf4.py
NF4 (4-bit NormalFloat) Quantization
Top-level definitions:
- class NF4Config
- class NF4Quantizer
- func quantize_nf4
- func dequantize_nf4

## llamatelemetry/server.py
llamatelemetry.server - Server Management for llama-server
Top-level definitions:
- class ServerManager

## llamatelemetry/telemetry/__init__.py
llamatelemetry.telemetry - OpenTelemetry integration layer
Top-level definitions:
- func is_otel_available
- func is_graphistry_available
- func get_metrics_collector
- func setup_telemetry

## llamatelemetry/telemetry/auto_instrument.py
llamatelemetry.telemetry.auto_instrument - Auto-instrumentation utilities.
Top-level definitions:
- func instrument_inference
- func _annotate_from_result
- func inference_span
- func batch_inference_span
- func _null_context
- func create_llm_attributes
- func annotate_span_from_result

## llamatelemetry/telemetry/exporter.py
llamatelemetry.telemetry.exporter - Span/Metric Exporters
Top-level definitions:
- func build_exporters

## llamatelemetry/telemetry/graphistry_export.py
llamatelemetry.telemetry.graphistry_export - Real-time trace graph visualization
Top-level definitions:
- class GraphistryTraceExporter

## llamatelemetry/telemetry/instrumentor.py
llamatelemetry.telemetry.instrumentor - OpenTelemetry instrumentor for LlamaCppClient.
Top-level definitions:
- class LlamaCppClientInstrumentor
- func _endpoint_to_operation
- func _add_request_attributes
- func _add_response_attributes
- func instrument_llamacpp_client
- func uninstrument_llamacpp_client

## llamatelemetry/telemetry/metrics.py
llamatelemetry.telemetry.metrics - GPU Metrics Collector
Top-level definitions:
- class GpuMetricsCollector

## llamatelemetry/telemetry/monitor.py
llamatelemetry.telemetry.monitor - Performance monitor for aggregating inference metrics.
Top-level definitions:
- class PerformanceSnapshot
- class InferenceRecord
- class PerformanceMonitor

## llamatelemetry/telemetry/resource.py
llamatelemetry.telemetry.resource - GPU-aware OpenTelemetry Resource
Top-level definitions:
- func _nvidia_smi_query
- func _cuda_version
- func _nccl_available
- func build_gpu_resource

## llamatelemetry/telemetry/tracer.py
llamatelemetry.telemetry.tracer - Inference-aware TracerProvider
Top-level definitions:
- class InferenceTracerProvider
- class _NoopTracer
- class _NoopSpanContext
- class _NoopSpan
- func annotate_inference_span

## llamatelemetry/unsloth/__init__.py
llamatelemetry Unsloth Integration

## llamatelemetry/unsloth/adapter.py
LoRA Adapter Management
Top-level definitions:
- class AdapterConfig
- class LoRAAdapter
- func merge_lora_adapters
- func extract_base_model

## llamatelemetry/unsloth/exporter.py
Unsloth to GGUF Exporter
Top-level definitions:
- class ExportConfig
- class UnslothExporter
- func export_to_llamatelemetry
- func export_to_gguf

## llamatelemetry/unsloth/loader.py
Unsloth Model Loader
Top-level definitions:
- func check_unsloth_available
- class UnslothModelLoader
- func load_unsloth_model

## llamatelemetry/utils.py
llamatelemetry.utils - Utility Functions
Top-level definitions:
- func detect_cuda
- func check_gpu_compatibility
- func get_llama_cpp_cuda_path
- func setup_environment
- func find_gguf_models
- func print_system_info
- func create_config_file
- func load_config
- func get_recommended_gpu_layers
- func validate_model_path
- func auto_configure_for_model
