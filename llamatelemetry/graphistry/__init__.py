"""
llamatelemetry Graphistry Integration

GPU-accelerated graph visualization with PyGraphistry and RAPIDS.
Designed for split-GPU architecture on Kaggle (GPU 0: LLM, GPU 1: Graphs).

Features:
    - Seamless integration with PyGraphistry for GPU graph visualization
    - RAPIDS acceleration with cuDF, cuGraph, cuML
    - LLM-powered graph analysis with Louie.AI
    - Split-GPU workload support (LLM on GPU 0, Graphs on GPU 1)

Example:
    >>> from llamatelemetry.graphistry import GraphWorkload, create_graph_from_llm_output
    >>> 
    >>> # Set GPU 1 for graph operations
    >>> workload = GraphWorkload(gpu_id=1)
    >>> 
    >>> # Create graph from LLM-generated knowledge
    >>> g = workload.create_knowledge_graph(entities, relationships)
    >>> g.plot()  # GPU-accelerated visualization
"""

from .workload import (
    GraphWorkload,
    SplitGPUManager,
    create_graph_from_llm_output,
    visualize_knowledge_graph,
)

from .rapids import (
    RAPIDSBackend,
    create_cudf_dataframe,
    run_cugraph_algorithm,
    check_rapids_available,
)

from .connector import (
    GraphistryConnector,
    register_graphistry,
    plot_graph,
)

from .viz import (
    GraphistryViz,
    TraceVisualization,
    MetricsVisualization,
    create_graph_viz,
)

__all__ = [
    # Workload management
    'GraphWorkload',
    'SplitGPUManager',
    'create_graph_from_llm_output',
    'visualize_knowledge_graph',
    
    # RAPIDS backend
    'RAPIDSBackend',
    'create_cudf_dataframe',
    'run_cugraph_algorithm',
    'check_rapids_available',
    
    # Graphistry connector
    'GraphistryConnector',
    'register_graphistry',
    'plot_graph',

    # High-level visualization
    'GraphistryViz',
    'TraceVisualization',
    'MetricsVisualization',
    'create_graph_viz',
]
