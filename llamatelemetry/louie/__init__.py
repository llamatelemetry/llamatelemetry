"""
llamatelemetry Louie.AI Integration

AI-powered investigation platform integration for natural language graph analysis.
Combines LLM inference with GPU-accelerated graph visualization.

Features:
    - Natural language queries for graph data
    - LLM-powered knowledge extraction
    - Seamless integration with llamatelemetry inference and Graphistry
    - Split-GPU support (LLM on GPU 0, Graphs on GPU 1)

Example:
    >>> from llamatelemetry.louie import LouieClient, natural_query
    >>> 
    >>> # Natural language graph analysis
    >>> result = natural_query("Find all connections between Python and AI")
    >>> print(result.text)
    >>> result.graph.plot()  # GPU-accelerated visualization
"""

from .client import (
    LouieClient,
    natural_query,
    extract_entities,
    extract_relationships,
)

from .knowledge import (
    KnowledgeExtractor,
    build_knowledge_graph,
    EntityType,
    RelationType,
)

__all__ = [
    # Client
    'LouieClient',
    'natural_query',
    'extract_entities',
    'extract_relationships',
    
    # Knowledge extraction
    'KnowledgeExtractor',
    'build_knowledge_graph',
    'EntityType',
    'RelationType',
]
