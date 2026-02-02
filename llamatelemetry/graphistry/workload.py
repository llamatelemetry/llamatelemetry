"""
Split-GPU Workload Management

Manages GPU assignment for split workloads:
- GPU 0: LLM inference with llama-server
- GPU 1: Graph operations with RAPIDS/Graphistry
"""

import os
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import json


@dataclass
class GPUAssignment:
    """GPU assignment for a specific workload."""
    gpu_id: int
    workload_type: str  # "llm" or "graph"
    memory_limit_gb: Optional[float] = None


class SplitGPUManager:
    """
    Manages split-GPU architecture for LLM + Graph workloads.
    
    On Kaggle with 2× Tesla T4:
    - GPU 0 (15GB): llama-server with GGUF model
    - GPU 1 (15GB): RAPIDS (cuDF, cuGraph) + Graphistry
    
    Example:
        >>> from llamatelemetry.graphistry import SplitGPUManager
        >>> 
        >>> manager = SplitGPUManager()
        >>> manager.assign_llm(gpu_id=0)
        >>> manager.assign_graph(gpu_id=1)
        >>> 
        >>> # Get environment for graph operations
        >>> env = manager.get_graph_env()
        >>> # CUDA_VISIBLE_DEVICES=1
    """
    
    def __init__(self, auto_detect: bool = True):
        """
        Initialize split-GPU manager.
        
        Args:
            auto_detect: Auto-detect GPUs and assign defaults
        """
        self.assignments: Dict[str, GPUAssignment] = {}
        self.gpu_count = 0
        
        if auto_detect:
            self._detect_gpus()
            self._assign_defaults()
    
    def _detect_gpus(self) -> None:
        """Detect available GPUs."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5
            )
            self.gpu_count = len([
                l for l in result.stdout.strip().split("\n")
                if l.startswith("GPU")
            ])
        except Exception:
            self.gpu_count = 0
    
    def _assign_defaults(self) -> None:
        """Assign default GPU workloads."""
        if self.gpu_count >= 2:
            # Kaggle 2× T4 configuration
            self.assign_llm(0)
            self.assign_graph(1)
        elif self.gpu_count == 1:
            # Single GPU: share both workloads
            self.assign_llm(0)
            self.assign_graph(0)
    
    def assign_llm(self, gpu_id: int, memory_limit_gb: Optional[float] = None) -> None:
        """
        Assign GPU for LLM inference.
        
        Args:
            gpu_id: GPU device ID
            memory_limit_gb: Optional VRAM limit
        """
        self.assignments["llm"] = GPUAssignment(
            gpu_id=gpu_id,
            workload_type="llm",
            memory_limit_gb=memory_limit_gb
        )
    
    def assign_graph(self, gpu_id: int, memory_limit_gb: Optional[float] = None) -> None:
        """
        Assign GPU for graph operations.
        
        Args:
            gpu_id: GPU device ID
            memory_limit_gb: Optional VRAM limit
        """
        self.assignments["graph"] = GPUAssignment(
            gpu_id=gpu_id,
            workload_type="graph",
            memory_limit_gb=memory_limit_gb
        )
    
    def get_llm_env(self) -> Dict[str, str]:
        """Get environment variables for LLM workload."""
        if "llm" not in self.assignments:
            return {}
        
        return {"CUDA_VISIBLE_DEVICES": str(self.assignments["llm"].gpu_id)}
    
    def get_graph_env(self) -> Dict[str, str]:
        """Get environment variables for graph workload."""
        if "graph" not in self.assignments:
            return {}
        
        return {"CUDA_VISIBLE_DEVICES": str(self.assignments["graph"].gpu_id)}
    
    def get_llama_server_args(self, model_path: str) -> List[str]:
        """
        Generate llama-server command arguments.
        
        Args:
            model_path: Path to GGUF model
            
        Returns:
            List of command-line arguments
        """
        args = [
            "-m", model_path,
            "-ngl", "99",
            "-fa",
            "--host", "0.0.0.0",
            "--port", "8080",
        ]
        
        if "llm" in self.assignments:
            gpu_id = self.assignments["llm"].gpu_id
            args.extend(["--main-gpu", str(gpu_id)])
        
        return args
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dict."""
        return {
            "gpu_count": self.gpu_count,
            "assignments": {
                k: {
                    "gpu_id": v.gpu_id,
                    "workload_type": v.workload_type,
                    "memory_limit_gb": v.memory_limit_gb
                }
                for k, v in self.assignments.items()
            }
        }


class GraphWorkload:
    """
    GPU-accelerated graph workload with RAPIDS and Graphistry.
    
    Example:
        >>> from llamatelemetry.graphistry import GraphWorkload
        >>> 
        >>> # Initialize on GPU 1
        >>> workload = GraphWorkload(gpu_id=1)
        >>> 
        >>> # Create knowledge graph from LLM output
        >>> entities = [
        ...     {"id": "python", "type": "language", "properties": {"year": 1991}},
        ...     {"id": "ai", "type": "field", "properties": {"subfield": "ML"}}
        ... ]
        >>> relationships = [
        ...     {"source": "python", "target": "ai", "type": "used_for", "weight": 0.9}
        ... ]
        >>> 
        >>> g = workload.create_knowledge_graph(entities, relationships)
        >>> g.plot()
    """
    
    def __init__(
        self,
        gpu_id: int = 1,
        graphistry_username: Optional[str] = None,
        graphistry_password: Optional[str] = None,
        graphistry_server: str = "hub.graphistry.com",
    ):
        """
        Initialize graph workload.
        
        Args:
            gpu_id: GPU to use for graph operations
            graphistry_username: Graphistry Hub username
            graphistry_password: Graphistry Hub password
            graphistry_server: Graphistry server URL
        """
        self.gpu_id = gpu_id
        self.graphistry_username = graphistry_username
        self.graphistry_password = graphistry_password
        self.graphistry_server = graphistry_server
        
        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Initialize backends
        self._rapids_available = False
        self._graphistry_registered = False
        self._init_backends()
    
    def _init_backends(self) -> None:
        """Initialize RAPIDS and Graphistry backends."""
        # Check RAPIDS
        try:
            import cudf
            import cugraph
            self._rapids_available = True
        except ImportError:
            self._rapids_available = False
        
        # Register Graphistry
        if self.graphistry_username and self.graphistry_password:
            try:
                import graphistry
                graphistry.register(
                    api=3,
                    username=self.graphistry_username,
                    password=self.graphistry_password,
                    server=self.graphistry_server
                )
                self._graphistry_registered = True
            except Exception:
                pass
    
    @property
    def rapids_available(self) -> bool:
        """Check if RAPIDS is available."""
        return self._rapids_available
    
    @property
    def graphistry_registered(self) -> bool:
        """Check if Graphistry is registered."""
        return self._graphistry_registered
    
    def create_knowledge_graph(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        use_gpu: bool = True,
    ):
        """
        Create knowledge graph from entities and relationships.
        
        Args:
            entities: List of entity dicts with 'id', 'type', 'properties'
            relationships: List of relationship dicts with 'source', 'target', 'type'
            use_gpu: Use GPU-accelerated cuDF/cuGraph
            
        Returns:
            Graphistry graph object ready for visualization
        """
        import graphistry
        
        # Create edges dataframe
        edges_data = []
        for rel in relationships:
            edges_data.append({
                "src": rel["source"],
                "dst": rel["target"],
                "relationship": rel.get("type", "related"),
                "weight": rel.get("weight", 1.0)
            })
        
        # Create nodes dataframe
        nodes_data = []
        for entity in entities:
            node = {"id": entity["id"], "type": entity.get("type", "entity")}
            if "properties" in entity:
                node.update(entity["properties"])
            nodes_data.append(node)
        
        # Use GPU or CPU dataframes
        if use_gpu and self._rapids_available:
            import cudf
            edges_df = cudf.DataFrame(edges_data)
            nodes_df = cudf.DataFrame(nodes_data)
        else:
            import pandas as pd
            edges_df = pd.DataFrame(edges_data)
            nodes_df = pd.DataFrame(nodes_data)
        
        # Create Graphistry graph
        g = graphistry.edges(edges_df, "src", "dst")
        g = g.nodes(nodes_df, "id")
        
        # Style by type
        g = g.encode_point_color("type", categorical_mapping={
            "entity": "#1f77b4",
            "concept": "#ff7f0e",
            "language": "#2ca02c",
            "field": "#d62728",
        })
        
        g = g.encode_edge_color("weight", ["#cccccc", "#ff0000"], as_continuous=True)
        
        return g
    
    def run_pagerank(self, edges_df, damping: float = 0.85, max_iter: int = 100):
        """
        Run PageRank algorithm on graph.
        
        Args:
            edges_df: Edge dataframe with 'src' and 'dst' columns
            damping: Damping factor
            max_iter: Maximum iterations
            
        Returns:
            DataFrame with 'vertex' and 'pagerank' columns
        """
        if not self._rapids_available:
            raise RuntimeError("RAPIDS (cuGraph) is required for PageRank")
        
        import cugraph
        import cudf
        
        # Ensure GPU dataframe
        if not isinstance(edges_df, cudf.DataFrame):
            edges_df = cudf.DataFrame(edges_df)
        
        # Create graph
        G = cugraph.Graph(directed=True)
        G.from_cudf_edgelist(edges_df, source="src", destination="dst")
        
        # Run PageRank
        pr = cugraph.pagerank(G, alpha=damping, max_iter=max_iter)
        
        return pr
    
    def run_community_detection(self, edges_df, resolution: float = 1.0):
        """
        Run Louvain community detection.
        
        Args:
            edges_df: Edge dataframe with 'src', 'dst', and optionally 'weight'
            resolution: Resolution parameter
            
        Returns:
            DataFrame with 'vertex' and 'partition' columns
        """
        if not self._rapids_available:
            raise RuntimeError("RAPIDS (cuGraph) is required for community detection")
        
        import cugraph
        import cudf
        
        # Ensure GPU dataframe
        if not isinstance(edges_df, cudf.DataFrame):
            edges_df = cudf.DataFrame(edges_df)
        
        # Create graph
        G = cugraph.Graph()
        if "weight" in edges_df.columns:
            G.from_cudf_edgelist(edges_df, source="src", destination="dst", edge_attr="weight")
        else:
            G.from_cudf_edgelist(edges_df, source="src", destination="dst")
        
        # Run Louvain
        parts, modularity = cugraph.louvain(G, resolution=resolution)
        
        return parts


def create_graph_from_llm_output(
    llm_response: str,
    workload: Optional[GraphWorkload] = None,
) -> Any:
    """
    Parse LLM output and create knowledge graph.
    
    Expects JSON format with 'entities' and 'relationships' keys.
    
    Args:
        llm_response: LLM output containing graph data
        workload: Optional GraphWorkload instance
        
    Returns:
        Graphistry graph object
        
    Example:
        >>> from llamatelemetry import InferenceEngine
        >>> from llamatelemetry.graphistry import create_graph_from_llm_output
        >>> 
        >>> engine = InferenceEngine()
        >>> response = engine.infer('''
        ...     Extract entities and relationships from:
        ...     "Python is used for AI and machine learning."
        ...     Return JSON with entities and relationships.
        ... ''')
        >>> 
        >>> g = create_graph_from_llm_output(response.text)
        >>> g.plot()
    """
    # Try to parse JSON from response
    try:
        # Find JSON in response
        import re
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if json_match:
            data = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in LLM response")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from LLM response: {e}")
    
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])
    
    if not entities and not relationships:
        raise ValueError("No entities or relationships found in LLM response")
    
    if workload is None:
        workload = GraphWorkload(gpu_id=1)
    
    return workload.create_knowledge_graph(entities, relationships)


def visualize_knowledge_graph(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    gpu_id: int = 1,
    **graphistry_kwargs,
) -> Any:
    """
    Quick visualization of knowledge graph.
    
    Args:
        entities: List of entity dicts
        relationships: List of relationship dicts
        gpu_id: GPU for graph operations
        **graphistry_kwargs: Additional Graphistry plot() arguments
        
    Returns:
        Graphistry graph URL or embed
    """
    workload = GraphWorkload(gpu_id=gpu_id)
    g = workload.create_knowledge_graph(entities, relationships)
    return g.plot(**graphistry_kwargs)
