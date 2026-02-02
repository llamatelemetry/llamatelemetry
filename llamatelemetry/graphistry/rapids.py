"""
RAPIDS Backend for GPU-Accelerated Graph Operations

Provides utilities for working with RAPIDS libraries:
- cuDF: GPU DataFrames
- cuGraph: GPU Graph Analytics
- cuML: GPU Machine Learning
"""

import os
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass


def check_rapids_available() -> Dict[str, bool]:
    """
    Check which RAPIDS components are available.
    
    Returns:
        Dict with component availability
        
    Example:
        >>> from llamatelemetry.graphistry import check_rapids_available
        >>> status = check_rapids_available()
        >>> print(status)
        {'cudf': True, 'cugraph': True, 'cuml': True}
    """
    status = {
        "cudf": False,
        "cugraph": False,
        "cuml": False,
        "pylibraft": False,
    }
    
    try:
        import cudf
        status["cudf"] = True
    except ImportError:
        pass
    
    try:
        import cugraph
        status["cugraph"] = True
    except ImportError:
        pass
    
    try:
        import cuml
        status["cuml"] = True
    except ImportError:
        pass
    
    try:
        import pylibraft
        status["pylibraft"] = True
    except ImportError:
        pass
    
    return status


class RAPIDSBackend:
    """
    RAPIDS backend for GPU-accelerated operations.
    
    Provides a unified interface for cuDF, cuGraph, and cuML operations.
    
    Example:
        >>> from llamatelemetry.graphistry import RAPIDSBackend
        >>> 
        >>> backend = RAPIDSBackend(gpu_id=1)
        >>> 
        >>> # Create GPU dataframe
        >>> gdf = backend.create_dataframe({
        ...     "src": ["a", "b", "c"],
        ...     "dst": ["b", "c", "a"]
        ... })
        >>> 
        >>> # Run PageRank
        >>> pr = backend.pagerank(gdf, "src", "dst")
    """
    
    def __init__(self, gpu_id: int = 1):
        """
        Initialize RAPIDS backend.
        
        Args:
            gpu_id: GPU device to use
        """
        self.gpu_id = gpu_id
        
        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Check availability
        self._status = check_rapids_available()
        
        if not any(self._status.values()):
            raise ImportError(
                "RAPIDS is not installed. Install with:\n"
                "  pip install --extra-index-url=https://pypi.nvidia.com "
                "cudf-cu12 cuml-cu12 cugraph-cu12"
            )
    
    @property
    def cudf_available(self) -> bool:
        return self._status["cudf"]
    
    @property
    def cugraph_available(self) -> bool:
        return self._status["cugraph"]
    
    @property
    def cuml_available(self) -> bool:
        return self._status["cuml"]
    
    def create_dataframe(self, data: Union[Dict, List[Dict]]) -> Any:
        """
        Create cuDF DataFrame.
        
        Args:
            data: Dict of columns or list of row dicts
            
        Returns:
            cuDF DataFrame
        """
        if not self.cudf_available:
            raise ImportError("cuDF is not available")
        
        import cudf
        return cudf.DataFrame(data)
    
    def from_pandas(self, pdf) -> Any:
        """
        Convert Pandas DataFrame to cuDF.
        
        Args:
            pdf: Pandas DataFrame
            
        Returns:
            cuDF DataFrame
        """
        if not self.cudf_available:
            raise ImportError("cuDF is not available")
        
        import cudf
        return cudf.from_pandas(pdf)
    
    def to_pandas(self, gdf) -> Any:
        """
        Convert cuDF DataFrame to Pandas.
        
        Args:
            gdf: cuDF DataFrame
            
        Returns:
            Pandas DataFrame
        """
        return gdf.to_pandas()
    
    def pagerank(
        self,
        edges_df,
        source_col: str = "src",
        dest_col: str = "dst",
        damping: float = 0.85,
        max_iter: int = 100,
    ) -> Any:
        """
        Run PageRank on graph.
        
        Args:
            edges_df: Edge DataFrame (cuDF or Pandas)
            source_col: Source vertex column name
            dest_col: Destination vertex column name
            damping: Damping factor
            max_iter: Maximum iterations
            
        Returns:
            DataFrame with 'vertex' and 'pagerank' columns
        """
        if not self.cugraph_available:
            raise ImportError("cuGraph is not available")
        
        import cugraph
        import cudf
        
        # Convert to cuDF if needed
        if not isinstance(edges_df, cudf.DataFrame):
            edges_df = cudf.DataFrame(edges_df)
        
        # Create graph
        G = cugraph.Graph(directed=True)
        G.from_cudf_edgelist(edges_df, source=source_col, destination=dest_col)
        
        # Run PageRank
        return cugraph.pagerank(G, alpha=damping, max_iter=max_iter)
    
    def louvain(
        self,
        edges_df,
        source_col: str = "src",
        dest_col: str = "dst",
        weight_col: Optional[str] = None,
        resolution: float = 1.0,
    ) -> Any:
        """
        Run Louvain community detection.
        
        Args:
            edges_df: Edge DataFrame
            source_col: Source vertex column name
            dest_col: Destination vertex column name
            weight_col: Optional edge weight column
            resolution: Resolution parameter
            
        Returns:
            Tuple of (partition DataFrame, modularity score)
        """
        if not self.cugraph_available:
            raise ImportError("cuGraph is not available")
        
        import cugraph
        import cudf
        
        # Convert to cuDF if needed
        if not isinstance(edges_df, cudf.DataFrame):
            edges_df = cudf.DataFrame(edges_df)
        
        # Create graph
        G = cugraph.Graph()
        if weight_col and weight_col in edges_df.columns:
            G.from_cudf_edgelist(
                edges_df,
                source=source_col,
                destination=dest_col,
                edge_attr=weight_col
            )
        else:
            G.from_cudf_edgelist(edges_df, source=source_col, destination=dest_col)
        
        # Run Louvain
        return cugraph.louvain(G, resolution=resolution)
    
    def betweenness_centrality(
        self,
        edges_df,
        source_col: str = "src",
        dest_col: str = "dst",
        k: Optional[int] = None,
        normalized: bool = True,
    ) -> Any:
        """
        Calculate betweenness centrality.
        
        Args:
            edges_df: Edge DataFrame
            source_col: Source vertex column name
            dest_col: Destination vertex column name
            k: Number of sample vertices (None = all)
            normalized: Normalize centrality values
            
        Returns:
            DataFrame with 'vertex' and 'betweenness_centrality' columns
        """
        if not self.cugraph_available:
            raise ImportError("cuGraph is not available")
        
        import cugraph
        import cudf
        
        # Convert to cuDF if needed
        if not isinstance(edges_df, cudf.DataFrame):
            edges_df = cudf.DataFrame(edges_df)
        
        # Create graph
        G = cugraph.Graph()
        G.from_cudf_edgelist(edges_df, source=source_col, destination=dest_col)
        
        # Run betweenness centrality
        return cugraph.betweenness_centrality(G, k=k, normalized=normalized)
    
    def connected_components(
        self,
        edges_df,
        source_col: str = "src",
        dest_col: str = "dst",
    ) -> Any:
        """
        Find connected components.
        
        Args:
            edges_df: Edge DataFrame
            source_col: Source vertex column name
            dest_col: Destination vertex column name
            
        Returns:
            DataFrame with 'vertex' and 'labels' columns
        """
        if not self.cugraph_available:
            raise ImportError("cuGraph is not available")
        
        import cugraph
        import cudf
        
        # Convert to cuDF if needed
        if not isinstance(edges_df, cudf.DataFrame):
            edges_df = cudf.DataFrame(edges_df)
        
        # Create graph
        G = cugraph.Graph()
        G.from_cudf_edgelist(edges_df, source=source_col, destination=dest_col)
        
        # Find connected components
        return cugraph.connected_components(G)
    
    def umap(
        self,
        data,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> Any:
        """
        Run UMAP dimensionality reduction.
        
        Args:
            data: Input data (cuDF or Pandas DataFrame/array)
            n_components: Number of output dimensions
            n_neighbors: Number of neighbors for local approximation
            min_dist: Minimum distance between points
            
        Returns:
            Reduced embeddings
        """
        if not self.cuml_available:
            raise ImportError("cuML is not available")
        
        import cuml
        import cudf
        
        # Convert to cuDF if needed
        if hasattr(data, 'values'):
            data = data.values
        
        # Run UMAP
        umap = cuml.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        
        return umap.fit_transform(data)


def create_cudf_dataframe(data: Union[Dict, List[Dict]]) -> Any:
    """
    Quick helper to create cuDF DataFrame.
    
    Args:
        data: Dict of columns or list of row dicts
        
    Returns:
        cuDF DataFrame
    """
    import cudf
    return cudf.DataFrame(data)


def run_cugraph_algorithm(
    algorithm: str,
    edges_df,
    source_col: str = "src",
    dest_col: str = "dst",
    **kwargs,
) -> Any:
    """
    Run cuGraph algorithm by name.
    
    Args:
        algorithm: Algorithm name ('pagerank', 'louvain', etc.)
        edges_df: Edge DataFrame
        source_col: Source column name
        dest_col: Destination column name
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Algorithm result
    """
    backend = RAPIDSBackend()
    
    algorithms = {
        "pagerank": backend.pagerank,
        "louvain": backend.louvain,
        "betweenness_centrality": backend.betweenness_centrality,
        "connected_components": backend.connected_components,
    }
    
    if algorithm not in algorithms:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Available: {list(algorithms.keys())}"
        )
    
    return algorithms[algorithm](edges_df, source_col, dest_col, **kwargs)
