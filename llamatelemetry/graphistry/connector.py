"""
Graphistry Connector

Connection and visualization utilities for PyGraphistry.
"""

import os
from typing import Optional, Dict, Any, Union


def register_graphistry(
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_key: Optional[str] = None,
    server: str = "hub.graphistry.com",
) -> bool:
    """
    Register with Graphistry service.
    
    Uses environment variables if credentials not provided:
    - GRAPHISTRY_USERNAME
    - GRAPHISTRY_PASSWORD
    - GRAPHISTRY_API_KEY
    - GRAPHISTRY_SERVER
    
    Args:
        username: Graphistry Hub username
        password: Graphistry Hub password
        api_key: API key (alternative to username/password)
        server: Graphistry server URL
        
    Returns:
        True if registration successful
        
    Example:
        >>> from llamatelemetry.graphistry import register_graphistry
        >>> 
        >>> # Using environment variables
        >>> os.environ["GRAPHISTRY_USERNAME"] = "myuser"
        >>> os.environ["GRAPHISTRY_PASSWORD"] = "mypass"
        >>> register_graphistry()
        True
        >>> 
        >>> # Or directly
        >>> register_graphistry(username="myuser", password="mypass")
        True
    """
    try:
        import graphistry
    except ImportError:
        raise ImportError(
            "PyGraphistry is not installed. Install with:\n"
            "  pip install graphistry[ai]"
        )
    
    # Get from env if not provided
    username = username or os.environ.get("GRAPHISTRY_USERNAME")
    password = password or os.environ.get("GRAPHISTRY_PASSWORD")
    api_key = api_key or os.environ.get("GRAPHISTRY_API_KEY")
    server = server or os.environ.get("GRAPHISTRY_SERVER", "hub.graphistry.com")
    
    try:
        if api_key:
            graphistry.register(api=3, token=api_key, server=server)
        elif username and password:
            graphistry.register(
                api=3,
                username=username,
                password=password,
                server=server
            )
        else:
            raise ValueError(
                "Graphistry credentials required. Set GRAPHISTRY_USERNAME and "
                "GRAPHISTRY_PASSWORD environment variables, or pass directly."
            )
        
        return True
    except Exception as e:
        print(f"Warning: Graphistry registration failed: {e}")
        return False


class GraphistryConnector:
    """
    Connector for Graphistry visualization service.
    
    Example:
        >>> from llamatelemetry.graphistry import GraphistryConnector
        >>> import pandas as pd
        >>> 
        >>> connector = GraphistryConnector()
        >>> connector.register(username="myuser", password="mypass")
        >>> 
        >>> edges = pd.DataFrame({
        ...     "src": ["a", "b", "c"],
        ...     "dst": ["b", "c", "a"],
        ...     "weight": [0.5, 0.8, 0.3]
        ... })
        >>> 
        >>> g = connector.create_graph(edges)
        >>> g.plot()
    """
    
    def __init__(
        self,
        auto_register: bool = True,
        server: str = "hub.graphistry.com",
    ):
        """
        Initialize Graphistry connector.
        
        Args:
            auto_register: Try to register using environment variables
            server: Graphistry server URL
        """
        self.server = server
        self._registered = False
        
        try:
            import graphistry
            self._graphistry = graphistry
        except ImportError:
            raise ImportError(
                "PyGraphistry is not installed. Install with:\n"
                "  pip install graphistry[ai]"
            )
        
        if auto_register:
            self._registered = register_graphistry(server=server)
    
    @property
    def is_registered(self) -> bool:
        """Check if registered with Graphistry."""
        return self._registered
    
    def register(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> bool:
        """
        Register with Graphistry service.
        
        Args:
            username: Graphistry Hub username
            password: Graphistry Hub password
            api_key: API key (alternative to username/password)
            
        Returns:
            True if registration successful
        """
        self._registered = register_graphistry(
            username=username,
            password=password,
            api_key=api_key,
            server=self.server
        )
        return self._registered
    
    def create_graph(
        self,
        edges_df,
        source: str = "src",
        destination: str = "dst",
        nodes_df=None,
        node_id: str = "id",
    ):
        """
        Create Graphistry graph from edge/node dataframes.
        
        Args:
            edges_df: Edge DataFrame with source/destination columns
            source: Source column name
            destination: Destination column name
            nodes_df: Optional node DataFrame
            node_id: Node ID column name
            
        Returns:
            Graphistry graph object
        """
        g = self._graphistry.edges(edges_df, source, destination)
        
        if nodes_df is not None:
            g = g.nodes(nodes_df, node_id)
        
        return g
    
    def plot(
        self,
        edges_df,
        source: str = "src",
        destination: str = "dst",
        nodes_df=None,
        node_id: str = "id",
        **plot_kwargs,
    ):
        """
        Quick plot of graph data.
        
        Args:
            edges_df: Edge DataFrame
            source: Source column name
            destination: Destination column name
            nodes_df: Optional node DataFrame
            node_id: Node ID column name
            **plot_kwargs: Additional plot() arguments
            
        Returns:
            Plot result (URL or embed)
        """
        g = self.create_graph(edges_df, source, destination, nodes_df, node_id)
        return g.plot(**plot_kwargs)
    
    def compute_igraph(
        self,
        edges_df,
        source: str = "src",
        destination: str = "dst",
        algorithm: str = "pagerank",
    ):
        """
        Run igraph algorithm and add to graph.
        
        Args:
            edges_df: Edge DataFrame
            source: Source column name
            destination: Destination column name
            algorithm: igraph algorithm name
            
        Returns:
            Graphistry graph with computed values
        """
        g = self.create_graph(edges_df, source, destination)
        return g.compute_igraph(algorithm)
    
    def compute_cugraph(
        self,
        edges_df,
        source: str = "src",
        destination: str = "dst",
        algorithm: str = "pagerank",
    ):
        """
        Run cuGraph algorithm (GPU) and add to graph.
        
        Args:
            edges_df: Edge DataFrame (should be cuDF for GPU)
            source: Source column name
            destination: Destination column name
            algorithm: cuGraph algorithm name
            
        Returns:
            Graphistry graph with computed values
        """
        g = self.create_graph(edges_df, source, destination)
        return g.compute_cugraph(algorithm)


def plot_graph(
    edges_df,
    source: str = "src",
    destination: str = "dst",
    nodes_df=None,
    node_id: str = "id",
    **kwargs,
):
    """
    Quick helper to plot graph data.
    
    Args:
        edges_df: Edge DataFrame
        source: Source column name
        destination: Destination column name
        nodes_df: Optional node DataFrame
        node_id: Node ID column name
        **kwargs: Additional Graphistry arguments
        
    Returns:
        Plot result
    """
    connector = GraphistryConnector()
    return connector.plot(
        edges_df,
        source=source,
        destination=destination,
        nodes_df=nodes_df,
        node_id=node_id,
        **kwargs
    )
