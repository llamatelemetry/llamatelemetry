"""
Louie.AI Client Integration

Natural language interface for graph analysis using llamatelemetry inference.
"""

import os
import json
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class QueryResult:
    """Result from a natural language query."""
    text: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    raw_response: str
    graph: Optional[Any] = None  # Graphistry graph if available


class LouieClient:
    """
    Client for natural language graph analysis.
    
    Combines llamatelemetry LLM inference with Louie.AI-style natural language
    processing for graph data analysis.
    
    Example:
        >>> from llamatelemetry.louie import LouieClient
        >>> 
        >>> client = LouieClient()
        >>> result = client.query("What entities are mentioned in this text?")
        >>> print(result.text)
        >>> 
        >>> # With custom model
        >>> client = LouieClient(model="gemma-3-1b-Q4_K_M")
        >>> result = client.query("Extract knowledge graph from: ...")
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        server_url: Optional[str] = None,
        use_local_llm: bool = True,
        graphistry_username: Optional[str] = None,
        graphistry_password: Optional[str] = None,
    ):
        """
        Initialize Louie client.
        
        Args:
            model: Model name for local inference (e.g., "gemma-3-1b-Q4_K_M")
            server_url: llama-server URL if using server mode
            use_local_llm: Use local llamatelemetry inference (vs Louie.AI cloud)
            graphistry_username: Graphistry Hub username for visualization
            graphistry_password: Graphistry Hub password
        """
        self.model = model or "gemma-3-1b-Q4_K_M"
        self.server_url = server_url or "http://localhost:8080"
        self.use_local_llm = use_local_llm
        self.graphistry_username = graphistry_username
        self.graphistry_password = graphistry_password
        
        self._engine = None
        self._client = None
        self._graphistry = None
    
    def _init_llm(self) -> None:
        """Initialize LLM backend."""
        if self._engine is not None or self._client is not None:
            return
        
        if self.use_local_llm:
            try:
                import llamatelemetry
                self._engine = llamatelemetry.InferenceEngine()
                self._engine.load_model(self.model, silent=True)
            except Exception as e:
                print(f"Warning: Failed to load local LLM: {e}")
                self._init_server_client()
        else:
            self._init_server_client()
    
    def _init_server_client(self) -> None:
        """Initialize server client."""
        try:
            from llamatelemetry.api import LlamaCppClient
            self._client = LlamaCppClient(self.server_url)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM backend: {e}")
    
    def _init_graphistry(self) -> None:
        """Initialize Graphistry connection."""
        if self._graphistry is not None:
            return
        
        try:
            from llamatelemetry.graphistry import GraphWorkload
            self._graphistry = GraphWorkload(
                gpu_id=1,
                graphistry_username=self.graphistry_username,
                graphistry_password=self.graphistry_password,
            )
        except Exception:
            pass
    
    def query(
        self,
        question: str,
        context: Optional[str] = None,
        extract_graph: bool = True,
        max_tokens: int = 1000,
    ) -> QueryResult:
        """
        Execute natural language query.
        
        Args:
            question: Natural language question
            context: Optional context/data to analyze
            extract_graph: Try to extract graph from response
            max_tokens: Maximum response tokens
            
        Returns:
            QueryResult with text, entities, relationships, and optional graph
            
        Example:
            >>> client = LouieClient()
            >>> result = client.query(
            ...     "Extract entities and relationships",
            ...     context="Python is used for AI. TensorFlow is a Python library."
            ... )
            >>> print(result.entities)
            [{"id": "Python", "type": "language"}, ...]
        """
        self._init_llm()
        
        # Build prompt for knowledge extraction
        prompt = self._build_extraction_prompt(question, context)
        
        # Get LLM response
        if self._engine:
            response = self._engine.infer(prompt, max_tokens=max_tokens)
            raw_response = response.text
        else:
            response = self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            raw_response = response.choices[0].message.content
        
        # Parse response
        entities, relationships = self._parse_graph_response(raw_response)
        
        # Build graph if requested and data available
        graph = None
        if extract_graph and (entities or relationships):
            self._init_graphistry()
            if self._graphistry:
                try:
                    graph = self._graphistry.create_knowledge_graph(
                        entities, relationships
                    )
                except Exception:
                    pass
        
        return QueryResult(
            text=raw_response,
            entities=entities,
            relationships=relationships,
            raw_response=raw_response,
            graph=graph
        )
    
    def _build_extraction_prompt(
        self,
        question: str,
        context: Optional[str] = None,
    ) -> str:
        """Build prompt for knowledge extraction."""
        prompt = """You are a knowledge extraction assistant. Extract entities and relationships from the given text.

Output format (JSON):
{
  "answer": "Your natural language answer",
  "entities": [
    {"id": "entity_name", "type": "entity_type", "properties": {}}
  ],
  "relationships": [
    {"source": "entity1", "target": "entity2", "type": "relationship_type", "weight": 1.0}
  ]
}

Entity types: person, organization, location, concept, technology, language, product, event, other
Relationship types: uses, creates, belongs_to, related_to, part_of, located_in, works_for, other

"""
        if context:
            prompt += f"Context:\n{context}\n\n"
        
        prompt += f"Question: {question}\n\nOutput:"
        
        return prompt
    
    def _parse_graph_response(
        self,
        response: str,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse entities and relationships from LLM response."""
        entities = []
        relationships = []
        
        try:
            # Try to find JSON in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get("entities", [])
                relationships = data.get("relationships", [])
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return entities, relationships
    
    def extract(self, text: str, **kwargs) -> QueryResult:
        """
        Shorthand for extraction query.
        
        Args:
            text: Text to extract knowledge from
            **kwargs: Additional query arguments
            
        Returns:
            QueryResult with extracted entities and relationships
        """
        return self.query(
            "Extract all entities and relationships from this text.",
            context=text,
            **kwargs
        )


def natural_query(
    question: str,
    context: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> QueryResult:
    """
    Quick natural language query.
    
    Args:
        question: Natural language question
        context: Optional context/data
        model: Model name for inference
        **kwargs: Additional LouieClient arguments
        
    Returns:
        QueryResult
        
    Example:
        >>> from llamatelemetry.louie import natural_query
        >>> 
        >>> result = natural_query(
        ...     "What are the main technologies?",
        ...     context="Python and TensorFlow are used for machine learning."
        ... )
        >>> print(result.text)
    """
    client = LouieClient(model=model, **kwargs)
    return client.query(question, context)


def extract_entities(
    text: str,
    model: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Extract entities from text.
    
    Args:
        text: Text to analyze
        model: Model name for inference
        **kwargs: Additional arguments
        
    Returns:
        List of entity dicts
        
    Example:
        >>> from llamatelemetry.louie import extract_entities
        >>> 
        >>> entities = extract_entities("Python is a programming language.")
        >>> print(entities)
        [{"id": "Python", "type": "language", "properties": {}}]
    """
    client = LouieClient(model=model, **kwargs)
    result = client.extract(text)
    return result.entities


def extract_relationships(
    text: str,
    model: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Extract relationships from text.
    
    Args:
        text: Text to analyze
        model: Model name for inference
        **kwargs: Additional arguments
        
    Returns:
        List of relationship dicts
        
    Example:
        >>> from llamatelemetry.louie import extract_relationships
        >>> 
        >>> rels = extract_relationships("Python is used for AI development.")
        >>> print(rels)
        [{"source": "Python", "target": "AI", "type": "used_for"}]
    """
    client = LouieClient(model=model, **kwargs)
    result = client.extract(text)
    return result.relationships
