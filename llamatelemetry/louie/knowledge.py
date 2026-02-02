"""
Knowledge Extraction and Graph Building

Extract structured knowledge from text using LLM inference.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class EntityType(Enum):
    """Standard entity types for knowledge extraction."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    LANGUAGE = "language"
    PRODUCT = "product"
    EVENT = "event"
    DATE = "date"
    NUMBER = "number"
    OTHER = "other"


class RelationType(Enum):
    """Standard relationship types for knowledge graphs."""
    USES = "uses"
    CREATES = "creates"
    BELONGS_TO = "belongs_to"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    WORKS_FOR = "works_for"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    EXTENDS = "extends"
    CONTAINS = "contains"
    OTHER = "other"


@dataclass
class Entity:
    """Extracted entity."""
    id: str
    type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "properties": self.properties
        }


@dataclass
class Relationship:
    """Extracted relationship."""
    source: str
    target: str
    type: RelationType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "weight": self.weight,
            "properties": self.properties
        }


@dataclass
class KnowledgeGraph:
    """Knowledge graph with entities and relationships."""
    entities: List[Entity]
    relationships: List[Relationship]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "metadata": self.metadata
        }
    
    def to_graphistry(self, gpu_id: int = 1):
        """
        Convert to Graphistry graph.
        
        Args:
            gpu_id: GPU for graph operations
            
        Returns:
            Graphistry graph object
        """
        from llamatelemetry.graphistry import GraphWorkload
        
        workload = GraphWorkload(gpu_id=gpu_id)
        return workload.create_knowledge_graph(
            [e.to_dict() for e in self.entities],
            [r.to_dict() for r in self.relationships]
        )


class KnowledgeExtractor:
    """
    Extract knowledge graphs from text using LLM.
    
    Example:
        >>> from llamatelemetry.louie import KnowledgeExtractor
        >>> 
        >>> extractor = KnowledgeExtractor(model="gemma-3-1b-Q4_K_M")
        >>> kg = extractor.extract('''
        ...     Python is a programming language created by Guido van Rossum.
        ...     It is widely used for machine learning with libraries like TensorFlow.
        ... ''')
        >>> 
        >>> print(kg.entities)
        >>> print(kg.relationships)
        >>> 
        >>> # Visualize with Graphistry
        >>> g = kg.to_graphistry()
        >>> g.plot()
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        server_url: Optional[str] = None,
        entity_types: Optional[List[EntityType]] = None,
        relationship_types: Optional[List[RelationType]] = None,
    ):
        """
        Initialize knowledge extractor.
        
        Args:
            model: Model name for inference
            server_url: llama-server URL if using server mode
            entity_types: Entity types to extract (None = all)
            relationship_types: Relationship types to extract (None = all)
        """
        self.model = model or "gemma-3-1b-Q4_K_M"
        self.server_url = server_url
        self.entity_types = entity_types or list(EntityType)
        self.relationship_types = relationship_types or list(RelationType)
        
        self._engine = None
        self._client = None
    
    def _init_llm(self) -> None:
        """Initialize LLM backend."""
        if self._engine is not None or self._client is not None:
            return
        
        if self.server_url:
            from llamatelemetry.api import LlamaCppClient
            self._client = LlamaCppClient(self.server_url)
        else:
            try:
                import llamatelemetry
                self._engine = llamatelemetry.InferenceEngine()
                self._engine.load_model(self.model, silent=True)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize LLM: {e}")
    
    def extract(
        self,
        text: str,
        max_entities: int = 50,
        max_relationships: int = 100,
        max_tokens: int = 2000,
    ) -> KnowledgeGraph:
        """
        Extract knowledge graph from text.
        
        Args:
            text: Text to analyze
            max_entities: Maximum entities to extract
            max_relationships: Maximum relationships to extract
            max_tokens: Maximum response tokens
            
        Returns:
            KnowledgeGraph with entities and relationships
        """
        self._init_llm()
        
        # Build extraction prompt
        prompt = self._build_prompt(text, max_entities, max_relationships)
        
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
        
        # Parse response into knowledge graph
        return self._parse_response(raw_response, text)
    
    def _build_prompt(
        self,
        text: str,
        max_entities: int,
        max_relationships: int,
    ) -> str:
        """Build extraction prompt."""
        entity_types_str = ", ".join([t.value for t in self.entity_types])
        rel_types_str = ", ".join([t.value for t in self.relationship_types])
        
        prompt = f"""Extract a knowledge graph from the following text.

Entity types: {entity_types_str}
Relationship types: {rel_types_str}

Rules:
- Extract up to {max_entities} entities
- Extract up to {max_relationships} relationships
- Each entity must have a unique id, type, and optional properties
- Each relationship must have source, target, type, and optional weight (0-1)

Output format (JSON only, no additional text):
{{
  "entities": [
    {{"id": "entity_name", "type": "entity_type", "properties": {{}}}}
  ],
  "relationships": [
    {{"source": "entity1", "target": "entity2", "type": "relationship_type", "weight": 1.0}}
  ]
}}

Text to analyze:
{text}

JSON output:"""
        
        return prompt
    
    def _parse_response(self, response: str, original_text: str) -> KnowledgeGraph:
        """Parse LLM response into KnowledgeGraph."""
        import json
        import re
        
        entities = []
        relationships = []
        
        try:
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                # Parse entities
                for e in data.get("entities", []):
                    try:
                        entity_type = EntityType(e.get("type", "other"))
                    except ValueError:
                        entity_type = EntityType.OTHER
                    
                    entities.append(Entity(
                        id=e["id"],
                        type=entity_type,
                        properties=e.get("properties", {})
                    ))
                
                # Parse relationships
                for r in data.get("relationships", []):
                    try:
                        rel_type = RelationType(r.get("type", "related_to"))
                    except ValueError:
                        rel_type = RelationType.OTHER
                    
                    relationships.append(Relationship(
                        source=r["source"],
                        target=r["target"],
                        type=rel_type,
                        weight=r.get("weight", 1.0),
                        properties=r.get("properties", {})
                    ))
        
        except (json.JSONDecodeError, KeyError) as e:
            # Return empty graph on parse error
            pass
        
        return KnowledgeGraph(
            entities=entities,
            relationships=relationships,
            metadata={
                "source_text_length": len(original_text),
                "entity_count": len(entities),
                "relationship_count": len(relationships),
            }
        )


def build_knowledge_graph(
    text: str,
    model: Optional[str] = None,
    **kwargs,
) -> KnowledgeGraph:
    """
    Quick helper to build knowledge graph from text.
    
    Args:
        text: Text to analyze
        model: Model name for inference
        **kwargs: Additional KnowledgeExtractor arguments
        
    Returns:
        KnowledgeGraph
        
    Example:
        >>> from llamatelemetry.louie import build_knowledge_graph
        >>> 
        >>> kg = build_knowledge_graph('''
        ...     Python is used for data science.
        ...     TensorFlow and PyTorch are popular ML frameworks.
        ... ''')
        >>> 
        >>> # Visualize
        >>> g = kg.to_graphistry()
        >>> g.plot()
    """
    extractor = KnowledgeExtractor(model=model, **kwargs)
    return extractor.extract(text)
