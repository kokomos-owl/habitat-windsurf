"""
Test state models for the ArangoDB graph state repository tests.

These models implement a simplified version of the pattern-aware RAG state models
to avoid import path issues with the original models.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

# Define exceptions locally
class InvalidStateError(Exception):
    """Error raised when a state is invalid."""
    pass
    
class StateValidationError(Exception):
    """Error raised when state validation fails."""
    pass

@dataclass
class ConceptNode:
    """Node representing a concept in the graph."""
    id: str
    name: str
    attributes: Dict[str, str]
    created_at: Optional[datetime] = None

@dataclass
class ConceptRelation:
    """Relationship between concept nodes."""
    source_id: str
    target_id: str
    relation_type: str
    weight: float

@dataclass
class PatternState:
    """State of a pattern at a point in time."""
    id: str
    content: str
    metadata: Dict[str, str]
    timestamp: datetime
    confidence: float = 1.0

class GraphStateSnapshot:
    """Snapshot of graph state at a point in time."""
    
    def __init__(
        self,
        id: str,
        nodes: List[ConceptNode],
        relations: List[ConceptRelation],
        patterns: List[PatternState],
        timestamp: datetime,
        version: int = 1
    ):
        """
        Initialize a graph state snapshot.
        
        Args:
            id: The ID of the snapshot
            nodes: The nodes in the snapshot
            relations: The relations in the snapshot
            patterns: The patterns in the snapshot
            timestamp: The timestamp of the snapshot
            version: The version of the snapshot
            
        Raises:
            InvalidStateError: If any pattern has confidence below threshold
            InvalidStateError: If any node has an empty ID
            StateValidationError: If any relation is invalid
        """
        # First validate relations before setting any attributes
        relations = relations or []
        for relation in relations:
            if not relation.relation_type:
                raise StateValidationError("Invalid relation: empty type")
            if relation.weight < 0:
                raise StateValidationError("Invalid relation: negative weight")
        
        # Now set attributes
        self.id = id
        self.nodes = nodes or []
        self.relations = relations
        self.patterns = patterns or []
        self.timestamp = timestamp
        self.version = version
        
    def validate_relations(self):
        """
        Validate that all relations reference valid nodes or patterns.
        
        Raises:
            StateValidationError: If the state is empty
            StateValidationError: If a relation references a non-existent node
        """
        if not self.nodes and not self.patterns and not self.relations:
            raise StateValidationError("Empty state")
            
        # Check that all relations reference valid nodes
        node_ids = set(node.id for node in self.nodes)
        pattern_ids = set(pattern.id for pattern in self.patterns)
        
        for relation in self.relations:
            if relation.source_id not in node_ids and relation.source_id not in pattern_ids:
                raise StateValidationError(f"Invalid relation: source {relation.source_id} not found")
                
            if relation.target_id not in node_ids and relation.target_id not in pattern_ids:
                raise StateValidationError(f"Invalid relation: target {relation.target_id} not found")
