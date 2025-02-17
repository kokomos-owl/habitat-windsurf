"""
Test state models for pattern-aware RAG.

These models implement a strict validation hierarchy to ensure data integrity:

1. Semantic Validation (During Initialization):
   - Relations must have non-empty types and non-negative weights
   - Pattern confidence must be above threshold (0.5)
   - Node IDs must be non-empty

2. Structural Validation (During validate_relations):
   - State cannot be empty (no nodes, patterns, or relations)
   - Required components must exist (nodes and patterns)
   - Relations must reference valid nodes or patterns

This approach ensures that invalid data is caught as early as possible during object
creation, while structural completeness is validated when needed.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

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
    """Snapshot of graph state at a point in time.
    
    This class implements a two-level validation hierarchy:
    1. Semantic validation during initialization to catch invalid data early
    2. Structural validation via validate_relations() to ensure completeness
    
    The separation ensures that objects cannot be created with invalid data,
    while allowing flexibility in when to check structural requirements.
    """
    
    def __init__(self, id: str, nodes: List[ConceptNode], relations: List[ConceptRelation], 
                 patterns: List[PatternState], timestamp: datetime, version: int = 1):
        """Initialize a graph state snapshot.
        
        Args:
            id: Unique identifier for this state
            nodes: List of concept nodes
            relations: List of relations between nodes
            patterns: List of pattern states
            timestamp: When this state was created
            version: State version number
            
        Raises:
            InvalidStateError: If any pattern has confidence below threshold
            InvalidStateError: If any node has an empty ID
            StateValidationError: If any relation is invalid
        """
        from habitat_evolution.pattern_aware_rag.core.exceptions import InvalidStateError, StateValidationError
        
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
        
        # Check pattern confidence during initialization
        for pattern in self.patterns:
            if pattern.confidence < 0.5:  # Below threshold
                raise InvalidStateError(f"Pattern {pattern.id} has confidence below threshold: {pattern.confidence}")
        
        # Check node IDs during initialization
        for node in self.nodes:
            if not node.id:
                raise InvalidStateError("Node ID cannot be empty")
    
    def is_graph_ready(self) -> bool:
        """Check if state is ready for graph operations.
        
        Returns:
            bool: True if state is ready for graph operations, False otherwise
        """
        return bool(self.nodes and self.patterns and self.relations)

    def validate_relations(self) -> None:
        """Validate the structural completeness of the graph state.
        
        This method performs the second level of validation, focusing on structural
        completeness rather than semantic validity (which is done during initialization).
        
        The validation checks:
        1. Empty state (no nodes, patterns, or relations)
        2. Missing required components (nodes and patterns)
        3. Missing relations between components
        4. Invalid relation references
        
        This separation of concerns allows us to:
        1. Catch invalid data immediately during object creation
        2. Validate structural requirements when needed
        3. Support partial states during construction
        
        Raises:
            StateValidationError: If any structural requirement is not met
            InvalidStateError: If the state is completely empty
        """
        from habitat_evolution.pattern_aware_rag.core.exceptions import InvalidStateError, StateValidationError
        
        # First validate any existing relations
        # This must happen before empty state check, as invalid relations should never exist
        for relation in self.relations:
            if not relation.relation_type:
                raise StateValidationError("Invalid relation: empty type")
            if relation.weight < 0:
                raise StateValidationError("Invalid relation: negative weight")
        
        # Then check for empty state
        if not self.nodes and not self.patterns and not self.relations:
            raise InvalidStateError("Empty state")
        
        # Check for missing required components
        if not self.nodes:
            raise StateValidationError("Missing nodes")
        if not self.patterns:
            raise StateValidationError("Missing patterns")
        
        # Check for missing relations when we have both nodes and patterns
        if self.nodes and self.patterns and not self.relations:
            raise StateValidationError("Missing required relations between nodes and patterns")
        
        # Finally validate relation references
        if self.relations:
            valid_ids = {p.id for p in self.patterns} | {n.id for n in self.nodes}
            for relation in self.relations:
                if relation.source_id not in valid_ids:
                    raise StateValidationError(f"Invalid relation: source {relation.source_id} not found")
                if relation.target_id not in valid_ids:
                    raise StateValidationError(f"Invalid relation: target {relation.target_id} not found")

    def get_provenance(self) -> Dict[str, Any]:
        """Get provenance information for this state."""
        # Ensure timezone-aware datetime
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.astimezone()
        return {
            "state_id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "node_count": len(self.nodes),
            "relation_count": len(self.relations),
            "pattern_count": len(self.patterns),
            "patterns": [{
                "id": p.id,
                "source": p.metadata.get("source", ""),
                "timestamp": p.metadata.get("timestamp", "")
            } for p in self.patterns]
        }
