"""
Test state models for pattern-aware RAG.
These models are used for testing and development.
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

@dataclass
class GraphStateSnapshot:
    """Snapshot of graph state at a point in time."""
    id: str
    nodes: List[ConceptNode]
    relations: List[ConceptRelation]
    patterns: List[PatternState]
    timestamp: datetime
    version: int = 1
    
    def is_graph_ready(self) -> bool:
        """Check if state is ready for graph operations."""
        return bool(self.nodes and self.patterns)

    def get_provenance(self) -> Dict[str, Any]:
        """Get provenance information for this state."""
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
