"""Test state definitions for Pattern-Aware RAG testing.

These states represent the concept-relationship frame that must be maintained
throughout the RAG process.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from datetime import datetime

@dataclass
class ConceptNode:
    """Represents a concept in the graph state."""
    id: str
    content: str
    type: str
    confidence: float
    emergence_time: datetime
    properties: Dict[str, any]

@dataclass
class ConceptRelation:
    """Represents a relationship between concepts."""
    source_id: str
    target_id: str
    type: str
    strength: float
    properties: Dict[str, any]

@dataclass
class PatternState:
    """Represents a pattern's current state."""
    pattern_id: str
    concepts: Set[str]
    relations: List[ConceptRelation]
    coherence: float
    stability: float
    emergence_stage: str

@dataclass
class GraphStateSnapshot:
    """Complete snapshot of graph state at a point in time."""
    id: str
    timestamp: datetime
    concepts: Dict[str, ConceptNode]
    relations: List[ConceptRelation]
    patterns: Dict[str, PatternState]
    metrics: Dict[str, float]
    temporal_context: Dict[str, any]

def create_test_graph_state() -> GraphStateSnapshot:
    """Create a test graph state with meaningful concept-relationship structure."""
    now = datetime.now()
    
    # Create test concepts
    concepts = {
        "c1": ConceptNode(
            id="c1",
            content="climate change",
            type="environmental_concept",
            confidence=0.95,
            emergence_time=now,
            properties={"domain": "climate"}
        ),
        "c2": ConceptNode(
            id="c2",
            content="global warming",
            type="environmental_concept",
            confidence=0.92,
            emergence_time=now,
            properties={"domain": "climate"}
        ),
        "c3": ConceptNode(
            id="c3",
            content="greenhouse gases",
            type="environmental_factor",
            confidence=0.88,
            emergence_time=now,
            properties={"domain": "climate"}
        )
    }
    
    # Create meaningful relationships
    relations = [
        ConceptRelation(
            source_id="c1",
            target_id="c2",
            type="strongly_related",
            strength=0.95,
            properties={"bidirectional": True}
        ),
        ConceptRelation(
            source_id="c2",
            target_id="c3",
            type="causal",
            strength=0.85,
            properties={"direction": "forward"}
        ),
        ConceptRelation(
            source_id="c3",
            target_id="c1",
            type="contributory",
            strength=0.82,
            properties={"impact": "high"}
        )
    ]
    
    # Create pattern states
    patterns = {
        "p1": PatternState(
            pattern_id="p1",
            concepts={"c1", "c2", "c3"},
            relations=relations,
            coherence=0.88,
            stability=0.92,
            emergence_stage="stable"
        )
    }
    
    return GraphStateSnapshot(
        id="test-state-1",
        timestamp=now,
        concepts=concepts,
        relations=relations,
        patterns=patterns,
        metrics={
            "coherence": 0.88,
            "stability": 0.92,
            "relationship_strength": 0.87
        },
        temporal_context={
            "stage": "evolving",
            "stability": 0.92,
            "recent_changes": ["pattern_stabilized"]
        }
    )
