"""
Pattern processor for sequential foundation of pattern-aware RAG.
"""
from typing import Dict, Optional, List, Any
from datetime import datetime
from dataclasses import dataclass
from ..state.test_states import PatternState, GraphStateSnapshot, ConceptNode
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

@dataclass
class EventCoordinator:
    """Coordinates pattern-aware events and state transitions."""
    events: List[Any] = None
    
    def __post_init__(self):
        self.events = [] if self.events is None else self.events
    
    def register_event(self, event: Any) -> None:
        """Register a new event."""
        self.events.append(event)
    
    def get_events(self) -> List[Any]:
        """Get all registered events."""
        return self.events

class PatternProcessor:
    """Processes patterns through the sequential foundation stages."""
    
    async def extract_pattern(self, document: Dict) -> PatternState:
        """Extract pattern from document with provenance tracking."""
        pattern = PatternState(
            id="temp_" + str(hash(document["content"])),
            content=document["content"],
            metadata=document["metadata"],
            timestamp=datetime.fromisoformat(document["metadata"]["timestamp"])
        )
        return pattern
    
    async def assign_adaptive_id(self, pattern: PatternState) -> AdaptiveID:
        """Assign Adaptive ID to pattern."""
        # Create deterministic base concept from pattern content
        base_concept = pattern.content[:10].lower().replace(" ", "_")
        # Create an AdaptiveID with the pattern's metadata
        return AdaptiveID(
            base_concept=base_concept,
            creator_id=pattern.metadata.get("source", "system"),
            weight=1.0,
            confidence=1.0,
            uncertainty=0.0
        )
    
    async def prepare_graph_state(self, pattern: PatternState, 
                                adaptive_id: Optional[AdaptiveID] = None) -> GraphStateSnapshot:
        """Prepare pattern for graph integration."""
        # Create concept node from pattern
        node = ConceptNode(
            id=pattern.id,
            name=pattern.content[:50],  # Use first 50 chars as name
            attributes={
                "source": pattern.metadata.get("source", ""),
                "timestamp": pattern.metadata.get("timestamp", "")
            }
        )
        if not adaptive_id:
            raise ValueError("Adaptive ID required for graph state preparation")
            
        # Create graph state with node and pattern
        return GraphStateSnapshot(
            id=f"state_{pattern.id}",
            nodes=[node],
            relations=[],
            patterns=[pattern],
            timestamp=pattern.timestamp
        )
        # Create graph state with pattern
        state = GraphStateSnapshot(
            id=f"state_{adaptive_id.id}",
            nodes=[],
            relations=[],
            patterns=[pattern],
            timestamp=pattern.timestamp
        )
        return state
