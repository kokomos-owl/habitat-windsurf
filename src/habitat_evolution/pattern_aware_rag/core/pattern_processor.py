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
        
    async def prepare_graph_state(self, pattern: PatternState, adaptive_id: AdaptiveID) -> GraphStateSnapshot:
        """Prepare graph-ready state from pattern and adaptive ID.
        
        Args:
            pattern: Pattern to prepare state for
            adaptive_id: Assigned adaptive ID
            
        Returns:
            Graph-ready state
        """
        # Create concept node from pattern
        concept = ConceptNode(
            id=str(adaptive_id),
            name=pattern.content[:50],  # Use first 50 chars as name
            attributes={
                "source": pattern.metadata.get("source", ""),
                "timestamp": pattern.metadata.get("timestamp", "")
            }
        )
        
        # Create initial state
        return GraphStateSnapshot(
            id=f"state_{pattern.id}",
            nodes=[concept],
            relations=[],
            patterns=[pattern],
            timestamp=pattern.timestamp,
            version=1
        )
    


