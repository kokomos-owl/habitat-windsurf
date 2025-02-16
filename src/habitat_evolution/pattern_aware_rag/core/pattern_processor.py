"""
Pattern processor for sequential foundation of pattern-aware RAG.
"""
from typing import Dict, Optional
from datetime import datetime
from ..state.test_states import PatternState, GraphStateSnapshot
from ..adaptive_core.id.adaptive_id import AdaptiveID

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
        return AdaptiveID(
            id=f"pattern_{base_concept}_{pattern.id[-8:]}",
            base_concept=base_concept,
            timestamp=pattern.timestamp
        )
    
    async def prepare_graph_state(self, pattern: PatternState, 
                                adaptive_id: Optional[AdaptiveID] = None) -> GraphStateSnapshot:
        """Prepare pattern for graph integration."""
        if not adaptive_id:
            raise ValueError("Adaptive ID required for graph state preparation")
            
        # Create graph state with pattern
        state = GraphStateSnapshot(
            id=f"state_{adaptive_id.id}",
            nodes=[],
            relations=[],
            patterns=[pattern],
            timestamp=pattern.timestamp
        )
        return state
