"""
Core pattern model for the Adaptive Core system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class Pattern:
    """
    Represents a pattern in the system with its associated properties and metrics.
    """
    id: str
    base_concept: str
    creator_id: str
    weight: float = 1.0
    confidence: float = 1.0
    uncertainty: float = 0.0
    coherence: float = 0.0
    phase_stability: float = 0.0
    signal_strength: float = 0.0
    
    # Dynamic properties
    properties: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)
    
    # Context tracking
    temporal_context: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    spatial_context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    state: str = "EMERGING"
    version: str = "1.0"

    def update_metrics(self, new_metrics: Dict[str, float]) -> None:
        """Update pattern metrics"""
        self.metrics.update(new_metrics)
        self.last_modified = datetime.now().isoformat()

    def add_relationship(self, relationship_id: str) -> None:
        """Add a relationship to this pattern"""
        if relationship_id not in self.relationships:
            self.relationships.append(relationship_id)
            self.last_modified = datetime.now().isoformat()

    def remove_relationship(self, relationship_id: str) -> None:
        """Remove a relationship from this pattern"""
        if relationship_id in self.relationships:
            self.relationships.remove(relationship_id)
            self.last_modified = datetime.now().isoformat()

    def update_temporal_context(self, key: str, value: Any) -> None:
        """Update temporal context"""
        if key not in self.temporal_context:
            self.temporal_context[key] = {}
        
        timestamp = datetime.now().isoformat()
        self.temporal_context[key][timestamp] = value
        self.last_modified = timestamp

    def update_spatial_context(self, updates: Dict[str, Any]) -> None:
        """Update spatial context"""
        self.spatial_context.update(updates)
        self.last_modified = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary representation"""
        return {
            "id": self.id,
            "base_concept": self.base_concept,
            "creator_id": self.creator_id,
            "weight": self.weight,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "coherence": self.coherence,
            "phase_stability": self.phase_stability,
            "signal_strength": self.signal_strength,
            "properties": self.properties,
            "metrics": self.metrics,
            "relationships": self.relationships,
            "temporal_context": self.temporal_context,
            "spatial_context": self.spatial_context,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "state": self.state,
            "version": self.version
        }
