"""
Core relationship model for the Adaptive Core system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class Relationship:
    """
    Represents a relationship between two patterns in the system.
    """
    id: str
    source_id: str
    target_id: str
    type: str
    strength: float = 0.0
    bidirectional: bool = False
    phase_locked: bool = False
    coherence_impact: float = 0.0
    
    # Dynamic properties
    properties: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Context tracking
    temporal_context: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    spatial_context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    state: str = "ACTIVE"
    version: str = "1.0"

    def update_metrics(self, new_metrics: Dict[str, float]) -> None:
        """Update relationship metrics"""
        self.metrics.update(new_metrics)
        self.last_modified = datetime.now().isoformat()

    def update_strength(self, new_strength: float) -> None:
        """Update relationship strength"""
        self.strength = new_strength
        self.last_modified = datetime.now().isoformat()

    def update_phase_lock(self, is_locked: bool) -> None:
        """Update phase lock status"""
        self.phase_locked = is_locked
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
        """Convert relationship to dictionary representation"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "strength": self.strength,
            "bidirectional": self.bidirectional,
            "phase_locked": self.phase_locked,
            "coherence_impact": self.coherence_impact,
            "properties": self.properties,
            "metrics": self.metrics,
            "temporal_context": self.temporal_context,
            "spatial_context": self.spatial_context,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "state": self.state,
            "version": self.version
        }
