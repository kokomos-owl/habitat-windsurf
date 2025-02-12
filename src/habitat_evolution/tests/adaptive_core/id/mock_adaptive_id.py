"""
Mock implementation of AdaptiveID for testing purposes.
Provides core functionality without external dependencies.
"""

import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

@dataclass
class AdaptiveState:
    """State information for adaptive processing."""
    version: int = 1
    confidence: float = 0.8
    adaptations: int = 0
    coherence_score: float = 0.7
    pattern_count: int = 0
    last_evolution: Optional[str] = None

class MockAdaptiveID:
    """Mock implementation of AdaptiveID for testing."""
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize with optional data."""
        self.id = str(uuid.uuid4())
        self.data = data or {}
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
        self.state = AdaptiveState()
        self.state_history = []
        self.pattern_evolution = {}
        self.coherence_history = []
        self.bidirectional_influences = []
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state of the adaptive ID."""
        return {
            "id": self.id,
            "data": self.data,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.state.version,
            "confidence": self.state.confidence,
            "adaptations": self.state.adaptations,
            "coherence_score": self.state.coherence_score,
            "pattern_count": self.state.pattern_count,
            "last_evolution": self.state.last_evolution
        }
    
    def update(self, new_data: Dict[str, Any]) -> None:
        """Update adaptive ID with new data."""
        self.data.update(new_data)
        self.updated_at = datetime.utcnow().isoformat()
        self.state.version += 1
        self.state.adaptations += 1
        self.state_history.append(self.get_current_state())
    
    def record_pattern_evolution(self, pattern_id: str, evolution_data: Dict[str, Any]) -> None:
        """Record pattern evolution data."""
        if pattern_id not in self.pattern_evolution:
            self.pattern_evolution[pattern_id] = []
        
        evolution_data["timestamp"] = datetime.utcnow().isoformat()
        self.pattern_evolution[pattern_id].append(evolution_data)
        self.state.pattern_count += 1
        self.state.last_evolution = evolution_data["timestamp"]
    
    def update_coherence(self, coherence_value: float, source: str) -> None:
        """Update coherence score."""
        self.state.coherence_score = coherence_value
        self.coherence_history.append({
            "value": coherence_value,
            "source": source,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def add_bidirectional_influence(self, influence_data: Dict[str, Any]) -> None:
        """Add bidirectional influence data."""
        influence_data["timestamp"] = datetime.utcnow().isoformat()
        self.bidirectional_influences.append(influence_data)
    
    def get_evolution_history(self, pattern_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get evolution history for a specific pattern or all patterns."""
        if pattern_id:
            return {pattern_id: self.pattern_evolution.get(pattern_id, [])}
        return self.pattern_evolution
    
    def get_coherence_history(self) -> List[Dict[str, Any]]:
        """Get coherence history."""
        return self.coherence_history
    
    def get_bidirectional_influences(self) -> List[Dict[str, Any]]:
        """Get bidirectional influence history."""
        return self.bidirectional_influences
