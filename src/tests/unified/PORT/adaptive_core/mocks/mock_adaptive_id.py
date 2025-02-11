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
        """Update data and increment version."""
        self.data.update(new_data)
        self.updated_at = datetime.utcnow().isoformat()
        self.state.version += 1
    
    def update_state(self, data: Dict[str, Any]) -> AdaptiveState:
        """Update adaptive state."""
        # Store previous state in history
        self.state_history.append(self.state)
        
        # Update state
        self.state.version += 1
        self.state.adaptations += 1
        
        # Update coherence and pattern metrics
        if "coherence_score" in data:
            self.state.coherence_score = data["coherence_score"]
            self.coherence_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "score": data["coherence_score"],
                "version": self.state.version
            })
        
        if "pattern_count" in data:
            self.state.pattern_count = data["pattern_count"]
        
        if "confidence" in data:
            self.state.confidence = data["confidence"]
            
        if "evolution_type" in data:
            self.state.last_evolution = data["evolution_type"]
            self.pattern_evolution[self.state.version] = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": data["evolution_type"],
                "patterns_affected": data.get("patterns_affected", [])
            }
            
        return self.state
    
    def record_bidirectional_influence(self, source: str, target: str, strength: float) -> None:
        """Record bidirectional influence between components."""
        self.bidirectional_influences.append({
            "timestamp": datetime.utcnow().isoformat(),
            "source": source,
            "target": target,
            "strength": strength,
            "version": self.state.version
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "data": self.data,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.get_current_state(),
            "state_history": self.state_history,
            "pattern_evolution": self.pattern_evolution,
            "coherence_history": self.coherence_history,
            "bidirectional_influences": self.bidirectional_influences
        }
    
    def process(self) -> None:
        """Process adaptive ID and increment version."""
        self.state.version += 1
        self.updated_at = datetime.utcnow().isoformat()
    
    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get history of state changes."""
        return [{
            "version": state.version,
            "confidence": state.confidence,
            "adaptations": state.adaptations,
            "coherence_score": state.coherence_score,
            "pattern_count": state.pattern_count,
            "last_evolution": state.last_evolution
        } for state in self.state_history]
    
    def process_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a document and return results.
        
        Args:
            doc_id: Unique document identifier
            content: Document content to process
            metadata: Optional metadata about the document
            
        Returns:
            Dict containing processing results
        """
        # Update state with mock processing
        self.update_state({
            "coherence_score": 0.85,
            "pattern_count": len(content.split()),
            "confidence": 0.9,
            "evolution_type": "document_processing"
        })
        
        # Record mock bidirectional influence
        self.record_bidirectional_influence(
            source="document_processor",
            target="adaptive_id",
            strength=0.75
        )
        
        # Return mock processing results
        return {
            "status": "success",
            "doc_id": doc_id,
            "adaptive_id": self.id,
            "state": self.get_current_state(),
            "coherence_score": self.state.coherence_score,
            "confidence": self.state.confidence,
            "patterns_found": self.state.pattern_count,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MockAdaptiveID':
        """Create instance from dictionary."""
        instance = cls(data.get("data"))
        instance.id = data["id"]
        instance.created_at = data["created_at"]
        instance.updated_at = data["updated_at"]
        instance.state = AdaptiveState(**data["state"])
        instance.state_history = data.get("state_history", [])
        instance.pattern_evolution = data.get("pattern_evolution", {})
        instance.coherence_history = data.get("coherence_history", [])
        instance.bidirectional_influences = data.get("bidirectional_influences", [])
        return instance
