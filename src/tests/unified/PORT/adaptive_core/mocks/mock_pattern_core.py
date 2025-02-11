"""Mock implementation of PatternCore for testing."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from .mock_adaptive_id import MockAdaptiveID

@dataclass
class PatternEvidence:
    """Evidence supporting a pattern."""
    source: str
    confidence: float
    timestamp: str = datetime.utcnow().isoformat()
    coherence_impact: float = 0.0
    bidirectional_strength: float = 0.0

class MockPatternCore:
    """Mock implementation of PatternCore for testing."""
    
    def __init__(self):
        """Initialize mock pattern core."""
        self.patterns = {}
        self.evidence = {}
        self.adaptive_id = MockAdaptiveID()
        self.coherence_threshold = 0.7
        self.evolution_history = []
        
    def register_pattern(self, pattern_id: str, pattern_data: Dict[str, Any]) -> str:
        """Register a new pattern."""
        self.patterns[pattern_id] = pattern_data
        
        # Track pattern registration in adaptive ID
        self.adaptive_id.update_state({
            "pattern_count": len(self.patterns),
            "evolution_type": "pattern_registration",
            "patterns_affected": [pattern_id]
        })
        
        return pattern_id
        
    def add_evidence(self, pattern_id: str, evidence: PatternEvidence) -> None:
        """Add evidence to a pattern."""
        if pattern_id not in self.evidence:
            self.evidence[pattern_id] = []
        
        self.evidence[pattern_id].append(evidence)
        
        # Track bidirectional influence
        if evidence.bidirectional_strength > 0:
            self.adaptive_id.record_bidirectional_influence(
                source=evidence.source,
                target=pattern_id,
                strength=evidence.bidirectional_strength
            )
        
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get pattern by ID."""
        return self.patterns.get(pattern_id)
        
    def get_evidence(self, pattern_id: str) -> List[PatternEvidence]:
        """Get evidence for a pattern."""
        return self.evidence.get(pattern_id, [])
        
    def calculate_confidence(self, pattern_id: str) -> float:
        """Calculate confidence score for a pattern."""
        evidence_list = self.evidence.get(pattern_id, [])
        if not evidence_list:
            return 0.0
            
        # Calculate weighted confidence based on evidence
        total_weight = 0.0
        weighted_sum = 0.0
        
        for evidence in evidence_list:
            # Weight more recent evidence higher
            recency_weight = 1.0
            if evidence.coherence_impact > 0:
                recency_weight *= (1.0 + evidence.coherence_impact)
                
            weighted_sum += evidence.confidence * recency_weight
            total_weight += recency_weight
            
        confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return confidence
        
    def evolve_pattern(self, pattern_id: str, new_data: Dict[str, Any]) -> None:
        """Evolve a pattern with new data."""
        if pattern_id not in self.patterns:
            return
            
        # Calculate current coherence
        current_confidence = self.calculate_confidence(pattern_id)
        
        # Update pattern data
        self.patterns[pattern_id].update(new_data)
        
        # Track evolution
        evolution_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "pattern_id": pattern_id,
            "confidence_before": current_confidence,
            "data_update": new_data
        }
        
        # Recalculate confidence after update
        new_confidence = self.calculate_confidence(pattern_id)
        evolution_record["confidence_after"] = new_confidence
        
        # Update adaptive ID state
        self.adaptive_id.update_state({
            "confidence": new_confidence,
            "evolution_type": "pattern_evolution",
            "patterns_affected": [pattern_id],
            "coherence_score": max(current_confidence, new_confidence)
        })
        
        # Store evolution record
        self.evolution_history.append(evolution_record)
        
    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get pattern evolution history."""
        return self.evolution_history
        
    def get_coherence_metrics(self) -> Dict[str, Any]:
        """Get coherence-related metrics."""
        pattern_confidences = {
            pid: self.calculate_confidence(pid)
            for pid in self.patterns
        }
        
        return {
            "average_confidence": sum(pattern_confidences.values()) / len(pattern_confidences) if pattern_confidences else 0.0,
            "pattern_confidences": pattern_confidences,
            "coherence_threshold": self.coherence_threshold,
            "patterns_above_threshold": sum(1 for conf in pattern_confidences.values() if conf >= self.coherence_threshold),
            "total_patterns": len(self.patterns),
            "total_evidence": sum(len(evidence_list) for evidence_list in self.evidence.values()),
            "adaptive_state": self.adaptive_id.get_current_state()
        }
