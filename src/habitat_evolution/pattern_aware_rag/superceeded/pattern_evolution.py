"""Pattern evolution tracking for climate risk analysis."""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class PatternState:
    """Current state of a pattern in the evolution process."""
    pattern: str
    confidence: float
    temporal_context: Dict[str, any]
    related_patterns: Set[str] = field(default_factory=set)
    flow_velocity: float = 0.0
    flow_direction: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class EvolutionMetrics:
    """Metrics tracking pattern evolution over time."""
    coherence: float = 0.0
    stability: float = 0.0
    emergence_rate: float = 0.0
    cross_pattern_flow: float = 0.0

class PatternEvolutionTracker:
    """Tracks and analyzes pattern evolution in climate risk documents."""
    
    def __init__(self, state_file: Optional[str] = None):
        self.patterns: Dict[str, PatternState] = {}
        self.state_file = state_file or "pattern_evolution.json"
        self.load_state()
        
    def load_state(self):
        """Load pattern evolution state from file."""
        try:
            path = Path(self.state_file)
            if path.exists():
                with path.open() as f:
                    data = json.load(f)
                    for p_data in data.get('patterns', []):
                        pattern = PatternState(
                            pattern=p_data['pattern'],
                            confidence=p_data['confidence'],
                            temporal_context=p_data['temporal_context'],
                            related_patterns=set(p_data['related_patterns']),
                            flow_velocity=p_data['flow_velocity'],
                            flow_direction=p_data['flow_direction'],
                            last_updated=datetime.fromisoformat(p_data['last_updated'])
                        )
                        self.patterns[pattern.pattern] = pattern
        except Exception as e:
            logger.warning(f"Could not load pattern state: {e}")
    
    def save_state(self):
        """Save pattern evolution state to file."""
        try:
            data = {
                'patterns': [{
                    'pattern': p.pattern,
                    'confidence': p.confidence,
                    'temporal_context': p.temporal_context,
                    'related_patterns': list(p.related_patterns),
                    'flow_velocity': p.flow_velocity,
                    'flow_direction': p.flow_direction,
                    'last_updated': p.last_updated.isoformat()
                } for p in self.patterns.values()],
                'last_updated': datetime.utcnow().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save pattern state: {e}")
    
    def calculate_similarity(self, pattern1: str, pattern2: str) -> float:
        """Calculate semantic similarity between two patterns."""
        # TODO: Implement more sophisticated similarity (e.g. embeddings)
        words1 = set(pattern1.lower().split())
        words2 = set(pattern2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    def update_flow_dynamics(self, pattern: str, 
                           related_patterns: List[str], 
                           scores: List[float]):
        """Update flow dynamics for a pattern based on relationships."""
        if not scores:
            return
            
        state = self.patterns.get(pattern)
        if not state:
            return
            
        # Update velocity based on relationship strengths
        state.flow_velocity = sum(scores) / len(scores)
        
        # Update direction based on strongest relationships
        strongest_related = sorted(zip(related_patterns, scores), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:3]
        if strongest_related:
            related_patterns = [r[0] for r in strongest_related]
            state.flow_direction = sum(r[1] for r in strongest_related) / len(strongest_related)
            state.related_patterns.update(related_patterns)
    
    def observe_pattern(self, pattern: str, 
                       confidence: float,
                       temporal_context: Dict[str, any],
                       related_patterns: Optional[List[str]] = None) -> EvolutionMetrics:
        """Observe a pattern and update its evolution state."""
        # Get or create pattern state
        state = self.patterns.get(pattern)
        if not state:
            state = PatternState(
                pattern=pattern,
                confidence=confidence,
                temporal_context=temporal_context
            )
            self.patterns[pattern] = state
        
        # Calculate relationship scores
        if related_patterns:
            scores = [
                self.calculate_similarity(pattern, rel)
                for rel in related_patterns
            ]
            self.update_flow_dynamics(pattern, related_patterns, scores)
        
        # Update pattern state
        state.confidence = confidence
        state.temporal_context = temporal_context
        state.last_updated = datetime.utcnow()
        
        # Calculate evolution metrics
        metrics = EvolutionMetrics()
        
        # Coherence based on relationship strength
        if related_patterns:
            metrics.coherence = sum(scores) / len(scores)
        
        # Stability based on confidence consistency
        metrics.stability = min(1.0, state.confidence * 0.8 + 0.2)
        
        # Emergence rate based on flow dynamics
        metrics.emergence_rate = state.flow_velocity * state.flow_direction
        
        # Cross-pattern flow based on relationship network
        if state.related_patterns:
            cross_flows = []
            for rel in state.related_patterns:
                if rel in self.patterns:
                    rel_state = self.patterns[rel]
                    if pattern in rel_state.related_patterns:
                        cross_flows.append(
                            (state.flow_velocity + rel_state.flow_velocity) / 2
                        )
            if cross_flows:
                metrics.cross_pattern_flow = sum(cross_flows) / len(cross_flows)
        
        self.save_state()
        return metrics
    
    def get_evolved_patterns(self, min_confidence: float = 0.5, 
                           min_flow: float = 0.3) -> List[Tuple[str, float]]:
        """Get patterns that show significant evolution."""
        evolved = []
        for pattern, state in self.patterns.items():
            if (state.confidence >= min_confidence and 
                state.flow_velocity >= min_flow):
                evolved.append((pattern, state.confidence))
        return sorted(evolved, key=lambda x: x[1], reverse=True)
    
    def get_pattern_relationships(self, pattern: str, 
                                min_strength: float = 0.3) -> List[Tuple[str, float]]:
        """Get related patterns with relationship strengths."""
        state = self.patterns.get(pattern)
        if not state:
            return []
            
        relationships = []
        for rel in state.related_patterns:
            strength = self.calculate_similarity(pattern, rel)
            if strength >= min_strength:
                relationships.append((rel, strength))
        return sorted(relationships, key=lambda x: x[1], reverse=True)
