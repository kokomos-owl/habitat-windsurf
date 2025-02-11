"""
Core pattern-aware RAG module ported from habitat_poc.
Integrates natural pattern awareness into RAG behavior.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FlowState:
    """Flow state for RAG integration."""
    energy: float
    coherence: float
    patterns: List[str]
    context: Dict[str, Any]

@dataclass
class PatternContext:
    """Pattern context for RAG enhancement."""
    active_patterns: List[str]
    pattern_weights: Dict[str, float]
    temporal_context: Dict[str, Any]

class PatternAwareRAG:
    """RAG with natural pattern awareness."""
    
    def __init__(self):
        self.flow_state = FlowState(
            energy=0.0,
            coherence=0.0,
            patterns=[],
            context={}
        )
        self.pattern_context = PatternContext(
            active_patterns=[],
            pattern_weights={},
            temporal_context={}
        )
        self.history: List[Dict[str, Any]] = []
    
    def enhance_query(
        self,
        query: str,
        patterns: List[str],
        flow_state: FlowState
    ) -> Dict[str, Any]:
        """Enhance query with natural pattern influence."""
        # Calculate pattern weights naturally
        weights = {
            pattern: self._calculate_pattern_weight(
                pattern,
                flow_state.patterns,
                flow_state.energy
            )
            for pattern in patterns
        }
        
        # Create enhanced query naturally
        enhanced_query = {
            'original': query,
            'patterns': patterns,
            'weights': weights,
            'context': {
                'flow_energy': flow_state.energy,
                'flow_coherence': flow_state.coherence,
                'temporal': flow_state.context.get('timeframe')
            }
        }
        
        # Record enhancement state
        state = {
            'query': query,
            'enhanced': enhanced_query,
            'flow_state': flow_state,
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(state)
        
        return enhanced_query
    
    def apply_pattern_context(
        self,
        query: str,
        patterns: List[str]
    ) -> Dict[str, Any]:
        """Apply natural pattern context."""
        # Update active patterns naturally
        self.pattern_context.active_patterns = patterns
        
        # Calculate weights for active patterns
        for pattern in patterns:
            if pattern not in self.pattern_context.pattern_weights:
                self.pattern_context.pattern_weights[pattern] = self._calculate_initial_weight(pattern)
        
        return {
            'query': query,
            'active_patterns': patterns,
            'weights': self.pattern_context.pattern_weights,
            'context': self.pattern_context.temporal_context
        }
    
    def _calculate_pattern_weight(
        self,
        pattern: str,
        flow_patterns: List[str],
        flow_energy: float
    ) -> float:
        """Calculate natural pattern weight."""
        base_weight = self.pattern_context.pattern_weights.get(pattern, 0.5)
        flow_bonus = 0.2 if pattern in flow_patterns else 0.0
        energy_factor = flow_energy * 0.3
        
        return min(1.0, base_weight + flow_bonus + energy_factor)
    
    def _calculate_initial_weight(self, pattern: str) -> float:
        """Calculate natural initial pattern weight."""
        temporal_bonus = 0.2 if 'temporal' in pattern.lower() else 0.0
        adaptation_bonus = 0.15 if 'adaptation' in pattern.lower() else 0.0
        
        return 0.5 + temporal_bonus + adaptation_bonus
