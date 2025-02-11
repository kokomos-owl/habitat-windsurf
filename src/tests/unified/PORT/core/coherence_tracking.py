"""
Core coherence tracking module ported from habitat_poc.
Preserves natural coherence emergence and tracking.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CoherenceState:
    """Natural coherence state."""
    energy: float = 0.0
    velocity: float = 0.0
    direction: float = 0.0
    propensity: float = 0.0
    coherence: float = 0.0
    patterns: List[str] = None

    def __post_init__(self):
        self.patterns = [] if self.patterns is None else self.patterns

class CoherenceTracker:
    """Tracks naturally emerging coherence."""
    
    def __init__(self):
        self.history: List[CoherenceState] = []
        self.thresholds = {
            'energy': 0.3,
            'coherence': 0.4,
            'velocity': 0.2,
            'propensity': 0.25
        }
    
    def track_coherence(self, state: CoherenceState) -> Dict[str, Any]:
        """Track coherence as it emerges naturally."""
        # Calculate coherence from state
        coherence = self._calculate_coherence(state)
        self.history.append(coherence)
        
        # Adapt thresholds naturally
        self.thresholds = self._adapt_thresholds(
            coherence,
            self.history
        )
        
        return {
            'coherence': coherence,
            'thresholds': self.thresholds,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_coherence(self, state: CoherenceState) -> float:
        """Calculate coherence naturally from state."""
        # Base coherence from energy and velocity
        base_coherence = (
            state.energy * 0.4 +
            abs(state.velocity) * 0.3 +
            state.propensity * 0.3
        )
        
        # Direction influences coherence
        direction_factor = 1.0 + (0.5 * state.direction if state.direction > 0 else 0)
        
        return base_coherence * direction_factor
    
    def _adapt_thresholds(
        self,
        coherence: float,
        history: List[CoherenceState]
    ) -> Dict[str, float]:
        """Adapt thresholds based on natural emergence."""
        if len(history) < 2:
            return self.thresholds
            
        # Calculate rate of change
        recent = history[-10:] if len(history) > 10 else history
        avg_coherence = sum(s.coherence for s in recent) / len(recent)
        
        # Adapt thresholds naturally
        return {
            'energy': min(0.4, self.thresholds['energy'] * (1 + (coherence - avg_coherence))),
            'coherence': min(0.5, self.thresholds['coherence'] * (1 + (coherence - avg_coherence))),
            'velocity': self.thresholds['velocity'],
            'propensity': self.thresholds['propensity']
        }
