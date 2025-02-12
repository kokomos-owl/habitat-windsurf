"""
Pattern quality analysis system.
"""
from dataclasses import dataclass
from typing import Dict, Any

from ..pattern.types import Pattern, FieldState

@dataclass
class QualityConfig:
    """Configuration for quality analysis."""
    signal_threshold: float = 0.2
    stability_window: int = 10
    energy_decay: float = 0.8

class PatternQualityAnalyzer:
    """Analyzes pattern quality through field lens."""
    
    def __init__(self, config: QualityConfig = None):
        self.config = config or QualityConfig()
        self._history: Dict[str, list] = {}
    
    def analyze_pattern(self,
                       pattern: Pattern,
                       field_state: FieldState) -> Dict[str, float]:
        """Analyze pattern quality through field lens.
        
        Args:
            pattern: Pattern to analyze
            field_state: Current field state
            
        Returns:
            Quality metrics
        """
        # Calculate signal strength
        signal_strength = self._calculate_signal_strength(pattern, field_state)
        
        # Calculate stability
        stability = self._calculate_stability(pattern)
        
        # Calculate energy state
        energy_state = self._calculate_energy_state(pattern, field_state)
        
        return {
            'signal_strength': signal_strength,
            'stability': stability,
            'energy_state': energy_state
        }
    
    def _calculate_signal_strength(self,
                                 pattern: Pattern,
                                 field_state: FieldState) -> float:
        """Calculate pattern signal strength."""
        coherence = pattern['coherence']
        field_coherence = field_state.gradients.coherence
        
        # Strong signal if pattern coherence aligns with field
        signal = abs(coherence - field_coherence)
        return max(0.0, 1.0 - signal)
    
    def _calculate_stability(self, pattern: Pattern) -> float:
        """Calculate pattern stability."""
        pattern_id = pattern['id']
        
        # Initialize history if needed
        if pattern_id not in self._history:
            self._history[pattern_id] = []
        
        # Update history
        history = self._history[pattern_id]
        history.append(pattern['coherence'])
        
        # Keep fixed window
        if len(history) > self.config.stability_window:
            history.pop(0)
        
        # Calculate stability from variance
        if len(history) < 2:
            return 0.0
            
        variance = sum((x - pattern['coherence']) ** 2 
                      for x in history) / len(history)
        return max(0.0, 1.0 - variance)
    
    def _calculate_energy_state(self,
                              pattern: Pattern,
                              field_state: FieldState) -> float:
        """Calculate pattern energy state."""
        base_energy = pattern['energy']
        field_energy = field_state.gradients.energy
        
        # Energy decays toward field energy
        energy_diff = field_energy - base_energy
        new_energy = base_energy + (energy_diff * (1.0 - self.config.energy_decay))
        
        return max(0.0, min(1.0, new_energy))
