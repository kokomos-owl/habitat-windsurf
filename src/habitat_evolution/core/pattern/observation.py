"""Pattern observation and phase relationship calculations."""

import numpy as np
from typing import Dict, Optional

class PatternObserver:
    """Observes and analyzes pattern behavior in field space."""
    
    def __init__(self):
        self.phase_history = []
        
    def calculate_phase_relationship(self, potential: float, position: Dict[str, float]) -> float:
        """Calculate phase relationship based on potential and position.
        
        Uses wave mechanics analogy where phase is determined by:
        - Position in field (spatial component)
        - Potential energy (amplitude component)
        """
        x, y = position['x'], position['y']
        
        # Calculate spatial phase component (0 to 2π)
        spatial_phase = np.arctan2(y, x) % (2 * np.pi)
        
        # Calculate amplitude phase component (0 to π/2)
        amplitude_phase = np.arcsin(max(0.0, min(1.0, potential))) 
        
        # Combine phases with weighting
        combined_phase = (0.7 * spatial_phase + 0.3 * amplitude_phase) / (2 * np.pi)
        
        self.phase_history.append(combined_phase)
        if len(self.phase_history) > 100:
            self.phase_history.pop(0)
            
        return combined_phase
        
    def get_phase_coherence(self) -> float:
        """Calculate phase coherence from recent history."""
        if not self.phase_history:
            return 0.0
            
        # Calculate circular variance
        phases = np.array(self.phase_history) * 2 * np.pi
        mean_sin = np.mean(np.sin(phases))
        mean_cos = np.mean(np.cos(phases))
        R = np.sqrt(mean_sin**2 + mean_cos**2)
        
        return R  # 1.0 = perfectly coherent, 0.0 = incoherent
