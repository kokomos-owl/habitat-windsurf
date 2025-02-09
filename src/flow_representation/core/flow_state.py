"""Core state management for flow representation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

@dataclass
class FlowDimensions:
    """Vector space dimensions for flow state."""
    stability: float
    coherence: float
    emergence_rate: float
    cross_pattern_flow: float
    energy_state: float
    adaptation_rate: float
    
    def to_vector(self) -> Any:
        """Convert dimensions to numpy vector."""
        return np.array([
            self.stability,
            self.coherence,
            self.emergence_rate,
            self.cross_pattern_flow,
            self.energy_state,
            self.adaptation_rate
        ])

@dataclass
class FlowState:
    """State representation for a flow pattern."""
    dimensions: FlowDimensions
    pattern_id: str
    timestamp: datetime
    related_patterns: List[str] = None
    metadata: Dict = None
    
    def calculate_evolution_metrics(self, previous_state: Optional['FlowState'] = None) -> Dict[str, float]:
        """Calculate evolution metrics between current and previous state."""
        if not previous_state:
            return {
                "velocity": 0.0,
                "acceleration": 0.0,
                "direction_change": 0.0
            }
            
        current_vec = self.dimensions.to_vector()
        previous_vec = previous_state.dimensions.to_vector()
        
        # Calculate velocity (magnitude of change)
        velocity = np.linalg.norm(current_vec - previous_vec)
        
        # Calculate acceleration if time difference available
        time_diff = (self.timestamp - previous_state.timestamp).total_seconds()
        acceleration = velocity / time_diff if time_diff > 0 else 0.0
        
        # Calculate direction change (cosine similarity)
        cos_sim = np.dot(current_vec, previous_vec) / (np.linalg.norm(current_vec) * np.linalg.norm(previous_vec))
        direction_change = np.arccos(np.clip(cos_sim, -1.0, 1.0))
        
        return {
            "velocity": velocity,
            "acceleration": acceleration,
            "direction_change": direction_change
        }
