"""Social field dynamics implementation."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class FieldState:
    """Represents the state of a social field."""
    energy: float  # Field energy level
    coherence: float  # Pattern coherence
    flow: float  # Resource/information flow
    stability: float  # Practice stability
    
@dataclass
class FieldConfig:
    """Configuration for social field dynamics."""
    # Thresholds
    coherence_threshold: float = 0.3  # Min coherence for pattern detection
    stability_threshold: float = 0.6  # Min stability for practice formation
    flow_threshold: float = 0.4  # Min flow for pattern spread
    
    # Evolution parameters
    energy_decay: float = 0.1  # Energy decay rate
    flow_resistance: float = 0.2  # Flow resistance
    coupling_strength: float = 0.3  # Inter-field coupling

class SocialField:
    """Manages social field dynamics."""
    
    def __init__(self, config: FieldConfig):
        self.config = config
        self.state = FieldState(
            energy=0.0,
            coherence=0.0,
            flow=0.0,
            stability=0.0
        )
    
    def update(self, dt: float, external_fields: Dict[str, 'SocialField']):
        """Update field state based on internal dynamics and external fields."""
        # Internal evolution
        self._evolve_internal(dt)
        
        # External coupling
        self._couple_fields(external_fields)
        
        # Update stability
        self._update_stability()
    
    def _evolve_internal(self, dt: float):
        """Evolve internal field state."""
        # Energy decay
        self.state.energy *= (1 - self.config.energy_decay * dt)
        
        # Flow dynamics
        flow_delta = -self.config.flow_resistance * self.state.flow
        self.state.flow = max(0, self.state.flow + flow_delta * dt)
        
        # Coherence evolution
        self.state.coherence = self._calculate_coherence()
    
    def _couple_fields(self, external_fields: Dict[str, 'SocialField']):
        """Handle coupling with external fields."""
        for field in external_fields.values():
            energy_exchange = self.config.coupling_strength * (
                field.state.energy - self.state.energy
            )
            self.state.energy += energy_exchange
    
    def _calculate_coherence(self) -> float:
        """Calculate pattern coherence based on energy and flow."""
        return min(1.0, (
            self.state.energy * 0.6 + 
            self.state.flow * 0.4
        ))
    
    def _update_stability(self):
        """Update practice stability based on coherence and flow."""
        self.state.stability = (
            self.state.coherence * 0.7 +
            (1 - self.state.flow) * 0.3
        )
