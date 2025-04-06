"""
Simplified implementation of the gradient service for testing.
"""

from typing import Dict, Any, Optional
from datetime import datetime

class GradientVector:
    """Represents a gradient vector in a field"""
    
    def __init__(self, direction: Dict[str, float], magnitude: float, stability: float = 0.5):
        self.direction = direction
        self.magnitude = magnitude
        self.stability = stability
        
    def __dict__(self):
        return {
            "direction": self.direction,
            "magnitude": self.magnitude,
            "stability": self.stability
        }

class GradientService:
    """
    Service for calculating gradients in fields.
    """
    
    async def calculate_gradient(self, field_id: str, position: Dict[str, float]) -> GradientVector:
        """Calculate the gradient vector at a position in the field"""
        # Simplified implementation for testing
        return GradientVector(
            direction={'x': 0.5, 'y': 0.5}, 
            magnitude=0.5,
            stability=0.7
        )
    
    async def get_flow_direction(self, field_id: str, position: Dict[str, float]) -> Dict[str, float]:
        """Get the flow direction at a position in the field"""
        # Simplified implementation for testing
        return {'x': 0.5, 'y': 0.5}
        
    async def calculate_potential_difference(self, field_id: str, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """Calculate the potential difference between two positions"""
        # Simplified implementation for testing
        return 0.2
        
    def _calculate_gradient_stability(self, gradient_components: Dict[str, float], field_stability: float) -> float:
        """Calculate the stability of a gradient based on its components and field stability"""
        # Simplified implementation for testing
        return 0.7
