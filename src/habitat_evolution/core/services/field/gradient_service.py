"""
Concrete implementation of the gradient service.
"""

from typing import Dict
from datetime import datetime
import numpy as np

from .interfaces import GradientService, GradientVector
from ...storage.field_repository import FieldRepository
from ...services.event_bus import EventBus

class ConcreteGradientService(GradientService):
    """
    Concrete implementation of GradientService that handles gradient calculations
    and field flow dynamics.
    """
    
    def __init__(
        self,
        field_repository: FieldRepository,
        event_bus: EventBus
    ):
        self.repository = field_repository
        self.event_bus = event_bus
        
    async def calculate_gradient(self, field_id: str, position: Dict[str, float]) -> GradientVector:
        """Calculate the gradient vector at a position in the field"""
        try:
            # Get field state
            field_state = await self.repository.get_field_state(field_id)
            if not field_state:
                raise ValueError(f"No state found for field {field_id}")
            
            # Calculate gradient components using central differences
            dx = 0.01  # Small delta for numerical differentiation
            gradient_components = {}
            for axis in position.keys():
                pos_forward = position.copy()
                pos_backward = position.copy()
                pos_forward[axis] += dx
                pos_backward[axis] -= dx
                
                # Calculate potential difference
                potential_diff = await self.calculate_potential_difference(
                    field_id, 
                    pos_backward,
                    pos_forward
                )
                gradient_components[axis] = potential_diff / (2 * dx)
            
            # Calculate magnitude and stability
            magnitude = np.sqrt(sum(x*x for x in gradient_components.values()))
            stability = self._calculate_gradient_stability(gradient_components, field_state.stability)
            
            gradient_vector = GradientVector(
                direction=gradient_components,
                magnitude=magnitude,
                stability=stability
            )
            
            # Emit gradient calculation event
            await self.event_bus.emit("field.gradient.calculated", {
                "field_id": field_id,
                "position": position,
                "gradient": gradient_vector.__dict__,
                "timestamp": datetime.now().isoformat()
            })
            
            return gradient_vector
            
        except Exception as e:
            await self.event_bus.emit("field.gradient.error", {
                "field_id": field_id,
                "position": position,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise
    
    async def get_flow_direction(self, field_id: str, position: Dict[str, float]) -> Dict[str, float]:
        """Get the flow direction at a position in the field"""
        try:
            gradient = await self.calculate_gradient(field_id, position)
            
            # Flow direction is opposite to gradient direction
            flow_direction = {
                axis: -component 
                for axis, component in gradient.direction.items()
            }
            
            # Normalize flow direction
            magnitude = np.sqrt(sum(x*x for x in flow_direction.values()))
            if magnitude > 0:
                flow_direction = {
                    axis: component/magnitude 
                    for axis, component in flow_direction.items()
                }
            
            # Emit flow direction event
            await self.event_bus.emit("field.flow.direction", {
                "field_id": field_id,
                "position": position,
                "flow_direction": flow_direction,
                "timestamp": datetime.now().isoformat()
            })
            
            return flow_direction
            
        except Exception as e:
            await self.event_bus.emit("field.flow.error", {
                "field_id": field_id,
                "position": position,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise
    
    async def calculate_potential_difference(
        self, 
        field_id: str, 
        position1: Dict[str, float],
        position2: Dict[str, float]
    ) -> float:
        """Calculate the potential difference between two points in the field"""
        try:
            field_state = await self.repository.get_field_state(field_id)
            if not field_state:
                raise ValueError(f"No state found for field {field_id}")
            
            # Calculate distance between points
            distance = np.sqrt(sum(
                (position2[axis] - position1[axis])**2 
                for axis in position1.keys()
            ))
            
            # Calculate potential difference using field properties
            base_potential = field_state.potential
            gradient_effect = sum(
                field_state.gradient[axis] * (position2[axis] - position1[axis])
                for axis in position1.keys()
            )
            
            potential_diff = base_potential * gradient_effect / (distance + 1e-6)
            
            # Emit potential difference event
            await self.event_bus.emit("field.potential.difference", {
                "field_id": field_id,
                "position1": position1,
                "position2": position2,
                "potential_difference": potential_diff,
                "timestamp": datetime.now().isoformat()
            })
            
            return potential_diff
            
        except Exception as e:
            await self.event_bus.emit("field.potential.error", {
                "field_id": field_id,
                "position1": position1,
                "position2": position2,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise
            
    def _calculate_gradient_stability(
        self,
        gradient_components: Dict[str, float],
        field_stability: float
    ) -> float:
        """Calculate gradient stability based on components and field stability"""
        # Stability decreases with gradient magnitude
        magnitude = np.sqrt(sum(x*x for x in gradient_components.values()))
        gradient_factor = 1.0 / (1.0 + magnitude)
        
        # Combine with field stability
        stability = 0.7 * field_stability + 0.3 * gradient_factor
        return max(0.0, min(1.0, stability))
