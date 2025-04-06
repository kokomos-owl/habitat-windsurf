"""
Field service interfaces for the Habitat Evolution system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FieldState:
    """Represents the state of a field at a point in time"""
    field_id: str
    timestamp: datetime = datetime.now()
    state_vector: Dict[str, float] = None
    pressure: float = 0.0
    stability: float = 0.0
    
    def __post_init__(self):
        if self.state_vector is None:
            self.state_vector = {}

@dataclass
class GradientVector:
    """Represents a gradient vector in the field"""
    direction: Dict[str, float]
    magnitude: float
    stability: float

class FieldStateService(ABC):
    """Service interface for managing field states"""
    
    @abstractmethod
    async def get_field_state(self, field_id: str) -> Optional[FieldState]:
        """Get the current state of a field"""
        pass
    
    @abstractmethod
    async def update_field_state(self, field_id: str, state: FieldState) -> None:
        """Update the state of a field"""
        pass
    
    @abstractmethod
    async def calculate_field_stability(self, field_id: str) -> float:
        """Calculate the stability metric for a field"""
        pass

class GradientService(ABC):
    """Service interface for gradient calculations"""
    
    @abstractmethod
    async def calculate_gradient(self, field_id: str, position: Dict[str, float]) -> GradientVector:
        """Calculate the gradient vector at a position in the field"""
        pass
    
    @abstractmethod
    async def get_flow_direction(self, field_id: str, position: Dict[str, float]) -> Dict[str, float]:
        """Get the flow direction at a position in the field"""
        pass
    
    @abstractmethod
    async def calculate_potential_difference(
        self, 
        field_id: str, 
        position1: Dict[str, float],
        position2: Dict[str, float]
    ) -> float:
        """Calculate the potential difference between two points in the field"""
        pass

class FlowDynamicsService(ABC):
    """Service interface for flow dynamics calculations"""
    
    @abstractmethod
    async def calculate_viscosity(self, field_id: str, position: Dict[str, float]) -> float:
        """Calculate the viscosity at a position in the field"""
        pass
    
    @abstractmethod
    async def calculate_turbulence(self, field_id: str, position: Dict[str, float]) -> float:
        """Calculate the turbulence at a position in the field"""
        pass
    
    @abstractmethod
    async def calculate_flow_rate(
        self, 
        field_id: str, 
        start_position: Dict[str, float],
        end_position: Dict[str, float]
    ) -> float:
        """Calculate the flow rate between two points in the field"""
        pass
