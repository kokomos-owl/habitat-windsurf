"""
Gradient service interface for Habitat Evolution.

This module defines the interface for the gradient service, which is
responsible for calculating and managing gradients in the semantic field.
"""

from typing import Protocol, Any, Dict, List, Optional, Tuple
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface


class GradientServiceInterface(ServiceInterface, Protocol):
    """
    Interface for the gradient service in Habitat Evolution.
    
    The gradient service is responsible for calculating and managing gradients
    in the semantic field, which represent directional forces in pattern evolution.
    It supports the pattern evolution and co-evolution principles of Habitat by
    identifying how patterns are likely to evolve based on the current field state.
    """
    
    @abstractmethod
    def calculate_gradient(self, field_state: Dict[str, Any], 
                          position: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate the gradient at a position in the semantic field.
        
        Args:
            field_state: The current state of the semantic field
            position: Optional position in the field (defaults to center)
            
        Returns:
            The calculated gradient
        """
        ...
        
    @abstractmethod
    def get_gradient_field(self, field_state: Dict[str, Any], 
                          resolution: int = 10) -> List[Dict[str, Any]]:
        """
        Get a gradient field for the semantic field.
        
        Args:
            field_state: The current state of the semantic field
            resolution: The resolution of the gradient field
            
        Returns:
            A list of gradient vectors
        """
        ...
        
    @abstractmethod
    def find_potential_wells(self, gradient_field: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find potential wells in the gradient field.
        
        Potential wells are areas where gradients converge, indicating
        stable attractors in the semantic field.
        
        Args:
            gradient_field: The gradient field to analyze
            
        Returns:
            A list of potential wells
        """
        ...
        
    @abstractmethod
    def calculate_flow_direction(self, source_position: Dict[str, float], 
                                target_position: Dict[str, float],
                                field_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate the flow direction between two positions.
        
        Args:
            source_position: The source position
            target_position: The target position
            field_state: The current state of the semantic field
            
        Returns:
            The flow direction vector
        """
        ...
        
    @abstractmethod
    def calculate_gradient_magnitude(self, gradient: Dict[str, float]) -> float:
        """
        Calculate the magnitude of a gradient.
        
        Args:
            gradient: The gradient to calculate magnitude for
            
        Returns:
            The magnitude of the gradient
        """
        ...
