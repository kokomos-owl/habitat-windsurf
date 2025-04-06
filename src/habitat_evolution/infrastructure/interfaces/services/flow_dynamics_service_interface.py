"""
Flow dynamics service interface for Habitat Evolution.

This module defines the interface for the flow dynamics service, which is
responsible for analyzing and managing flow dynamics in the semantic field.
"""

from typing import Protocol, Any, Dict, List, Optional, Tuple
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface


class FlowDynamicsServiceInterface(ServiceInterface, Protocol):
    """
    Interface for the flow dynamics service in Habitat Evolution.
    
    The flow dynamics service is responsible for analyzing and managing flow dynamics
    in the semantic field, which represent how patterns and concepts flow and interact.
    It supports the pattern evolution and co-evolution principles of Habitat by
    tracking how information and meaning flow through the system.
    """
    
    @abstractmethod
    def calculate_flow_coherence(self, patterns: List[Dict[str, Any]], 
                                field_state: Dict[str, Any]) -> float:
        """
        Calculate the coherence of flow between patterns.
        
        Args:
            patterns: The patterns to analyze
            field_state: The current state of the semantic field
            
        Returns:
            The calculated flow coherence
        """
        ...
        
    @abstractmethod
    def identify_flow_channels(self, field_state: Dict[str, Any], 
                              threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Identify channels of coherent flow in the semantic field.
        
        Args:
            field_state: The current state of the semantic field
            threshold: The minimum coherence threshold for channels
            
        Returns:
            A list of flow channels
        """
        ...
        
    @abstractmethod
    def calculate_cross_pattern_flow(self, source_pattern: Dict[str, Any], 
                                    target_pattern: Dict[str, Any],
                                    field_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the flow between two patterns.
        
        Args:
            source_pattern: The source pattern
            target_pattern: The target pattern
            field_state: The current state of the semantic field
            
        Returns:
            The calculated cross-pattern flow
        """
        ...
        
    @abstractmethod
    def detect_flow_turbulence(self, field_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect turbulence in the semantic flow.
        
        Args:
            field_state: The current state of the semantic field
            
        Returns:
            Information about detected turbulence
        """
        ...
        
    @abstractmethod
    def calculate_flow_stability(self, current_state: Dict[str, Any], 
                                previous_state: Dict[str, Any]) -> float:
        """
        Calculate the stability of flow between two field states.
        
        Args:
            current_state: The current state of the semantic field
            previous_state: The previous state of the semantic field
            
        Returns:
            The calculated flow stability
        """
        ...
