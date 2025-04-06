"""
Field state service interface for Habitat Evolution.

This module defines the interface for the field state service, which is
responsible for managing the semantic field state in the Habitat Evolution system.
"""

from typing import Protocol, Any, Dict, List, Optional
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface


class FieldStateServiceInterface(ServiceInterface, Protocol):
    """
    Interface for the field state service in Habitat Evolution.
    
    The field state service is responsible for managing the semantic field state,
    including density, coherence, stability, and other field metrics. It supports
    the pattern evolution and co-evolution principles of Habitat by tracking how
    the semantic field evolves over time and how it influences pattern emergence.
    """
    
    @abstractmethod
    def get_field_state(self, field_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the current state of the semantic field.
        
        Args:
            field_id: Optional ID of the specific field to get
            
        Returns:
            The current field state
        """
        ...
        
    @abstractmethod
    def update_field_state(self, state_updates: Dict[str, Any], field_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Update the state of the semantic field.
        
        Args:
            state_updates: The updates to apply to the field state
            field_id: Optional ID of the specific field to update
            
        Returns:
            The updated field state
        """
        ...
        
    @abstractmethod
    def calculate_field_metrics(self, patterns: List[Dict[str, Any]], 
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate metrics for the semantic field based on patterns.
        
        Args:
            patterns: The patterns to use for metric calculation
            context: Optional context for metric calculation
            
        Returns:
            The calculated field metrics
        """
        ...
        
    @abstractmethod
    def get_density_centers(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Get density centers in the semantic field.
        
        Args:
            threshold: The minimum density threshold for centers
            
        Returns:
            A list of density centers
        """
        ...
        
    @abstractmethod
    def create_snapshot(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a snapshot of the current field state.
        
        Args:
            metadata: Optional metadata for the snapshot
            
        Returns:
            The ID of the created snapshot
        """
        ...
        
    @abstractmethod
    def get_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Get a field state snapshot by ID.
        
        Args:
            snapshot_id: The ID of the snapshot to get
            
        Returns:
            The snapshot
        """
        ...
