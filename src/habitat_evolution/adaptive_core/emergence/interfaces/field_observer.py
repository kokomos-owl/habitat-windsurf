"""
Field Observer Interface for Vector-Tonic Persistence Integration.

This module defines the interface for observing field state events
in the Vector-Tonic Window system. It supports the pattern evolution and 
co-evolution principles of Habitat Evolution by enabling the observation
of semantic change across the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class FieldObserverInterface(ABC):
    """
    Interface for observers of field state events.
    
    This interface defines the methods that must be implemented by any class
    that wants to observe field state events in the Vector-Tonic Window system.
    It enables components to react to field state changes, density shifts, and
    topology changes.
    """
    
    @abstractmethod
    def on_field_state_change(self, field_id: str, previous_state: Dict[str, Any], 
                             new_state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when the field state changes.
        
        Args:
            field_id: The ID of the field.
            previous_state: The previous state of the field.
            new_state: The new state of the field.
            metadata: Optional metadata about the state change.
        """
        pass
    
    @abstractmethod
    def on_field_coherence_change(self, field_id: str, previous_coherence: float, 
                                 new_coherence: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when the field coherence changes.
        
        Args:
            field_id: The ID of the field.
            previous_coherence: The previous coherence value.
            new_coherence: The new coherence value.
            metadata: Optional metadata about the coherence change.
        """
        pass
    
    @abstractmethod
    def on_field_stability_change(self, field_id: str, previous_stability: float, 
                                 new_stability: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when the field stability changes.
        
        Args:
            field_id: The ID of the field.
            previous_stability: The previous stability value.
            new_stability: The new stability value.
            metadata: Optional metadata about the stability change.
        """
        pass
    
    @abstractmethod
    def on_density_center_shift(self, field_id: str, previous_centers: List[Dict[str, Any]], 
                               new_centers: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when density centers shift.
        
        Args:
            field_id: The ID of the field.
            previous_centers: The previous density centers.
            new_centers: The new density centers.
            metadata: Optional metadata about the shift.
        """
        pass
    
    @abstractmethod
    def on_eigenspace_change(self, field_id: str, previous_eigenspace: Dict[str, Any], 
                            new_eigenspace: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when the eigenspace changes.
        
        Args:
            field_id: The ID of the field.
            previous_eigenspace: The previous eigenspace properties.
            new_eigenspace: The new eigenspace properties.
            metadata: Optional metadata about the change.
        """
        pass
    
    @abstractmethod
    def on_topology_change(self, field_id: str, previous_topology: Dict[str, Any], 
                          new_topology: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when the field topology changes.
        
        Args:
            field_id: The ID of the field.
            previous_topology: The previous topology.
            new_topology: The new topology.
            metadata: Optional metadata about the change.
        """
        pass
