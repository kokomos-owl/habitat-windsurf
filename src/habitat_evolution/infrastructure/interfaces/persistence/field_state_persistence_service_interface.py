"""
Field state persistence service interface for Habitat Evolution.

This module defines the interface for field state persistence services in the Habitat Evolution system,
providing a consistent approach to field state persistence across the application.
"""

from typing import Protocol, Dict, List, Any, Optional
from abc import abstractmethod
from datetime import datetime

from src.habitat_evolution.infrastructure.interfaces.persistence.persistence_service_interface import PersistenceServiceInterface


class FieldStatePersistenceServiceInterface(PersistenceServiceInterface[Dict[str, Any]], Protocol):
    """
    Interface for field state persistence services in Habitat Evolution.
    
    Field state persistence services provide a consistent approach to field state persistence,
    abstracting the details of field state storage and retrieval. This supports the pattern
    evolution and co-evolution principles of Habitat by enabling flexible field state
    persistence mechanisms while maintaining a consistent interface.
    """
    
    @abstractmethod
    def find_by_timestamp(self, start_time: datetime, end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Find field states by timestamp.
        
        Args:
            start_time: The start of the time range
            end_time: The end of the time range (defaults to now)
            
        Returns:
            A list of field states created in the time range
        """
        ...
        
    @abstractmethod
    def find_by_density(self, min_density: float) -> List[Dict[str, Any]]:
        """
        Find field states by density.
        
        Args:
            min_density: The minimum density threshold
            
        Returns:
            A list of field states with density above the threshold
        """
        ...
        
    @abstractmethod
    def find_by_coherence(self, min_coherence: float) -> List[Dict[str, Any]]:
        """
        Find field states by coherence.
        
        Args:
            min_coherence: The minimum coherence threshold
            
        Returns:
            A list of field states with coherence above the threshold
        """
        ...
        
    @abstractmethod
    def create_snapshot(self, field_state: Dict[str, Any], 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a snapshot of a field state.
        
        Args:
            field_state: The field state to snapshot
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
        
    @abstractmethod
    def find_snapshots_by_metadata(self, metadata_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find snapshots by metadata criteria.
        
        Args:
            metadata_criteria: The metadata criteria to filter by
            
        Returns:
            A list of matching snapshots
        """
        ...
        
    @abstractmethod
    def update_field_metrics(self, field_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Update the metrics of a field state.
        
        Args:
            field_id: The ID of the field state to update
            metrics: The new metrics values
            
        Returns:
            The updated field state
        """
        ...
