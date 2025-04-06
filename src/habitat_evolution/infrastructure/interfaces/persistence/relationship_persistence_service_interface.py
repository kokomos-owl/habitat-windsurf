"""
Relationship persistence service interface for Habitat Evolution.

This module defines the interface for relationship persistence services in the Habitat Evolution system,
providing a consistent approach to relationship persistence across the application.
"""

from typing import Protocol, Dict, List, Any, Optional
from abc import abstractmethod
from datetime import datetime

from src.habitat_evolution.infrastructure.interfaces.persistence.persistence_service_interface import PersistenceServiceInterface


class RelationshipPersistenceServiceInterface(PersistenceServiceInterface[Dict[str, Any]], Protocol):
    """
    Interface for relationship persistence services in Habitat Evolution.
    
    Relationship persistence services provide a consistent approach to relationship persistence,
    abstracting the details of relationship storage and retrieval. This supports the pattern
    evolution and co-evolution principles of Habitat by enabling flexible relationship
    persistence mechanisms while maintaining a consistent interface.
    """
    
    @abstractmethod
    def find_by_source(self, source_id: str, relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find relationships by source entity.
        
        Args:
            source_id: The ID of the source entity
            relationship_type: Optional type of relationship to filter by
            
        Returns:
            A list of matching relationships
        """
        ...
        
    @abstractmethod
    def find_by_target(self, target_id: str, relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find relationships by target entity.
        
        Args:
            target_id: The ID of the target entity
            relationship_type: Optional type of relationship to filter by
            
        Returns:
            A list of matching relationships
        """
        ...
        
    @abstractmethod
    def find_by_type(self, relationship_type: str) -> List[Dict[str, Any]]:
        """
        Find relationships by type.
        
        Args:
            relationship_type: The type of relationship to find
            
        Returns:
            A list of matching relationships
        """
        ...
        
    @abstractmethod
    def find_by_timestamp(self, start_time: datetime, end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Find relationships by timestamp.
        
        Args:
            start_time: The start of the time range
            end_time: The end of the time range (defaults to now)
            
        Returns:
            A list of relationships created in the time range
        """
        ...
        
    @abstractmethod
    def find_by_strength(self, min_strength: float) -> List[Dict[str, Any]]:
        """
        Find relationships by strength.
        
        Args:
            min_strength: The minimum strength threshold
            
        Returns:
            A list of relationships with strength above the threshold
        """
        ...
        
    @abstractmethod
    def update_relationship_strength(self, relationship_id: str, strength: float) -> Dict[str, Any]:
        """
        Update the strength of a relationship.
        
        Args:
            relationship_id: The ID of the relationship to update
            strength: The new strength value
            
        Returns:
            The updated relationship
        """
        ...
        
    @abstractmethod
    def find_path(self, source_id: str, target_id: str, 
                 max_depth: int = 3,
                 relationship_types: Optional[List[str]] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Find a path between two entities.
        
        Args:
            source_id: The ID of the source entity
            target_id: The ID of the target entity
            max_depth: The maximum path length
            relationship_types: Optional types of relationships to consider
            
        Returns:
            The path if found, None otherwise
        """
        ...
