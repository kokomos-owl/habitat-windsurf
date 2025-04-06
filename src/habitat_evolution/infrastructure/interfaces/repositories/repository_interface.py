"""
Repository interface for Habitat Evolution.

This module defines the base interface for repositories in the Habitat Evolution system,
providing a consistent approach to data access across the application.
"""

from typing import Protocol, TypeVar, Generic, Dict, List, Any, Optional
from abc import abstractmethod

T = TypeVar('T')


class RepositoryInterface(Generic[T], Protocol):
    """
    Base interface for repositories in Habitat Evolution.
    
    Repositories provide a consistent approach to data access, abstracting the
    underlying storage technology and providing domain-specific methods for
    retrieving and persisting entities. This supports the pattern evolution and
    co-evolution principles of Habitat by enabling flexible storage mechanisms
    while maintaining a consistent interface.
    """
    
    @abstractmethod
    def find_by_id(self, entity_id: str) -> Optional[T]:
        """
        Find an entity by its ID.
        
        Args:
            entity_id: The ID of the entity to find
            
        Returns:
            The entity if found, None otherwise
        """
        ...
        
    @abstractmethod
    def find_all(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[T]:
        """
        Find all entities matching the filter criteria.
        
        Args:
            filter_criteria: Optional criteria to filter entities by
            
        Returns:
            A list of matching entities
        """
        ...
        
    @abstractmethod
    def save(self, entity: T) -> T:
        """
        Save an entity to the repository.
        
        If the entity already exists, it will be updated. Otherwise, it will be created.
        
        Args:
            entity: The entity to save
            
        Returns:
            The saved entity with any generated IDs or timestamps
        """
        ...
        
    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """
        Delete an entity from the repository.
        
        Args:
            entity_id: The ID of the entity to delete
            
        Returns:
            True if the entity was deleted, False otherwise
        """
        ...
        
    @abstractmethod
    def count(self, filter_criteria: Optional[Dict[str, Any]] = None) -> int:
        """
        Count entities matching the filter criteria.
        
        Args:
            filter_criteria: Optional criteria to filter entities by
            
        Returns:
            The number of matching entities
        """
        ...
