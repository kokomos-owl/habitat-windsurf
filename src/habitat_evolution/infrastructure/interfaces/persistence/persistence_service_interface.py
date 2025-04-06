"""
Persistence service interface for Habitat Evolution.

This module defines the interface for persistence services in the Habitat Evolution system,
providing a consistent approach to data persistence across the application.
"""

from typing import Protocol, Dict, List, Any, Optional, TypeVar, Generic
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface

T = TypeVar('T')


class PersistenceServiceInterface(ServiceInterface, Protocol, Generic[T]):
    """
    Interface for persistence services in Habitat Evolution.
    
    Persistence services provide a consistent approach to data persistence,
    abstracting the details of storage and retrieval. This supports the pattern
    evolution and co-evolution principles of Habitat by enabling flexible
    persistence mechanisms while maintaining a consistent interface.
    """
    
    @abstractmethod
    def save(self, entity: T) -> T:
        """
        Save an entity to the persistence store.
        
        If the entity already exists, it will be updated. Otherwise, it will be created.
        
        Args:
            entity: The entity to save
            
        Returns:
            The saved entity with any generated IDs or timestamps
        """
        ...
        
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
    def delete(self, entity_id: str) -> bool:
        """
        Delete an entity from the persistence store.
        
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
        
    @abstractmethod
    def save_batch(self, entities: List[T]) -> List[T]:
        """
        Save a batch of entities to the persistence store.
        
        Args:
            entities: The entities to save
            
        Returns:
            The saved entities with any generated IDs or timestamps
        """
        ...
        
    @abstractmethod
    def delete_batch(self, entity_ids: List[str]) -> int:
        """
        Delete a batch of entities from the persistence store.
        
        Args:
            entity_ids: The IDs of the entities to delete
            
        Returns:
            The number of entities deleted
        """
        ...
