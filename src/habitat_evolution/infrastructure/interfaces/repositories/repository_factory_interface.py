"""
Repository factory interface for Habitat Evolution.

This module defines the interface for repository factories in the Habitat Evolution system,
providing a consistent approach to repository creation across the application.
"""

from typing import Protocol, TypeVar, Dict, Any, Optional
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.repositories.repository_interface import RepositoryInterface
from src.habitat_evolution.infrastructure.interfaces.repositories.graph_repository_interface import GraphRepositoryInterface

T = TypeVar('T')
R = TypeVar('R', bound=RepositoryInterface)


class RepositoryFactoryInterface(Protocol):
    """
    Interface for repository factories in Habitat Evolution.
    
    Repository factories provide a consistent approach to repository creation,
    abstracting the details of repository instantiation and configuration.
    This supports the pattern evolution and co-evolution principles of Habitat
    by enabling flexible repository creation while maintaining a consistent interface.
    """
    
    @abstractmethod
    def create_repository(self, repository_type: str, config: Optional[Dict[str, Any]] = None) -> RepositoryInterface:
        """
        Create a repository of the specified type.
        
        Args:
            repository_type: The type of repository to create
            config: Optional configuration for the repository
            
        Returns:
            The created repository
        """
        ...
        
    @abstractmethod
    def create_graph_repository(self, entity_type: str, config: Optional[Dict[str, Any]] = None) -> GraphRepositoryInterface:
        """
        Create a graph repository for the specified entity type.
        
        Args:
            entity_type: The type of entity the repository will manage
            config: Optional configuration for the repository
            
        Returns:
            The created graph repository
        """
        ...
        
    @abstractmethod
    def create_pattern_repository(self, config: Optional[Dict[str, Any]] = None) -> RepositoryInterface:
        """
        Create a pattern repository.
        
        Args:
            config: Optional configuration for the repository
            
        Returns:
            The created pattern repository
        """
        ...
        
    @abstractmethod
    def create_field_state_repository(self, config: Optional[Dict[str, Any]] = None) -> RepositoryInterface:
        """
        Create a field state repository.
        
        Args:
            config: Optional configuration for the repository
            
        Returns:
            The created field state repository
        """
        ...
        
    @abstractmethod
    def create_relationship_repository(self, config: Optional[Dict[str, Any]] = None) -> RepositoryInterface:
        """
        Create a relationship repository.
        
        Args:
            config: Optional configuration for the repository
            
        Returns:
            The created relationship repository
        """
        ...
        
    @abstractmethod
    def create_topology_repository(self, config: Optional[Dict[str, Any]] = None) -> RepositoryInterface:
        """
        Create a topology repository.
        
        Args:
            config: Optional configuration for the repository
            
        Returns:
            The created topology repository
        """
        ...
