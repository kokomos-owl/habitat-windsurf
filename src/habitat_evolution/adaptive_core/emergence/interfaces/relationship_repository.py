"""
Relationship Repository Interface for Vector-Tonic Persistence Integration.

This module defines the interface for persisting and retrieving relationships
in the Vector-Tonic Window system. It supports the pattern evolution and 
co-evolution principles of Habitat Evolution by enabling the observation
of semantic change across the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime


class RelationshipRepositoryInterface(ABC):
    """
    Interface for a repository that manages relationship objects.
    
    This repository is responsible for persisting and retrieving relationships,
    which represent connections between patterns in the vector-tonic window.
    Relationships capture the semantic connections between patterns and are
    crucial for tracking pattern co-evolution.
    """
    
    @abstractmethod
    def save(self, relationship: Dict[str, Any]) -> str:
        """
        Save a relationship to the repository.
        
        Args:
            relationship: The relationship to save.
            
        Returns:
            The ID of the saved relationship.
        """
        pass
    
    @abstractmethod
    def find_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Find a relationship by its ID.
        
        Args:
            id: The ID of the relationship to find.
            
        Returns:
            The relationship if found, None otherwise.
        """
        pass
    
    @abstractmethod
    def find_by_source(self, source_id: str) -> List[Dict[str, Any]]:
        """
        Find relationships by their source pattern.
        
        Args:
            source_id: The ID of the source pattern.
            
        Returns:
            A list of relationships with the specified source pattern.
        """
        pass
    
    @abstractmethod
    def find_by_target(self, target_id: str) -> List[Dict[str, Any]]:
        """
        Find relationships by their target pattern.
        
        Args:
            target_id: The ID of the target pattern.
            
        Returns:
            A list of relationships with the specified target pattern.
        """
        pass
    
    @abstractmethod
    def find_by_predicate(self, predicate: str) -> List[Dict[str, Any]]:
        """
        Find relationships by their predicate.
        
        Args:
            predicate: The predicate to search for.
            
        Returns:
            A list of relationships with the specified predicate.
        """
        pass
    
    @abstractmethod
    def find_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Find relationships within a time range.
        
        Args:
            start_time: The start of the time range.
            end_time: The end of the time range.
            
        Returns:
            A list of relationships within the specified time range.
        """
        pass
    
    @abstractmethod
    def find_by_strength_range(self, min_strength: float, max_strength: float) -> List[Dict[str, Any]]:
        """
        Find relationships within a strength range.
        
        Args:
            min_strength: The minimum strength value.
            max_strength: The maximum strength value.
            
        Returns:
            A list of relationships within the specified strength range.
        """
        pass
    
    @abstractmethod
    def find_by_pattern_pair(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        """
        Find relationships between a specific pair of patterns.
        
        Args:
            source_id: The ID of the source pattern.
            target_id: The ID of the target pattern.
            
        Returns:
            A list of relationships between the specified patterns.
        """
        pass
    
    @abstractmethod
    def find_by_pattern_and_predicate(self, pattern_id: str, predicate: str, is_source: bool = True) -> List[Dict[str, Any]]:
        """
        Find relationships by pattern and predicate.
        
        Args:
            pattern_id: The ID of the pattern.
            predicate: The predicate to search for.
            is_source: Whether the pattern is the source (True) or target (False).
            
        Returns:
            A list of relationships with the specified pattern and predicate.
        """
        pass
