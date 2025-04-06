"""
Pattern persistence service interface for Habitat Evolution.

This module defines the interface for pattern persistence services in the Habitat Evolution system,
providing a consistent approach to pattern persistence across the application.
"""

from typing import Protocol, Dict, List, Any, Optional
from abc import abstractmethod
from datetime import datetime

from src.habitat_evolution.infrastructure.interfaces.persistence.persistence_service_interface import PersistenceServiceInterface


class PatternPersistenceServiceInterface(PersistenceServiceInterface[Dict[str, Any]], Protocol):
    """
    Interface for pattern persistence services in Habitat Evolution.
    
    Pattern persistence services provide a consistent approach to pattern persistence,
    abstracting the details of pattern storage and retrieval. This supports the pattern
    evolution and co-evolution principles of Habitat by enabling flexible pattern
    persistence mechanisms while maintaining a consistent interface.
    """
    
    @abstractmethod
    def find_by_name(self, name: str, exact_match: bool = True) -> List[Dict[str, Any]]:
        """
        Find patterns by name.
        
        Args:
            name: The name to search for
            exact_match: Whether to require an exact match
            
        Returns:
            A list of matching patterns
        """
        ...
        
    @abstractmethod
    def find_by_quality(self, min_quality: float) -> List[Dict[str, Any]]:
        """
        Find patterns by quality.
        
        Args:
            min_quality: The minimum quality threshold
            
        Returns:
            A list of patterns with quality above the threshold
        """
        ...
        
    @abstractmethod
    def find_by_creation_time(self, start_time: datetime, end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Find patterns by creation time.
        
        Args:
            start_time: The start of the time range
            end_time: The end of the time range (defaults to now)
            
        Returns:
            A list of patterns created in the time range
        """
        ...
        
    @abstractmethod
    def find_related_patterns(self, pattern_id: str, relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find patterns related to a pattern.
        
        Args:
            pattern_id: The ID of the pattern to find related patterns for
            relationship_type: Optional type of relationship to filter by
            
        Returns:
            A list of related patterns
        """
        ...
        
    @abstractmethod
    def save_pattern_relationship(self, source_id: str, target_id: str, 
                                 relationship_type: str, 
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a relationship between two patterns.
        
        Args:
            source_id: The ID of the source pattern
            target_id: The ID of the target pattern
            relationship_type: The type of relationship
            metadata: Optional metadata for the relationship
            
        Returns:
            The ID of the created relationship
        """
        ...
        
    @abstractmethod
    def update_pattern_quality(self, pattern_id: str, quality: float, 
                              quality_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Update the quality of a pattern.
        
        Args:
            pattern_id: The ID of the pattern to update
            quality: The new quality value
            quality_metrics: Optional detailed quality metrics
            
        Returns:
            The updated pattern
        """
        ...
