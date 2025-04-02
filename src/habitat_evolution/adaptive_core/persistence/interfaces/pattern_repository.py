"""
Pattern Repository Interface for Vector-Tonic Persistence Integration.

This module defines the interface for persisting and retrieving patterns
in the Vector-Tonic Window system. It supports the pattern evolution and 
co-evolution principles of Habitat Evolution by enabling the observation
of semantic change across the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime


class PatternRepositoryInterface(ABC):
    """
    Interface for a repository that manages pattern objects.
    
    This repository is responsible for persisting and retrieving patterns,
    which represent coherent structures detected in the vector-tonic window.
    Patterns capture eigenspace properties, temporal characteristics, and
    oscillatory properties that are crucial for tracking pattern evolution.
    """
    
    @abstractmethod
    def save(self, pattern: Dict[str, Any]) -> str:
        """
        Save a pattern to the repository.
        
        Args:
            pattern: The pattern to save.
            
        Returns:
            The ID of the saved pattern.
        """
        pass
    
    @abstractmethod
    def find_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Find a pattern by its ID.
        
        Args:
            id: The ID of the pattern to find.
            
        Returns:
            The pattern if found, None otherwise.
        """
        pass
    
    @abstractmethod
    def find_by_name(self, name: str) -> List[Dict[str, Any]]:
        """
        Find patterns by their name.
        
        Args:
            name: The name to search for.
            
        Returns:
            A list of patterns with the specified name.
        """
        pass
    
    @abstractmethod
    def find_by_coherence_range(self, min_coherence: float, max_coherence: float) -> List[Dict[str, Any]]:
        """
        Find patterns within a coherence range.
        
        Args:
            min_coherence: The minimum coherence value.
            max_coherence: The maximum coherence value.
            
        Returns:
            A list of patterns within the specified coherence range.
        """
        pass
    
    @abstractmethod
    def find_by_stability_range(self, min_stability: float, max_stability: float) -> List[Dict[str, Any]]:
        """
        Find patterns within a stability range.
        
        Args:
            min_stability: The minimum stability value.
            max_stability: The maximum stability value.
            
        Returns:
            A list of patterns within the specified stability range.
        """
        pass
    
    @abstractmethod
    def find_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Find patterns within a time range.
        
        Args:
            start_time: The start of the time range.
            end_time: The end of the time range.
            
        Returns:
            A list of patterns within the specified time range.
        """
        pass
    
    @abstractmethod
    def find_by_relationship(self, source: str, predicate: str, target: str) -> List[Dict[str, Any]]:
        """
        Find patterns by their relationship structure.
        
        Args:
            source: The source of the relationship.
            predicate: The predicate of the relationship.
            target: The target of the relationship.
            
        Returns:
            A list of patterns with the specified relationship structure.
        """
        pass
    
    @abstractmethod
    def find_by_eigenspace_position(self, position: List[float], radius: float) -> List[Dict[str, Any]]:
        """
        Find patterns by their position in eigenspace.
        
        Args:
            position: The position in eigenspace.
            radius: The radius around the position to search.
            
        Returns:
            A list of patterns within the specified radius of the position.
        """
        pass
    
    @abstractmethod
    def find_by_resonance(self, pattern_id: str, min_resonance: float) -> List[Dict[str, Any]]:
        """
        Find patterns that resonate with a specific pattern.
        
        Args:
            pattern_id: The ID of the pattern to find resonances for.
            min_resonance: The minimum resonance value.
            
        Returns:
            A list of patterns that resonate with the specified pattern.
        """
        pass
