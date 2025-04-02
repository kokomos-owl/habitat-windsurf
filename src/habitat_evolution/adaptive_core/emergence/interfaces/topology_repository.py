"""
Topology Repository Interface for Vector-Tonic Persistence Integration.

This module defines the interface for persisting and retrieving topology constructs
in the Vector-Tonic Window system. It supports the pattern evolution and 
co-evolution principles of Habitat Evolution by enabling the observation
of semantic change across the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime


class TopologyRepositoryInterface(ABC):
    """
    Interface for a repository that manages topology constructs.
    
    This repository is responsible for persisting and retrieving topology constructs,
    which represent the topological features of the vector-tonic field.
    Topology constructs capture frequency domains, boundaries, and resonance points
    that are crucial for tracking field evolution.
    """
    
    @abstractmethod
    def save(self, topology: Dict[str, Any]) -> str:
        """
        Save a topology construct to the repository.
        
        Args:
            topology: The topology construct to save.
            
        Returns:
            The ID of the saved topology construct.
        """
        pass
    
    @abstractmethod
    def find_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Find a topology construct by its ID.
        
        Args:
            id: The ID of the topology construct to find.
            
        Returns:
            The topology construct if found, None otherwise.
        """
        pass
    
    @abstractmethod
    def find_by_type(self, topology_type: str) -> List[Dict[str, Any]]:
        """
        Find topology constructs by their type.
        
        Args:
            topology_type: The type to search for (e.g., 'frequency_domain', 'boundary', 'resonance_point').
            
        Returns:
            A list of topology constructs with the specified type.
        """
        pass
    
    @abstractmethod
    def find_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Find topology constructs within a time range.
        
        Args:
            start_time: The start of the time range.
            end_time: The end of the time range.
            
        Returns:
            A list of topology constructs within the specified time range.
        """
        pass
    
    @abstractmethod
    def find_by_pattern_id(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Find topology constructs that involve a specific pattern.
        
        Args:
            pattern_id: The ID of the pattern to search for.
            
        Returns:
            A list of topology constructs that involve the specified pattern.
        """
        pass
    
    @abstractmethod
    def find_by_eigenspace_region(self, center: List[float], radius: float) -> List[Dict[str, Any]]:
        """
        Find topology constructs within a region of eigenspace.
        
        Args:
            center: The center of the region in eigenspace.
            radius: The radius of the region.
            
        Returns:
            A list of topology constructs within the specified region.
        """
        pass
    
    @abstractmethod
    def find_boundaries_between_patterns(self, pattern_id_1: str, pattern_id_2: str) -> List[Dict[str, Any]]:
        """
        Find boundaries between two specific patterns.
        
        Args:
            pattern_id_1: The ID of the first pattern.
            pattern_id_2: The ID of the second pattern.
            
        Returns:
            A list of boundary constructs between the specified patterns.
        """
        pass
    
    @abstractmethod
    def find_frequency_domains_by_frequency_range(self, min_frequency: float, max_frequency: float) -> List[Dict[str, Any]]:
        """
        Find frequency domains within a frequency range.
        
        Args:
            min_frequency: The minimum frequency.
            max_frequency: The maximum frequency.
            
        Returns:
            A list of frequency domain constructs within the specified frequency range.
        """
        pass
    
    @abstractmethod
    def find_resonance_points_by_strength(self, min_strength: float) -> List[Dict[str, Any]]:
        """
        Find resonance points with a minimum strength.
        
        Args:
            min_strength: The minimum resonance strength.
            
        Returns:
            A list of resonance point constructs with at least the specified strength.
        """
        pass
    
    @abstractmethod
    def get_latest_topology_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest overall topology state.
        
        Returns:
            The latest topology state if available, None otherwise.
        """
        pass
