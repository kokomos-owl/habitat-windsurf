"""
Field State Repository Interface for Vector-Tonic Persistence Integration.

This module defines the interface for persisting and retrieving field states
in the Vector-Tonic Window system. It supports the pattern evolution and 
co-evolution principles of Habitat Evolution by enabling the observation
of semantic change across the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime


class FieldStateRepositoryInterface(ABC):
    """
    Interface for a repository that manages field state objects.
    
    This repository is responsible for persisting and retrieving field states,
    which represent the state of the vector-tonic field at a specific point in time.
    Field states capture eigenspace properties, resonance relationships, and metrics
    that are crucial for tracking field state evolution.
    """
    
    @abstractmethod
    def save(self, field_state: Dict[str, Any]) -> str:
        """
        Save a field state to the repository.
        
        Args:
            field_state: The field state to save.
            
        Returns:
            The ID of the saved field state.
        """
        pass
    
    @abstractmethod
    def find_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Find a field state by its ID.
        
        Args:
            id: The ID of the field state to find.
            
        Returns:
            The field state if found, None otherwise.
        """
        pass
    
    @abstractmethod
    def find_by_timestamp(self, timestamp: datetime) -> List[Dict[str, Any]]:
        """
        Find field states by their timestamp.
        
        Args:
            timestamp: The timestamp to search for.
            
        Returns:
            A list of field states with the specified timestamp.
        """
        pass
    
    @abstractmethod
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest field state.
        
        Returns:
            The latest field state if available, None otherwise.
        """
        pass
    
    @abstractmethod
    def find_by_coherence_range(self, min_coherence: float, max_coherence: float) -> List[Dict[str, Any]]:
        """
        Find field states within a coherence range.
        
        Args:
            min_coherence: The minimum coherence value.
            max_coherence: The maximum coherence value.
            
        Returns:
            A list of field states within the specified coherence range.
        """
        pass
    
    @abstractmethod
    def find_by_stability_range(self, min_stability: float, max_stability: float) -> List[Dict[str, Any]]:
        """
        Find field states within a stability range.
        
        Args:
            min_stability: The minimum stability value.
            max_stability: The maximum stability value.
            
        Returns:
            A list of field states within the specified stability range.
        """
        pass
    
    @abstractmethod
    def find_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Find field states within a time range.
        
        Args:
            start_time: The start of the time range.
            end_time: The end of the time range.
            
        Returns:
            A list of field states within the specified time range.
        """
        pass
    
    @abstractmethod
    def find_by_pattern_id(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Find field states that contain a specific pattern.
        
        Args:
            pattern_id: The ID of the pattern to search for.
            
        Returns:
            A list of field states that contain the specified pattern.
        """
        pass
