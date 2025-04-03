"""
Event Repository Interface for Habitat Evolution.

This module defines the interface for persisting and retrieving events
in the Habitat Evolution system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime


class EventRepository(ABC):
    """
    Interface for a repository that manages events.
    
    This repository is responsible for persisting and retrieving events
    related to pattern evolution and system changes.
    """
    
    @abstractmethod
    def save(self, event: Dict[str, Any]) -> str:
        """
        Save an event to the repository.
        
        Args:
            event: The event to save.
            
        Returns:
            The ID of the saved event.
        """
        pass
    
    @abstractmethod
    def find_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Find an event by its ID.
        
        Args:
            id: The ID of the event to find.
            
        Returns:
            The event if found, None otherwise.
        """
        pass
    
    @abstractmethod
    def find_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """
        Find events by their type.
        
        Args:
            event_type: The type of events to find.
            
        Returns:
            A list of events of the specified type.
        """
        pass
    
    @abstractmethod
    def find_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Find events within a time range.
        
        Args:
            start_time: The start of the time range.
            end_time: The end of the time range.
            
        Returns:
            A list of events within the specified time range.
        """
        pass
    
    @abstractmethod
    def find_all(self) -> List[Dict[str, Any]]:
        """
        Find all events.
        
        Returns:
            A list of all events.
        """
        pass
