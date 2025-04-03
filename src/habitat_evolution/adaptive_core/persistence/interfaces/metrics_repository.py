"""
Metrics Repository Interface for Habitat Evolution.

This module defines the interface for persisting and retrieving metrics
in the Habitat Evolution system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime


class MetricsRepository(ABC):
    """
    Interface for a repository that manages metrics.
    
    This repository is responsible for persisting and retrieving metrics
    related to pattern evolution and system performance.
    """
    
    @abstractmethod
    def save(self, metrics: Dict[str, Any]) -> str:
        """
        Save metrics to the repository.
        
        Args:
            metrics: The metrics to save.
            
        Returns:
            The ID of the saved metrics.
        """
        pass
    
    @abstractmethod
    def find_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Find metrics by their ID.
        
        Args:
            id: The ID of the metrics to find.
            
        Returns:
            The metrics if found, None otherwise.
        """
        pass
    
    @abstractmethod
    def find_by_type(self, metric_type: str) -> List[Dict[str, Any]]:
        """
        Find metrics by their type.
        
        Args:
            metric_type: The type of metrics to find.
            
        Returns:
            A list of metrics of the specified type.
        """
        pass
    
    @abstractmethod
    def find_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Find metrics within a time range.
        
        Args:
            start_time: The start of the time range.
            end_time: The end of the time range.
            
        Returns:
            A list of metrics within the specified time range.
        """
        pass
    
    @abstractmethod
    def find_all(self) -> List[Dict[str, Any]]:
        """
        Find all metrics.
        
        Returns:
            A list of all metrics.
        """
        pass
