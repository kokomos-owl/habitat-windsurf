"""
Metrics service interface for Habitat Evolution.

This module defines the interface for the metrics service, which is
responsible for tracking and analyzing various metrics in the system.
"""

from typing import Protocol, Any, Dict, List, Optional, Tuple
from abc import abstractmethod
from datetime import datetime

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface


class MetricsServiceInterface(ServiceInterface, Protocol):
    """
    Interface for the metrics service in Habitat Evolution.
    
    The metrics service is responsible for tracking and analyzing various metrics
    in the system, providing insights into pattern evolution, field dynamics,
    and system performance. It supports the pattern evolution and co-evolution
    principles of Habitat by enabling the observation and analysis of how patterns
    and the semantic field evolve over time.
    """
    
    @abstractmethod
    def track_metric(self, metric_name: str, value: float, 
                    context: Optional[Dict[str, Any]] = None) -> None:
        """
        Track a metric value.
        
        Args:
            metric_name: The name of the metric to track
            value: The value of the metric
            context: Optional context for the metric
        """
        ...
        
    @abstractmethod
    def get_metric_history(self, metric_name: str, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get the history of a metric.
        
        Args:
            metric_name: The name of the metric to get history for
            start_time: Optional start time for the history
            end_time: Optional end time for the history
            
        Returns:
            A list of metric values with timestamps
        """
        ...
        
    @abstractmethod
    def calculate_metric_statistics(self, metric_name: str, 
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None) -> Dict[str, float]:
        """
        Calculate statistics for a metric.
        
        Args:
            metric_name: The name of the metric to calculate statistics for
            start_time: Optional start time for the calculation
            end_time: Optional end time for the calculation
            
        Returns:
            Statistics for the metric (mean, median, min, max, etc.)
        """
        ...
        
    @abstractmethod
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get the current values of all tracked metrics.
        
        Returns:
            A dictionary of current metric values
        """
        ...
        
    @abstractmethod
    def calculate_correlation(self, metric_name1: str, metric_name2: str,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> float:
        """
        Calculate the correlation between two metrics.
        
        Args:
            metric_name1: The name of the first metric
            metric_name2: The name of the second metric
            start_time: Optional start time for the calculation
            end_time: Optional end time for the calculation
            
        Returns:
            The correlation coefficient between the metrics
        """
        ...
