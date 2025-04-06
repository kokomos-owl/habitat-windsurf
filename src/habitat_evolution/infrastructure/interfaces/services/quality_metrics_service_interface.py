"""
Quality metrics service interface for Habitat Evolution.

This module defines the interface for the quality metrics service, which is
responsible for tracking and analyzing quality metrics for patterns.
"""

from typing import Protocol, Any, Dict, List, Optional, Tuple
from abc import abstractmethod
from datetime import datetime

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface


class QualityMetricsServiceInterface(ServiceInterface, Protocol):
    """
    Interface for the quality metrics service in Habitat Evolution.
    
    The quality metrics service is responsible for tracking and analyzing quality
    metrics for patterns, providing insights into pattern quality, stability,
    coherence, and other quality-related aspects. It supports the pattern evolution
    and co-evolution principles of Habitat by enabling the observation and analysis
    of how pattern quality evolves over time.
    """
    
    @abstractmethod
    def calculate_pattern_quality(self, pattern: Dict[str, Any], 
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate quality metrics for a pattern.
        
        Args:
            pattern: The pattern to calculate quality for
            context: Optional context for the calculation
            
        Returns:
            A dictionary of quality metrics
        """
        ...
        
    @abstractmethod
    def track_quality_transition(self, pattern_id: str, 
                               previous_quality: str,
                               new_quality: str,
                               metrics: Dict[str, float]) -> None:
        """
        Track a quality state transition for a pattern.
        
        Args:
            pattern_id: The ID of the pattern
            previous_quality: The previous quality state
            new_quality: The new quality state
            metrics: The quality metrics associated with the transition
        """
        ...
        
    @abstractmethod
    def get_quality_history(self, pattern_id: str,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get the quality history for a pattern.
        
        Args:
            pattern_id: The ID of the pattern
            start_time: Optional start time for the history
            end_time: Optional end time for the history
            
        Returns:
            A list of quality states with timestamps
        """
        ...
        
    @abstractmethod
    def calculate_quality_distribution(self) -> Dict[str, int]:
        """
        Calculate the distribution of patterns across quality states.
        
        Returns:
            A dictionary mapping quality states to pattern counts
        """
        ...
        
    @abstractmethod
    def get_patterns_by_quality_state(self, quality_state: str) -> List[str]:
        """
        Get patterns with a specific quality state.
        
        Args:
            quality_state: The quality state to filter by
            
        Returns:
            A list of pattern IDs with the specified quality state
        """
        ...
        
    @abstractmethod
    def calculate_quality_transition_probabilities(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate transition probabilities between quality states.
        
        Returns:
            A dictionary mapping source states to target states with probabilities
        """
        ...
