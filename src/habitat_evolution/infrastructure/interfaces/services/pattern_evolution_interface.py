"""
Pattern evolution interface for Habitat Evolution.

This module defines the interface for pattern evolution in the Habitat Evolution system,
providing a consistent approach to tracking and evolving patterns based on usage and feedback.
"""

from typing import Protocol, Dict, List, Any, Optional
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface


class PatternEvolutionInterface(ServiceInterface, Protocol):
    """
    Interface for pattern evolution in Habitat Evolution.
    
    Pattern evolution provides a consistent approach to tracking and evolving patterns
    based on usage and feedback, enabling the system to improve over time. This supports
    the pattern evolution and co-evolution principles of Habitat by enabling patterns to
    evolve through quality states based on contextual evidence.
    """
    
    @abstractmethod
    def track_pattern_usage(self, pattern_id: str, context: Dict[str, Any]) -> None:
        """
        Track the usage of a pattern.
        
        This method records when a pattern is used in a query or document processing,
        updating its usage statistics and potentially its quality state.
        
        Args:
            pattern_id: The ID of the pattern to track
            context: The context in which the pattern was used
        """
        ...
        
    @abstractmethod
    def track_pattern_feedback(self, pattern_id: str, feedback: Dict[str, Any]) -> None:
        """
        Track feedback for a pattern.
        
        This method records feedback for a pattern, updating its quality metrics
        and potentially its quality state.
        
        Args:
            pattern_id: The ID of the pattern to track
            feedback: The feedback to record
        """
        ...
        
    @abstractmethod
    def get_pattern_evolution(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get the evolution history for a pattern.
        
        This method retrieves the evolution history for a pattern, including quality
        state transitions, usage statistics, and relationship changes.
        
        Args:
            pattern_id: The ID of the pattern to get evolution history for
            
        Returns:
            A dictionary containing the pattern evolution history
        """
        ...
        
    @abstractmethod
    def get_pattern_quality(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get the quality metrics for a pattern.
        
        This method retrieves the quality metrics for a pattern, including its
        current quality state, score, and feedback count.
        
        Args:
            pattern_id: The ID of the pattern to get quality metrics for
            
        Returns:
            A dictionary containing the pattern quality metrics
        """
        ...
        
    @abstractmethod
    def update_pattern_quality(self, pattern_id: str, quality_metrics: Dict[str, Any]) -> None:
        """
        Update the quality metrics for a pattern.
        
        This method updates the quality metrics for a pattern, potentially
        transitioning it to a new quality state.
        
        Args:
            pattern_id: The ID of the pattern to update
            quality_metrics: The new quality metrics
        """
        ...
        
    @abstractmethod
    def identify_emerging_patterns(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Identify emerging patterns based on usage and quality metrics.
        
        This method identifies patterns that are emerging as important based on
        their usage and quality metrics.
        
        Args:
            threshold: The threshold for identifying emerging patterns
            
        Returns:
            A list of emerging patterns
        """
        ...
