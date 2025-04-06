"""
Pattern evolution service interface for Habitat Evolution.

This module defines the interface for the pattern evolution service, which is
responsible for managing pattern detection, evolution, and quality assessment.
"""

from typing import Protocol, Any, Dict, List, Optional
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface


class PatternEvolutionServiceInterface(ServiceInterface, Protocol):
    """
    Interface for the pattern evolution service in Habitat Evolution.
    
    The pattern evolution service is responsible for managing pattern detection,
    evolution, and quality assessment. It supports the pattern evolution and
    co-evolution principles of Habitat by tracking how patterns emerge and evolve
    over time, and how they influence each other.
    """
    
    @abstractmethod
    def detect_patterns(self, content: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect patterns in the provided content.
        
        Args:
            content: The content to detect patterns in
            context: Optional context for pattern detection
            
        Returns:
            A list of detected patterns
        """
        ...
        
    @abstractmethod
    def evolve_pattern(self, pattern_id: str, quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Evolve a pattern based on quality metrics.
        
        Args:
            pattern_id: The ID of the pattern to evolve
            quality_metrics: Metrics to use for evolution
            
        Returns:
            The evolved pattern
        """
        ...
        
    @abstractmethod
    def get_pattern(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get a pattern by ID.
        
        Args:
            pattern_id: The ID of the pattern to get
            
        Returns:
            The pattern
        """
        ...
        
    @abstractmethod
    def get_patterns_by_quality(self, min_quality: float) -> List[Dict[str, Any]]:
        """
        Get patterns with quality above the specified threshold.
        
        Args:
            min_quality: The minimum quality threshold
            
        Returns:
            A list of patterns with quality above the threshold
        """
        ...
        
    @abstractmethod
    def create_relationship(self, source_id: str, target_id: str, 
                           relationship_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a relationship between two patterns.
        
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
    def get_relationships(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Get relationships for a pattern.
        
        Args:
            pattern_id: The ID of the pattern to get relationships for
            
        Returns:
            A list of relationships
        """
        ...
