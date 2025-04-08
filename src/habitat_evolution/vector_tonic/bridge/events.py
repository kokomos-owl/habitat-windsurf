"""
Events for the vector-tonic bridge module.

This module defines events for communicating between the vector-tonic
statistical pattern domain and the semantic pattern domain. These events
use Habitat's event bus to enable loose coupling between domains.
"""

from typing import Dict, List, Any, Optional
from src.habitat_evolution.core.services.event_bus import Event


class StatisticalPatternDetectedEvent(Event):
    """Event fired when a statistical pattern is detected in time-series data."""
    
    def __init__(self, 
                pattern_id: str, 
                pattern_data: Dict[str, Any],
                source: str = "vector_tonic"):
        """
        Initialize a new statistical pattern detected event.
        
        Args:
            pattern_id: Unique identifier for the pattern
            pattern_data: Data describing the pattern
            source: Source of the pattern detection
        """
        super().__init__(event_type="statistical_pattern_detected")
        self.pattern_id = pattern_id
        self.pattern_data = pattern_data
        self.source = source


class StatisticalPatternQualityChangedEvent(Event):
    """Event fired when a statistical pattern's quality state changes."""
    
    def __init__(self, 
                pattern_id: str, 
                previous_state: str,
                new_state: str,
                source: str = "vector_tonic"):
        """
        Initialize a new statistical pattern quality changed event.
        
        Args:
            pattern_id: Unique identifier for the pattern
            previous_state: Previous quality state
            new_state: New quality state
            source: Source of the pattern quality change
        """
        super().__init__(event_type="statistical_pattern_quality_changed")
        self.pattern_id = pattern_id
        self.previous_state = previous_state
        self.new_state = new_state
        self.source = source


class StatisticalRelationshipDetectedEvent(Event):
    """Event fired when a relationship between statistical patterns is detected."""
    
    def __init__(self, 
                relationship_id: str,
                source_pattern_id: str,
                target_pattern_id: str,
                relationship_type: str,
                relationship_data: Dict[str, Any],
                source: str = "vector_tonic"):
        """
        Initialize a new statistical relationship detected event.
        
        Args:
            relationship_id: Unique identifier for the relationship
            source_pattern_id: ID of the source pattern
            target_pattern_id: ID of the target pattern
            relationship_type: Type of relationship
            relationship_data: Data describing the relationship
            source: Source of the relationship detection
        """
        super().__init__(event_type="statistical_relationship_detected")
        self.relationship_id = relationship_id
        self.source_pattern_id = source_pattern_id
        self.target_pattern_id = target_pattern_id
        self.relationship_type = relationship_type
        self.relationship_data = relationship_data
        self.source = source


class CrossDomainCorrelationDetectedEvent(Event):
    """Event fired when a correlation between statistical and semantic patterns is detected."""
    
    def __init__(self, 
                correlation_id: str,
                statistical_pattern_id: str,
                semantic_pattern_id: str,
                correlation_strength: float,
                correlation_type: str,
                correlation_data: Dict[str, Any],
                source: str = "pattern_bridge"):
        """
        Initialize a new cross-domain correlation detected event.
        
        Args:
            correlation_id: Unique identifier for the correlation
            statistical_pattern_id: ID of the statistical pattern
            semantic_pattern_id: ID of the semantic pattern
            correlation_strength: Strength of the correlation (0-1)
            correlation_type: Type of correlation
            correlation_data: Data describing the correlation
            source: Source of the correlation detection
        """
        super().__init__(event_type="cross_domain_correlation_detected")
        self.correlation_id = correlation_id
        self.statistical_pattern_id = statistical_pattern_id
        self.semantic_pattern_id = semantic_pattern_id
        self.correlation_strength = correlation_strength
        self.correlation_type = correlation_type
        self.correlation_data = correlation_data
        self.source = source


class CoPatternCreatedEvent(Event):
    """Event fired when a co-pattern is created from correlated patterns."""
    
    def __init__(self, 
                co_pattern_id: str,
                statistical_pattern_id: str,
                semantic_pattern_id: str,
                quality_state: str,
                co_pattern_data: Dict[str, Any],
                source: str = "pattern_bridge"):
        """
        Initialize a new co-pattern created event.
        
        Args:
            co_pattern_id: Unique identifier for the co-pattern
            statistical_pattern_id: ID of the statistical pattern
            semantic_pattern_id: ID of the semantic pattern
            quality_state: Quality state of the co-pattern
            co_pattern_data: Data describing the co-pattern
            source: Source of the co-pattern creation
        """
        super().__init__(event_type="co_pattern_created")
        self.co_pattern_id = co_pattern_id
        self.statistical_pattern_id = statistical_pattern_id
        self.semantic_pattern_id = semantic_pattern_id
        self.quality_state = quality_state
        self.co_pattern_data = co_pattern_data
        self.source = source
