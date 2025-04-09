"""
Events for the vector-tonic bridge module.

This module defines events for communicating between the vector-tonic
statistical pattern domain and the semantic pattern domain. These events
use Habitat's event bus to enable loose coupling between domains.
"""

from typing import Dict, List, Any, Optional
from src.habitat_evolution.core.services.event_bus import Event
from src.habitat_evolution.core.services.time_provider import TimeProvider


def create_statistical_pattern_detected_event(pattern_id: str, 
                                     pattern_data: Dict[str, Any],
                                     source: str = "vector_tonic") -> Event:
    """
    Create a statistical pattern detected event.
    
    Args:
        pattern_id: Unique identifier for the pattern
        pattern_data: Data describing the pattern
        source: Source of the pattern detection
        
    Returns:
        Event object for the statistical pattern detection
    """
    data = {
        "pattern_id": pattern_id,
        "pattern_data": pattern_data
    }
    
    return Event.create(
        type="statistical_pattern_detected",
        data=data,
        source=source
    )


def create_statistical_pattern_quality_changed_event(pattern_id: str, 
                                           previous_state: str,
                                           new_state: str,
                                           source: str = "vector_tonic") -> Event:
    """
    Create a statistical pattern quality changed event.
    
    Args:
        pattern_id: Unique identifier for the pattern
        previous_state: Previous quality state
        new_state: New quality state
        source: Source of the pattern quality change
        
    Returns:
        Event object for the statistical pattern quality change
    """
    data = {
        "pattern_id": pattern_id,
        "previous_state": previous_state,
        "new_state": new_state
    }
    
    return Event.create(
        type="statistical_pattern_quality_changed",
        data=data,
        source=source
    )


def create_statistical_relationship_detected_event(relationship_id: str,
                                            source_pattern_id: str,
                                            target_pattern_id: str,
                                            relationship_type: str,
                                            relationship_data: Dict[str, Any],
                                            source: str = "vector_tonic") -> Event:
    """
    Create a statistical relationship detected event.
    
    Args:
        relationship_id: Unique identifier for the relationship
        source_pattern_id: ID of the source pattern
        target_pattern_id: ID of the target pattern
        relationship_type: Type of relationship
        relationship_data: Data describing the relationship
        source: Source of the relationship detection
        
    Returns:
        Event object for the statistical relationship detection
    """
    data = {
        "relationship_id": relationship_id,
        "source_pattern_id": source_pattern_id,
        "target_pattern_id": target_pattern_id,
        "relationship_type": relationship_type,
        "relationship_data": relationship_data
    }
    
    return Event.create(
        type="statistical_relationship_detected",
        data=data,
        source=source
    )


def create_cross_domain_correlation_detected_event(correlation_id: str,
                                              statistical_pattern_id: str,
                                              semantic_pattern_id: str,
                                              correlation_strength: float,
                                              correlation_type: str,
                                              correlation_data: Dict[str, Any],
                                              source: str = "pattern_bridge") -> Event:
    """
    Create a cross-domain correlation detected event.
    
    Args:
        correlation_id: Unique identifier for the correlation
        statistical_pattern_id: ID of the statistical pattern
        semantic_pattern_id: ID of the semantic pattern
        correlation_strength: Strength of the correlation (0-1)
        correlation_type: Type of correlation
        correlation_data: Data describing the correlation
        source: Source of the correlation detection
        
    Returns:
        Event object for the cross-domain correlation detection
    """
    data = {
        "correlation_id": correlation_id,
        "statistical_pattern_id": statistical_pattern_id,
        "semantic_pattern_id": semantic_pattern_id,
        "correlation_strength": correlation_strength,
        "correlation_type": correlation_type,
        "correlation_data": correlation_data
    }
    
    return Event.create(
        type="cross_domain_correlation_detected",
        data=data,
        source=source
    )


def create_co_pattern_created_event(co_pattern_id: str,
                                  statistical_pattern_id: str,
                                  semantic_pattern_id: str,
                                  quality_state: str,
                                  co_pattern_data: Dict[str, Any],
                                  source: str = "pattern_bridge") -> Event:
    """
    Create a co-pattern created event.
    
    Args:
        co_pattern_id: Unique identifier for the co-pattern
        statistical_pattern_id: ID of the statistical pattern
        semantic_pattern_id: ID of the semantic pattern
        quality_state: Quality state of the co-pattern
        co_pattern_data: Data describing the co-pattern
        source: Source of the co-pattern creation
        
    Returns:
        Event object for the co-pattern creation
    """
    data = {
        "co_pattern_id": co_pattern_id,
        "statistical_pattern_id": statistical_pattern_id,
        "semantic_pattern_id": semantic_pattern_id,
        "quality_state": quality_state,
        "co_pattern_data": co_pattern_data
    }
    
    return Event.create(
        type="co_pattern_created",
        data=data,
        source=source
    )
