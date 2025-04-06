"""
Pattern bridge for Habitat Evolution.

This module provides a bridge between different Pattern implementations
in the Habitat Evolution system, ensuring compatibility between components
without requiring modifications to the original classes.
"""

import logging
from typing import Dict, Any, List, Optional, TypeVar, Generic, Type, Union
from datetime import datetime

from src.habitat_evolution.adaptive_core.models.pattern import Pattern as AdaptiveCorePattern
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_pattern_repository import Pattern as ArangoDBPattern
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface

logger = logging.getLogger(__name__)

# Type variable for the pattern type
P = TypeVar('P')


class PatternBridge(Generic[P]):
    """
    Bridge for different Pattern implementations.
    
    This class provides a bridge between different Pattern implementations,
    ensuring compatibility between components without requiring modifications
    to the original classes.
    """
    
    def __init__(self, 
                 event_service: EventServiceInterface,
                 pattern_class: Type[P]):
        """
        Initialize a new pattern bridge.
        
        Args:
            event_service: The event service for publishing events
            pattern_class: The pattern class to bridge to
        """
        self._event_service = event_service
        self._pattern_class = pattern_class
        self._metadata_cache = {}  # id -> metadata
        logger.debug(f"PatternBridge created for {pattern_class.__name__}")
    
    def get_metadata(self, pattern: P) -> Dict[str, Any]:
        """
        Get metadata for a pattern.
        
        Args:
            pattern: The pattern to get metadata for
            
        Returns:
            The metadata for the pattern
        """
        # Check if pattern already has metadata attribute
        if hasattr(pattern, 'metadata'):
            return getattr(pattern, 'metadata')
            
        # Check if pattern is in cache
        pattern_id = self._get_pattern_id(pattern)
        if pattern_id in self._metadata_cache:
            return self._metadata_cache[pattern_id]
            
        # Create metadata based on pattern type
        if isinstance(pattern, AdaptiveCorePattern):
            metadata = self._create_metadata_for_adaptive_core_pattern(pattern)
        elif isinstance(pattern, ArangoDBPattern):
            metadata = pattern.metadata or {}
        else:
            # Default metadata
            metadata = self._create_default_metadata(pattern)
            
        # Cache metadata
        self._metadata_cache[pattern_id] = metadata
        
        return metadata
    
    def set_metadata(self, pattern: P, metadata: Dict[str, Any]) -> None:
        """
        Set metadata for a pattern.
        
        Args:
            pattern: The pattern to set metadata for
            metadata: The metadata to set
        """
        # Check if pattern already has metadata attribute
        if hasattr(pattern, 'metadata') and hasattr(pattern, 'metadata.update'):
            # If pattern has a metadata attribute that's a dict, update it
            getattr(pattern, 'metadata').update(metadata)
        else:
            # Otherwise, store in cache
            pattern_id = self._get_pattern_id(pattern)
            self._metadata_cache[pattern_id] = metadata
    
    def get_text(self, pattern: P) -> str:
        """
        Get text for a pattern.
        
        Args:
            pattern: The pattern to get text for
            
        Returns:
            The text for the pattern
        """
        # Check if pattern already has text attribute
        if hasattr(pattern, 'text'):
            return getattr(pattern, 'text')
            
        # Check if pattern has name attribute
        if hasattr(pattern, 'name'):
            return getattr(pattern, 'name')
            
        # Check if pattern has base_concept attribute
        if hasattr(pattern, 'base_concept'):
            return getattr(pattern, 'base_concept')
            
        # Default to string representation
        return str(pattern)
    
    def enhance_pattern(self, pattern: P) -> P:
        """
        Enhance a pattern with metadata and text attributes.
        
        This method adds metadata and text attributes to a pattern object
        without modifying the original class. The attributes are added
        dynamically to the instance.
        
        Args:
            pattern: The pattern to enhance
            
        Returns:
            The enhanced pattern
        """
        # Get metadata
        metadata = self.get_metadata(pattern)
        
        # Get text
        text = self.get_text(pattern)
        
        # Add metadata attribute if not present
        if not hasattr(pattern, 'metadata'):
            setattr(pattern, 'metadata', metadata)
            
        # Add text attribute if not present
        if not hasattr(pattern, 'text'):
            setattr(pattern, 'text', text)
            
        return pattern
    
    def enhance_patterns(self, patterns: List[P]) -> List[P]:
        """
        Enhance a list of patterns with metadata and text attributes.
        
        Args:
            patterns: The patterns to enhance
            
        Returns:
            The enhanced patterns
        """
        return [self.enhance_pattern(p) for p in patterns]
    
    def _get_pattern_id(self, pattern: P) -> str:
        """
        Get the ID of a pattern.
        
        Args:
            pattern: The pattern to get the ID for
            
        Returns:
            The ID of the pattern
        """
        # Check if pattern has id attribute
        if hasattr(pattern, 'id'):
            return getattr(pattern, 'id')
            
        # Check if pattern has _id attribute
        if hasattr(pattern, '_id'):
            return getattr(pattern, '_id')
            
        # Default to object ID
        return str(id(pattern))
    
    def _create_metadata_for_adaptive_core_pattern(self, pattern: AdaptiveCorePattern) -> Dict[str, Any]:
        """
        Create metadata for an AdaptiveCore Pattern.
        
        Args:
            pattern: The AdaptiveCore Pattern to create metadata for
            
        Returns:
            The metadata for the pattern
        """
        return {
            "coherence": pattern.coherence,
            "quality": pattern.confidence,
            "stability": pattern.phase_stability,
            "uncertainty": pattern.uncertainty,
            "signal_strength": pattern.signal_strength,
            "state": pattern.state,
            "version": pattern.version,
            "creator_id": pattern.creator_id,
            "weight": pattern.weight,
            "metrics": pattern.metrics,
            "properties": pattern.properties,
            "text": pattern.base_concept
        }
    
    def _create_default_metadata(self, pattern: P) -> Dict[str, Any]:
        """
        Create default metadata for a pattern.
        
        Args:
            pattern: The pattern to create metadata for
            
        Returns:
            The default metadata for the pattern
        """
        # Create basic metadata
        metadata = {
            "coherence": 0.0,
            "quality": 0.5,
            "stability": 0.0,
            "uncertainty": 1.0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Add text if available
        text = self.get_text(pattern)
        if text:
            metadata["text"] = text
            
        return metadata
