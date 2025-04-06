"""
Pattern adapter for Habitat Evolution.

This module provides adapters for converting between different Pattern implementations
in the Habitat Evolution system, ensuring compatibility between components.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.habitat_evolution.adaptive_core.models.pattern import Pattern as AdaptiveCorePattern
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_pattern_repository import Pattern as ArangoDBPattern

logger = logging.getLogger(__name__)


class PatternAdapter:
    """
    Adapter for converting between different Pattern implementations.
    
    This adapter provides methods for converting between the AdaptiveCore Pattern
    and the ArangoDB Pattern implementations, ensuring compatibility between
    components that use different Pattern models.
    """
    
    @staticmethod
    def adaptive_core_to_arangodb(pattern: AdaptiveCorePattern) -> ArangoDBPattern:
        """
        Convert an AdaptiveCore Pattern to an ArangoDB Pattern.
        
        Args:
            pattern: The AdaptiveCore Pattern to convert
            
        Returns:
            The equivalent ArangoDB Pattern
        """
        # Create metadata from AdaptiveCore Pattern properties
        metadata = {
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
            "properties": pattern.properties
        }
        
        # Create ArangoDB Pattern
        return ArangoDBPattern(
            id=pattern.id,
            name=pattern.base_concept,
            pattern_type="adaptive_core",
            description=f"Pattern for {pattern.base_concept}",
            metadata=metadata,
            created_at=pattern.created_at,
            updated_at=pattern.last_modified
        )
    
    @staticmethod
    def arangodb_to_adaptive_core(pattern: ArangoDBPattern) -> AdaptiveCorePattern:
        """
        Convert an ArangoDB Pattern to an AdaptiveCore Pattern.
        
        Args:
            pattern: The ArangoDB Pattern to convert
            
        Returns:
            The equivalent AdaptiveCore Pattern
        """
        # Extract metadata from ArangoDB Pattern
        metadata = pattern.metadata or {}
        
        # Create AdaptiveCore Pattern
        return AdaptiveCorePattern(
            id=pattern.id or "",
            base_concept=pattern.name or "",
            creator_id=metadata.get("creator_id", "system"),
            weight=metadata.get("weight", 1.0),
            confidence=metadata.get("quality", 1.0),
            uncertainty=metadata.get("uncertainty", 0.0),
            coherence=metadata.get("coherence", 0.0),
            phase_stability=metadata.get("stability", 0.0),
            signal_strength=metadata.get("signal_strength", 0.0),
            properties=metadata.get("properties", {}),
            metrics=metadata.get("metrics", {}),
            relationships=[],
            created_at=pattern.created_at or datetime.utcnow().isoformat(),
            last_modified=pattern.updated_at or datetime.utcnow().isoformat(),
            state=metadata.get("state", "EMERGING"),
            version=metadata.get("version", "1.0")
        )
    
    @staticmethod
    def add_metadata_to_adaptive_core_pattern(pattern: AdaptiveCorePattern) -> AdaptiveCorePattern:
        """
        Add a metadata property to an AdaptiveCore Pattern for compatibility.
        
        This method adds a metadata property to an AdaptiveCore Pattern to make it
        compatible with code that expects a Pattern with a metadata attribute.
        
        Args:
            pattern: The AdaptiveCore Pattern to modify
            
        Returns:
            The modified AdaptiveCore Pattern
        """
        # Create metadata dictionary
        metadata = {
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
            "properties": pattern.properties
        }
        
        # Add metadata property to pattern
        setattr(pattern, 'metadata', metadata)
        
        return pattern
    
    @staticmethod
    def patch_adaptive_core_pattern_class():
        """
        Patch the AdaptiveCore Pattern class to add a metadata property.
        
        This method adds a metadata property to the AdaptiveCore Pattern class
        to make it compatible with code that expects a Pattern with a metadata
        attribute.
        """
        # Define metadata property getter
        def get_metadata(self):
            return {
                "coherence": self.coherence,
                "quality": self.confidence,
                "stability": self.phase_stability,
                "uncertainty": self.uncertainty,
                "signal_strength": self.signal_strength,
                "state": self.state,
                "version": self.version,
                "creator_id": self.creator_id,
                "weight": self.weight,
                "metrics": self.metrics,
                "properties": self.properties
            }
        
        # Add metadata property to AdaptiveCore Pattern class
        if not hasattr(AdaptiveCorePattern, 'metadata'):
            AdaptiveCorePattern.metadata = property(get_metadata)
            logger.info("Added metadata property to AdaptiveCore Pattern class")


# Patch the AdaptiveCore Pattern class when this module is imported
PatternAdapter.patch_adaptive_core_pattern_class()
