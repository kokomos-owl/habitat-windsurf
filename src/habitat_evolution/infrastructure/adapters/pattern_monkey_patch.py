"""
Pattern adapter integration for Habitat Evolution.

This module provides compatibility between the Pattern class and the pattern-aware RAG components
by using the PatternAdaptiveIDAdapter, which enhances patterns with versioning, relationship tracking,
and context management capabilities from AdaptiveID.
"""

import logging
from typing import Dict, Any, Optional
from functools import wraps

from src.habitat_evolution.adaptive_core.models.pattern import Pattern
from src.habitat_evolution.infrastructure.adapters.pattern_adaptive_id_adapter import PatternAdaptiveIDAdapter, PatternAdaptiveIDFactory

logger = logging.getLogger(__name__)


def apply_pattern_metadata_patch():
    """
    Apply a compatibility patch to the Pattern class for pattern-aware RAG components.
    
    Instead of directly monkey patching the Pattern class, this function now ensures
    that the Pattern class is compatible with the PatternAdaptiveIDAdapter, which
    provides enhanced capabilities through AdaptiveID integration.
    
    This approach is cleaner and more maintainable than direct monkey patching.
    """
    # Check if the patch has already been applied
    if hasattr(Pattern, '_metadata_patch_applied'):
        return
    
    # Add metadata property that delegates to the adapter pattern
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Get the metadata for this pattern.
        
        This property provides compatibility with the pattern-aware RAG components
        by creating a minimal metadata dictionary. For full capabilities, use
        the PatternAdaptiveIDAdapter.
        
        Returns:
            A dictionary containing the pattern's metadata
        """
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
            "properties": self.properties,
            "text": self.base_concept
        }
    
    # Add text property
    @property
    def text(self) -> str:
        """
        Get the text representation of this pattern.
        
        This property provides compatibility with the quality_enhanced_retrieval.py
        which expects patterns to have a text attribute.
        
        Returns:
            The base concept as the text representation of the pattern
        """
        return self.base_concept
    
    # Add get_adapter method to easily convert a Pattern to a PatternAdaptiveIDAdapter
    def get_adapter(self) -> PatternAdaptiveIDAdapter:
        """
        Get a PatternAdaptiveIDAdapter for this pattern.
        
        This method provides an easy way to access the enhanced capabilities
        of AdaptiveID for any Pattern instance.
        
        Returns:
            A PatternAdaptiveIDAdapter instance for this pattern
        """
        return PatternAdaptiveIDFactory.create_adapter(self)
    
    # Apply the patch
    Pattern.metadata = metadata
    Pattern.text = text
    Pattern.get_adapter = get_adapter
    
    # Mark as patched to avoid duplicate patching
    Pattern._metadata_patch_applied = True
    
    logger.info("Added compatibility methods to Pattern class for AdaptiveID integration")


# The patch is now applied in adaptive_core/models/__init__.py
# Do not apply it here to avoid duplicate patching
