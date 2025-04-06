"""
Pattern metadata extension for Habitat Evolution.

This module adds metadata and text properties to the Pattern class in the adaptive_core module,
ensuring compatibility with the pattern-aware RAG components without modifying the
original class directly.
"""

import logging
from typing import Dict, Any, Optional

from habitat_evolution.adaptive_core.models.pattern import Pattern

logger = logging.getLogger(__name__)


def apply_pattern_metadata_patch():
    """
    Apply a monkey patch to the Pattern class to add metadata and text properties.
    
    This function adds metadata and text properties to the Pattern class without
    modifying the original class file, ensuring backward compatibility while
    providing the necessary functionality for pattern-aware RAG components.
    """
    # Check if the patch has already been applied
    if hasattr(Pattern, '_metadata_patch_applied'):
        return
    
    # Add metadata property
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Get the metadata for this pattern.
        
        This property provides compatibility with the pattern-aware RAG components.
        
        Returns:
            A dictionary containing the pattern's metadata
        """
        return {
            "coherence": getattr(self, "coherence", 0.5),
            "quality": getattr(self, "confidence", 0.5),
            "stability": getattr(self, "phase_stability", 0.5),
            "uncertainty": getattr(self, "uncertainty", 0.5),
            "signal_strength": getattr(self, "signal_strength", 0.5),
            "state": getattr(self, "state", "active"),
            "version": getattr(self, "version", 1),
            "creator_id": getattr(self, "creator_id", ""),
            "weight": getattr(self, "weight", 1.0),
            "metrics": getattr(self, "metrics", {}),
            "properties": getattr(self, "properties", {}),
            "text": getattr(self, "base_concept", "")
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
        return getattr(self, "base_concept", "")
    
    # Apply the patch
    Pattern.metadata = metadata
    Pattern.text = text
    
    # Mark as patched to avoid duplicate patching
    Pattern._metadata_patch_applied = True
    
    logger.info("Added metadata property to AdaptiveCore Pattern class")


# Apply the patch when this module is imported
apply_pattern_metadata_patch()
