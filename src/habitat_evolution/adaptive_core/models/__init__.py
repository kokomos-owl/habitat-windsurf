"""
Core models for the Adaptive Core system.
"""

from .pattern import Pattern
from .relationship import Relationship

# Import pattern_metadata to apply the patch
from .pattern_metadata import apply_pattern_metadata_patch

# Apply the patch to ensure Pattern has metadata and text properties
apply_pattern_metadata_patch()

__all__ = [
    'Pattern',
    'Relationship',
    'apply_pattern_metadata_patch'
]
