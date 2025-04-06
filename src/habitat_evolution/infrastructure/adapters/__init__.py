"""
Adapters for Habitat Evolution.

This package provides adapters for converting between different implementations
of core concepts in the Habitat Evolution system, ensuring compatibility between
components that use different models.
"""

from .pattern_adapter import PatternAdapter
from .pattern_bridge import PatternBridge
from .pattern_monkey_patch import apply_pattern_metadata_patch

# Apply the monkey patch when the package is imported
apply_pattern_metadata_patch()

__all__ = [
    'PatternAdapter',
    'PatternBridge',
    'apply_pattern_metadata_patch'
]
