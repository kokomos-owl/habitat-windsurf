"""
Adapters for Habitat Evolution.

This package provides adapters for converting between different implementations
of core concepts in the Habitat Evolution system, ensuring compatibility between
components that use different models.
"""

from .pattern_adapter import PatternAdapter
from .pattern_bridge import PatternBridge
# Import but don't apply the patch - it's already applied in adaptive_core
from .pattern_monkey_patch import apply_pattern_metadata_patch

__all__ = [
    'PatternAdapter',
    'PatternBridge',
    'apply_pattern_metadata_patch'
]
