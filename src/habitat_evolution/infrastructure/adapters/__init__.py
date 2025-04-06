"""
Adapters for Habitat Evolution.

This package provides adapters for converting between different implementations
of core concepts in the Habitat Evolution system, ensuring compatibility between
components that use different models.
"""

from .pattern_adapter import PatternAdapter
from .pattern_bridge import PatternBridge

__all__ = [
    'PatternAdapter',
    'PatternBridge'
]
