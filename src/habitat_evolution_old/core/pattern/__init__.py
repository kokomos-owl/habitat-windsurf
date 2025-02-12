"""
Pattern evolution core module.
Handles pattern lifecycle, evolution, and field interactions.
"""

from .evolution import FieldDrivenPatternManager
from .metrics import PatternMetrics
from .types import Pattern, FieldState

__all__ = ['FieldDrivenPatternManager', 'PatternMetrics', 'Pattern', 'FieldState']
