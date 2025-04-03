"""
Quality-aware pattern context for RAG operations.

This package extends the pattern-aware RAG system with quality assessment paths
and context-aware pattern extraction capabilities.
"""

from .quality_aware_context import QualityAwarePatternContext
from .quality_transitions import QualityTransitionTracker

__all__ = [
    'QualityAwarePatternContext',
    'QualityTransitionTracker',
]
