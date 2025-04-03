"""
Context-aware pattern extraction with quality assessment paths.

This package implements a self-reinforcing feedback mechanism for pattern extraction
that combines sliding window approaches with context awareness and quality assessment.
"""

from .context_aware_extractor import ContextAwareExtractor
from .entity_context_manager import EntityContextManager
from .quality_assessment import QualityAssessment

__all__ = [
    'ContextAwareExtractor',
    'EntityContextManager',
    'QualityAssessment',
]
