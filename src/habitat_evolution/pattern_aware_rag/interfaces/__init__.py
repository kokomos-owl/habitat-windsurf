"""
Pattern-Aware RAG Interface Components.

Provides interfaces for interacting with the pattern-aware RAG system,
including pattern emergence observation and agent integration.
"""

from .pattern_emergence import (
    PatternEmergenceInterface,
    EmergentPattern,
    PatternState,
    PatternMetrics,
    PatternFeedback,
    PatternEvent,
    PatternEventType
)

__all__ = [
    'PatternEmergenceInterface',
    'EmergentPattern',
    'PatternState',
    'PatternMetrics',
    'PatternFeedback',
    'PatternEvent',
    'PatternEventType'
]
