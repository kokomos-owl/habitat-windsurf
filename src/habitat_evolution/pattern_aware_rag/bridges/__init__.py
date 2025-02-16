"""Integration Bridges for Pattern-Aware RAG.

This module provides bridge components that integrate different parts of the system:

1. Adaptive State Bridge:
   - Connects graph state evolution with adaptive ID system
   - Manages state transitions and pattern evolution
   - Maintains coherence between different representations

Key Responsibilities:
- State transition management
- Pattern evolution tracking
- Coherence maintenance
- Version history tracking

Usage:
    from habitat_evolution.pattern_aware_rag.bridges import AdaptiveStateManager
"""

from .adaptive_state_bridge import AdaptiveStateManager

__all__ = ['AdaptiveStateManager']