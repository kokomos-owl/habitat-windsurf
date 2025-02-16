"""Learning Control System for Pattern-Aware RAG.

This module provides components for managing learning windows and system stability:

1. Learning Windows:
   - Temporal windows for pattern evolution
   - Change rate limiting
   - Coherence thresholds

2. Back Pressure:
   - Adaptive rate control
   - Stability management
   - System protection

3. Event Coordination:
   - Event sequencing
   - Window tracking
   - State coordination

Key Features:
- Temporal learning management
- Stability control
- Event processing
- Change rate limiting

Usage:
    from habitat_evolution.pattern_aware_rag.learning import (
        EventCoordinator,
        LearningWindow,
        BackPressureController
    )
"""

from .learning_control import (
    EventCoordinator,
    LearningWindow,
    BackPressureController
)

__all__ = [
    'EventCoordinator',
    'LearningWindow',
    'BackPressureController'
]