"""
Emergence Package

This package implements components for detecting and tracking emergent patterns
in the Habitat Evolution system without imposing predefined structures.
"""

from .semantic_current_observer import SemanticCurrentObserver
from .emergent_pattern_detector import EmergentPatternDetector
from .resonance_trail_observer import ResonanceTrailObserver
from .pattern_integration import (
    integrate_with_actant_journey_tracker,
    integrate_with_field_navigator,
    integrate_with_field_state,
    setup_emergent_pattern_system
)

# Event bus integration components
from .event_bus_integration import AdaptiveIDEventAdapter, PatternEventPublisher
from .event_aware_detector import EventAwarePatternDetector
from .integration_service import EventBusIntegrationService

__all__ = [
    'SemanticCurrentObserver',
    'EmergentPatternDetector',
    'ResonanceTrailObserver',
    'AdaptiveIDEventAdapter',
    'PatternEventPublisher',
    'EventAwarePatternDetector',
    'EventBusIntegrationService',
    'integrate_with_actant_journey_tracker',
    'integrate_with_field_navigator',
    'integrate_with_field_state',
    'setup_emergent_pattern_system'
]
