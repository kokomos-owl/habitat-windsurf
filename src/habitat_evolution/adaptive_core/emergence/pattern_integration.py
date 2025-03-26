"""
Pattern Integration

This module provides integration functions for connecting the emergent pattern
components with existing Habitat systems.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from ..id.adaptive_id import AdaptiveID
from ..transformation.actant_journey_tracker import ActantJourneyTracker
from ...field.field_navigator import FieldNavigator
from ...field.field_state import TonicHarmonicFieldState
from .semantic_current_observer import SemanticCurrentObserver
from .emergent_pattern_detector import EmergentPatternDetector
from .resonance_trail_observer import ResonanceTrailObserver


def integrate_with_actant_journey_tracker(
    semantic_observer: SemanticCurrentObserver, 
    journey_tracker: ActantJourneyTracker
) -> None:
    """
    Integrate the semantic observer with the actant journey tracker.
    
    Args:
        semantic_observer: Observer for semantic currents
        journey_tracker: Tracker for actant journeys
    """
    logger = logging.getLogger(__name__)
    logger.info("Integrating semantic observer with actant journey tracker")
    
    # Register the semantic observer's AdaptiveID with the journey tracker
    semantic_observer.adaptive_id.register_with_learning_window(journey_tracker)
    
    # Set up observation of actant journeys
    for actant_name, journey in journey_tracker.actant_journeys.items():
        if journey.adaptive_id:
            # Register the journey's AdaptiveID with the semantic observer
            journey.adaptive_id.register_with_field_observer(semantic_observer.field_navigator)
            logger.info(f"Registered actant journey for {actant_name} with semantic observer")
    
    # Set up journey tracker to notify semantic observer of new journeys
    class JourneyObserver:
        def __init__(self, semantic_observer):
            self.semantic_observer = semantic_observer
        
        def observe_pattern_evolution(self, context):
            if "new_journey" in context:
                journey = context["new_journey"]
                if journey.adaptive_id:
                    journey.adaptive_id.register_with_field_observer(
                        self.semantic_observer.field_navigator
                    )
    
    # Create and register the observer
    journey_observer = JourneyObserver(semantic_observer)
    journey_tracker.learning_windows.append(journey_observer)
    
    logger.info("Semantic observer successfully integrated with actant journey tracker")


def integrate_with_field_navigator(
    pattern_detector: EmergentPatternDetector, 
    field_navigator: FieldNavigator
) -> None:
    """
    Integrate the pattern detector with the field navigator.
    
    Args:
        pattern_detector: Detector for emergent patterns
        field_navigator: Navigator for the semantic field
    """
    logger = logging.getLogger(__name__)
    logger.info("Integrating pattern detector with field navigator")
    
    # Register the pattern detector's AdaptiveID with the field navigator
    pattern_detector.adaptive_id.register_with_field_observer(field_navigator)
    
    # Set up notification of field changes
    if hasattr(field_navigator, 'add_observer'):
        field_navigator.add_observer(pattern_detector.adaptive_id)
        logger.info("Added pattern detector as field navigator observer")
    else:
        # Alternative approach if add_observer doesn't exist
        class FieldObserver:
            def __init__(self, pattern_detector):
                self.pattern_detector = pattern_detector
            
            def notify(self, event_type, **kwargs):
                if event_type == "field_updated":
                    # Trigger pattern detection on field update
                    self.pattern_detector.detect_patterns()
        
        # Register with observers list if it exists
        if hasattr(field_navigator, 'observers'):
            field_navigator.observers.append(FieldObserver(pattern_detector))
            logger.info("Added pattern detector as field navigator observer (alternative method)")
    
    logger.info("Pattern detector successfully integrated with field navigator")


def integrate_with_field_state(
    resonance_observer: ResonanceTrailObserver, 
    field_state: TonicHarmonicFieldState
) -> None:
    """
    Integrate the resonance observer with the field state.
    
    Args:
        resonance_observer: Observer for resonance trails
        field_state: State of the tonic-harmonic field
    """
    logger = logging.getLogger(__name__)
    logger.info("Integrating resonance observer with field state")
    
    # Register the resonance observer's AdaptiveID with the field state
    resonance_observer.adaptive_id.register_with_field_observer(field_state)
    
    # Set up notification of field state changes
    if hasattr(field_state, 'add_observer'):
        field_state.add_observer(resonance_observer.adaptive_id)
        logger.info("Added resonance observer as field state observer")
    else:
        # Alternative approach if add_observer doesn't exist
        class FieldStateObserver:
            def __init__(self, resonance_observer):
                self.resonance_observer = resonance_observer
            
            def notify(self, event_type, **kwargs):
                if event_type == "field_state_updated" and "patterns" in kwargs:
                    # Track pattern movements
                    for pattern_id, pattern_data in kwargs["patterns"].items():
                        if "old_position" in pattern_data and "new_position" in pattern_data:
                            self.resonance_observer.observe_pattern_movement(
                                pattern_id,
                                pattern_data["old_position"],
                                pattern_data["new_position"],
                                datetime.now().isoformat()
                            )
        
        # Register with observers list if it exists
        if hasattr(field_state, 'observers'):
            field_state.observers.append(FieldStateObserver(resonance_observer))
            logger.info("Added resonance observer as field state observer (alternative method)")
    
    logger.info("Resonance observer successfully integrated with field state")


def setup_emergent_pattern_system(
    journey_tracker: ActantJourneyTracker,
    field_navigator: FieldNavigator,
    field_state: TonicHarmonicFieldState
) -> Dict[str, Any]:
    """
    Set up the complete emergent pattern system.
    
    Args:
        journey_tracker: Tracker for actant journeys
        field_navigator: Navigator for the semantic field
        field_state: State of the tonic-harmonic field
        
    Returns:
        Dictionary containing the created components
    """
    logger = logging.getLogger(__name__)
    logger.info("Setting up emergent pattern system")
    
    # Create components
    semantic_observer = SemanticCurrentObserver(field_navigator, journey_tracker)
    pattern_detector = EmergentPatternDetector(semantic_observer)
    resonance_observer = ResonanceTrailObserver(field_state)
    
    # Integrate components
    integrate_with_actant_journey_tracker(semantic_observer, journey_tracker)
    integrate_with_field_navigator(pattern_detector, field_navigator)
    integrate_with_field_state(resonance_observer, field_state)
    
    logger.info("Emergent pattern system successfully set up")
    
    return {
        "semantic_observer": semantic_observer,
        "pattern_detector": pattern_detector,
        "resonance_observer": resonance_observer
    }
