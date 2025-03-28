"""
Vector-Tonic-Harmonic Learning Window Integration

This module provides enhanced integration between the learning window system
and the vector-tonic-harmonic validation, enabling progressive preparation
during the OPENING state and creating a feedback loop between harmonic analysis
and learning window control.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
import numpy as np
from datetime import datetime, timedelta
from collections import deque

# Use absolute imports to avoid module path issues
from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from src.habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService, OperationType
from src.habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics
from src.habitat_evolution.adaptive_core.emergence.learning_window_integration import LearningWindowAwareDetector
from src.habitat_evolution.adaptive_core.emergence.tonic_harmonic_integration import TonicHarmonicPatternDetector
from src.habitat_evolution.pattern_aware_rag.learning.learning_control import WindowState, BackPressureController, LearningWindow

logger = logging.getLogger(__name__)


class FieldStateModulator:
    """
    Modulates field state based on tonic-harmonic resonance patterns.
    
    Instead of applying binary back pressure, this class modulates the field state
    to allow for natural emergence of patterns through resonance, dissonance, and
    field density interactions, preserving the errant, fuzzy, turbulent, and
    dissonant characteristics of Habitat.
    """
    
    def __init__(self):
        # Resonance history tracks pattern emergence over time
        self.resonance_history = deque(maxlen=50)
        
        # Field topology state with continuity tracking
        self._previous_field_metrics = {
            'density': 0.5,
            'turbulence': 0.3,
            'coherence': 0.7,
            'stability': 0.7
        }
        self.field_density = 0.5  # 0.0 to 1.0
        self.field_turbulence = 0.3  # 0.0 to 1.0
        self.field_coherence = 0.7  # 0.0 to 1.0
        self.field_stability = 0.7  # 0.0 to 1.0
        
        # Temporal evolution parameters
        self.continuity_factor = 0.7  # How much previous state influences current (higher = more continuous)
        self.resonance_decay_rate = 0.2  # Rate at which resonance decays
        
        # Resonance windows for unusual pattern emergence
        self.resonance_windows = []
        
        # Pattern relationship tracking
        self.pattern_relationships = {}
        self.relationship_strength_history = deque(maxlen=50)
        
        # Adaptive pattern type receptivity with learning
        self.pattern_receptivity = {
            'primary': 1.0,
            'secondary': 0.8,
            'meta': 0.7,
            'emergent': 0.6
        }
        self.receptivity_learning_rate = 0.05
        self.receptivity_history = {k: deque(maxlen=20) for k in self.pattern_receptivity.keys()}
        
        # Interference patterns from recently detected patterns
        self.interference_patterns = {}
        
        # Visualization data collection
        self.visualization_data = {
            'field_state_history': deque(maxlen=100),
            'pattern_emergence_points': [],
            'resonance_centers': {},
            'interference_patterns': {}
        }
        
        logger.info("Enhanced FieldStateModulator initialized with field continuity")
    
    def record_pattern_emergence(self, pattern_data: Dict[str, Any]):
        """
        Record pattern emergence to update field state.
        
        Args:
            pattern_data: Data about the emerged pattern
        """
        # Extract pattern metadata
        pattern_id = pattern_data.get('id', 'unknown')
        pattern_type = pattern_data.get('context', {}).get('cascade_type', 'secondary')
        confidence = pattern_data.get('confidence', 0.5)
        
        # Record in resonance history
        self.resonance_history.append({
            'timestamp': datetime.now(),
            'pattern_id': pattern_id,
            'pattern_type': pattern_type,
            'confidence': confidence
        })
        
        # Update field topology
        self._update_field_topology()
        
        # Create interference pattern
        self._create_interference_pattern(pattern_id, pattern_type, confidence)
    
    def _update_field_topology(self):
        """
        Update field topology based on resonance history with continuity.
        
        This method ensures the field evolves continuously rather than being
        recalculated from scratch with each detection cycle, creating a more
        natural flow dynamics rather than discrete snapshots.
        """
        if not self.resonance_history:
            return
        
        # Calculate recent pattern density (patterns per time unit)
        recent_count = len([r for r in self.resonance_history 
                           if (datetime.now() - r['timestamp']).total_seconds() < 5])
        
        # Calculate pattern type diversity
        pattern_types = set(r['pattern_type'] for r in self.resonance_history)
        type_diversity = len(pattern_types) / max(1, len(self.resonance_history))
        
        # Calculate current field metrics
        current_density = min(1.0, recent_count / 10)  # Normalize to 0-1
        
        # Calculate confidence variance for turbulence
        confidence_values = [r['confidence'] for r in self.resonance_history]
        confidence_variance = np.var(confidence_values) if len(confidence_values) > 1 else 0
        current_turbulence = (type_diversity * 0.5) + (confidence_variance * 0.5)
        
        # Calculate coherence based on pattern relationships
        relationship_strength = [v for v in self.pattern_relationships.values()]
        current_coherence = sum(relationship_strength) / max(1, len(relationship_strength))
        current_coherence = 0.3 + (current_coherence * 0.7)  # Baseline of 0.3
        
        # Calculate stability as inverse of turbulence with smoothing
        current_stability = 1.0 - (current_turbulence * 0.8)
        
        # Apply field state continuity - blend current with previous state
        # This creates a continuous flow rather than discrete snapshots
        self.field_density = ((self._previous_field_metrics['density'] * self.continuity_factor) + 
                             (current_density * (1 - self.continuity_factor)))
        
        self.field_turbulence = ((self._previous_field_metrics['turbulence'] * self.continuity_factor) + 
                                (current_turbulence * (1 - self.continuity_factor)))
        
        self.field_coherence = ((self._previous_field_metrics['coherence'] * self.continuity_factor) + 
                               (current_coherence * (1 - self.continuity_factor)))
        
        self.field_stability = ((self._previous_field_metrics['stability'] * self.continuity_factor) + 
                               (current_stability * (1 - self.continuity_factor)))
        
        # Update previous metrics for next cycle - maintaining continuity
        self._previous_field_metrics['density'] = self.field_density
        self._previous_field_metrics['turbulence'] = self.field_turbulence
        self._previous_field_metrics['coherence'] = self.field_coherence
        self._previous_field_metrics['stability'] = self.field_stability
        
        # Store field state history for visualization
        self.visualization_data['field_state_history'].append({
            'timestamp': datetime.now(),
            'density': self.field_density,
            'turbulence': self.field_turbulence,
            'coherence': self.field_coherence,
            'stability': self.field_stability
        })
        
        # Decay existing interference patterns
        self._decay_interference_patterns()
        
        # Create resonance windows during periods of high turbulence
        # Adaptive threshold based on field density
        resonance_threshold = 0.7 - (self.field_density * 0.2)
        if self.field_turbulence > resonance_threshold and np.random.random() < 0.3:
            self._create_resonance_window()
    
    def _create_interference_pattern(self, pattern_id: str, pattern_type: str, confidence: float):
        """
        Create an interference pattern from a detected pattern.
        
        Args:
            pattern_id: ID of the pattern
            pattern_type: Type of the pattern
            confidence: Confidence score of the pattern
        """
        # Interference strength based on confidence and pattern type
        type_factor = {
            'primary': 1.0,
            'secondary': 0.8,
            'meta': 1.2,
            'emergent': 0.7
        }.get(pattern_type, 0.5)
        
        strength = confidence * type_factor
        
        # Interference duration based on pattern importance
        duration = 5 + (confidence * 10)  # 5-15 seconds
        
        # Record interference pattern
        self.interference_patterns[pattern_id] = {
            'created_at': datetime.now(),
            'duration': duration,
            'strength': strength,
            'pattern_type': pattern_type
        }
    
    def _decay_interference_patterns(self):
        """
        Decay existing interference patterns over time.
        """
        now = datetime.now()
        expired_patterns = []
        
        for pattern_id, interference in self.interference_patterns.items():
            # Calculate age of interference pattern
            age = (now - interference['created_at']).total_seconds()
            
            # Check if expired
            if age > interference['duration']:
                expired_patterns.append(pattern_id)
                continue
            
            # Decay strength based on age
            decay_factor = 1 - (age / interference['duration'])
            interference['strength'] *= decay_factor
        
        # Remove expired patterns
        for pattern_id in expired_patterns:
            self.interference_patterns.pop(pattern_id, None)
    
    def _create_resonance_window(self):
        """
        Create a resonance window for unusual pattern emergence.
        """
        # Resonance windows allow unusual patterns to emerge
        # during periods of high field density
        window_duration = 2 + (np.random.random() * 3)  # 2-5 seconds
        
        self.resonance_windows.append({
            'created_at': datetime.now(),
            'duration': window_duration,
            'receptivity_boost': 0.3 + (np.random.random() * 0.4)  # 0.3-0.7
        })
        
        logger.info(f"Created resonance window for {window_duration:.1f}s with "  
                   f"receptivity boost {self.resonance_windows[-1]['receptivity_boost']:.2f}")
    
    def get_pattern_receptivity(self, pattern_type: str) -> float:
        """
        Get current receptivity for a pattern type.
        
        Args:
            pattern_type: Type of pattern
            
        Returns:
            Receptivity value (0.0-1.0)
        """
        # Base receptivity for this pattern type
        base_receptivity = self.pattern_receptivity.get(pattern_type, 0.5)
        
        # Adjust for field density (higher density = lower receptivity)
        density_factor = 1 - (self.field_density * 0.5)  # 0.5-1.0
        
        # Adjust for field turbulence (higher turbulence = higher receptivity for unusual patterns)
        turbulence_factor = 1.0
        if pattern_type in ['emergent', 'meta']:
            turbulence_factor = 1 + (self.field_turbulence * 0.5)  # 1.0-1.5
        
        # Check for active resonance windows
        window_boost = 0.0
        now = datetime.now()
        active_windows = [w for w in self.resonance_windows 
                         if (now - w['created_at']).total_seconds() < w['duration']]
        
        if active_windows:
            window_boost = max(w['receptivity_boost'] for w in active_windows)
        
        # Calculate final receptivity
        receptivity = base_receptivity * density_factor * turbulence_factor + window_boost
        
        # Ensure within bounds
        receptivity = max(0.1, min(1.0, receptivity))
        
        return receptivity
    
    def should_detect_pattern(self, pattern_data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Determine if a pattern should be detected based on field state.
        
        This enhanced method incorporates pattern relationship tracking and adaptive
        receptivity to create a more natural emergence of patterns through resonance,
        dissonance, and field density interactions.
        
        Args:
            pattern_data: Pattern data
            
        Returns:
            Tuple of (should_detect, confidence_modifier)
        """
        pattern_type = pattern_data.get('context', {}).get('cascade_type', 'secondary')
        pattern_confidence = pattern_data.get('confidence', 0.5)
        pattern_id = pattern_data.get('id', str(id(pattern_data)))
        
        # Get current receptivity for this pattern type
        receptivity = self.get_pattern_receptivity(pattern_type)
        
        # Calculate interference from existing patterns with relationship awareness
        interference = 0.0
        related_patterns = []
        
        for other_id, interference_data in self.interference_patterns.items():
            # Check for relationship between patterns
            relationship_key = f"{pattern_id}_{other_id}"
            relationship_strength = self.pattern_relationships.get(relationship_key, 0.0)
            
            # Patterns with established relationships interfere more
            if relationship_strength > 0:
                related_patterns.append(other_id)
                interference += interference_data['strength'] * relationship_strength
            else:
                # Patterns of the same type interfere more
                type_match = interference_data['pattern_type'] == pattern_type
                type_factor = 1.2 if type_match else 0.8
                interference += interference_data['strength'] * type_factor
        
        # Cap interference
        interference = min(0.9, interference)
        
        # Calculate detection threshold - adaptive based on pattern relationships
        # Patterns with established relationships have lower thresholds
        relationship_factor = 0.1 if related_patterns else 0.0
        coherence_factor = self.field_coherence * 0.2  # Higher coherence = lower threshold
        threshold = 0.3 + (interference * 0.5) - relationship_factor - coherence_factor
        
        # Adjust confidence based on receptivity and field state
        field_state_factor = (self.field_coherence * 0.3) + (self.field_stability * 0.2)
        adjusted_confidence = pattern_confidence * receptivity * (1 + field_state_factor)
        
        # Determine if pattern should be detected
        should_detect = adjusted_confidence > threshold
        
        # Add some randomness for errant behavior (more likely with high turbulence)
        if self.field_turbulence > 0.5:
            # Probability of errant behavior scales with turbulence
            errant_probability = self.field_turbulence * 0.2
            if np.random.random() < errant_probability:
                # Randomly flip the decision with low probability
                should_detect = not should_detect
                logger.info(f"Field turbulence ({self.field_turbulence:.2f}) caused errant detection behavior")
        
        # Calculate confidence modifier with field state influence
        confidence_modifier = receptivity * (1 - interference) * (1 + (self.field_coherence * 0.2))
        
        # If detected, update pattern relationships
        if should_detect:
            self._update_pattern_relationships(pattern_id, pattern_type)
            
            # Record for visualization
            self.visualization_data['pattern_emergence_points'].append({
                'pattern_id': pattern_id,
                'pattern_type': pattern_type,
                'timestamp': datetime.now(),
                'confidence': pattern_confidence * confidence_modifier,
                'position': [np.random.random(), np.random.random()]  # Placeholder for actual field position
            })
        
        # Update receptivity history for adaptive learning
        if pattern_type in self.receptivity_history:
            self.receptivity_history[pattern_type].append(1.0 if should_detect else 0.0)
            self._update_adaptive_receptivity()
        
        return should_detect, confidence_modifier
        
    def _update_pattern_relationships(self, pattern_id: str, pattern_type: str):
        """
        Update pattern relationships when a pattern is detected.
        
        Args:
            pattern_id: ID of the detected pattern
            pattern_type: Type of the detected pattern
        """
        # Look for recently detected patterns to establish relationships
        now = datetime.now()
        recent_patterns = [p for p in self.resonance_history 
                         if (now - p['timestamp']).total_seconds() < 10]
        
        for recent in recent_patterns:
            if recent['pattern_id'] != pattern_id:  # Don't relate to self
                relationship_key = f"{pattern_id}_{recent['pattern_id']}"
                reverse_key = f"{recent['pattern_id']}_{pattern_id}"
                
                # Get current relationship strength or initialize
                current_strength = self.pattern_relationships.get(relationship_key, 0.0)
                
                # Strengthen relationship - patterns detected close together are related
                # Patterns of same type have stronger relationships
                type_factor = 1.2 if pattern_type == recent['pattern_type'] else 0.8
                time_factor = 1.0 - ((now - recent['timestamp']).total_seconds() / 10)
                
                # Update relationship strength
                new_strength = min(1.0, current_strength + (0.1 * type_factor * time_factor))
                self.pattern_relationships[relationship_key] = new_strength
                self.pattern_relationships[reverse_key] = new_strength  # Bidirectional
                
                # Record relationship strength for history
                self.relationship_strength_history.append({
                    'key': relationship_key,
                    'strength': new_strength,
                    'timestamp': now
                })
                
    def _update_adaptive_receptivity(self):
        """
        Update adaptive receptivity based on detection history.
        """
        for pattern_type, history in self.receptivity_history.items():
            if len(history) > 5:
                # Calculate recent detection rate
                detection_rate = sum(history) / len(history)
                
                # If rarely detected, increase receptivity
                if detection_rate < 0.2:
                    self.pattern_receptivity[pattern_type] = min(
                        1.0, 
                        self.pattern_receptivity[pattern_type] + self.receptivity_learning_rate
                    )
                # If frequently detected, decrease receptivity
                elif detection_rate > 0.8:
                    self.pattern_receptivity[pattern_type] = max(
                        0.1, 
                        self.pattern_receptivity[pattern_type] - self.receptivity_learning_rate
                    )
                    
                logger.debug(f"Updated receptivity for {pattern_type}: {self.pattern_receptivity[pattern_type]:.2f}")
    
    def get_field_state(self) -> Dict[str, Any]:
        """
        Get current field state metrics with enhanced field metrics and visualization data.
        
        Returns:
            Dictionary with comprehensive field state metrics
        """
        # Count active resonance windows
        now = datetime.now()
        active_windows = [w for w in self.resonance_windows 
                         if (now - w['created_at']).total_seconds() < w['duration']]
        
        # Get significant pattern relationships (strength > 0.3)
        significant_relationships = {
            k: v for k, v in self.pattern_relationships.items() if v > 0.3
        }
        
        # Get recent patterns
        recent_patterns = [p for p in self.resonance_history 
                          if (now - p['timestamp']).total_seconds() < 30]
        
        return {
            # Core field metrics with continuity
            'field_density': self.field_density,
            'field_turbulence': self.field_turbulence,
            'field_coherence': self.field_coherence,
            'field_stability': self.field_stability,
            
            # Activity metrics
            'active_interference_patterns': len(self.interference_patterns),
            'active_resonance_windows': len(active_windows),
            'recent_pattern_count': len(recent_patterns),
            'significant_relationship_count': len(significant_relationships),
            
            # Adaptive receptivity
            'pattern_receptivity': self.pattern_receptivity,
            
            # Field evolution metrics
            'continuity_factor': self.continuity_factor,
            'field_metrics_history': list(self.visualization_data['field_state_history'])[-5:] if self.visualization_data['field_state_history'] else [],
            
            # Pattern relationship metrics
            'top_relationships': list(significant_relationships.items())[:5] if significant_relationships else []
        }


class VectorTonicWindowIntegrator:
    """
    Integrates vector-tonic-harmonic validation with learning window control.
    
    This class enhances the learning window system by:
    1. Using the OPENING state for progressive preparation
    2. Creating a feedback loop between harmonic analysis and window control
    3. Making field metrics influence both window states and harmonic analysis
    """
    
    def __init__(
        self,
        tonic_detector: TonicHarmonicPatternDetector,
        event_bus: LocalEventBus,
        harmonic_io_service: HarmonicIOService,
        metrics: Optional[TonicHarmonicMetrics] = None,
        adaptive_soak_period: bool = True
    ):
        """
        Initialize the vector-tonic window integrator.
        
        Args:
            tonic_detector: The tonic-harmonic pattern detector
            event_bus: Event bus for publishing and subscribing to events
            harmonic_io_service: Service for harmonic I/O operations
            metrics: Optional metrics service for tonic-harmonic analysis
            adaptive_soak_period: Whether to use adaptive soak periods
        """
        self.tonic_detector = tonic_detector
        self.base_detector = tonic_detector.base_detector
        self.event_bus = event_bus
        self.harmonic_io_service = harmonic_io_service
        self.metrics = metrics or TonicHarmonicMetrics()
        self.adaptive_soak_period = adaptive_soak_period
        
        # Vector cache for progressive preparation
        self.vector_cache = {}
        self.cache_warming_level = 0.0  # 0.0 to 1.0
        
        # Pattern candidates identified during OPENING
        self.pattern_candidates = []
        
        # Adaptive soak period parameters
        self.min_soak_period = 2  # seconds (reduced for testing)
        self.max_soak_period = 10  # seconds (reduced for testing)
        self.field_history = deque(maxlen=20)
        
        # Register event handlers
        self._register_event_handlers()
        
        # Create field state modulator to replace binary back pressure
        self.field_modulator = FieldStateModulator()
        
        logger.info("VectorTonicWindowIntegrator initialized")
    
    def _register_event_handlers(self):
        """Register handlers for relevant events."""
        self.event_bus.subscribe("field.state.updated", self._on_field_state_updated)
        self.event_bus.subscribe("learning.window.state", self._on_window_state_changed)
        self.event_bus.subscribe("vector.gradient.update", self._on_vector_gradient_updated)
    
    def _on_field_state_updated(self, event: Event):
        """
        Handle field state update events.
        
        Args:
            event: Field state update event
        """
        field_state_data = event.data.get('field_state', {})
        if not field_state_data:
            return
            
        # Update field history for adaptive soak period calculation
        # Extract metrics from field_state_data, handling possible nested structure
        metrics = {}
        if isinstance(field_state_data, dict):
            # Try to get metrics directly
            if 'metrics' in field_state_data and isinstance(field_state_data['metrics'], dict):
                metrics = field_state_data['metrics']
            # If not found, check if metrics is nested in field_properties
            elif 'field_properties' in field_state_data and isinstance(field_state_data['field_properties'], dict):
                # Extract relevant metrics from field_properties
                field_props = field_state_data['field_properties']
                metrics = {
                    'coherence': field_props.get('coherence', 0.5),
                    'stability': field_props.get('stability', 0.5),
                    'turbulence': 1.0 - field_props.get('stability', 0.5),
                    'density': field_props.get('density', 0.5)
                }
        
        # Add to field history with proper structure
        self.field_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        # If window is in OPENING state, use this update for progressive preparation
        if self.base_detector.window_state == WindowState.OPENING:
            self._progress_preparation(field_state_data)
    
    def _on_window_state_changed(self, event: Event):
        """
        Handle window state change events.
        
        Args:
            event: Window state change event
        """
        state = event.data.get('state')
        if not state:
            return
            
        # If transitioning to OPENING, start progressive preparation
        if state == WindowState.OPENING.value:
            self._start_progressive_preparation()
            
            # Calculate adaptive soak period if enabled
            if self.adaptive_soak_period:
                soak_period = self._calculate_adaptive_soak_period()
                
                # Update next transition time
                if self.base_detector.current_window:
                    self.base_detector.current_window.next_transition_time = (
                        datetime.now() + timedelta(seconds=soak_period)
                    )
                    logger.info(f"Set adaptive soak period to {soak_period} seconds")
    
    def _on_vector_gradient_updated(self, event: Event):
        """
        Handle vector gradient update events.
        
        Args:
            event: Vector gradient update event
        """
        try:
            # Log the structure of the event data for debugging
            logger.info(f"Vector gradient update event data structure: {type(event.data)}")
            for key, value in event.data.items():
                logger.info(f"Key: {key}, Type: {type(value)}")
                
            gradient_data = event.data.get('gradient', {})
            if not gradient_data:
                logger.warning("No gradient data found in event")
                return
                
            # Log the structure of the gradient data for debugging
            logger.info(f"Gradient data structure: {type(gradient_data)}")
            for key, value in gradient_data.items():
                logger.info(f"Gradient key: {key}, Type: {type(value)}")
                
            # Use gradient data to inform window state decisions
            if self.base_detector.window_state == WindowState.OPENING:
                try:
                    # Update detection thresholds based on gradient
                    self._adjust_detection_thresholds(gradient_data)
                except Exception as e:
                    logger.error(f"Error in _adjust_detection_thresholds: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                try:
                    # Warm vector cache with gradient data
                    self._warm_vector_cache(gradient_data)
                except Exception as e:
                    logger.error(f"Error in _warm_vector_cache: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error in vector gradient update handler: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _start_progressive_preparation(self):
        """Start progressive preparation during OPENING state."""
        logger.info("Starting progressive preparation during OPENING state")
        
        # Reset preparation state
        self.vector_cache = {}
        self.cache_warming_level = 0.0
        self.pattern_candidates = []
        
        # Request vector gradients to warm cache
        # Request vector gradients to warm cache through the HarmonicIOService
        if self.harmonic_io_service:
            try:
                # The error was due to passing the event_bus as repository
                # We should pass self as the repository since we have the _warm_vector_cache method
                self.harmonic_io_service.schedule_operation(
                    operation_type="process",  # Use the string value, not the enum
                    repository=self,  # Pass self as the repository
                    method_name='_warm_vector_cache',
                    args=(),  # No arguments needed
                    kwargs={},  # No keyword arguments needed
                    data_context={
                        'priority': 'high',
                        'purpose': 'cache_warming',
                        'stability': 0.8,  # High stability for predictable warming
                        'coherence': 0.7,   # Good coherence for meaningful patterns
                        'field_state_id': self.field_modulator.field_state_id if hasattr(self.field_modulator, 'field_state_id') else None
                    }
                )
                logger.info("Scheduled vector gradient analysis for cache warming")
            except Exception as e:
                logger.warning(f"Error scheduling vector gradient analysis: {e}")
                # If scheduling fails, still warm the cache directly as a fallback
                self._warm_vector_cache()
                logger.info("Directly warmed vector cache as fallback")
    
    def _progress_preparation(self, field_state_data: Dict[str, Any]):
        """
        Progress the preparation based on field state updates.
        
        Args:
            field_state_data: Field state data
        """
        # Extract metrics
        metrics = field_state_data.get('metrics', {})
        coherence = metrics.get('coherence', 0.5)
        stability = metrics.get('stability', 0.5)
        
        # Update cache warming level based on time elapsed and field metrics
        if self.base_detector.current_window and self.base_detector.current_window.next_transition_time:
            time_remaining = (self.base_detector.current_window.next_transition_time - datetime.now()).total_seconds()
            total_soak_period = self._calculate_adaptive_soak_period()
            
            # Calculate progress as percentage of soak period elapsed
            elapsed_percentage = max(0, min(1, 1 - (time_remaining / total_soak_period)))
            
            # Adjust for field metrics (faster warming with higher coherence/stability)
            metric_factor = (coherence + stability) / 2
            
            # Update cache warming level
            self.cache_warming_level = min(1.0, elapsed_percentage * (1 + metric_factor))
            
            logger.info(f"Preparation progress: {self.cache_warming_level:.2f}, Time remaining: {time_remaining:.1f}s")
            
            # If we're over 50% prepared, start identifying pattern candidates
            if self.cache_warming_level > 0.5 and not self.pattern_candidates:
                self._identify_pattern_candidates()
    
    def _warm_vector_cache(self, gradient_data: Dict[str, Any] = None):
        """
        Warm the vector cache with gradient data.
        
        Args:
            gradient_data: Vector gradient data (optional)
        """
        # If gradient data is provided, use it
        if gradient_data and gradient_data.get('vectors'):
            # Extract vectors from gradient data
            vectors = gradient_data.get('vectors', {})
            
            # Handle both dictionary and list types for vectors
            if isinstance(vectors, dict):
                for key, vector in vectors.items():
                    self.vector_cache[key] = vector
                logger.info(f"Vector cache warmed with {len(vectors)} vectors from gradient data")
            elif isinstance(vectors, list):
                # If vectors is a list, use index as key
                for i, vector in enumerate(vectors):
                    self.vector_cache[f"vector_{i}"] = vector
                logger.info(f"Vector cache warmed with {len(vectors)} vectors from gradient data")
            else:
                logger.warning(f"Unexpected vectors type: {type(vectors)}")
            return
        
        # If no gradient data provided, try to load climate risk data
        try:
            # Find climate risk data directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 
                                "data", "climate_risk")
            
            # Check if the directory exists
            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"Climate risk data directory not found: {data_dir}")
                
            # Import the ClimateDataLoader
            from src.habitat_evolution.adaptive_core.emergence.climate_data_loader import ClimateDataLoader
            
            # Create a loader and load the data
            climate_loader = ClimateDataLoader(data_dir)
            climate_loader.load_all_files()
            
            # Get relationships and convert them to vectors
            relationships = climate_loader.relationships
            logger.info(f"Found {len(relationships)} relationships from climate risk data")
            
            # Create vectors from relationships
            # In a real implementation, we would use embedding models
            # For now, we'll create deterministic vectors based on the relationship content
            vectors = {}
            for i, rel in enumerate(relationships[:20]):  # Limit to 20 for testing
                # Create a deterministic vector based on the relationship content
                source = rel.get('source', '')
                predicate = rel.get('predicate', '')
                target = rel.get('target', '')
                
                # Create a simple hash-based vector (in real implementation, use embeddings)
                seed = hash(f"{source}_{predicate}_{target}") % 10000
                np.random.seed(seed)
                vector = np.random.rand(128).tolist()
                
                # Add to vector cache
                key = f"climate_vector_{source}_{predicate}_{target}"
                vectors[key] = vector
            
            self.vector_cache.update(vectors)
            logger.info(f"Vector cache warmed with {len(vectors)} vectors from climate risk data")
            
            # Generate some additional synthetic vectors to ensure we have enough
            if len(vectors) < 10:
                synthetic_count = 10 - len(vectors)
                synthetic_vectors = {
                    f"synthetic_vector_{i}": np.random.rand(128).tolist() 
                    for i in range(synthetic_count)
                }
                self.vector_cache.update(synthetic_vectors)
                logger.info(f"Added {len(synthetic_vectors)} synthetic vectors to supplement climate data")
            
        except Exception as e:
            logger.warning(f"Error loading climate risk data: {e}. Falling back to synthetic vectors.")
            # Generate synthetic vectors for testing as fallback
            synthetic_vectors = {
                f"synthetic_vector_{i}": np.random.rand(128).tolist() 
                for i in range(10)
            }
            self.vector_cache.update(synthetic_vectors)
            logger.info(f"Vector cache warmed with {len(synthetic_vectors)} synthetic vectors (fallback)")
        
        # Force cache warming level to increase
        self.cache_warming_level = max(0.5, self.cache_warming_level)
    
    def _adjust_detection_thresholds(self, gradient_data: Dict[str, Any]):
        """
        Adjust detection thresholds based on gradient data.
        
        Args:
            gradient_data: Vector gradient data
        """
        # Extract gradient metrics
        gradient_metrics = gradient_data.get('metrics', {})
        density = gradient_metrics.get('density', 0.5)
        variance = gradient_metrics.get('variance', 0.5)
        
        # Extract field state metrics
        field_coherence = gradient_metrics.get('coherence', 0.5)
        field_stability = gradient_metrics.get('stability', 0.5)
        
        # Update field state in the detector if it supports the new field-aware threshold system
        if hasattr(self.base_detector.detector, 'update_field_state'):
            # Update field state to influence threshold calculations
            self.base_detector.detector.update_field_state(field_coherence, field_stability)
            logger.info(f"Updated detector field state: coherence={field_coherence:.2f}, stability={field_stability:.2f}")
        # Fall back to legacy threshold adjustment if needed
        elif hasattr(self.base_detector.detector, 'threshold'):
            base_threshold = self.base_detector.detector.threshold
            
            # Higher density = lower threshold (more sensitive)
            # Higher variance = higher threshold (less sensitive)
            # Higher coherence = lower threshold (more sensitive)
            # Higher stability = lower threshold (more sensitive)
            coherence_factor = field_coherence * 0.2
            stability_factor = field_stability * 0.1
            adjusted_threshold = base_threshold * (1 - (density * 0.2) - coherence_factor - stability_factor + (variance * 0.2))
            
            # Ensure threshold is within reasonable bounds
            adjusted_threshold = max(1, min(base_threshold * 1.5, adjusted_threshold))
            
            # Set as dynamic threshold (will be used when window opens)
            self.base_detector.detector.dynamic_threshold = adjusted_threshold
            
            logger.info(f"Adjusted detection threshold to {adjusted_threshold:.2f} based on gradient metrics")
    
    def _identify_pattern_candidates(self):
        """Identify pattern candidates during OPENING state."""
        logger.info("Identifying pattern candidates during OPENING state")
        
        try:
            # Use the detector with a higher threshold to find strong candidates
            original_threshold = self.base_detector.detector.threshold
            self.base_detector.detector.threshold *= 1.5  # 50% higher threshold for candidates
            
            # Temporarily override window state check
            original_detect = self.base_detector.detect_patterns
            
            def override_detect():
                # Skip window state check just for candidate identification
                try:
                    # Get raw patterns from detector
                    raw_patterns = self.base_detector.detector.detect_patterns()
                    
                    # Apply field state modulation to each pattern
                    modulated_patterns = []
                    for pattern in raw_patterns:
                        should_detect, confidence_modifier = self.field_modulator.should_detect_pattern(pattern)
                        
                        if should_detect:
                            # Adjust confidence based on field state
                            pattern['confidence'] = pattern.get('confidence', 0.5) * confidence_modifier
                            modulated_patterns.append(pattern)
                            
                            # Record pattern emergence to update field state
                            self.field_modulator.record_pattern_emergence(pattern)
                    
                    return modulated_patterns
                except AttributeError as e:
                    # Handle missing attributes gracefully
                    logger.warning(f"Error during pattern detection: {e}")
                    return []
                    
            self.base_detector.detect_patterns = override_detect
            
            # Identify candidates
            try:
                candidates = self.base_detector.detect_patterns()
                self.pattern_candidates = candidates
                logger.info(f"Identified {len(candidates)} pattern candidates during OPENING state")
            except Exception as e:
                logger.warning(f"Error identifying pattern candidates: {e}")
                self.pattern_candidates = []
            
            # Restore original behavior
            self.base_detector.detect_patterns = original_detect
            self.base_detector.detector.threshold = original_threshold
            
        except Exception as e:
            logger.warning(f"Failed to identify pattern candidates: {e}")
            self.pattern_candidates = []
    
    def _calculate_adaptive_soak_period(self) -> float:
        """
        Calculate adaptive soak period based on field history.
        
        Returns:
            Soak period in seconds
        """
        if not self.field_history or len(self.field_history) < 2:
            return self.max_soak_period
            
        # Calculate field volatility
        coherence_values = []
        stability_values = []
        
        for entry in self.field_history:
            # Handle both dictionary and list types for field history entries
            if isinstance(entry, dict) and 'metrics' in entry:
                metrics = entry['metrics']
                if isinstance(metrics, dict):
                    coherence_values.append(metrics.get('coherence', 0.5))
                    stability_values.append(metrics.get('stability', 0.5))
                elif isinstance(metrics, list) and len(metrics) > 0:
                    # If metrics is a list, try to find coherence and stability in the first item
                    if isinstance(metrics[0], dict):
                        coherence_values.append(metrics[0].get('coherence', 0.5))
                        stability_values.append(metrics[0].get('stability', 0.5))
            # Log unexpected entry type for debugging
            elif not isinstance(entry, dict):
                logger.warning(f"Unexpected field history entry type: {type(entry)}")
        
        if not coherence_values or not stability_values:
            return self.max_soak_period
            
        # Calculate variance of metrics
        coherence_variance = np.var(coherence_values) if len(coherence_values) > 1 else 0
        stability_variance = np.var(stability_values) if len(stability_values) > 1 else 0
        
        # Higher variance = more volatile = longer soak period
        volatility = (coherence_variance + stability_variance) / 2
        
        # Scale to soak period range
        soak_period = self.min_soak_period + (self.max_soak_period - self.min_soak_period) * volatility
        
        # Ensure within bounds
        soak_period = max(self.min_soak_period, min(self.max_soak_period, soak_period))
        
        return soak_period
    
    def get_preparation_status(self) -> Dict[str, Any]:
        """
        Get the current preparation status.
        
        Returns:
            Dictionary with preparation status
        """
        # Get field state from modulator
        field_state = self.field_modulator.get_field_state()
        
        status = {
            'cache_warming_level': self.cache_warming_level,
            'vector_cache_size': len(self.vector_cache),
            'pattern_candidates': len(self.pattern_candidates),
            'soak_period': self._calculate_adaptive_soak_period() if self.adaptive_soak_period else 30,
            'window_state': self.base_detector.window_state.value if hasattr(self.base_detector, 'window_state') else None,
            'field_state': field_state
        }
        
        return status


def create_vector_tonic_window_integrator(
    tonic_detector: TonicHarmonicPatternDetector,
    event_bus: LocalEventBus,
    harmonic_io_service: Optional[HarmonicIOService] = None,
    metrics: Optional[TonicHarmonicMetrics] = None,
    adaptive_soak_period: bool = True
) -> VectorTonicWindowIntegrator:
    """
    Create a vector-tonic window integrator.
    
    Args:
        tonic_detector: The tonic-harmonic pattern detector
        event_bus: Event bus for publishing and subscribing to events
        harmonic_io_service: Optional harmonic I/O service
        metrics: Optional metrics service for tonic-harmonic analysis
        adaptive_soak_period: Whether to use adaptive soak periods
        
    Returns:
        Configured vector-tonic window integrator
    """
    # Create harmonic I/O service if not provided
    if not harmonic_io_service:
        harmonic_io_service = HarmonicIOService(event_bus)
    
    # Create metrics service if not provided
    if not metrics:
        metrics = TonicHarmonicMetrics()
    
    # Create and return integrator
    integrator = VectorTonicWindowIntegrator(
        tonic_detector=tonic_detector,
        event_bus=event_bus,
        harmonic_io_service=harmonic_io_service,
        metrics=metrics,
        adaptive_soak_period=adaptive_soak_period
    )
    
    logger.info("Created vector-tonic window integrator")
    return integrator
