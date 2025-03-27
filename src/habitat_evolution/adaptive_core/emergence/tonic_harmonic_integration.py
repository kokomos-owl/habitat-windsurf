"""
Tonic-Harmonic Integration for Pattern Detection

This module integrates the vector+ tonic-harmonics system with pattern detection,
enabling semantic boundary detection and enhanced pattern identification through
harmonic resonance analysis.

It creates a bridge between the pattern detection system and the field-based
harmonic analysis, allowing patterns to evolve naturally while respecting
the underlying semantic structure of the data.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
import numpy as np
from datetime import datetime
from collections import deque

# Use absolute imports to avoid module path issues
from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from src.habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService, OperationType
from src.habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics
from src.habitat_evolution.adaptive_core.emergence.learning_window_integration import LearningWindowAwareDetector
from src.habitat_evolution.adaptive_core.emergence.event_aware_detector import EventAwarePatternDetector
from src.habitat_evolution.pattern_aware_rag.learning.learning_control import WindowState, BackPressureController

logger = logging.getLogger(__name__)


class TonicHarmonicPatternDetector:
    """
    Enhances pattern detection with tonic-harmonic analysis for semantic boundary detection.
    
    This detector integrates with the vector+ system to identify semantic boundaries
    and enhance pattern detection through harmonic resonance analysis. It uses the
    tonic-harmonic field state to guide pattern evolution and detection sensitivity.
    """
    
    def __init__(self, 
                 base_detector: LearningWindowAwareDetector,
                 harmonic_io_service: HarmonicIOService,
                 event_bus: LocalEventBus,
                 field_bridge: Optional[HarmonicFieldIOBridge] = None,
                 metrics: Optional[TonicHarmonicMetrics] = None):
        """
        Initialize the tonic-harmonic pattern detector.
        
        Args:
            base_detector: The base learning window aware detector
            harmonic_io_service: Service for harmonic I/O operations
            event_bus: Event bus for publishing and subscribing to events
            field_bridge: Bridge to the harmonic field I/O system
            metrics: Metrics for tonic-harmonic analysis
        """
        self.base_detector = base_detector
        self.harmonic_io_service = harmonic_io_service
        self.event_bus = event_bus
        self.field_bridge = field_bridge or HarmonicFieldIOBridge(harmonic_io_service)
        self.metrics = metrics or TonicHarmonicMetrics()
        
        # Current field state
        self.field_state = None
        
        # Semantic boundary detection
        self.boundary_thresholds = {
            "coherence": 0.7,
            "resonance": 0.65,
            "turbulence": 0.4
        }
        
        # Pattern evolution parameters
        self.evolution_parameters = {
            "adaptation_rate": 0.15,
            "boundary_sensitivity": 0.8,
            "field_influence": 0.6
        }
        
        # Field state continuity tracking
        self._previous_field_metrics = {
            'density': 0.5,
            'turbulence': 0.3,
            'coherence': 0.7,
            'stability': 0.7
        }
        
        # Pattern relationship tracking
        self.pattern_relationships = {}
        self.relationship_strength_history = deque(maxlen=50)
        
        # Adaptive receptivity learning
        self.pattern_type_receptivity = {
            'primary': 1.0,
            'secondary': 0.8,
            'meta': 0.7,
            'emergent': 0.6
        }
        self.receptivity_learning_rate = 0.05
        self.receptivity_history = {k: deque(maxlen=20) for k in self.pattern_type_receptivity.keys()}
        
        # Visualization data collection
        self.visualization_data = {
            'field_state_history': deque(maxlen=100),
            'pattern_emergence_points': [],
            'resonance_centers': {},
            'interference_patterns': {}
        }
        
        # Register for field state updates
        self._register_field_state_handler()
        
        # Register for pattern events
        self._register_pattern_event_handlers()
        
        logger.info("TonicHarmonicPatternDetector initialized with enhanced field continuity")
    
    def _register_field_state_handler(self):
        """Register handler for field state updates."""
        self.event_bus.subscribe("field.state.updated", self._on_field_state_updated)
    
    def _register_pattern_event_handlers(self):
        """Register handlers for pattern-related events."""
        self.event_bus.subscribe("pattern.detected", self._on_pattern_detected)
        self.event_bus.subscribe("pattern.evolved", self._on_pattern_evolved)
    
    def _on_field_state_updated(self, event: Event):
        """
        Handle field state update events.
        
        Args:
            event: Field state update event
        """
        field_state_data = event.data.get('field_state', {})
        if field_state_data:
            # Update local field state
            self.field_state = TonicHarmonicFieldState(field_state_data)
            
            # Update detection parameters based on field state
            self._update_detection_parameters()
            
            logger.info(f"Field state updated: dimensionality={self.field_state.effective_dimensionality}")
    
    def _on_pattern_detected(self, event: Event):
        """
        Handle pattern detection events.
        
        Args:
            event: Pattern detection event
        """
        pattern_id = event.data.get('pattern_id')
        pattern_data = event.data.get('pattern_data', {})
        
        if pattern_id and pattern_data and self.field_state:
            # Schedule harmonic analysis of the pattern
            self._schedule_pattern_analysis(pattern_id, pattern_data)
    
    def _on_pattern_evolved(self, event: Event):
        """
        Handle pattern evolution events.
        
        Args:
            event: Pattern evolution event
        """
        pattern_id = event.data.get('pattern_id')
        from_state = event.data.get('from_state', {})
        to_state = event.data.get('to_state', {})
        
        if pattern_id and from_state and to_state and self.field_state:
            # Analyze pattern evolution in harmonic context
            self._analyze_pattern_evolution(pattern_id, from_state, to_state)
    
    def _update_detection_parameters(self):
        """Update pattern detection parameters based on current field state."""
        if not self.field_state:
            return
        
        # Extract field metrics
        coherence = self.field_state.coherence
        # Turbulence might be derived from metrics or field_properties
        turbulence = 1.0 - self.field_state.stability  # Estimate turbulence as inverse of stability
        stability = self.field_state.stability
        
        # Adjust detection threshold based on field metrics
        base_threshold = self.base_detector.detector.threshold
        field_factor = (coherence * 0.5) + (stability * 0.3) - (turbulence * 0.2)
        
        # Ensure field factor is within reasonable bounds
        field_factor = max(0.5, min(1.5, field_factor))
        
        # Update detector threshold
        new_threshold = base_threshold / field_factor
        self.base_detector.detector.threshold = new_threshold
        
        logger.info(f"Updated detection threshold to {new_threshold:.2f} based on field state")
        
        # Update learning window parameters if needed
        if hasattr(self.base_detector, 'window_state'):
            current_state = self.base_detector.window_state
            
            # Determine optimal window state based on field metrics
            if coherence > self.boundary_thresholds["coherence"] and stability > 0.6:
                optimal_state = WindowState.OPEN
            elif turbulence > self.boundary_thresholds["turbulence"]:
                optimal_state = WindowState.CLOSED
            else:
                optimal_state = WindowState.NARROWING if current_state == WindowState.OPEN else WindowState.WIDENING
            
            # Only update if different from current state
            if optimal_state != current_state:
                self.base_detector.update_window_state(optimal_state)
                logger.info(f"Updated learning window state to {optimal_state} based on field metrics")
    
    def _schedule_pattern_analysis(self, pattern_id: str, pattern_data: Dict[str, Any]):
        """
        Schedule harmonic analysis of a detected pattern.
        
        Args:
            pattern_id: ID of the detected pattern
            pattern_data: Data associated with the pattern
        """
        # Create operation context
        context = {
            "operation_type": "pattern_analysis",
            "pattern_id": pattern_id,
            "pattern_data": pattern_data,
            "field_state_id": self.field_state.id if self.field_state else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Schedule through harmonic I/O
        self.harmonic_io_service.schedule_operation(
            operation_type=OperationType.PROCESS.value,
            repository=self,
            method_name="_process_pattern_analysis",
            args=(),
            kwargs={"context": context},
            data_context=context
        )
    
    def _process_pattern_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process pattern analysis in harmonic context.
        
        Args:
            context: Operation context
            
        Returns:
            Analysis results
        """
        pattern_id = context.get("pattern_id")
        pattern_data = context.get("pattern_data", {})
        
        if not pattern_id or not pattern_data:
            logger.warning("Invalid pattern analysis context")
            return {"success": False, "error": "Invalid context"}
        
        # Extract relationship components
        source = pattern_data.get("source", "")
        predicate = pattern_data.get("predicate", "")
        target = pattern_data.get("target", "")
        
        # Create domain data for harmonic analysis with wave properties
        domain_data = [
            {
                "id": f"source_{pattern_id}", 
                "name": source, 
                "type": "source",
                "frequency": 1.0,  # Base frequency for source
                "amplitude": 1.0,  # Standard amplitude
                "phase": 0.0,      # Initial phase
                "stability": 0.8,  # Assumed stability
                "tonic_value": 0.7 # Tonic value for source
            },
            {
                "id": f"predicate_{pattern_id}", 
                "name": predicate, 
                "type": "predicate",
                "frequency": 1.5,  # Higher frequency for predicates (relations)
                "amplitude": 0.9,  # Slightly lower amplitude
                "phase": np.pi/4,  # Phase shift
                "stability": 0.7,  # Slightly lower stability
                "tonic_value": 0.9 # Higher tonic value for predicates
            },
            {
                "id": f"target_{pattern_id}", 
                "name": target, 
                "type": "target",
                "frequency": 2.0,  # Highest frequency for targets
                "amplitude": 1.0,  # Standard amplitude
                "phase": np.pi/2, # Larger phase shift
                "stability": 0.8,  # Assumed stability
                "tonic_value": 0.7 # Tonic value for target
            }
        ]
        
        # Calculate harmonic coherence
        coherence_results = self.metrics.calculate_harmonic_coherence(domain_data)
        
        # Calculate resonance stability
        stability_results = self.metrics.calculate_resonance_stability(domain_data)
        
        # Calculate phase alignment
        alignment_results = self.metrics.calculate_phase_alignment(domain_data)
        
        # Calculate overall tonic-harmonic score if method exists
        overall_score = None
        if hasattr(self.metrics, 'calculate_overall_score'):
            # Create resonance data structure expected by the method
            resonance_data = {
                "coherence_matrix": self._create_coherence_matrix(domain_data, coherence_results),
                "alignment_matrix": self._create_alignment_matrix(domain_data, alignment_results),
                "stability_values": [result["stability"] for result in stability_results] if stability_results else []
            }
            overall_score = self.metrics.calculate_overall_score(domain_data, resonance_data)
        
        # Combine results
        analysis_results = {
            "pattern_id": pattern_id,
            "harmonic_coherence": coherence_results,
            "resonance_stability": stability_results,
            "phase_alignment": alignment_results,
            "overall_score": overall_score,
            "timestamp": datetime.now().isoformat(),
            "wave_interference_type": self._determine_wave_interference_type(alignment_results),
            "dimensional_alignment": self._calculate_dimensional_alignment(domain_data)
        }
        
        # Publish analysis results
        self.event_bus.publish(Event(
            "pattern.harmonic_analysis",
            {
                "pattern_id": pattern_id,
                "analysis": analysis_results
            },
            source="tonic_harmonic_detector"
        ))
        
        return {"success": True, "analysis": analysis_results}
        
    def _create_coherence_matrix(self, domain_data: List[Dict[str, Any]], 
                                coherence_results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create a coherence matrix from coherence results.
        
        Args:
            domain_data: List of domain dictionaries
            coherence_results: List of coherence metrics for domain pairs
            
        Returns:
            Coherence matrix as numpy array
        """
        n = len(domain_data)
        matrix = np.zeros((n, n))
        
        # Fill diagonal with 1.0 (perfect coherence with self)
        np.fill_diagonal(matrix, 1.0)
        
        # Map domain IDs to indices
        id_to_idx = {domain["id"]: i for i, domain in enumerate(domain_data)}
        
        # Fill matrix with coherence values
        for result in coherence_results:
            i = id_to_idx.get(result["domain1_id"])
            j = id_to_idx.get(result["domain2_id"])
            if i is not None and j is not None:
                matrix[i, j] = result["coherence"]
                matrix[j, i] = result["coherence"]  # Symmetric matrix
        
        return matrix
    
    def _create_alignment_matrix(self, domain_data: List[Dict[str, Any]], 
                                alignment_results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create an alignment matrix from alignment results.
        
        Args:
            domain_data: List of domain dictionaries
            alignment_results: List of alignment metrics for domain pairs
            
        Returns:
            Alignment matrix as numpy array
        """
        n = len(domain_data)
        matrix = np.zeros((n, n))
        
        # Fill diagonal with 1.0 (perfect alignment with self)
        np.fill_diagonal(matrix, 1.0)
        
        # Map domain IDs to indices
        id_to_idx = {domain["id"]: i for i, domain in enumerate(domain_data)}
        
        # Fill matrix with alignment values
        for result in alignment_results:
            i = id_to_idx.get(result["domain1_id"])
            j = id_to_idx.get(result["domain2_id"])
            if i is not None and j is not None:
                matrix[i, j] = result["alignment"]
                matrix[j, i] = result["alignment"]  # Symmetric matrix
        
        return matrix
    
    def _determine_wave_interference_type(self, alignment_results: List[Dict[str, Any]]) -> str:
        """
        Determine the overall wave interference type from alignment results.
        
        Args:
            alignment_results: List of alignment metrics for domain pairs
            
        Returns:
            Wave interference type (CONSTRUCTIVE, DESTRUCTIVE, or PARTIAL)
        """
        if not alignment_results:
            return "NEUTRAL"
        
        # Calculate average interference potential
        avg_potential = sum(result["interference_potential"] for result in alignment_results) / len(alignment_results)
        
        if avg_potential > 0.5:
            return "CONSTRUCTIVE"
        elif avg_potential < -0.5:
            return "DESTRUCTIVE"
        else:
            return "PARTIAL"
    
    def _calculate_dimensional_alignment(self, domain_data: List[Dict[str, Any]]) -> float:
        """
        Calculate dimensional alignment for the pattern.
        
        Args:
            domain_data: List of domain dictionaries
            
        Returns:
            Dimensional alignment score (0.0 to 1.0)
        """
        # In a real implementation, this would use eigenspace analysis
        # For now, we'll use a simplified approach based on frequency ratios
        
        if len(domain_data) < 2:
            return 0.0
        
        # Extract frequencies
        frequencies = [domain.get("frequency", 1.0) for domain in domain_data]
        
        # Calculate pairwise ratios
        ratios = []
        for i in range(len(frequencies)):
            for j in range(i+1, len(frequencies)):
                if frequencies[i] <= frequencies[j]:
                    ratio = frequencies[j] / frequencies[i]
                else:
                    ratio = frequencies[i] / frequencies[j]
                ratios.append(ratio)
        
        # Check how close ratios are to simple fractions
        alignment_scores = []
        for ratio in ratios:
            # Perfect unison (1:1)
            if abs(ratio - 1.0) < 0.05:
                alignment_scores.append(1.0)
            # Perfect octave (1:2)
            elif abs(ratio - 2.0) < 0.05:
                alignment_scores.append(0.95)
            # Perfect fifth (2:3)
            elif abs(ratio - 1.5) < 0.05:
                alignment_scores.append(0.9)
            # Perfect fourth (3:4)
            elif abs(ratio - 1.33) < 0.05:
                alignment_scores.append(0.85)
            else:
                # Default score for other ratios
                alignment_scores.append(0.7)
        
        # Return average alignment score
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
    
    def _analyze_pattern_evolution(self, pattern_id: str, from_state: Dict[str, Any], to_state: Dict[str, Any]):
        """
        Analyze pattern evolution in harmonic context.
        
        Args:
            pattern_id: ID of the evolved pattern
            from_state: Previous pattern state
            to_state: New pattern state
        """
        if not self.field_state:
            logger.warning("Cannot analyze pattern evolution without field state")
            return
        
        # Calculate semantic boundary crossing
        boundary_crossed = self._detect_semantic_boundary(from_state, to_state)
        
        if boundary_crossed:
            logger.info(f"Pattern {pattern_id} crossed semantic boundary during evolution")
            
            # Publish semantic boundary event
            self.event_bus.publish(Event.create(
                "pattern.semantic_boundary",
                {
                    "pattern_id": pattern_id,
                    "from_state": from_state,
                    "to_state": to_state,
                    "boundary_type": boundary_crossed,
                    "field_state_id": self.field_state.id
                },
                source="tonic_harmonic_detector"
            ))
    
    def _detect_semantic_boundary(self, from_state: Dict[str, Any], to_state: Dict[str, Any]) -> Optional[str]:
        """
        Detect if a pattern evolution crossed a semantic boundary.
        
        Args:
            from_state: Previous pattern state
            to_state: New pattern state
            
        Returns:
            Type of boundary crossed, or None if no boundary was crossed
        """
        if not self.field_state:
            return None
        
        # Extract components from states
        from_source = from_state.get("source", "")
        from_predicate = from_state.get("predicate", "")
        from_target = from_state.get("target", "")
        
        to_source = to_state.get("source", "")
        to_predicate = to_state.get("predicate", "")
        to_target = to_state.get("target", "")
        
        # Check if predicate changed (semantic role boundary)
        if from_predicate != to_predicate:
            return "predicate_boundary"
        
        # Check if both source and target changed (domain boundary)
        if from_source != to_source and from_target != to_target:
            return "domain_boundary"
        
        # Check if either source or target changed significantly (entity boundary)
        if from_source != to_source or from_target != to_target:
            return "entity_boundary"
        
        return None
    
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect patterns with tonic-harmonic awareness and field state modulation.
        
        Instead of binary back pressure, this method uses field state modulation to
        allow for natural emergence of patterns through resonance, dissonance, and
        field density interactions, preserving the errant, fuzzy, turbulent, and
        dissonant characteristics of Habitat.
        
        Returns:
            List of detected patterns
        """
        # First update detection parameters based on current field state
        if self.field_state:
            self._update_detection_parameters()
        
        # Delegate to base detector to get candidate patterns
        raw_patterns = self.base_detector.detect_patterns()
        
        # If no patterns or no field state, return as is
        if not raw_patterns or not self.field_state:
            return raw_patterns
        
        # Apply field state modulation
        modulated_patterns = self._apply_field_state_modulation(raw_patterns)
        
        # Apply tonic-harmonic filtering to the modulated patterns
        if modulated_patterns:
            modulated_patterns = self._apply_harmonic_filtering(modulated_patterns)
            
        # Export visualization data periodically (every 5th call)
        if hasattr(self, '_detect_call_count'):
            self._detect_call_count += 1
        else:
            self._detect_call_count = 1
            
        if self._detect_call_count % 5 == 0:
            self._export_visualization_data()
        
        return modulated_patterns
        
    def _apply_field_state_modulation(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply field state modulation to detected patterns.
        
        This replaces the binary back pressure mechanism with a more nuanced approach
        that considers field topology, resonance patterns, and interference effects.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Modulated list of patterns
        """
        # Extract field metrics from current state
        coherence = self.field_state.coherence
        stability = self.field_state.stability
        current_turbulence = 1.0 - stability  # Estimate turbulence as inverse of stability
        
        # Apply field state continuity - blend current with previous state
        # This creates a continuous flow rather than discrete snapshots
        turbulence = (self._previous_field_metrics['turbulence'] * 0.7) + (current_turbulence * 0.3)
        
        # Calculate current field density based on recent pattern history
        pattern_history = getattr(self.base_detector, 'pattern_history', [])
        recent_pattern_count = len([p for p in pattern_history 
                                  if hasattr(p, 'timestamp') and 
                                  (datetime.now() - p.timestamp).total_seconds() < 10])
        current_density = min(1.0, recent_pattern_count / 20)  # Normalize to 0-1
        
        # Apply field density continuity
        field_density = (self._previous_field_metrics['density'] * 0.7) + (current_density * 0.3)
        
        # Update previous metrics for next cycle - maintaining continuity
        self._previous_field_metrics['turbulence'] = turbulence
        self._previous_field_metrics['density'] = field_density
        self._previous_field_metrics['coherence'] = coherence
        self._previous_field_metrics['stability'] = stability
        
        # Store field state history for visualization
        self.visualization_data['field_state_history'].append({
            'timestamp': datetime.now(),
            'density': field_density,
            'turbulence': turbulence,
            'coherence': coherence,
            'stability': stability
        })
        
        # Track interference patterns from recently detected patterns
        interference_patterns = {}
        for p in pattern_history[-10:] if hasattr(self.base_detector, 'pattern_history') else []:
            pattern_type = p.get('context', {}).get('cascade_type', 'secondary')
            pattern_id = p.get('id', str(id(p)))
            confidence = p.get('confidence', 0.5)
            
            # Calculate interference strength based on pattern type and confidence
            type_factor = self.pattern_type_receptivity.get(pattern_type, 0.5)
            
            interference_patterns[pattern_id] = {
                'strength': confidence * type_factor,
                'pattern_type': pattern_type
            }
            
            # Store for visualization
            self.visualization_data['interference_patterns'][pattern_id] = {
                'timestamp': datetime.now(),
                'strength': confidence * type_factor,
                'pattern_type': pattern_type,
                'position': [np.random.random(), np.random.random()]  # Placeholder for actual field position
            }
        
        # Determine if we're in a resonance window (period of high receptivity)
        # Higher turbulence = more likely to have resonance windows
        resonance_threshold = 0.7 - (field_density * 0.2)  # Adaptive threshold based on density
        in_resonance_window = turbulence > resonance_threshold and np.random.random() < (turbulence * 0.3)
        
        if in_resonance_window:
            logger.info(f"Field turbulence ({turbulence:.2f}) created resonance window")
            # Record resonance window for visualization
            window_id = f"window_{len(self.visualization_data['resonance_centers']) + 1}"
            self.visualization_data['resonance_centers'][window_id] = {
                'timestamp': datetime.now(),
                'turbulence': turbulence,
                'duration': 2 + (np.random.random() * 3),  # 2-5 seconds
                'position': [np.random.random(), np.random.random()]  # Placeholder
            }
        
        # Apply modulation to each pattern
        modulated_patterns = []
        detected_pattern_types = set()
        
        for pattern in patterns:
            pattern_type = pattern.get('context', {}).get('cascade_type', 'secondary')
            pattern_confidence = pattern.get('confidence', 0.5)
            pattern_id = pattern.get('id', str(id(pattern)))
            
            # Use adaptive receptivity for this pattern type
            base_receptivity = self.pattern_type_receptivity.get(pattern_type, 0.5)
            
            # Adjust for field density (higher density = lower receptivity)
            density_factor = 1 - (field_density * 0.5)  # 0.5-1.0
            
            # Adjust for field turbulence (higher turbulence = higher receptivity for unusual patterns)
            turbulence_factor = 1.0
            if pattern_type in ['emergent', 'meta']:
                turbulence_factor = 1 + (turbulence * 0.5)  # 1.0-1.5
            
            # Check for resonance window boost
            window_boost = 0.3 if in_resonance_window else 0.0
            
            # Calculate interference from existing patterns
            interference = 0.0
            related_patterns = []
            
            for other_id, interference_data in interference_patterns.items():
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
            
            # Calculate final receptivity
            receptivity = base_receptivity * density_factor * turbulence_factor + window_boost
            receptivity = max(0.1, min(1.0, receptivity))
            
            # Calculate detection threshold - adaptive based on pattern relationships
            relationship_factor = 0.1 if related_patterns else 0.0
            threshold = 0.3 + (interference * 0.5) - relationship_factor
            
            # Adjust confidence based on receptivity
            adjusted_confidence = pattern_confidence * receptivity
            
            # Determine if pattern should be detected
            should_detect = adjusted_confidence > threshold
            
            # Add some randomness for errant behavior (more likely with high turbulence)
            if turbulence > 0.5 and np.random.random() < (turbulence * 0.2):
                # Randomly flip the decision with low probability
                should_detect = not should_detect
                logger.info(f"Field turbulence caused errant detection behavior")
            
            if should_detect:
                # Adjust confidence based on field state
                confidence_modifier = receptivity * (1 - interference)
                pattern['confidence'] = pattern_confidence * confidence_modifier
                modulated_patterns.append(pattern)
                detected_pattern_types.add(pattern_type)
                
                # Record pattern emergence for visualization
                self.visualization_data['pattern_emergence_points'].append({
                    'pattern_id': pattern_id,
                    'pattern_type': pattern_type,
                    'timestamp': datetime.now(),
                    'confidence': pattern['confidence'],
                    'position': [np.random.random(), np.random.random()]  # Placeholder
                })
                
                # Update pattern relationships
                for other_id in interference_patterns.keys():
                    relationship_key = f"{pattern_id}_{other_id}"
                    # Strengthen relationship if both patterns detected together
                    if other_id in [p.get('id', '') for p in modulated_patterns]:
                        current_strength = self.pattern_relationships.get(relationship_key, 0.0)
                        self.pattern_relationships[relationship_key] = min(1.0, current_strength + 0.1)
                        
                        # Record relationship strength for history
                        self.relationship_strength_history.append({
                            'key': relationship_key,
                            'strength': self.pattern_relationships[relationship_key],
                            'timestamp': datetime.now()
                        })
                
                # Log the modulation effect
                logger.info(f"Field state modulation: pattern {pattern_id} ({pattern_type}) "
                           f"passed with adjusted confidence {pattern['confidence']:.2f}")
        
        # Update adaptive receptivity based on detected patterns
        self._update_adaptive_receptivity(detected_pattern_types)
        
        return modulated_patterns
        
    def _update_adaptive_receptivity(self, detected_types: Set[str]):
        """
        Update adaptive receptivity based on detected pattern types.
        
        Args:
            detected_types: Set of detected pattern types
        """
        # Record detection for each pattern type
        for pattern_type in self.pattern_type_receptivity.keys():
            # 1.0 if detected, 0.0 if not
            detection_value = 1.0 if pattern_type in detected_types else 0.0
            self.receptivity_history[pattern_type].append(detection_value)
            
            # Calculate recent detection rate
            if len(self.receptivity_history[pattern_type]) > 5:
                detection_rate = sum(self.receptivity_history[pattern_type]) / len(self.receptivity_history[pattern_type])
                
                # If rarely detected, increase receptivity
                if detection_rate < 0.2:
                    self.pattern_type_receptivity[pattern_type] = min(
                        1.0, 
                        self.pattern_type_receptivity[pattern_type] + self.receptivity_learning_rate
                    )
                # If frequently detected, decrease receptivity
                elif detection_rate > 0.8:
                    self.pattern_type_receptivity[pattern_type] = max(
                        0.1, 
                        self.pattern_type_receptivity[pattern_type] - self.receptivity_learning_rate
                    )
                    
        # Log adaptive receptivity changes
        logger.info(f"Updated pattern type receptivity: {self.pattern_type_receptivity}")
        
    def _export_visualization_data(self):
        """
        Export visualization data to the harmonic I/O service for external visualization tools.
        
        This method prepares and exports field state, pattern relationships, and other
        visualization data to be consumed by external visualization tools.
        """
        try:
            # Prepare visualization data package
            timestamp = datetime.now().isoformat()
            visualization_package = {
                'timestamp': timestamp,
                'field_state': {
                    'current': {
                        'density': self._previous_field_metrics['density'],
                        'turbulence': self._previous_field_metrics['turbulence'],
                        'coherence': self._previous_field_metrics['coherence'],
                        'stability': self._previous_field_metrics['stability']
                    },
                    'history': list(self.visualization_data['field_state_history'])
                },
                'pattern_relationships': {
                    'current': self.pattern_relationships,
                    'history': list(self.relationship_strength_history)
                },
                'pattern_emergence': self.visualization_data['pattern_emergence_points'][-20:],
                'resonance_centers': self.visualization_data['resonance_centers'],
                'interference_patterns': self.visualization_data['interference_patterns'],
                'receptivity': {
                    'current': self.pattern_type_receptivity,
                    'history': {k: list(v) for k, v in self.receptivity_history.items()}
                }
            }
            
            # Schedule export operation through harmonic I/O service
            self.harmonic_io_service.schedule_operation(
                operation_type=OperationType.WRITE,
                repository=self.field_bridge,
                method_name='write_visualization_data',
                data_context={
                    'visualization_data': visualization_package,
                    'source': 'tonic_harmonic_detector',
                    'timestamp': timestamp,
                    'field_state_id': self.field_state.id if self.field_state else None
                }
            )
            
            logger.info(f"Exported field visualization data at {timestamp}")
            
            # Clear old data to prevent memory bloat
            if len(self.visualization_data['pattern_emergence_points']) > 100:
                self.visualization_data['pattern_emergence_points'] = self.visualization_data['pattern_emergence_points'][-50:]
                
            # Clear old interference patterns
            current_time = datetime.now()
            old_patterns = []
            for pattern_id, data in self.visualization_data['interference_patterns'].items():
                if (current_time - data['timestamp']).total_seconds() > 300:  # 5 minutes
                    old_patterns.append(pattern_id)
            
            for pattern_id in old_patterns:
                del self.visualization_data['interference_patterns'][pattern_id]
                
        except Exception as e:
            logger.error(f"Error exporting visualization data: {str(e)}")
    
    def get_field_visualization_data(self) -> Dict[str, Any]:
        """
        Get the current field visualization data for external visualization tools.
        
        This method provides a snapshot of the current field state, pattern relationships,
        and other visualization data for external tools to use.
        
        Returns:
            Dictionary containing visualization data
        """
        # Prepare current visualization data snapshot
        return {
            'timestamp': datetime.now().isoformat(),
            'field_state': {
                'density': self._previous_field_metrics['density'],
                'turbulence': self._previous_field_metrics['turbulence'],
                'coherence': self._previous_field_metrics['coherence'],
                'stability': self._previous_field_metrics['stability']
            },
            'pattern_type_receptivity': self.pattern_type_receptivity,
            'active_relationships': {
                k: v for k, v in self.pattern_relationships.items() 
                if v > 0.3  # Only include significant relationships
            },
            'recent_patterns': self.visualization_data['pattern_emergence_points'][-10:],
            'active_resonance_centers': {
                k: v for k, v in self.visualization_data['resonance_centers'].items()
                if (datetime.now() - v['timestamp']).total_seconds() < 30  # Only recent resonance centers
            }
        }
    
    def _apply_harmonic_filtering(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply harmonic filtering to detected patterns.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Filtered list of patterns
        """
        if not self.field_state:
            return patterns
        
        filtered_patterns = []
        
        for pattern in patterns:
            # Extract pattern components
            source = pattern.get("source", "")
            predicate = pattern.get("predicate", "")
            target = pattern.get("target", "")
            
            # Skip patterns with missing components
            if not source or not predicate or not target:
                continue
            
            # Create domain data for quick harmonic analysis
            domain_data = [
                {"id": f"source_{id(pattern)}", "name": source, "type": "source"},
                {"id": f"predicate_{id(pattern)}", "name": predicate, "type": "predicate"},
                {"id": f"target_{id(pattern)}", "name": target, "type": "target"}
            ]
            
            # Quick harmonic coherence check
            coherence = self._quick_coherence_check(domain_data)
            
            # Include pattern if coherence is above threshold
            if coherence >= self.boundary_thresholds["coherence"]:
                filtered_patterns.append(pattern)
        
        return filtered_patterns
    
    def _quick_coherence_check(self, domain_data: List[Dict[str, Any]]) -> float:
        """
        Perform a quick coherence check on domain data.
        
        Args:
            domain_data: Domain data to check
            
        Returns:
            Coherence score
        """
        # If we don't have enough domains for analysis
        if len(domain_data) < 3:
            return 0.0
        
        # Add required wave properties if missing
        for domain in domain_data:
            if "frequency" not in domain:
                # Generate frequency based on domain type
                if domain.get("type") == "source":
                    domain["frequency"] = 1.0
                elif domain.get("type") == "predicate":
                    domain["frequency"] = 1.5
                elif domain.get("type") == "target":
                    domain["frequency"] = 2.0
                else:
                    domain["frequency"] = 1.0
            
            if "amplitude" not in domain:
                domain["amplitude"] = 1.0
                
            if "phase" not in domain:
                domain["phase"] = 0.0
        
        # Use the actual TonicHarmonicMetrics implementation
        coherence_results = self.metrics.calculate_harmonic_coherence(domain_data)
        
        # Calculate average coherence from results
        if coherence_results:
            avg_coherence = sum(result["coherence"] for result in coherence_results) / len(coherence_results)
        else:
            avg_coherence = 0.5  # Default value
        
        # Use field state eigenvalues to weight the coherence if available
        if self.field_state and hasattr(self.field_state, 'eigenvalues'):
            # Get top 3 eigenvalues (or fewer if not available)
            top_eigenvalues = self.field_state.eigenvalues[:min(3, len(self.field_state.eigenvalues))]
            
            # Normalize eigenvalues
            if sum(top_eigenvalues) > 0:
                weights = [val / sum(top_eigenvalues) for val in top_eigenvalues]
                # Apply eigenvalue weighting
                field_factor = sum(w * (i+1)/len(weights) for i, w in enumerate(weights))
                # Blend coherence with field factor
                coherence = 0.7 * avg_coherence + 0.3 * field_factor
            else:
                coherence = avg_coherence
        else:
            coherence = avg_coherence
        
        return coherence


class VectorPlusFieldBridge:
    """
    Bridge between vector+ system and field-based pattern detection.
    
    This bridge enables bidirectional communication between the vector+
    system and the field-based pattern detection, ensuring that semantic
    boundaries are properly detected and respected during pattern evolution.
    """
    
    def __init__(self, 
                 event_bus: LocalEventBus,
                 harmonic_io_service: HarmonicIOService,
                 field_state: Optional[TonicHarmonicFieldState] = None):
        """
        Initialize the vector+ field bridge.
        
        Args:
            event_bus: Event bus for publishing and subscribing to events
            harmonic_io_service: Service for harmonic I/O operations
            field_state: Initial field state
        """
        self.event_bus = event_bus
        self.harmonic_io_service = harmonic_io_service
        self.field_state = field_state
        
        # Vector+ processing contexts
        self.processing_contexts = {}
        
        # Register for field events
        self._register_event_handlers()
        
        logger.info("VectorPlusFieldBridge initialized")
    
    def _register_event_handlers(self):
        """Register handlers for field and vector+ events."""
        self.event_bus.subscribe("field.state.updated", self._on_field_state_updated)
        self.event_bus.subscribe("vector.gradient.updated", self._on_vector_gradient_updated)
        self.event_bus.subscribe("pattern.detected", self._on_pattern_detected)
    
    def _on_field_state_updated(self, event: Event):
        """
        Handle field state update events.
        
        Args:
            event: Field state update event
        """
        field_state_data = event.data.get('field_state', {})
        if field_state_data:
            # Update local field state
            self.field_state = TonicHarmonicFieldState(field_state_data)
            
            # Publish vector+ gradient update based on field state
            self._publish_vector_gradient()
    
    def _on_vector_gradient_updated(self, event: Event):
        """
        Handle vector gradient update events.
        
        Args:
            event: Vector gradient update event
        """
        gradient_data = event.data.get('gradient', {})
        if gradient_data and self.field_state:
            # Update field state with vector gradient information
            self._update_field_with_gradient(gradient_data)
    
    def _on_pattern_detected(self, event: Event):
        """
        Handle pattern detection events.
        
        Args:
            event: Pattern detection event
        """
        pattern_id = event.data.get('pattern_id')
        pattern_data = event.data.get('pattern_data', {})
        
        if pattern_id and pattern_data:
            # Schedule vector+ analysis of the pattern
            self._schedule_vector_analysis(pattern_id, pattern_data)
    
    def _publish_vector_gradient(self):
        """Publish vector+ gradient update based on field state."""
        if not self.field_state:
            return
        
        # Extract field metrics
        coherence = self.field_state.coherence
        # Turbulence might be derived from metrics or field_properties
        turbulence = 1.0 - self.field_state.stability  # Estimate turbulence as inverse of stability
        stability = self.field_state.stability
        
        # Create gradient data
        gradient_data = {
            "coherence": coherence,
            "turbulence": turbulence,
            "stability": stability,
            "field_state_id": self.field_state.id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add eigenspace information if available
        if hasattr(self.field_state, 'eigenvalues') and hasattr(self.field_state, 'eigenvectors'):
            gradient_data["eigenspace"] = {
                "eigenvalues": self.field_state.eigenvalues[:min(3, len(self.field_state.eigenvalues))],
                "effective_dimensionality": self.field_state.effective_dimensionality
            }
        
        # Publish gradient update
        self.event_bus.publish(Event.create(
            "field.gradient.update",
            {
                "gradients": gradient_data
            },
            source="vector_plus_bridge"
        ))
        
        logger.info(f"Published field gradient update: coherence={coherence:.2f}, turbulence={turbulence:.2f}")
    
    def _update_field_with_gradient(self, gradient_data: Dict[str, Any]):
        """
        Update field state with vector gradient information.
        
        Args:
            gradient_data: Vector gradient data
        """
        if not self.field_state:
            return
        
        # Schedule field update through harmonic I/O
        context = {
            "operation_type": "field_update",
            "gradient_data": gradient_data,
            "field_state_id": self.field_state.id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.harmonic_io_service.schedule_operation(
            operation_type=OperationType.UPDATE.value,
            repository=self,
            method_name="_process_field_update",
            args=(),
            kwargs={"context": context},
            data_context=context
        )
    
    def _process_field_update(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process field update with vector gradient information.
        
        Args:
            context: Operation context
            
        Returns:
            Update results
        """
        gradient_data = context.get("gradient_data", {})
        field_state_id = context.get("field_state_id")
        
        if not gradient_data or not field_state_id or not self.field_state:
            logger.warning("Invalid field update context")
            return {"success": False, "error": "Invalid context"}
        
        # Check if field state matches
        if field_state_id != self.field_state.id:
            logger.warning(f"Field state mismatch: {field_state_id} != {self.field_state.id}")
            return {"success": False, "error": "Field state mismatch"}
        
        # Update field state with gradient information
        # This would typically involve more complex field mathematics
        # For now, we'll just update some basic metrics
        
        # Create updated field state
        updated_field_state = {
            "id": self.field_state.id,
            "version_id": str(uuid.uuid4()),  # New version
            "created_at": self.field_state.created_at,
            "last_modified": datetime.now().isoformat(),
            "topology": {
                "effective_dimensionality": self.field_state.effective_dimensionality,
                "principal_dimensions": self.field_state.principal_dimensions,
                "eigenvalues": self.field_state.eigenvalues,
                "eigenvectors": self.field_state.eigenvectors
            },
            "metrics": {
                "coherence": gradient_data.get("coherence", 0.5),
                "turbulence": gradient_data.get("turbulence", 0.5),
                "stability": gradient_data.get("stability", 0.5)
            }
        }
        
        # Publish updated field state
        self.event_bus.publish(Event(
            "field.state.updated",
            {
                "field_state": updated_field_state
            },
            source="vector_plus_bridge"
        ))
        
        return {"success": True, "field_state_id": self.field_state.id}
    
    def _schedule_vector_analysis(self, pattern_id: str, pattern_data: Dict[str, Any]):
        """
        Schedule vector+ analysis of a detected pattern.
        
        Args:
            pattern_id: ID of the detected pattern
            pattern_data: Data associated with the pattern
        """
        # Create operation context
        context = {
            "operation_type": "vector_analysis",
            "pattern_id": pattern_id,
            "pattern_data": pattern_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Schedule through harmonic I/O
        self.harmonic_io_service.schedule_operation(
            operation_type=OperationType.PROCESS.value,
            repository=self,
            method_name="_process_vector_analysis",
            args=(),
            kwargs={"context": context},
            data_context=context
        )
    
    def _process_vector_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process vector+ analysis of a pattern.
        
        Args:
            context: Operation context
            
        Returns:
            Analysis results
        """
        pattern_id = context.get("pattern_id")
        pattern_data = context.get("pattern_data", {})
        
        if not pattern_id or not pattern_data:
            logger.warning("Invalid vector analysis context")
            return {"success": False, "error": "Invalid context"}
        
        # Extract relationship components
        source = pattern_data.get("source", "")
        predicate = pattern_data.get("predicate", "")
        target = pattern_data.get("target", "")
        
        # In a real implementation, this would perform vector+ analysis
        # For now, we'll just create a simple vector representation
        
        # Create vector representation (simplified)
        vector_representation = {
            "pattern_id": pattern_id,
            "components": {
                "source": {"text": source, "vector": [0.1, 0.2, 0.3]},  # Placeholder
                "predicate": {"text": predicate, "vector": [0.4, 0.5, 0.6]},  # Placeholder
                "target": {"text": target, "vector": [0.7, 0.8, 0.9]}  # Placeholder
            },
            "combined_vector": [0.4, 0.5, 0.6],  # Placeholder
            "timestamp": datetime.now().isoformat()
        }
        
        # Publish vector representation
        self.event_bus.publish(Event(
            "pattern.vector_representation",
            {
                "pattern_id": pattern_id,
                "vector_representation": vector_representation
            },
            source="vector_plus_bridge"
        ))
        
        return {"success": True, "vector_representation": vector_representation}


def create_tonic_harmonic_detector(
    base_detector: LearningWindowAwareDetector,
    event_bus: LocalEventBus,
    harmonic_io_service: Optional[HarmonicIOService] = None,
    validator: Optional[Any] = None
) -> Tuple[TonicHarmonicPatternDetector, Any]:
    """
    Create a tonic-harmonic pattern detector with optional validator.
    
    Args:
        base_detector: Base learning window aware detector
        event_bus: Event bus for publishing and subscribing to events
        harmonic_io_service: Optional harmonic I/O service
        validator: Optional tonic-harmonic validator
        
    Returns:
        Tuple of (configured tonic-harmonic pattern detector, validator)
    """
    # Create harmonic I/O service if not provided
    if not harmonic_io_service:
        harmonic_io_service = HarmonicIOService(base_frequency=0.2, harmonics=3)
    
    # Create field bridge
    field_bridge = HarmonicFieldIOBridge(harmonic_io_service)
    
    # Create metrics with appropriate configuration
    metrics_config = {
        "harmonic_coherence_weight": 0.4,
        "phase_alignment_weight": 0.3,
        "resonance_stability_weight": 0.3,
        "frequency_bands": [
            {"name": "low", "min": 0.0, "max": 0.5},
            {"name": "medium", "min": 0.5, "max": 1.0},
            {"name": "high", "min": 1.0, "max": 2.0}
        ]
    }
    metrics = TonicHarmonicMetrics(config=metrics_config)
    
    # Create detector
    detector = TonicHarmonicPatternDetector(
        base_detector=base_detector,
        harmonic_io_service=harmonic_io_service,
        event_bus=event_bus,
        field_bridge=field_bridge,
        metrics=metrics
    )
    
    # Create vector+ bridge
    vector_bridge = VectorPlusFieldBridge(
        event_bus=event_bus,
        harmonic_io_service=harmonic_io_service
    )
    
    # Create validator if not provided
    if validator is None:
        try:
            # Import here to avoid circular imports
            from ...pattern_aware_rag.validation.tonic_harmonic_validator import TonicHarmonicValidator
            
            # Create a simple vector-only detector for comparison
            vector_only_detector = EventAwarePatternDetector(
                semantic_observer=base_detector.detector.semantic_observer,
                threshold=base_detector.detector.threshold * 1.5,  # Higher threshold for vector-only
                event_bus=event_bus,
                entity_id="vector_only_detector"
            )
            
            # Create the validator
            validator = TonicHarmonicValidator(
                vector_only_detector=vector_only_detector,
                resonance_detector=detector
            )
            
            logger.info("Created TonicHarmonicValidator for comparison testing")
        except ImportError:
            logger.warning("TonicHarmonicValidator not available, skipping validator creation")
            validator = None
    
    # Return the detector and validator
    return detector, validator
