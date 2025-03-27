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

from ...core.services.event_bus import LocalEventBus, Event
from ..id.adaptive_id import AdaptiveID
from ...field.field_state import TonicHarmonicFieldState
from ...field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from ..io.harmonic_io_service import HarmonicIOService, OperationType
from ..resonance.tonic_harmonic_metrics import TonicHarmonicMetrics
from .learning_window_integration import LearningWindowAwareDetector
from .event_aware_detector import EventAwarePatternDetector
from ...pattern_aware_rag.learning.learning_control import WindowState, BackPressureController

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
        
        # Register for field state updates
        self._register_field_state_handler()
        
        # Register for pattern events
        self._register_pattern_event_handlers()
        
        logger.info("TonicHarmonicPatternDetector initialized")
    
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
        coherence = self.field_state.get_coherence_metric()
        turbulence = self.field_state.get_turbulence_metric()
        stability = self.field_state.get_stability_metric()
        
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
            operation_type=OperationType.PROCESS,
            callback=self._process_pattern_analysis,
            context=context,
            priority=0.7  # Medium-high priority
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
            self.event_bus.publish(Event(
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
        Detect patterns with tonic-harmonic awareness.
        
        Returns:
            List of detected patterns
        """
        # First update detection parameters based on current field state
        if self.field_state:
            self._update_detection_parameters()
        
        # Delegate to base detector
        patterns = self.base_detector.detect_patterns()
        
        # Apply tonic-harmonic filtering if field state is available
        if patterns and self.field_state:
            patterns = self._apply_harmonic_filtering(patterns)
        
        return patterns
    
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
        coherence = self.field_state.get_coherence_metric()
        turbulence = self.field_state.get_turbulence_metric()
        stability = self.field_state.get_stability_metric()
        
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
        self.event_bus.publish(Event(
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
            operation_type=OperationType.UPDATE,
            callback=self._process_field_update,
            context=context,
            priority=0.8  # High priority
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
            operation_type=OperationType.PROCESS,
            callback=self._process_vector_analysis,
            context=context,
            priority=0.6  # Medium priority
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
