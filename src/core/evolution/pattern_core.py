"""
Core pattern service for Habitat knowledge evolution.

Natural observation service that works with knowledge coherence to track
pattern emergence without enforcing structure. Maintains alignment with adaptive_core
evolution tracking while providing enhanced evidence chains and temporal mapping.
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock, RLock
import logging
from uuid import uuid4

from core.interfaces.base_states import BaseProjectState
from core.events.event_manager import EventManager
from core.events.event_manager import EventType
from core.utils.timestamp_service import TimestampService
from core.utils.version_service import VersionService
from core.utils.logging_config import get_logger
from core.types import DensityMetrics

@dataclass
class LearningWindow:
    """Container for pattern evolution within a bounded context."""
    window_id: str
    start_time: datetime
    patterns: List[str] = field(default_factory=list)
    density_metrics: DensityMetrics = field(default_factory=DensityMetrics)
    coherence_level: float = 0.0
    viscosity_gradient: float = 0.0
    
    def __post_init__(self):
        """Post initialization validation."""
        if not isinstance(self.density_metrics, DensityMetrics):
            self.density_metrics = DensityMetrics()
        self.coherence_level = float(self.coherence_level)
        self.viscosity_gradient = float(self.viscosity_gradient)
    
    def update_viscosity(self, new_patterns: List[str]) -> float:
        """Update viscosity gradient based on pattern changes."""
        if not self.patterns:
            self.viscosity_gradient = 0.35  # Base viscosity
            return self.viscosity_gradient
            
        # Calculate pattern overlap
        current_set = set(self.patterns)
        new_set = set(new_patterns)
        intersection = len(current_set & new_set)
        union = len(current_set | new_set)
        
        # Calculate viscosity gradient
        self.viscosity_gradient = (
            0.35 +  # Base viscosity
            0.45 * (intersection / union if union > 0 else 0) +  # Pattern stability
            0.20 * (len(new_set) / len(current_set) if current_set else 1)  # Growth factor
        )
        return self.viscosity_gradient

@dataclass
class TemporalContext:
    """Temporal context for pattern tracking."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    sequence_order: Optional[int] = None
    confidence: float = 1.0
    learning_window: Optional[LearningWindow] = None

@dataclass
class UncertaintyMetrics:
    """Uncertainty metrics for pattern evidence."""
    confidence_score: float = 1.0
    uncertainty_value: float = 0.0
    reliability_score: float = 1.0
    source_quality: float = 1.0
    temporal_stability: float = 1.0
    cross_reference_score: float = 1.0
    interface_confidence: float = 1.0  # Confidence in interface recognition
    viscosity_stability: float = 1.0   # Stability of viscosity measurements
    
    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence with interface and viscosity factors."""
        return min(1.0, (
            0.3 * self.confidence_score +
            0.2 * self.reliability_score +
            0.2 * self.temporal_stability +
            0.15 * self.interface_confidence +
            0.15 * self.viscosity_stability
        ))

@dataclass
class PatternEvolutionMetrics:
    """Metrics for pattern evolution tracking."""
    gradient: float = 0.0
    interface_strength: float = 0.0
    stability: float = 0.0
    emergence_rate: float = 0.0
    coherence_level: float = 0.0
    
    def calculate_recognition_threshold(self) -> float:
        """Calculate pattern recognition threshold."""
        return (
            0.4 * self.coherence_level +
            0.3 * self.stability +
            0.3 * self.interface_strength
        )

@dataclass
class PatternEvidence:
    """Enhanced evidence structure for pattern tracking."""
    evidence_id: str
    timestamp: str
    pattern_type: str
    source_data: Dict[str, Any]
    temporal_context: Optional[TemporalContext] = None
    uncertainty_metrics: Optional[UncertaintyMetrics] = None
    cross_references: List[str] = None
    stability_score: float = 1.0
    emergence_rate: float = 0.0
    version: str = "1.0"
    evolution_metrics: Optional[PatternEvolutionMetrics] = None
    density_metrics: Optional[DensityMetrics] = None

    def __post_init__(self):
        if self.cross_references is None:
            self.cross_references = []
        if self.temporal_context is None:
            self.temporal_context = TemporalContext()
        if self.uncertainty_metrics is None:
            self.uncertainty_metrics = UncertaintyMetrics()

    def validate(self) -> bool:
        """Validate pattern evidence."""
        return all([
            0 <= self.stability_score <= 1,
            0 <= self.emergence_rate <= 1,
            bool(self.pattern_type),
            bool(self.source_data)
        ])
        
    def calculate_density_score(self) -> float:
        """Calculate overall density score from metrics."""
        if not self.density_metrics:
            return 0.0
        return (
            0.4 * self.density_metrics.global_density +
            0.4 * self.density_metrics.local_density +
            0.2 * self.density_metrics.cross_domain_strength
        )

class PatternCore:
    """
    Core pattern observation service that naturally expresses coherence through
    evidence-based pattern emergence. Maintains alignment with adaptive_core
    evolution tracking while providing enhanced temporal mapping.
    """
    def __init__(
        self,
        timestamp_service: Optional[TimestampService] = None,
        event_manager: Optional[EventManager] = None,
        version_service: Optional[VersionService] = None
    ):
        self.timestamp_service = timestamp_service or TimestampService()
        self.event_manager = event_manager or EventManager()
        self.version_service = version_service or VersionService()
        self._lock = RLock()
        
        # Enhanced pattern observation storage
        self.evidence_chains: Dict[str, List[PatternEvidence]] = {}
        self.temporal_maps: Dict[str, Dict[str, Any]] = {}
        self.pattern_versions: Dict[str, List[Dict[str, Any]]] = {}
        
        # Pattern evolution tracking
        self.pattern_metrics: Dict[str, UncertaintyMetrics] = {}
        self.temporal_contexts: Dict[str, TemporalContext] = {}
        self.stability_scores: Dict[str, float] = {}
        self.emergence_rates: Dict[str, float] = {}
        
        # Learning window management
        self.active_windows: Dict[str, LearningWindow] = {}
        self.window_metrics: Dict[str, DensityMetrics] = {}
        
        # Evolution metrics
        self.evolution_metrics: Dict[str, PatternEvolutionMetrics] = {}
        
        # Cross-reference tracking
        self.pattern_references: Dict[str, Set[str]] = {}
        self.reference_strengths: Dict[str, Dict[str, float]] = {}
        
        # Thresholds
        self.evidence_threshold = 0.3
        self.temporal_threshold = 0.3
        self.density_threshold = 0.3
        self.viscosity_threshold = 0.35
        self.stability_threshold = 0.3
        
        # Observation history
        self.pattern_history: List[Dict[str, Any]] = []
        self.latest_observations: Dict[str, Any] = {}
        
        # Logger initialization
        self.logger = get_logger(__name__)
        
    def create_learning_window(self, initial_patterns: List[str] = None) -> str:
        """Create a new learning window for pattern evolution tracking.
        
        Args:
            initial_patterns: Optional list of patterns to initialize window with
            
        Returns:
            str: ID of created learning window
        """
        window_id = str(uuid4())
        window = LearningWindow(
            window_id=window_id,
            start_time=datetime.now(),
            patterns=initial_patterns or []
        )
        
        with self._lock:
            self.active_windows[window_id] = window
            self.window_metrics[window_id] = DensityMetrics()
            
        self.logger.info(
            f"Created learning window {window_id} with {len(initial_patterns or [])} initial patterns"
        )
        return window_id
    
    def update_window_patterns(self, window_id: str, new_patterns: List[str]) -> Dict[str, Any]:
        """Update patterns in a learning window and recalculate metrics.
        
        Args:
            window_id: ID of learning window to update
            new_patterns: New patterns to process
            
        Returns:
            Dict containing updated metrics
        """
        with self._lock:
            if window_id not in self.active_windows:
                raise ValueError(f"Learning window {window_id} not found")
                
            window = self.active_windows[window_id]
            
            # Update viscosity gradient
            viscosity = window.update_viscosity(new_patterns)
            
            # Update density metrics
            metrics = self.window_metrics[window_id]
            metrics.local_density = len(new_patterns) / max(len(window.patterns), 1)
            metrics.viscosity = viscosity
            
            # Calculate cross-domain strength through pattern overlap
            cross_domain = self._calculate_cross_domain_strength(new_patterns)
            metrics.cross_domain_strength = cross_domain
            
            # Update interface recognition based on metrics
            metrics.interface_recognition = metrics.calculate_interface_strength()
            
            # Update window patterns
            window.patterns = new_patterns
            window.density_metrics = metrics
            
            return {
                "viscosity": viscosity,
                "local_density": metrics.local_density,
                "cross_domain_strength": cross_domain,
                "interface_recognition": metrics.interface_recognition
            }
    
    def _calculate_cross_domain_strength(self, patterns: List[str]) -> float:
        """Calculate cross-domain strength based on pattern relationships.
        
        Args:
            patterns: List of patterns to analyze
            
        Returns:
            float: Cross-domain strength score
        """
        if not patterns:
            return 0.0
            
        # Get all related patterns through references
        related_patterns = set()
        for pattern in patterns:
            if pattern in self.pattern_references:
                related_patterns.update(self.pattern_references[pattern])
        
        # Calculate strength based on relationship overlap
        if not related_patterns:
            return 0.0
            
        pattern_set = set(patterns)
        overlap = len(pattern_set & related_patterns)
        total = len(pattern_set | related_patterns)
        
        return overlap / total if total > 0 else 0.0
    
    def calculate_window_metrics(self, window_id: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a learning window.
        
        Args:
            window_id: ID of learning window to analyze
            
        Returns:
            Dict containing calculated metrics
        """
        with self._lock:
            if window_id not in self.active_windows:
                raise ValueError(f"Learning window {window_id} not found")
                
            window = self.active_windows[window_id]
            metrics = self.window_metrics[window_id]
            
            # Calculate global density across all windows
            total_patterns = sum(len(w.patterns) for w in self.active_windows.values())
            if total_patterns > 0:
                metrics.global_density = len(window.patterns) / total_patterns
            
            # Get evidence chains for patterns in window
            evidence_scores = []
            for pattern in window.patterns:
                if pattern in self.evidence_chains:
                    chain = self.evidence_chains[pattern]
                    if chain and chain[-1].density_metrics:
                        evidence_scores.append(chain[-1].calculate_density_score())
            
            # Update coherence level based on evidence and metrics
            if evidence_scores:
                window.coherence_level = (
                    0.4 * sum(evidence_scores) / len(evidence_scores) +
                    0.3 * metrics.interface_recognition +
                    0.3 * metrics.viscosity
                )
            
            return {
                "global_density": metrics.global_density,
                "local_density": metrics.local_density,
                "cross_domain_strength": metrics.cross_domain_strength,
                "interface_recognition": metrics.interface_recognition,
                "viscosity": metrics.viscosity,
                "coherence_level": window.coherence_level,
                "viscosity_gradient": window.viscosity_gradient
            }
            
    def track_pattern_evolution(self, pattern_id: str, window_id: str) -> Dict[str, Any]:
        """Track pattern evolution within a learning window.
        
        Args:
            pattern_id: ID of pattern to track
            window_id: ID of learning window context
            
        Returns:
            Dict containing evolution metrics
        """
        with self._lock:
            if pattern_id not in self.evidence_chains:
                raise ValueError(f"Pattern {pattern_id} not found")
            if window_id not in self.active_windows:
                raise ValueError(f"Learning window {window_id} not found")
                
            # Get current evidence and window
            evidence_chain = self.evidence_chains[pattern_id]
            window = self.active_windows[window_id]
            
            if not evidence_chain:
                return {}
                
            current_evidence = evidence_chain[-1]
            
            # Initialize evolution metrics if needed
            if not current_evidence.evolution_metrics:
                current_evidence.evolution_metrics = PatternEvolutionMetrics()
            
            evolution_metrics = current_evidence.evolution_metrics
            
            # Calculate gradient from window viscosity
            evolution_metrics.gradient = window.viscosity_gradient
            
            # Update interface strength from window metrics
            evolution_metrics.interface_strength = (
                window.density_metrics.interface_recognition
            )
            
            # Calculate stability based on evidence chain
            if len(evidence_chain) > 1:
                prev_evidence = evidence_chain[-2]
                evolution_metrics.stability = self._calculate_stability(
                    pattern_id, window
                )
            
            # Update emergence rate based on window context
            evolution_metrics.emergence_rate = self._calculate_emergence_rate(
                pattern_id, window
            )
            
            # Set coherence level from window
            evolution_metrics.coherence_level = window.coherence_level
            
            # Store evolution metrics
            self.evolution_metrics[pattern_id] = evolution_metrics
            
            return {
                "gradient": evolution_metrics.gradient,
                "interface_strength": evolution_metrics.interface_strength,
                "stability": evolution_metrics.stability,
                "emergence_rate": evolution_metrics.emergence_rate,
                "coherence_level": evolution_metrics.coherence_level,
                "recognition_threshold": evolution_metrics.calculate_recognition_threshold()
            }
    
    def _calculate_stability(self, pattern_id: str, window: LearningWindow) -> float:
        """Calculate pattern stability in window context.
        
        Args:
            pattern_id: Pattern to analyze
            window: Learning window context
            
        Returns:
            float: Stability score
        """
        if not window.patterns:
            return 0.0
            
        # Calculate temporal stability based on pattern presence
        temporal_stability = 1.0 if pattern_id in window.patterns else 0.0
        
        # Calculate interface stability
        interface_stability = window.density_metrics.interface_recognition
        
        # Calculate viscosity stability
        viscosity_stability = 1.0 - window.viscosity_gradient  # Lower gradient means higher stability
        
        # Weighted combination
        stability = (
            0.4 * temporal_stability +
            0.3 * interface_stability +
            0.3 * viscosity_stability
        )
        
        # Update stability tracking
        self.stability_scores[pattern_id] = stability
        
        return min(1.0, max(0.0, stability))  # Ensure stability is between 0 and 1
    
    def _calculate_emergence_rate(self, pattern_id: str, window: LearningWindow) -> float:
        """Calculate pattern emergence rate in window context.
        
        Args:
            pattern_id: Pattern to analyze
            window: Learning window context
            
        Returns:
            float: Emergence rate
        """
        if not window.patterns:
            return 0.0
            
        # Get pattern references in window
        window_refs = set()
        for pattern in window.patterns:
            if pattern in self.pattern_references:
                window_refs.update(self.pattern_references[pattern])
        
        # Calculate emergence factors
        density_factor = window.density_metrics.local_density if window.density_metrics else 0.0
        reference_factor = (
            len(window_refs) / len(window.patterns)
            if pattern_id in window_refs else 0.0
        )
        viscosity_factor = window.viscosity_gradient
        
        # Weighted emergence rate
        emergence_rate = (
            0.4 * density_factor +
            0.3 * reference_factor +
            0.3 * viscosity_factor
        )
        
        return max(0.0, min(1.0, emergence_rate))
    
    def assess_interface_strength(self, pattern_id: str, window_id: str) -> Dict[str, Any]:
        """Assess interface strength for a pattern in a learning window.
        
        Args:
            pattern_id: Pattern to analyze
            window_id: Learning window context
            
        Returns:
            Dict containing interface metrics
        """
        with self._lock:
            if pattern_id not in self.evidence_chains:
                raise ValueError(f"Pattern {pattern_id} not found")
            if window_id not in self.active_windows:
                raise ValueError(f"Learning window {window_id} not found")
            
            window = self.active_windows[window_id]
            evidence = self.evidence_chains[pattern_id][-1]
            
            # Calculate interface metrics
            interface_metrics = self._calculate_interface_metrics(evidence, window)
            
            # Update uncertainty metrics
            if evidence.uncertainty_metrics:
                evidence.uncertainty_metrics.interface_confidence = interface_metrics["confidence"]
                evidence.uncertainty_metrics.viscosity_stability = interface_metrics["stability"]
            
            return interface_metrics
    
    def _calculate_interface_metrics(self, evidence: PatternEvidence, window: LearningWindow) -> Dict[str, float]:
        """Calculate detailed interface metrics.
        
        Args:
            evidence: Pattern evidence to analyze
            window: Learning window context
            
        Returns:
            Dict containing interface metric scores
        """
        # Base recognition strength
        recognition = window.density_metrics.interface_recognition
        
        # Calculate confidence based on evidence and window metrics
        confidence = min(1.0, (
            0.4 * recognition +
            0.3 * window.coherence_level +
            0.3 * (evidence.stability_score if evidence else 0.0)
        ))
        
        # Calculate stability based on viscosity
        stability = min(1.0, (
            0.5 * window.viscosity_gradient +
            0.5 * (1.0 - abs(window.viscosity_gradient - (
                evidence.evolution_metrics.gradient if evidence.evolution_metrics else 0.0
            )))
        ))
        
        # Calculate alignment score
        alignment = 0.0
        if evidence.density_metrics and window.density_metrics:
            alignment = 1.0 - abs(
                evidence.density_metrics.interface_recognition -
                window.density_metrics.interface_recognition
            )
        
        return {
            "recognition": recognition,
            "confidence": confidence,
            "stability": stability,
            "alignment": alignment,
            "overall_strength": (
                0.3 * recognition +
                0.3 * confidence +
                0.2 * stability +
                0.2 * alignment
            )
        }
        
        # Observation history
        self.pattern_history: List[Dict[str, Any]] = []
        self.latest_observations: Dict[str, Any] = {}
        
    def observe_pattern(self, pattern_id: str, observation: Dict[str, Any], window_id: Optional[str] = None) -> Dict[str, Any]:
        """Record a pattern observation and trigger evolution tracking.
        
        Args:
            pattern_id: ID of pattern being observed
            observation: Observation data
            window_id: Optional ID of learning window to use
            
        Returns:
            Dict containing observation results
        """
        with self._lock:
            timestamp = self.timestamp_service.get_timestamp()
            
            # Create or get learning window
            if not window_id:
                window_id = self.create_learning_window([pattern_id])
            elif window_id not in self.active_windows:
                raise ValueError(f"Learning window {window_id} not found")
            
            # Create pattern evidence
            evidence = PatternEvidence(
                evidence_id=str(uuid4()),
                timestamp=timestamp,
                pattern_type=observation.get("type", "unknown"),
                source_data=observation,
                temporal_context=TemporalContext(
                    start_time=datetime.now(),
                    learning_window=self.active_windows[window_id]
                ),
                uncertainty_metrics=UncertaintyMetrics()
            )
            
            # Initialize or update evidence chain
            if pattern_id not in self.evidence_chains:
                self.evidence_chains[pattern_id] = []
            self.evidence_chains[pattern_id].append(evidence)
            
            # Track pattern evolution
            evolution_metrics = self.track_pattern_evolution(pattern_id, window_id)
            
            # Assess interface strength
            interface_metrics = self.assess_interface_strength(pattern_id, window_id)
            
            # Update window patterns and calculate metrics
            window_metrics = self.update_window_patterns(
                window_id,
                self.active_windows[window_id].patterns + [pattern_id]
            )
            
            # Record observation
            observation_record = {
                "pattern_id": pattern_id,
                "timestamp": timestamp,
                "window_id": window_id,
                "evolution_metrics": evolution_metrics,
                "interface_metrics": interface_metrics,
                "window_metrics": window_metrics
            }
            
            self.pattern_history.append(observation_record)
            self.latest_observations[pattern_id] = observation_record
            
            # Emit observation event
            self._emit_observation_event(pattern_id, observation_record)
            
            return observation_record
    
    def _emit_observation_event(self, pattern_id: str, observation: Dict[str, Any]):
        """Emit pattern observation event.
        
        Args:
            pattern_id: ID of observed pattern
            observation: Observation record
        """
        event_data = {
            "pattern_id": pattern_id,
            "timestamp": observation["timestamp"],
            "window_id": observation["window_id"],
            "metrics": {
                "evolution": observation["evolution_metrics"],
                "interface": observation["interface_metrics"],
                "window": observation["window_metrics"]
            }
        }
        
        self.event_manager.emit_event(
            EventType.PATTERN_OBSERVED,
            event_data
        )
    
    def get_pattern_state(self, pattern_id: str) -> Dict[str, Any]:
        """Get current state of a pattern.
        
        Args:
            pattern_id: ID of pattern to check
            
        Returns:
            Dict containing pattern state
        """
        with self._lock:
            if pattern_id not in self.evidence_chains:
                raise ValueError(f"Pattern {pattern_id} not found")
                
            evidence_chain = self.evidence_chains[pattern_id]
            if not evidence_chain:
                return {}
                
            current_evidence = evidence_chain[-1]
            latest_observation = self.latest_observations.get(pattern_id, {})
            
            return {
                "evidence": {
                    "id": current_evidence.evidence_id,
                    "timestamp": current_evidence.timestamp,
                    "type": current_evidence.pattern_type,
                    "stability": current_evidence.stability_score,
                    "emergence_rate": current_evidence.emergence_rate
                },
                "evolution": latest_observation.get("evolution_metrics", {}),
                "interface": latest_observation.get("interface_metrics", {}),
                "window": latest_observation.get("window_metrics", {}),
                "chain_length": len(evidence_chain)
            }
    
    def get_window_state(self, window_id: str) -> Dict[str, Any]:
        """Get current state of a learning window.
        
        Args:
            window_id: ID of window to check
            
        Returns:
            Dict containing window state
        """
        with self._lock:
            if window_id not in self.active_windows:
                raise ValueError(f"Learning window {window_id} not found")
                
            window = self.active_windows[window_id]
            metrics = self.window_metrics[window_id]
            
            # Calculate current metrics
            current_metrics = self.calculate_window_metrics(window_id)
            
            # Get patterns in window
            pattern_states = {}
            for pattern_id in window.patterns:
                if pattern_id in self.evidence_chains:
                    pattern_states[pattern_id] = self.get_pattern_state(pattern_id)
            
            return {
                "window_id": window_id,
                "start_time": window.start_time.isoformat(),
                "pattern_count": len(window.patterns),
                "metrics": current_metrics,
                "patterns": pattern_states,
                "coherence_level": window.coherence_level,
                "viscosity_gradient": window.viscosity_gradient
            }
        
        # Thresholds for light validation
        self.evidence_threshold = 0.3
        self.temporal_threshold = 0.3
        self.density_threshold = 0.3
        self.viscosity_threshold = 0.35
        
    def process_pattern_batch(self, patterns: List[Dict[str, Any]], window_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a batch of patterns in a single learning window.
        
        Args:
            patterns: List of pattern observations
            window_id: Optional ID of learning window to use
            
        Returns:
            Dict containing batch processing results
        """
        with self._lock:
            # Create new learning window if needed
            if not window_id:
                window_id = self.create_learning_window()
            
            batch_results = {
                "window_id": window_id,
                "processed_count": 0,
                "pattern_results": {},
                "window_metrics": None
            }
            
            # Process each pattern
            for pattern in patterns:
                pattern_id = pattern.get("id") or str(uuid4())
                try:
                    result = self.observe_pattern(pattern_id, pattern, window_id)
                    batch_results["pattern_results"][pattern_id] = result
                    batch_results["processed_count"] += 1
                except Exception as e:
                    self.logger.error(f"Error processing pattern {pattern_id}: {str(e)}")
                    batch_results["pattern_results"][pattern_id] = {"error": str(e)}
            
            # Get final window state
            batch_results["window_metrics"] = self.calculate_window_metrics(window_id)
            
            return batch_results
    
    def cleanup_inactive_windows(self, max_age_minutes: int = 60) -> List[str]:
        """Remove inactive learning windows.
        
        Args:
            max_age_minutes: Maximum age in minutes for inactive windows
            
        Returns:
            List of removed window IDs
        """
        with self._lock:
            current_time = datetime.now()
            removed_windows = []
            
            for window_id, window in list(self.active_windows.items()):
                age = (current_time - window.start_time).total_seconds() / 60
                
                if age > max_age_minutes:
                    # Archive window metrics if needed
                    final_metrics = self.calculate_window_metrics(window_id)
                    
                    # Remove window
                    del self.active_windows[window_id]
                    del self.window_metrics[window_id]
                    
                    removed_windows.append(window_id)
                    
                    self.logger.info(f"Removed inactive window {window_id} (age: {age:.1f} minutes)")
            
            return removed_windows
    
    def prune_pattern_history(self, max_entries: int = 1000) -> int:
        """Prune pattern observation history to prevent unbounded growth.
        
        Args:
            max_entries: Maximum number of history entries to keep
            
        Returns:
            int: Number of entries removed
        """
        with self._lock:
            current_size = len(self.pattern_history)
            if current_size <= max_entries:
                return 0
            
            # Keep most recent entries
            entries_to_remove = current_size - max_entries
            self.pattern_history = self.pattern_history[-max_entries:]
            
            self.logger.info(f"Pruned {entries_to_remove} entries from pattern history")
            return entries_to_remove
        self.stability_threshold = 0.3
        
        self.logger = get_logger(__name__)
        self.logger.info("Initialized pattern observation service")

    def observe_evolution(
        self, 
        structural_change: Dict[str, Any],
        semantic_change: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Observe coherence in knowledge evolution without enforcing it.
        Enhanced to track pattern emergence and stability.
        """
        try:
            with self._lock:
                # Create pattern evidence
                pattern_evidence = self._create_pattern_evidence(
                    structural_change,
                    semantic_change,
                    evidence
                )
                
                # Update evidence chains
                self._update_evidence_chains(pattern_evidence)
                
                # Track temporal context
                self._update_temporal_context(pattern_evidence)
                
                # Calculate stability and emergence
                stability = self._calculate_stability(pattern_evidence)
                emergence = self._calculate_emergence_rate(pattern_evidence)
                
                # Update metrics
                pattern_evidence.stability_score = stability
                pattern_evidence.emergence_rate = emergence
                
                # Track cross-references
                self._update_cross_references(pattern_evidence)
                
                # Create observation result
                observation = {
                    "pattern_id": str(uuid4()),
                    "evidence": pattern_evidence,
                    "stability": stability,
                    "emergence_rate": emergence,
                    "timestamp": self.timestamp_service.get_timestamp(),
                    "cross_references": list(self.pattern_references.get(
                        pattern_evidence.evidence_id, set()
                    ))
                }
                
                # Track history
                self.pattern_history.append(observation)
                self.latest_observations[observation["pattern_id"]] = observation

                # Emit event
                self.event_manager.publish(EventType.PATTERN_OBSERVED, observation)

                return observation

        except Exception as e:
            self.logger.error(f"Error observing evolution: {str(e)}")
            return {}

    def _create_pattern_evidence(
        self,
        structural_change: Dict[str, Any],
        semantic_change: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None
    ) -> PatternEvidence:
        """Create pattern evidence from changes."""
        evidence_id = str(uuid4())
        timestamp = self.timestamp_service.get_timestamp()
        
        # Determine pattern type from changes
        pattern_type = self._determine_pattern_type(
            structural_change,
            semantic_change
        )
        
        # Combine source data
        source_data = {
            "structural": structural_change,
            "semantic": semantic_change,
            "additional": evidence or {}
        }
        
        # Create temporal context
        temporal_context = TemporalContext(
            start_time=datetime.now(),
            confidence=self._calculate_temporal_confidence(source_data)
        )
        
        # Calculate uncertainty metrics
        uncertainty_metrics = UncertaintyMetrics(
            confidence_score=self._calculate_confidence(source_data),
            uncertainty_value=self._calculate_uncertainty(source_data),
            reliability_score=self._calculate_reliability(source_data),
            source_quality=self._calculate_source_quality(source_data),
            temporal_stability=self._calculate_temporal_stability(source_data),
            cross_reference_score=self._calculate_cross_reference_score(source_data)
        )
        
        return PatternEvidence(
            evidence_id=evidence_id,
            timestamp=timestamp,
            pattern_type=pattern_type,
            source_data=source_data,
            temporal_context=temporal_context,
            uncertainty_metrics=uncertainty_metrics
        )

    def _update_evidence_chains(self, evidence: PatternEvidence) -> None:
        """Update evidence chains with new evidence."""
        pattern_type = evidence.pattern_type
        if pattern_type not in self.evidence_chains:
            self.evidence_chains[pattern_type] = []
        self.evidence_chains[pattern_type].append(evidence)

    def _update_temporal_context(self, evidence: PatternEvidence) -> None:
        """Update temporal context tracking."""
        pattern_type = evidence.pattern_type
        if pattern_type not in self.temporal_contexts:
            self.temporal_contexts[pattern_type] = evidence.temporal_context
        else:
            # Update existing context
            current = self.temporal_contexts[pattern_type]
            current.end_time = datetime.now()
            if current.start_time:
                current.duration = (current.end_time - current.start_time).total_seconds()
    def _update_cross_references(self, evidence: PatternEvidence) -> None:
        """Update pattern cross-references."""
        evidence_id = evidence.evidence_id
        pattern_type = evidence.pattern_type
        
        # Initialize tracking sets
        if evidence_id not in self.pattern_references:
            self.pattern_references[evidence_id] = set()
        
        if evidence_id not in self.reference_strengths:
            self.reference_strengths[evidence_id] = {}
            
        # Find related patterns
        for other_type, chain in self.evidence_chains.items():
            if other_type != pattern_type:
                for other_evidence in chain:
                    # Calculate reference strength
                    strength = self._calculate_reference_strength(
                        evidence,
                        other_evidence
                    )
                    
                    if strength > self.evidence_threshold:
                        # Add cross-reference
                        self.pattern_references[evidence_id].add(
                            other_evidence.evidence_id
                        )
                        
                        # Track reference strength
                        self.reference_strengths[evidence_id][
                            other_evidence.evidence_id
                        ] = strength

    def _calculate_reference_strength(
        self,
        evidence1: PatternEvidence,
        evidence2: PatternEvidence
    ) -> float:
        """Calculate strength of pattern reference."""
        # Temporal proximity
        temporal_diff = abs(
            datetime.fromisoformat(evidence1.timestamp) -
            datetime.fromisoformat(evidence2.timestamp)
        ).total_seconds()
        temporal_strength = 1.0 / (1.0 + temporal_diff / 3600)  # Hourly scale
        
        # Structural similarity
        structural_strength = self._calculate_structural_similarity(
            evidence1.source_data["structural"],
            evidence2.source_data["structural"]
        )
        
        # Semantic similarity
        semantic_strength = self._calculate_semantic_similarity(
            evidence1.source_data["semantic"],
            evidence2.source_data["semantic"]
        )
        
        # Weighted combination
        strength = (
            0.4 * temporal_strength +
            0.3 * structural_strength +
            0.3 * semantic_strength
        )
        
        return strength

    def _calculate_structural_similarity(
        self,
        struct1: Dict[str, Any],
        struct2: Dict[str, Any]
    ) -> float:
        """Calculate structural similarity between patterns."""
        # Simple implementation - can be enhanced
        common_keys = set(struct1.keys()) & set(struct2.keys())
        all_keys = set(struct1.keys()) | set(struct2.keys())
        
        return len(common_keys) / max(len(all_keys), 1)

    def _calculate_semantic_similarity(
        self,
        sem1: Dict[str, Any],
        sem2: Dict[str, Any]
    ) -> float:
        """Calculate semantic similarity between patterns."""
        # Simple implementation - can be enhanced
        common_keys = set(sem1.keys()) & set(sem2.keys())
        all_keys = set(sem1.keys()) | set(sem2.keys())
        
        return len(common_keys) / max(len(all_keys), 1)

    def _determine_pattern_type(
        self,
        structural_change: Dict[str, Any],
        semantic_change: Dict[str, Any]
    ) -> str:
        """Determine pattern type from changes."""
        # Simple implementation - can be enhanced
        if "type" in structural_change:
            return structural_change["type"]
        if "type" in semantic_change:
            return semantic_change["type"]
        return "unknown"

    def _calculate_temporal_confidence(self, source_data: Dict[str, Any]) -> float:
        """Calculate confidence in temporal aspects."""
        # Simple implementation - can be enhanced
        return 1.0

    def _calculate_confidence(self, source_data: Dict[str, Any]) -> float:
        """Calculate overall confidence score."""
        # Simple implementation - can be enhanced
        return 1.0

    def _calculate_uncertainty(self, source_data: Dict[str, Any]) -> float:
        """Calculate uncertainty value."""
        # Simple implementation - can be enhanced
        return 0.0

    def _calculate_reliability(self, source_data: Dict[str, Any]) -> float:
        """Calculate reliability score."""
        # Simple implementation - can be enhanced
        return 1.0

    def _calculate_source_quality(self, source_data: Dict[str, Any]) -> float:
        """Calculate source quality score."""
        # Simple implementation - can be enhanced
        return 1.0

    def _calculate_temporal_stability(self, source_data: Dict[str, Any]) -> float:
        """Calculate temporal stability score."""
        # Simple implementation - can be enhanced
        return 1.0

    def _calculate_cross_reference_score(self, source_data: Dict[str, Any]) -> float:
        """Calculate cross-reference quality score."""
        # Simple implementation - can be enhanced
        return 1.0
