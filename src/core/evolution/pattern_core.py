"""
Core pattern service for Habitat knowledge evolution.

Natural observation service that works with knowledge coherence to track
pattern emergence without enforcing structure. Maintains alignment with adaptive_core
evolution tracking while providing enhanced evidence chains and temporal mapping.
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
import logging
from uuid import uuid4

from core.interfaces.base_states import BaseProjectState
from core.events.event_manager import EventManager
from core.events.event_types import EventType
from core.utils.timestamp_service import TimestampService
from core.utils.version_service import VersionService
from core.utils.logging_config import get_logger

@dataclass
class TemporalContext:
    """Temporal context for pattern tracking."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    sequence_order: Optional[int] = None
    confidence: float = 1.0

@dataclass
class UncertaintyMetrics:
    """Uncertainty metrics for pattern evidence."""
    confidence_score: float = 1.0
    uncertainty_value: float = 0.0
    reliability_score: float = 1.0
    source_quality: float = 1.0
    temporal_stability: float = 1.0
    cross_reference_score: float = 1.0

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
        self._lock = threading.Lock()
        
        # Enhanced pattern observation storage
        self.evidence_chains: Dict[str, List[PatternEvidence]] = {}
        self.temporal_maps: Dict[str, Dict[str, Any]] = {}
        self.pattern_versions: Dict[str, List[Dict[str, Any]]] = {}
        
        # Pattern evolution tracking
        self.pattern_metrics: Dict[str, UncertaintyMetrics] = {}
        self.temporal_contexts: Dict[str, TemporalContext] = {}
        self.stability_scores: Dict[str, float] = {}
        self.emergence_rates: Dict[str, float] = {}
        
        # Cross-reference tracking
        self.pattern_references: Dict[str, Set[str]] = {}
        self.reference_strengths: Dict[str, Dict[str, float]] = {}
        
        # Observation history
        self.pattern_history: List[Dict[str, Any]] = []
        self.latest_observations: Dict[str, Any] = {}
        
        # Thresholds for light validation
        self.evidence_threshold = 0.3
        self.temporal_threshold = 0.3
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

    def _calculate_stability(self, evidence: PatternEvidence) -> float:
        """Calculate pattern stability score."""
        pattern_type = evidence.pattern_type
        
        # Get historical evidence
        historical = self.evidence_chains.get(pattern_type, [])
        if not historical:
            return 1.0
            
        # Calculate temporal stability
        temporal_stability = evidence.uncertainty_metrics.temporal_stability
        
        # Calculate cross-reference stability
        cross_ref_stability = len(evidence.cross_references) / max(
            len(self.pattern_references), 1
        )
        
        # Calculate evidence chain stability
        chain_stability = len(historical) / max(
            sum(len(chain) for chain in self.evidence_chains.values()), 1
        )
        
        # Weighted combination
        stability = (
            0.4 * temporal_stability +
            0.3 * cross_ref_stability +
            0.3 * chain_stability
        )
        
        # Update stability tracking
        self.stability_scores[pattern_type] = stability
        
        return stability

    def _calculate_emergence_rate(self, evidence: PatternEvidence) -> float:
        """Calculate pattern emergence rate."""
        pattern_type = evidence.pattern_type
        
        # Get historical evidence
        historical = self.evidence_chains.get(pattern_type, [])
        if not historical:
            return 0.0
            
        # Calculate rate of evidence accumulation
        total_patterns = sum(len(chain) for chain in self.evidence_chains.values())
        pattern_count = len(historical)
        
        emergence = pattern_count / max(total_patterns, 1)
        
        # Update emergence tracking
        self.emergence_rates[pattern_type] = emergence
        
        return emergence

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
