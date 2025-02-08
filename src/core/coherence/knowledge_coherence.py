"""
Light coherence observation service that works with PatternCore to track
natural knowledge evolution in Habitat.

Observes but doesn't enforce coherence between structure and meaning,
allowing natural pattern emergence while maintaining clean data structures.
Enhanced with pattern-aware coherence tracking and feedback loops.
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
from uuid import uuid4

from core.evolution.pattern_core import PatternCore, PatternEvidence
from core.events.event_manager import EventManager
from core.events.event_types import EventType
from core.utils.timestamp_service import TimestampService
from core.utils.version_service import VersionService
from core.utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class CoherenceMetrics:
    """Enhanced metrics for coherence assessment"""
    structure_meaning_alignment: float = 0.0
    pattern_alignment: float = 0.0
    temporal_consistency: float = 0.0
    domain_consistency: float = 0.0
    cross_pattern_coherence: float = 0.0
    feedback_incorporation: float = 0.0
    overall_coherence: float = 0.0

@dataclass
class CoherenceEvidence:
    """Evidence structure for coherence tracking"""
    evidence_id: str
    timestamp: str
    source_pattern: PatternEvidence
    coherence_metrics: CoherenceMetrics
    feedback_history: List[Dict[str, Any]]
    cross_validations: List[str]
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.feedback_history is None:
            self.feedback_history = []
        if self.cross_validations is None:
            self.cross_validations = []

class KnowledgeCoherence:
    """
    Light coherence observation service that works with PatternCore.
    Enhanced with pattern-aware coherence tracking and feedback loops.
    """
    def __init__(
        self,
        pattern_core: PatternCore,
        timestamp_service: Optional[TimestampService] = None,
        event_manager: Optional[EventManager] = None,
        version_service: Optional[VersionService] = None
    ):
        self.pattern_core = pattern_core
        self.timestamp_service = timestamp_service or TimestampService()
        self.event_manager = event_manager or EventManager()
        self.version_service = version_service or VersionService()
        self._lock = threading.Lock()

        # Enhanced coherence tracking
        self.coherence_history: List[CoherenceEvidence] = []
        self.latest_coherence: Dict[str, CoherenceEvidence] = {}
        
        # Pattern-aware tracking
        self.pattern_coherence: Dict[str, List[CoherenceEvidence]] = {}
        self.cross_pattern_coherence: Dict[str, Dict[str, float]] = {}
        
        # Feedback loops
        self.feedback_history: List[Dict[str, Any]] = []
        self.feedback_incorporation: Dict[str, float] = {}
        
        # Thresholds
        self.coherence_threshold = 0.3
        self.feedback_threshold = 0.3
        self.cross_validation_threshold = 0.3

        logger.info("Initialized KnowledgeCoherence with pattern awareness")

    def observe_evolution(
        self,
        structural_change: Dict[str, Any],
        semantic_change: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Observe coherence in knowledge evolution without enforcing it.
        Enhanced with pattern awareness and feedback loops.
        """
        try:
            with self._lock:
                # Let PatternCore observe patterns first
                pattern_observation = self.pattern_core.observe_evolution(
                    structural_change,
                    semantic_change,
                    evidence
                )
                
                if not pattern_observation:
                    return {}
                
                # Create coherence evidence
                coherence_evidence = self._create_coherence_evidence(
                    pattern_observation,
                    structural_change,
                    semantic_change
                )
                
                # Track coherence
                self._track_coherence(coherence_evidence)
                
                # Update cross-pattern coherence
                self._update_cross_pattern_coherence(coherence_evidence)
                
                # Process feedback
                self._process_feedback(coherence_evidence)
                
                # Create observation result
                observation = {
                    "coherence_id": coherence_evidence.evidence_id,
                    "pattern_id": pattern_observation["pattern_id"],
                    "coherence_metrics": vars(coherence_evidence.coherence_metrics),
                    "timestamp": self.timestamp_service.get_timestamp(),
                    "confidence": coherence_evidence.confidence,
                    "cross_validations": coherence_evidence.cross_validations
                }
                
                # Track history
                self.coherence_history.append(coherence_evidence)
                self.latest_coherence[observation["coherence_id"]] = coherence_evidence

                # Emit event
                self.event_manager.publish(EventType.COHERENCE_OBSERVED, observation)

                return observation

        except Exception as e:
            logger.error(f"Error observing coherence: {str(e)}")
            return {}

    def _create_coherence_evidence(
        self,
        pattern_observation: Dict[str, Any],
        structural_change: Dict[str, Any],
        semantic_change: Dict[str, Any]
    ) -> CoherenceEvidence:
        """Create coherence evidence from pattern observation."""
        # Calculate coherence metrics
        metrics = CoherenceMetrics(
            structure_meaning_alignment=self._calculate_structure_meaning_alignment(
                structural_change,
                semantic_change
            ),
            pattern_alignment=self._calculate_pattern_alignment(
                pattern_observation
            ),
            temporal_consistency=self._calculate_temporal_consistency(
                pattern_observation
            ),
            domain_consistency=self._calculate_domain_consistency(
                pattern_observation
            ),
            cross_pattern_coherence=self._calculate_cross_pattern_coherence(
                pattern_observation
            ),
            feedback_incorporation=self._calculate_feedback_incorporation(
                pattern_observation
            )
        )
        
        # Calculate overall coherence
        metrics.overall_coherence = self._calculate_overall_coherence(metrics)
        
        return CoherenceEvidence(
            evidence_id=str(uuid4()),
            timestamp=self.timestamp_service.get_timestamp(),
            source_pattern=pattern_observation["evidence"],
            coherence_metrics=metrics,
            feedback_history=[],
            cross_validations=[],
            confidence=self._calculate_confidence(metrics)
        )

    def _track_coherence(self, evidence: CoherenceEvidence) -> None:
        """Track coherence evidence."""
        pattern_id = evidence.source_pattern.evidence_id
        
        # Track by pattern
        if pattern_id not in self.pattern_coherence:
            self.pattern_coherence[pattern_id] = []
        self.pattern_coherence[pattern_id].append(evidence)
        
        # Update cross-validations
        self._update_cross_validations(evidence)

    def _update_cross_pattern_coherence(self, evidence: CoherenceEvidence) -> None:
        """Update cross-pattern coherence tracking."""
        pattern_id = evidence.source_pattern.evidence_id
        
        if pattern_id not in self.cross_pattern_coherence:
            self.cross_pattern_coherence[pattern_id] = {}
            
        # Calculate coherence with other patterns
        for other_id, other_evidence in self.latest_coherence.items():
            if other_id != evidence.evidence_id:
                coherence = self._calculate_pattern_pair_coherence(
                    evidence,
                    other_evidence
                )
                
                if coherence > self.coherence_threshold:
                    self.cross_pattern_coherence[pattern_id][other_id] = coherence

    def _process_feedback(self, evidence: CoherenceEvidence) -> None:
        """Process and incorporate feedback."""
        pattern_id = evidence.source_pattern.evidence_id
        
        # Get historical feedback
        historical_feedback = self.feedback_history
        
        # Calculate feedback incorporation
        incorporation = self._calculate_feedback_incorporation_rate(
            evidence,
            historical_feedback
        )
        
        # Update tracking
        self.feedback_incorporation[pattern_id] = incorporation
        
        # Add to feedback history
        self.feedback_history.append({
            "evidence_id": evidence.evidence_id,
            "pattern_id": pattern_id,
            "incorporation_rate": incorporation,
            "timestamp": self.timestamp_service.get_timestamp()
        })

    def _calculate_structure_meaning_alignment(
        self,
        structural_change: Dict[str, Any],
        semantic_change: Dict[str, Any]
    ) -> float:
        """Calculate alignment between structure and meaning."""
        # Simple implementation - can be enhanced
        common_keys = set(structural_change.keys()) & set(semantic_change.keys())
        all_keys = set(structural_change.keys()) | set(semantic_change.keys())
        
        return len(common_keys) / max(len(all_keys), 1)

    def _calculate_pattern_alignment(
        self,
        pattern_observation: Dict[str, Any]
    ) -> float:
        """Calculate pattern alignment score."""
        evidence = pattern_observation["evidence"]
        return evidence.stability_score

    def _calculate_temporal_consistency(
        self,
        pattern_observation: Dict[str, Any]
    ) -> float:
        """Calculate temporal consistency score."""
        evidence = pattern_observation["evidence"]
        return evidence.temporal_context.confidence

    def _calculate_domain_consistency(
        self,
        pattern_observation: Dict[str, Any]
    ) -> float:
        """Calculate domain consistency score."""
        evidence = pattern_observation["evidence"]
        return evidence.uncertainty_metrics.reliability_score

    def _calculate_cross_pattern_coherence(
        self,
        pattern_observation: Dict[str, Any]
    ) -> float:
        """Calculate cross-pattern coherence score."""
        pattern_id = pattern_observation["pattern_id"]
        
        if pattern_id not in self.cross_pattern_coherence:
            return 1.0
            
        coherence_scores = self.cross_pattern_coherence[pattern_id].values()
        if not coherence_scores:
            return 1.0
            
        return sum(coherence_scores) / len(coherence_scores)

    def _calculate_feedback_incorporation(
        self,
        pattern_observation: Dict[str, Any]
    ) -> float:
        """Calculate feedback incorporation score."""
        pattern_id = pattern_observation["pattern_id"]
        return self.feedback_incorporation.get(pattern_id, 1.0)

    def _calculate_overall_coherence(self, metrics: CoherenceMetrics) -> float:
        """Calculate overall coherence score."""
        weights = {
            "structure_meaning_alignment": 0.25,
            "pattern_alignment": 0.2,
            "temporal_consistency": 0.15,
            "domain_consistency": 0.15,
            "cross_pattern_coherence": 0.15,
            "feedback_incorporation": 0.1
        }
        
        overall = sum(
            getattr(metrics, attr) * weight
            for attr, weight in weights.items()
        )
        
        return overall

    def _calculate_confidence(self, metrics: CoherenceMetrics) -> float:
        """Calculate confidence in coherence assessment."""
        # Simple implementation - can be enhanced
        return metrics.overall_coherence

    def _calculate_pattern_pair_coherence(
        self,
        evidence1: CoherenceEvidence,
        evidence2: CoherenceEvidence
    ) -> float:
        """Calculate coherence between two patterns."""
        # Temporal proximity
        temporal_diff = abs(
            datetime.fromisoformat(evidence1.timestamp) -
            datetime.fromisoformat(evidence2.timestamp)
        ).total_seconds()
        temporal_coherence = 1.0 / (1.0 + temporal_diff / 3600)
        
        # Metric similarity
        metric_coherence = self._calculate_metric_similarity(
            evidence1.coherence_metrics,
            evidence2.coherence_metrics
        )
        
        # Pattern similarity from source patterns
        pattern_coherence = self._calculate_pattern_similarity(
            evidence1.source_pattern,
            evidence2.source_pattern
        )
        
        # Weighted combination
        coherence = (
            0.3 * temporal_coherence +
            0.3 * metric_coherence +
            0.4 * pattern_coherence
        )
        
        return coherence

    def _calculate_metric_similarity(
        self,
        metrics1: CoherenceMetrics,
        metrics2: CoherenceMetrics
    ) -> float:
        """Calculate similarity between coherence metrics."""
        # Simple implementation - can be enhanced
        m1 = vars(metrics1)
        m2 = vars(metrics2)
        
        differences = [
            abs(m1[key] - m2[key])
            for key in m1
            if key in m2
        ]
        
        return 1.0 - (sum(differences) / len(differences))

    def _calculate_pattern_similarity(
        self,
        pattern1: PatternEvidence,
        pattern2: PatternEvidence
    ) -> float:
        """Calculate similarity between patterns."""
        # Simple implementation - can be enhanced
        return pattern1.stability_score * pattern2.stability_score

    def _calculate_feedback_incorporation_rate(
        self,
        evidence: CoherenceEvidence,
        historical_feedback: List[Dict[str, Any]]
    ) -> float:
        """Calculate rate of feedback incorporation."""
        if not historical_feedback:
            return 1.0
            
        relevant_feedback = [
            f for f in historical_feedback
            if f["pattern_id"] == evidence.source_pattern.evidence_id
        ]
        
        if not relevant_feedback:
            return 1.0
            
        incorporation_rates = [f["incorporation_rate"] for f in relevant_feedback]
        return sum(incorporation_rates) / len(incorporation_rates)

    def _update_cross_validations(self, evidence: CoherenceEvidence) -> None:
        """Update cross-validations for coherence evidence."""
        pattern_id = evidence.source_pattern.evidence_id
        
        # Get related patterns from cross-pattern coherence
        related_patterns = self.cross_pattern_coherence.get(pattern_id, {})
        
        # Add cross-validations for highly coherent patterns
        for other_id, coherence in related_patterns.items():
            if coherence > self.cross_validation_threshold:
                evidence.cross_validations.append(other_id)
