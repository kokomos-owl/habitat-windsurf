"""
Core relationship modeling system for Habitat that maintains natural
evolution of relationships between adaptive concepts.

Enhanced with pattern-aware relationship tracking, coherence validation,
and identity-preserving evolution. Supports temporal versioning and
uncertainty propagation.
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import threading
from uuid import uuid4

from core.evolution.pattern_core import PatternCore, PatternEvidence
from core.coherence.knowledge_coherence import KnowledgeCoherence, CoherenceEvidence
from core.adaptive.adaptive_id import AdaptiveID, AdaptiveState, IdentityMetrics
from core.events.event_manager import EventManager
from core.events.event_types import EventType
from core.utils.timestamp_service import TimestampService
from core.utils.version_service import VersionService
from core.utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class RelationshipMetrics:
    """Enhanced metrics for relationship tracking"""
    strength: float = 1.0
    confidence: float = 1.0
    stability: float = 1.0
    coherence: float = 1.0
    temporal_consistency: float = 1.0
    pattern_support: float = 1.0
    identity_alignment: float = 1.0
    evolution_rate: float = 0.0

@dataclass
class RelationshipContext:
    """Context for tracking relationship evolution"""
    start_time: datetime
    current_version: str
    previous_versions: List[str] = field(default_factory=list)
    pattern_evidence: List[str] = field(default_factory=list)
    coherence_evidence: List[str] = field(default_factory=list)
    identity_states: List[str] = field(default_factory=list)
    confidence: float = 1.0

@dataclass
class RelationshipState:
    """Enhanced state tracking for relationships"""
    state_id: str
    version: str
    timestamp: str
    source_id: str
    target_id: str
    relationship_type: str
    patterns: List[PatternEvidence]
    coherence: List[CoherenceEvidence]
    source_state: AdaptiveState
    target_state: AdaptiveState
    metrics: RelationshipMetrics
    context: RelationshipContext
    data: Dict[str, Any]
    
    def __post_init__(self):
        if self.patterns is None:
            self.patterns = []
        if self.coherence is None:
            self.coherence = []

class RelationshipModel:
    """
    Enhanced relationship modeling system that maintains natural evolution
    of relationships between adaptive concepts with pattern awareness.
    """
    def __init__(
        self,
        source_concept: AdaptiveID,
        target_concept: AdaptiveID,
        relationship_type: str,
        pattern_core: PatternCore,
        knowledge_coherence: KnowledgeCoherence,
        timestamp_service: Optional[TimestampService] = None,
        event_manager: Optional[EventManager] = None,
        version_service: Optional[VersionService] = None,
        initial_data: Optional[Dict[str, Any]] = None
    ):
        self.source_concept = source_concept
        self.target_concept = target_concept
        self.relationship_type = relationship_type
        self.pattern_core = pattern_core
        self.knowledge_coherence = knowledge_coherence
        self.timestamp_service = timestamp_service or TimestampService()
        self.event_manager = event_manager or EventManager()
        self.version_service = version_service or VersionService()
        self._lock = threading.Lock()

        # Enhanced state management
        self.current_state: Optional[RelationshipState] = None
        self.state_history: List[RelationshipState] = []
        self.version_map: Dict[str, RelationshipState] = {}
        
        # Pattern and coherence tracking
        self.active_patterns: Dict[str, PatternEvidence] = {}
        self.active_coherence: Dict[str, CoherenceEvidence] = {}
        
        # Evolution tracking
        self.evolution_metrics: Dict[str, RelationshipMetrics] = {}
        self.evolution_context: Optional[RelationshipContext] = None
        
        # Uncertainty propagation
        self.uncertainty_chain: List[Dict[str, float]] = []
        self.confidence_history: List[Dict[str, float]] = []
        
        # Thresholds
        self.evolution_threshold = 0.3
        self.coherence_threshold = 0.3
        self.pattern_threshold = 0.3
        
        # Initialize
        self._initialize_state(initial_data or {})
        
        logger.info(
            f"Initialized RelationshipModel between {source_concept.concept_id} "
            f"and {target_concept.concept_id} of type {relationship_type}"
        )

    def evolve(
        self,
        new_data: Dict[str, Any],
        structural_change: Optional[Dict[str, Any]] = None,
        semantic_change: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evolve the relationship naturally while maintaining coherence.
        Enhanced with pattern awareness and identity preservation.
        """
        try:
            with self._lock:
                # Observe patterns first
                pattern_observation = self.pattern_core.observe_evolution(
                    structural_change or {},
                    semantic_change or {},
                    new_data
                )
                
                if not pattern_observation:
                    return {}
                    
                # Observe coherence
                coherence_observation = self.knowledge_coherence.observe_evolution(
                    structural_change or {},
                    semantic_change or {},
                    new_data
                )
                
                if not coherence_observation:
                    return {}
                
                # Create new state
                new_state = self._create_new_state(
                    new_data,
                    pattern_observation,
                    coherence_observation
                )
                
                # Validate evolution
                if not self._validate_evolution(new_state):
                    logger.warning(
                        f"Evolution validation failed for relationship "
                        f"{self.relationship_type}"
                    )
                    return {}
                
                # Update state
                self._update_state(new_state)
                
                # Propagate uncertainty
                self._propagate_uncertainty(new_state)
                
                # Create evolution result
                result = {
                    "relationship_id": new_state.state_id,
                    "source_id": self.source_concept.concept_id,
                    "target_id": self.target_concept.concept_id,
                    "type": self.relationship_type,
                    "version": new_state.version,
                    "timestamp": new_state.timestamp,
                    "metrics": vars(new_state.metrics),
                    "patterns": len(new_state.patterns),
                    "coherence": len(new_state.coherence),
                    "confidence": new_state.context.confidence
                }
                
                # Emit event
                self.event_manager.publish(EventType.RELATIONSHIP_EVOLVED, result)
                
                return result

        except Exception as e:
            logger.error(
                f"Error evolving relationship {self.relationship_type}: {str(e)}"
            )
            return {}

    def _initialize_state(self, initial_data: Dict[str, Any]) -> None:
        """Initialize relationship state."""
        # Create evolution context
        self.evolution_context = RelationshipContext(
            start_time=datetime.now(),
            current_version="1.0",
            previous_versions=[],
            pattern_evidence=[],
            coherence_evidence=[],
            identity_states=[]
        )
        
        # Create initial state
        initial_state = RelationshipState(
            state_id=str(uuid4()),
            version="1.0",
            timestamp=self.timestamp_service.get_timestamp(),
            source_id=self.source_concept.concept_id,
            target_id=self.target_concept.concept_id,
            relationship_type=self.relationship_type,
            patterns=[],
            coherence=[],
            source_state=self.source_concept.current_state,
            target_state=self.target_concept.current_state,
            metrics=RelationshipMetrics(),
            context=self.evolution_context,
            data=initial_data
        )
        
        # Set as current
        self.current_state = initial_state
        self.state_history.append(initial_state)
        self.version_map["1.0"] = initial_state

    def _create_new_state(
        self,
        new_data: Dict[str, Any],
        pattern_observation: Dict[str, Any],
        coherence_observation: Dict[str, Any]
    ) -> RelationshipState:
        """Create new relationship state."""
        # Update patterns
        current_patterns = list(self.active_patterns.values())
        current_patterns.append(pattern_observation["evidence"])
        
        # Update coherence
        current_coherence = list(self.active_coherence.values())
        current_coherence.append(coherence_observation["evidence"])
        
        # Calculate new version
        new_version = self.version_service.get_next_version(
            self.current_state.version
        )
        
        # Update context
        context = RelationshipContext(
            start_time=self.evolution_context.start_time,
            current_version=new_version,
            previous_versions=self.evolution_context.previous_versions + [
                self.current_state.version
            ],
            pattern_evidence=self.evolution_context.pattern_evidence + [
                pattern_observation["pattern_id"]
            ],
            coherence_evidence=self.evolution_context.coherence_evidence + [
                coherence_observation["coherence_id"]
            ],
            identity_states=self.evolution_context.identity_states + [
                self.source_concept.current_state.state_id,
                self.target_concept.current_state.state_id
            ],
            confidence=self._calculate_context_confidence(
                pattern_observation,
                coherence_observation
            )
        )
        
        # Calculate metrics
        metrics = self._calculate_relationship_metrics(
            pattern_observation,
            coherence_observation
        )
        
        return RelationshipState(
            state_id=str(uuid4()),
            version=new_version,
            timestamp=self.timestamp_service.get_timestamp(),
            source_id=self.source_concept.concept_id,
            target_id=self.target_concept.concept_id,
            relationship_type=self.relationship_type,
            patterns=current_patterns,
            coherence=current_coherence,
            source_state=self.source_concept.current_state,
            target_state=self.target_concept.current_state,
            metrics=metrics,
            context=context,
            data=new_data
        )

    def _validate_evolution(self, new_state: RelationshipState) -> bool:
        """Validate relationship evolution."""
        if not self.current_state:
            return True
            
        # Check metrics
        metrics_valid = all([
            new_state.metrics.strength >= self.evolution_threshold,
            new_state.metrics.coherence >= self.coherence_threshold,
            new_state.metrics.pattern_support >= self.pattern_threshold
        ])
        
        # Check patterns
        patterns_valid = any(
            pattern.validate()
            for pattern in new_state.patterns
        )
        
        # Check coherence
        coherence_valid = len(new_state.coherence) > 0
        
        # Check identity alignment
        identity_valid = self._validate_identity_alignment(new_state)
        
        return all([
            metrics_valid,
            patterns_valid,
            coherence_valid,
            identity_valid
        ])

    def _update_state(self, new_state: RelationshipState) -> None:
        """Update relationship state."""
        # Update current state
        self.current_state = new_state
        
        # Update history
        self.state_history.append(new_state)
        self.version_map[new_state.version] = new_state
        
        # Update evolution context
        self.evolution_context = new_state.context
        
        # Update active patterns and coherence
        self.active_patterns = {
            pattern.evidence_id: pattern
            for pattern in new_state.patterns
        }
        self.active_coherence = {
            evidence.evidence_id: evidence
            for evidence in new_state.coherence
        }
        
        # Update evolution metrics
        self.evolution_metrics[new_state.version] = new_state.metrics

    def _propagate_uncertainty(self, new_state: RelationshipState) -> None:
        """Propagate uncertainty through relationship chain."""
        # Calculate uncertainty from patterns
        pattern_uncertainty = 1.0 - sum(
            pattern.uncertainty_metrics.confidence_score
            for pattern in new_state.patterns
        ) / max(len(new_state.patterns), 1)
        
        # Calculate uncertainty from coherence
        coherence_uncertainty = 1.0 - sum(
            evidence.confidence
            for evidence in new_state.coherence
        ) / max(len(new_state.coherence), 1)
        
        # Calculate uncertainty from identity
        identity_uncertainty = 1.0 - (
            new_state.source_state.metrics.stability_score *
            new_state.target_state.metrics.stability_score
        )
        
        # Combined uncertainty
        uncertainty = {
            "state_id": new_state.state_id,
            "version": new_state.version,
            "pattern_uncertainty": pattern_uncertainty,
            "coherence_uncertainty": coherence_uncertainty,
            "identity_uncertainty": identity_uncertainty,
            "total_uncertainty": (
                pattern_uncertainty +
                coherence_uncertainty +
                identity_uncertainty
            ) / 3
        }
        
        # Update chain
        self.uncertainty_chain.append(uncertainty)
        
        # Track confidence
        self.confidence_history.append({
            "state_id": new_state.state_id,
            "version": new_state.version,
            "confidence": new_state.context.confidence,
            "timestamp": new_state.timestamp
        })

    def _calculate_relationship_metrics(
        self,
        pattern_observation: Dict[str, Any],
        coherence_observation: Dict[str, Any]
    ) -> RelationshipMetrics:
        """Calculate relationship metrics."""
        pattern_evidence = pattern_observation["evidence"]
        coherence_evidence = coherence_observation["evidence"]
        
        # Get source and target metrics
        source_metrics = self.source_concept.current_state.metrics
        target_metrics = self.target_concept.current_state.metrics
        
        return RelationshipMetrics(
            strength=self._calculate_relationship_strength(
                pattern_evidence,
                coherence_evidence
            ),
            confidence=self._calculate_relationship_confidence(
                pattern_evidence,
                coherence_evidence
            ),
            stability=min(
                pattern_evidence.stability_score,
                coherence_evidence.coherence_metrics.overall_coherence
            ),
            coherence=coherence_evidence.coherence_metrics.overall_coherence,
            temporal_consistency=coherence_evidence.coherence_metrics.temporal_consistency,
            pattern_support=pattern_evidence.stability_score,
            identity_alignment=self._calculate_identity_alignment(
                source_metrics,
                target_metrics
            ),
            evolution_rate=pattern_evidence.emergence_rate
        )

    def _calculate_context_confidence(
        self,
        pattern_observation: Dict[str, Any],
        coherence_observation: Dict[str, Any]
    ) -> float:
        """Calculate relationship context confidence."""
        pattern_confidence = pattern_observation["evidence"].uncertainty_metrics.confidence_score
        coherence_confidence = coherence_observation["evidence"].confidence
        
        source_confidence = self.source_concept.current_state.context.confidence
        target_confidence = self.target_concept.current_state.context.confidence
        
        return (
            pattern_confidence +
            coherence_confidence +
            source_confidence +
            target_confidence
        ) / 4

    def _calculate_relationship_strength(
        self,
        pattern_evidence: PatternEvidence,
        coherence_evidence: CoherenceEvidence
    ) -> float:
        """Calculate relationship strength."""
        pattern_strength = pattern_evidence.stability_score
        coherence_strength = coherence_evidence.confidence
        
        source_strength = self.source_concept.current_state.metrics.stability_score
        target_strength = self.target_concept.current_state.metrics.stability_score
        
        return (
            0.3 * pattern_strength +
            0.3 * coherence_strength +
            0.2 * source_strength +
            0.2 * target_strength
        )

    def _calculate_relationship_confidence(
        self,
        pattern_evidence: PatternEvidence,
        coherence_evidence: CoherenceEvidence
    ) -> float:
        """Calculate relationship confidence."""
        pattern_confidence = pattern_evidence.uncertainty_metrics.confidence_score
        coherence_confidence = coherence_evidence.confidence
        
        temporal_factor = pattern_evidence.temporal_context.confidence
        stability_factor = pattern_evidence.stability_score
        
        return (
            0.3 * pattern_confidence +
            0.3 * coherence_confidence +
            0.2 * temporal_factor +
            0.2 * stability_factor
        )

    def _calculate_identity_alignment(
        self,
        source_metrics: IdentityMetrics,
        target_metrics: IdentityMetrics
    ) -> float:
        """Calculate identity alignment between concepts."""
        alignments = [
            source_metrics.stability_score * target_metrics.stability_score,
            source_metrics.coherence_score * target_metrics.coherence_score,
            source_metrics.pattern_alignment * target_metrics.pattern_alignment,
            source_metrics.temporal_consistency * target_metrics.temporal_consistency
        ]
        
        return sum(alignments) / len(alignments)

    def _validate_identity_alignment(self, new_state: RelationshipState) -> bool:
        """Validate identity alignment in relationship."""
        source_valid = self.source_concept.current_state.metrics.stability_score >= self.evolution_threshold
        target_valid = self.target_concept.current_state.metrics.stability_score >= self.evolution_threshold
        
        alignment = self._calculate_identity_alignment(
            self.source_concept.current_state.metrics,
            self.target_concept.current_state.metrics
        )
        
        alignment_valid = alignment >= self.evolution_threshold
        
        return all([source_valid, target_valid, alignment_valid])
