"""
Core adaptive identity system for Habitat that maintains concept evolution
while preserving identity through natural knowledge evolution.

Enhanced with pattern-aware evolution, coherence tracking, and relationship
management. Supports versioning while maintaining identity continuity.
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import threading
from uuid import uuid4
import json

from core.evolution.pattern_core import PatternCore, PatternEvidence
from core.coherence.knowledge_coherence import KnowledgeCoherence, CoherenceEvidence
from core.events.event_manager import EventManager
from core.events.event_types import EventType
from core.utils.timestamp_service import TimestampService
from core.utils.version_service import VersionService
from core.utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class IdentityMetrics:
    """Enhanced metrics for identity tracking"""
    stability_score: float = 1.0
    coherence_score: float = 1.0
    evolution_rate: float = 0.0
    pattern_alignment: float = 1.0
    relationship_strength: float = 1.0
    temporal_consistency: float = 1.0
    version_confidence: float = 1.0

@dataclass
class EvolutionContext:
    """Context for tracking concept evolution"""
    start_time: datetime
    current_version: str
    previous_versions: List[str] = field(default_factory=list)
    pattern_history: List[str] = field(default_factory=list)
    coherence_history: List[str] = field(default_factory=list)
    relationship_history: List[str] = field(default_factory=list)
    confidence: float = 1.0

@dataclass
class AdaptiveState:
    """Enhanced state tracking for adaptive concepts"""
    state_id: str
    version: str
    timestamp: str
    patterns: List[PatternEvidence]
    coherence: List[CoherenceEvidence]
    metrics: IdentityMetrics
    context: EvolutionContext
    data: Dict[str, Any]
    
    def __post_init__(self):
        if self.patterns is None:
            self.patterns = []
        if self.coherence is None:
            self.coherence = []

class AdaptiveID:
    """
    Enhanced adaptive identity system that maintains concept evolution
    while preserving identity through natural knowledge evolution.
    """
    def __init__(
        self,
        concept_id: str,
        pattern_core: PatternCore,
        knowledge_coherence: KnowledgeCoherence,
        timestamp_service: Optional[TimestampService] = None,
        event_manager: Optional[EventManager] = None,
        version_service: Optional[VersionService] = None,
        initial_data: Optional[Dict[str, Any]] = None
    ):
        self.concept_id = concept_id
        self.pattern_core = pattern_core
        self.knowledge_coherence = knowledge_coherence
        self.timestamp_service = timestamp_service or TimestampService()
        self.event_manager = event_manager or EventManager()
        self.version_service = version_service or VersionService()
        self._lock = threading.Lock()

        # Enhanced state management
        self.current_state: Optional[AdaptiveState] = None
        self.state_history: List[AdaptiveState] = []
        self.version_map: Dict[str, AdaptiveState] = {}
        
        # Pattern and coherence tracking
        self.active_patterns: Dict[str, PatternEvidence] = {}
        self.active_coherence: Dict[str, CoherenceEvidence] = {}
        
        # Evolution tracking
        self.evolution_metrics: Dict[str, IdentityMetrics] = {}
        self.evolution_context: Optional[EvolutionContext] = None
        
        # Relationship tracking
        self.related_concepts: Dict[str, float] = {}
        self.relationship_history: List[Dict[str, Any]] = []
        
        # Thresholds
        self.evolution_threshold = 0.3
        self.coherence_threshold = 0.3
        self.relationship_threshold = 0.3
        
        # Initialize
        self._initialize_state(initial_data or {})
        
        logger.info(f"Initialized AdaptiveID {concept_id} with pattern awareness")

    def evolve(
        self,
        new_data: Dict[str, Any],
        structural_change: Optional[Dict[str, Any]] = None,
        semantic_change: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evolve the adaptive concept naturally while maintaining identity.
        Enhanced with pattern awareness and coherence tracking.
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
                    logger.warning(f"Evolution validation failed for {self.concept_id}")
                    return {}
                
                # Update state
                self._update_state(new_state)
                
                # Track relationships
                self._update_relationships(new_state)
                
                # Create evolution result
                result = {
                    "concept_id": self.concept_id,
                    "state_id": new_state.state_id,
                    "version": new_state.version,
                    "timestamp": new_state.timestamp,
                    "metrics": vars(new_state.metrics),
                    "patterns": len(new_state.patterns),
                    "coherence": len(new_state.coherence),
                    "relationships": len(self.related_concepts)
                }
                
                # Emit event
                self.event_manager.publish(EventType.CONCEPT_EVOLVED, result)
                
                return result

        except Exception as e:
            logger.error(f"Error evolving concept {self.concept_id}: {str(e)}")
            return {}

    def _initialize_state(self, initial_data: Dict[str, Any]) -> None:
        """Initialize adaptive state."""
        # Create evolution context
        self.evolution_context = EvolutionContext(
            start_time=datetime.now(),
            current_version="1.0",
            previous_versions=[],
            pattern_history=[],
            coherence_history=[],
            relationship_history=[]
        )
        
        # Create initial state
        initial_state = AdaptiveState(
            state_id=str(uuid4()),
            version="1.0",
            timestamp=self.timestamp_service.get_timestamp(),
            patterns=[],
            coherence=[],
            metrics=IdentityMetrics(),
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
    ) -> AdaptiveState:
        """Create new adaptive state."""
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
        context = EvolutionContext(
            start_time=self.evolution_context.start_time,
            current_version=new_version,
            previous_versions=self.evolution_context.previous_versions + [
                self.current_state.version
            ],
            pattern_history=self.evolution_context.pattern_history + [
                pattern_observation["pattern_id"]
            ],
            coherence_history=self.evolution_context.coherence_history + [
                coherence_observation["coherence_id"]
            ],
            relationship_history=self.evolution_context.relationship_history,
            confidence=self._calculate_context_confidence(
                pattern_observation,
                coherence_observation
            )
        )
        
        # Calculate metrics
        metrics = self._calculate_identity_metrics(
            pattern_observation,
            coherence_observation
        )
        
        return AdaptiveState(
            state_id=str(uuid4()),
            version=new_version,
            timestamp=self.timestamp_service.get_timestamp(),
            patterns=current_patterns,
            coherence=current_coherence,
            metrics=metrics,
            context=context,
            data=new_data
        )

    def _validate_evolution(self, new_state: AdaptiveState) -> bool:
        """Validate concept evolution."""
        if not self.current_state:
            return True
            
        # Check metrics
        metrics_valid = all([
            new_state.metrics.stability_score >= self.evolution_threshold,
            new_state.metrics.coherence_score >= self.coherence_threshold,
            new_state.metrics.relationship_strength >= self.relationship_threshold
        ])
        
        # Check patterns
        patterns_valid = any(
            pattern.validate()
            for pattern in new_state.patterns
        )
        
        # Check coherence
        coherence_valid = len(new_state.coherence) > 0
        
        return all([metrics_valid, patterns_valid, coherence_valid])

    def _update_state(self, new_state: AdaptiveState) -> None:
        """Update adaptive state."""
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

    def _update_relationships(self, new_state: AdaptiveState) -> None:
        """Update concept relationships."""
        # Get related patterns
        related_patterns = {}
        for pattern in new_state.patterns:
            for ref_id in pattern.cross_references:
                related_patterns[ref_id] = pattern.stability_score
                
        # Get related coherence
        related_coherence = {}
        for evidence in new_state.coherence:
            for ref_id in evidence.cross_validations:
                related_coherence[ref_id] = evidence.confidence
                
        # Combine relationships
        combined = {}
        all_refs = set(related_patterns.keys()) | set(related_coherence.keys())
        
        for ref_id in all_refs:
            pattern_strength = related_patterns.get(ref_id, 0.0)
            coherence_strength = related_coherence.get(ref_id, 0.0)
            
            combined[ref_id] = (pattern_strength + coherence_strength) / 2
            
        # Update related concepts
        self.related_concepts = {
            ref_id: strength
            for ref_id, strength in combined.items()
            if strength >= self.relationship_threshold
        }
        
        # Track relationship history
        self.relationship_history.append({
            "state_id": new_state.state_id,
            "version": new_state.version,
            "timestamp": new_state.timestamp,
            "relationships": self.related_concepts.copy()
        })

    def _calculate_identity_metrics(
        self,
        pattern_observation: Dict[str, Any],
        coherence_observation: Dict[str, Any]
    ) -> IdentityMetrics:
        """Calculate identity metrics."""
        pattern_evidence = pattern_observation["evidence"]
        coherence_evidence = coherence_observation["evidence"]
        
        return IdentityMetrics(
            stability_score=pattern_evidence.stability_score,
            coherence_score=coherence_evidence.coherence_metrics.overall_coherence,
            evolution_rate=pattern_evidence.emergence_rate,
            pattern_alignment=coherence_evidence.coherence_metrics.pattern_alignment,
            relationship_strength=self._calculate_relationship_strength(
                pattern_evidence,
                coherence_evidence
            ),
            temporal_consistency=coherence_evidence.coherence_metrics.temporal_consistency,
            version_confidence=self._calculate_version_confidence(
                pattern_evidence,
                coherence_evidence
            )
        )

    def _calculate_context_confidence(
        self,
        pattern_observation: Dict[str, Any],
        coherence_observation: Dict[str, Any]
    ) -> float:
        """Calculate evolution context confidence."""
        pattern_confidence = pattern_observation["evidence"].uncertainty_metrics.confidence_score
        coherence_confidence = coherence_observation["evidence"].confidence
        
        return (pattern_confidence + coherence_confidence) / 2

    def _calculate_relationship_strength(
        self,
        pattern_evidence: PatternEvidence,
        coherence_evidence: CoherenceEvidence
    ) -> float:
        """Calculate relationship strength."""
        pattern_refs = len(pattern_evidence.cross_references)
        coherence_refs = len(coherence_evidence.cross_validations)
        
        if not pattern_refs and not coherence_refs:
            return 1.0
            
        pattern_strength = pattern_evidence.stability_score
        coherence_strength = coherence_evidence.confidence
        
        return (pattern_strength + coherence_strength) / 2

    def _calculate_version_confidence(
        self,
        pattern_evidence: PatternEvidence,
        coherence_evidence: CoherenceEvidence
    ) -> float:
        """Calculate version confidence."""
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
