"""
Natural temporal evolution service for pattern dynamics.

Allows temporal patterns to emerge naturally while maintaining
coherence with pattern evolution and state space dynamics.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from threading import RLock

from src.core.events.event_manager import EventManager
from src.core.utils.timestamp_service import TimestampService
from src.core.utils.version_service import VersionService
from src.core.utils.logging_config import get_logger
from src.core.evolution.pattern_core import PatternEvidence, TemporalContext
from src.core.types import DensityMetrics

logger = get_logger(__name__)

@dataclass
class TemporalEvidence:
    """Natural temporal evidence structure."""
    evidence_id: str
    timestamp: datetime
    pattern_evidence: PatternEvidence
    temporal_context: TemporalContext
    stability_score: float = 1.0
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate temporal evidence."""
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if self.temporal_context is None:
            self.temporal_context = TemporalContext()

class TemporalCore:
    """Natural temporal evolution service."""
    
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
        
        # Temporal evidence tracking
        self.temporal_evidence: Dict[str, List[TemporalEvidence]] = {}
        self.pattern_timelines: Dict[str, List[datetime]] = {}
        self.stability_scores: Dict[str, float] = {}
        
        # Evolution thresholds
        self.stability_threshold = 0.3
        self.confidence_threshold = 0.3
        
        logger.info("Initialized TemporalCore with natural evolution")
    
    async def observe_temporal_evolution(
        self,
        pattern_evidence: PatternEvidence,
        temporal_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Observe temporal evolution naturally.
        
        Args:
            pattern_evidence: Evidence of pattern observation
            temporal_context: Optional temporal context
            
        Returns:
            Dict containing temporal evolution metrics
        """
        try:
            with self._lock:
                # Create temporal evidence
                evidence = self._create_temporal_evidence(
                    pattern_evidence,
                    temporal_context
                )
                
                # Track evidence
                self._track_temporal_evidence(evidence)
                
                # Update pattern timeline
                self._update_pattern_timeline(
                    evidence.pattern_evidence.evidence_id,
                    evidence.timestamp
                )
                
                # Calculate stability
                stability = self._calculate_stability(evidence)
                self.stability_scores[evidence.evidence_id] = stability
                
                # Create observation result
                result = {
                    "temporal_id": evidence.evidence_id,
                    "pattern_id": evidence.pattern_evidence.evidence_id,
                    "timestamp": evidence.timestamp.isoformat(),
                    "stability": stability,
                    "confidence": evidence.confidence
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Error in temporal evolution: {str(e)}")
            return {}
    
    def _create_temporal_evidence(
        self,
        pattern_evidence: PatternEvidence,
        temporal_context: Optional[Dict[str, Any]] = None
    ) -> TemporalEvidence:
        """Create temporal evidence from pattern evidence."""
        context = TemporalContext(
            start_time=self.timestamp_service.get_timestamp(),
            confidence=self._calculate_temporal_confidence(pattern_evidence)
        )
        
        if temporal_context:
            context.end_time = temporal_context.get("end_time")
            context.duration = temporal_context.get("duration")
            context.sequence_order = temporal_context.get("sequence_order")
        
        # Calculate confidence
        confidence = self._calculate_temporal_confidence(pattern_evidence)
        
        return TemporalEvidence(
            evidence_id=f"temporal_{pattern_evidence.evidence_id}",
            timestamp=self.timestamp_service.get_timestamp(),
            pattern_evidence=pattern_evidence,
            temporal_context=context,
            confidence=confidence
        )
    
    def _track_temporal_evidence(self, evidence: TemporalEvidence) -> None:
        """Track temporal evidence naturally."""
        pattern_id = evidence.pattern_evidence.evidence_id
        if pattern_id not in self.temporal_evidence:
            self.temporal_evidence[pattern_id] = []
        self.temporal_evidence[pattern_id].append(evidence)
    
    def _update_pattern_timeline(
        self,
        pattern_id: str,
        timestamp: datetime
    ) -> None:
        """Update pattern timeline naturally."""
        if pattern_id not in self.pattern_timelines:
            self.pattern_timelines[pattern_id] = []
        self.pattern_timelines[pattern_id].append(timestamp)
    
    def _calculate_stability(self, evidence: TemporalEvidence) -> float:
        """Calculate temporal stability naturally."""
        pattern_id = evidence.pattern_evidence.evidence_id
        if pattern_id not in self.temporal_evidence:
            return 1.0
            
        # Get temporal history
        history = self.temporal_evidence[pattern_id]
        if len(history) < 2:
            return 0.8  # Base stability for new patterns
            
        # Calculate time differences
        time_diffs = []
        for i in range(len(history) - 1):
            diff = (history[i+1].timestamp - history[i].timestamp).total_seconds()
            time_diffs.append(diff)
            
        # Calculate stability from consistency
        avg_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 1.0
        if avg_diff == 0:
            avg_diff = 1.0  # Prevent division by zero
            
        variance = sum((d - avg_diff) ** 2 for d in time_diffs) / len(time_diffs) if time_diffs else 0.0
        
        # Natural stability decay - higher variance means lower stability
        stability = 1.0 / (1.0 + variance / avg_diff) if avg_diff > 0 else 0.8
        return max(0.0, min(1.0, stability))
    
    def _calculate_temporal_confidence(
        self,
        pattern_evidence: PatternEvidence
    ) -> float:
        """Calculate temporal confidence naturally."""
        if not pattern_evidence.temporal_context:
            return 0.8  # Base confidence
            
        # Get pattern history
        pattern_id = pattern_evidence.evidence_id
        history = self.temporal_evidence.get(pattern_id, [])
        
        # Base confidence factors
        base_confidence = pattern_evidence.temporal_context.confidence
        stability = pattern_evidence.stability_score
        temporal_stability = (
            pattern_evidence.uncertainty_metrics.temporal_stability
            if pattern_evidence.uncertainty_metrics else 0.8
        )
        
        # Add dynamic factor based on history
        history_factor = len(history) / 10.0  # Increases with more observations
        
        # Calculate weighted confidence
        confidence = (
            0.4 * base_confidence +
            0.3 * stability +
            0.2 * temporal_stability +
            0.1 * history_factor
        )
        
        return max(0.0, min(1.0, confidence))
