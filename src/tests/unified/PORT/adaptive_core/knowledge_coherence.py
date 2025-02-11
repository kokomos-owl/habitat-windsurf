# knowledge_coherence.py

"""
Light coherence observation service that works in concert with PatternCore
to track natural knowledge evolution in Habitat.

Observes but doesn't enforce coherence between structure and meaning,
allowing natural pattern emergence while maintaining clean data structures
for future RGCN capabilities.
"""

from typing import Dict, Any, List, Optional, Set, Union
from uuid import uuid4
import threading
import logging
from datetime import datetime
from dependency_injector.wiring import inject, Provide

from config import AppContainer
from core.base_states import BaseProjectState
from events.event_manager import EventManager
from events.event_types import EventType
from utils.timestamp_service import TimestampService
from utils.version_service import VersionService
from utils.logging_config import get_logger
from .pattern_core import PatternCore

logger = get_logger(__name__)

class CoherenceError(Exception):
    """Base exception for coherence tracking errors."""
    pass

class KnowledgeCoherence:
    """
    Light coherence observation service that works with PatternCore.
    Observes natural alignment without enforcing structure.
    """
    @inject
    def __init__(
        self,
        pattern_core: PatternCore = Provide[AppContainer.pattern_core],
        timestamp_service: TimestampService = Provide[AppContainer.timestamp_service],
        event_manager: EventManager = Provide[AppContainer.event_manager],
        version_service: VersionService = Provide[AppContainer.version_service]
    ):
        self.pattern_core = pattern_core
        self.timestamp_service = timestamp_service
        self.event_manager = event_manager
        self.version_service = version_service
        self._lock = threading.Lock()

        # Coherence tracking
        self.coherence_history: List[Dict[str, Any]] = []
        self.latest_coherence: Dict[str, Any] = {}
        
        # Natural pattern buffers
        self.recent_patterns: List[Dict[str, Any]] = []
        self.pattern_buffer_size = 5  # Keep last 5 patterns for context
        
        logger.info("Initialized KnowledgeCoherence")

    def observe_evolution(
        self,
        structural_change: Dict[str, Any],
        semantic_change: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Observe coherence in knowledge evolution without enforcing it.
        Works with PatternCore to track natural pattern emergence.
        """
        try:
            with self._lock:
                # Let PatternCore observe patterns
                pattern = self.pattern_core.observe_evolution(
                    structural_change,
                    semantic_change,
                    evidence
                )
                
                # Observe coherence from pattern
                coherence = self._observe_coherence(pattern)
                
                # Track history
                self.coherence_history.append(coherence)
                self.latest_coherence = coherence

                # Emit event but don't enforce
                self.event_manager.publish(EventType.COHERENCE_OBSERVED, {
                    "coherence": coherence,
                    "pattern": pattern
                })

                return coherence

        except Exception as e:
            logger.error(f"Error observing evolution coherence: {str(e)}")
            return {}

    def _observe_coherence(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observe natural coherence from pattern without enforcing structure.
        """
        try:
            # Update pattern buffer
            self.recent_patterns.append(pattern)
            if len(self.recent_patterns) > self.pattern_buffer_size:
                self.recent_patterns.pop(0)

            # Calculate coherence metrics
            coherence = {
                "timestamp": self.timestamp_service.get_timestamp(),
                "version": self.version_service.get_new_version(),
                "metrics": {
                    "evidence_strength": self._calculate_evidence_strength(pattern),
                    "temporal_alignment": self._check_temporal_alignment(pattern),
                    "structural_consistency": self._check_structural_consistency(pattern),
                    "semantic_continuity": self._check_semantic_continuity(pattern)
                }
            }

            # Add pattern context
            coherence["context"] = {
                "pattern_type": pattern.get("type"),
                "recent_patterns": len(self.recent_patterns),
                "evidence_base": pattern.get("evidence", {}).get("source")
            }

            return coherence

        except Exception as e:
            logger.error(f"Error calculating coherence: {str(e)}")
            return {}

    def _calculate_evidence_strength(self, pattern: Dict[str, Any]) -> float:
        """Calculate evidence strength without enforcing requirements."""
        if not pattern.get("evidence"):
            return 0.1  # Allow patterns without strong evidence
            
        evidence = pattern["evidence"]
        base_score = 0.5  # Start with moderate confidence
        
        # Strengthen based on evidence quality
        if evidence.get("source"):
            base_score += 0.2
        if evidence.get("confidence"):
            base_score += evidence["confidence"] * 0.3
            
        return min(base_score, 1.0)

    def _check_temporal_alignment(self, pattern: Dict[str, Any]) -> float:
        """Check temporal alignment without enforcing it."""
        if not self.recent_patterns:
            return 1.0  # No context to check against
            
        # Check if temporal progression makes sense
        prev_pattern = self.recent_patterns[-1]
        if "temporal_context" in prev_pattern and "temporal_context" in pattern:
            prev_time = prev_pattern["temporal_context"].get("timestamp")
            curr_time = pattern["temporal_context"].get("timestamp")
            
            if prev_time and curr_time and prev_time <= curr_time:
                return 1.0
            return 0.5
            
        return 1.0  # No temporal context to check

    def _check_structural_consistency(self, pattern: Dict[str, Any]) -> float:
        """Check structural consistency without enforcing it."""
        if not self.recent_patterns:
            return 1.0  # No context to check against
            
        # Simple structural evolution check
        prev_pattern = self.recent_patterns[-1]
        if "structure" in prev_pattern and "structure" in pattern:
            return self._check_structure_evolution(
                prev_pattern["structure"],
                pattern["structure"]
            )
            
        return 1.0  # No structure to check

    def _check_semantic_continuity(self, pattern: Dict[str, Any]) -> float:
        """Check semantic continuity without enforcing it."""
        if not self.recent_patterns:
            return 1.0  # No context to check against
            
        # Check semantic evolution
        prev_pattern = self.recent_patterns[-1]
        if "meaning" in prev_pattern and "meaning" in pattern:
            return self._check_meaning_evolution(
                prev_pattern["meaning"],
                pattern["meaning"]
            )
            
        return 1.0  # No meaning to check

    def _check_structure_evolution(
        self,
        prev_structure: Dict[str, Any],
        curr_structure: Dict[str, Any]
    ) -> float:
        """Check if structural evolution seems reasonable."""
        # Simple check for POC - could be more sophisticated
        return 1.0  # Allow evolution by default

    def _check_meaning_evolution(
        self,
        prev_meaning: Dict[str, Any],
        curr_meaning: Dict[str, Any]
    ) -> float:
        """Check if meaning evolution seems reasonable."""
        # Simple check for POC - could be more sophisticated
        return 1.0  # Allow evolution by default

    def get_current_coherence(self) -> Dict[str, Any]:
        """Get latest coherence observations."""
        return self.latest_coherence

    def get_coherence_history(self) -> List[Dict[str, Any]]:
        """Get complete coherence history."""
        return self.coherence_history

    def cleanup(self) -> None:
        """Clean up coherence tracking resources."""
        try:
            with self._lock:
                self.coherence_history.clear()
                self.latest_coherence = {}
                self.recent_patterns.clear()
                
                logger.info("Cleaned up KnowledgeCoherence resources")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise CoherenceError(f"Cleanup failed: {str(e)}")