# pattern_core.py

"""
Core pattern service for Habitat knowledge evolution.

Natural observation service that works with knowledge coherence to track
pattern emergence without enforcing structure. Aligns with adaptive_core
components to maintain consistent evolution evidence and temporal mapping.
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from uuid import uuid4
import threading
import logging
from datetime import datetime

# Local imports
from habitat_test.core.interfaces.base_states import BaseProjectState  # Update import
from events.event_manager import EventManager
from events.event_types import EventType
from utils.timestamp_service import TimestampService
from utils.version_service import VersionService
from utils.logging_config import get_logger

from adaptive_core.relationship_model import (
    RelationshipModel, 
    TemporalContext,
    UncertaintyMetrics
)

# Third-party imports
from dependency_injector import containers, providers
from dependency_injector.wiring import inject, Provide
from config import AppContainer
from dataclasses import dataclass

logger = get_logger(__name__)

class PatternError(Exception):
    """Base exception for pattern tracking errors."""
    pass

@dataclass
class PatternEvidence:
    """Evidence structure aligned with adaptive_core tracking."""
    evidence_id: str
    timestamp: str
    pattern_type: str
    source_data: Dict[str, Any]
    temporal_context: Optional[TemporalContext] = None
    uncertainty_metrics: Optional[UncertaintyMetrics] = None
    version: str = "1.0"

@dataclass
class Version:
    """Version information for pattern tracking."""
    version_id: str
    data: Dict[str, Any]
    timestamp: str
    origin: str

class PatternCore:
    """
    Core pattern observation service that naturally expresses coherence through
    evidence-based pattern emergence. Maintains alignment with adaptive_core
    evolution tracking.
    """
    @inject
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
        
        # Pattern observation storage aligned with adaptive_core
        self.evidence_chains: Dict[str, PatternEvidence] = {}
        self.temporal_maps: Dict[str, Dict[str, Any]] = {}
        self.pattern_versions: Dict[str, List[Version]] = {}
        
        # Evolution tracking matching relationship_model
        self.pattern_metrics: Dict[str, UncertaintyMetrics] = {}
        self.temporal_contexts: Dict[str, TemporalContext] = {}
        
        # Observation history (aligned with version history in adaptive_id)
        self.pattern_history: List[Dict[str, Any]] = []
        self.latest_observations: Dict[str, Any] = {}
        
        # Light coherence thresholds
        self.evidence_threshold = 0.3  # Light evidence validation
        self.temporal_threshold = 0.3  # Light temporal alignment check
        
        logger.info("Initialized pattern observation service")

    def observe_evolution(
        self, 
        structural_change: Dict[str, Any],
        semantic_change: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Observe natural pattern emergence without enforcing structure.
        Maintains alignment with adaptive_core evolution tracking.
        """
        try:
            with self._lock:
                # Extract patterns with temporal context
                structure_pattern = self._extract_structure_pattern(
                    structural_change,
                    self._get_temporal_context(structural_change)
                )
                
                meaning_pattern = self._extract_meaning_pattern(
                    semantic_change,
                    self._get_temporal_context(semantic_change)
                )
                
                # Record aligned evidence if provided
                evidence_id = None
                if evidence:
                    evidence_id = self._record_evidence_chain(
                        structure_pattern=structure_pattern,
                        meaning_pattern=meaning_pattern,
                        evidence=evidence,
                        temporal_context=self._get_temporal_context(evidence)
                    )
                    
                    # Track uncertainty metrics
                    self._update_pattern_metrics(
                        evidence_id,
                        structure_pattern,
                        meaning_pattern
                    )

                # Map to timeline preserving temporal alignment
                timestamp = self.timestamp_service.get_timestamp()
                self._map_to_timeline(
                    timestamp,
                    structure_pattern,
                    meaning_pattern,
                    evidence_id
                )

                # Record observation matching version history pattern
                observation = {
                    "observation_id": str(uuid4()),
                    "timestamp": timestamp,
                    "version": self.version_service.get_new_version(),
                    "patterns": {
                        "structure": structure_pattern,
                        "meaning": meaning_pattern
                    },
                    "evidence_id": evidence_id,
                    "temporal_context": self._get_temporal_context({
                        "structure": structural_change,
                        "semantic": semantic_change,
                        "evidence": evidence or {}
                    }),
                    "uncertainty_metrics": self._get_uncertainty_metrics(
                        evidence_id
                    ) if evidence_id else None
                }

                self.pattern_history.append(observation)
                self.latest_observations = observation

                # Emit aligned observation event
                self.event_manager.publish(
                    EventType.PATTERN_OBSERVED,
                    observation
                )

                return observation

        except Exception as e:
            logger.error(f"Error observing evolution: {str(e)}")
            raise PatternError(f"Evolution observation failed: {str(e)}")

    def _extract_structure_pattern(
        self,
        structural_change: Dict[str, Any],
        temporal_context: Optional[TemporalContext] = None
    ) -> Dict[str, Any]:
        """
        Extract structural patterns while maintaining relationship alignment.
        """
        try:
            # Structure analysis aligned with relationship patterns
            relationships = {
                "from": structural_change.get("from_state", []),
                "to": structural_change.get("to_state", [])
            }

            # Calculate metrics matching relationship confidence model
            metrics = {
                "relationship_density": len(relationships["to"]),
                "node_change_rate": len(
                    structural_change.get("modified_nodes", [])
                ),
                "structural_confidence": self._calculate_structural_confidence(
                    relationships,
                    temporal_context
                )
            }

            pattern = {
                "type": "structure_evolution",
                "pattern_id": str(uuid4()),
                "relationships": relationships,
                "metrics": metrics,
                "temporal_context": temporal_context.__dict__ if temporal_context else None
            }

            return pattern

        except Exception as e:
            logger.error(f"Error extracting structure pattern: {str(e)}")
            return {}

    def _extract_meaning_pattern(
        self,
        semantic_change: Dict[str, Any],
        temporal_context: Optional[TemporalContext] = None
    ) -> Dict[str, Any]:
        """
        Extract meaning patterns while maintaining AdaptiveID alignment.
        """
        try:
            # Semantic analysis aligned with AdaptiveID evolution
            concept_changes = {
                "from": semantic_change.get("from_state", {}),
                "to": semantic_change.get("to_state", {})
            }

            # Calculate metrics matching adaptive confidence
            metrics = {
                "semantic_shift_rate": self._calculate_semantic_shift(
                    concept_changes
                ),
                "meaning_confidence": self._calculate_meaning_confidence(
                    concept_changes,
                    temporal_context
                )
            }

            pattern = {
                "type": "meaning_evolution",
                "pattern_id": str(uuid4()),
                "concept_changes": concept_changes,
                "metrics": metrics,
                "temporal_context": temporal_context.__dict__ if temporal_context else None
            }

            return pattern

        except Exception as e:
            logger.error(f"Error extracting meaning pattern: {str(e)}")
            return {}

    def _record_evidence_chain(
        self,
        structure_pattern: Dict[str, Any],
        meaning_pattern: Dict[str, Any],
        evidence: Dict[str, Any],
        temporal_context: Optional[TemporalContext] = None
    ) -> str:
        """
        Record evidence chain aligned with version and relationship tracking.
        """
        try:
            evidence_id = str(uuid4())
            timestamp = self.timestamp_service.get_timestamp()

            # Create aligned evidence record
            evidence_record = PatternEvidence(
                evidence_id=evidence_id,
                timestamp=timestamp,
                pattern_type="evolution",
                source_data={
                    "structure_pattern": structure_pattern,
                    "meaning_pattern": meaning_pattern,
                    "evidence": evidence
                },
                temporal_context=temporal_context,
                uncertainty_metrics=UncertaintyMetrics(
                    confidence_score=self._calculate_evidence_confidence(
                        structure_pattern,
                        meaning_pattern,
                        evidence
                    ),
                    uncertainty_value=self._calculate_evidence_uncertainty(
                        structure_pattern,
                        meaning_pattern,
                        evidence
                    ),
                    reliability_score=evidence.get("reliability", 1.0),
                    source_quality=evidence.get("source_quality", 1.0)
                ),
                version=self.version_service.get_new_version()
            )

            # Store with temporal ordering
            self.evidence_chains[evidence_id] = evidence_record
            
            # Create version record matching adaptive_core pattern
            version = Version(
                version_id=self.version_service.get_new_version_id(),
                data=evidence_record.__dict__,
                timestamp=timestamp,
                origin="pattern_observation"
            )
            
            if evidence_id not in self.pattern_versions:
                self.pattern_versions[evidence_id] = []
            self.pattern_versions[evidence_id].append(version)

            return evidence_id

        except Exception as e:
            logger.error(f"Error recording evidence chain: {str(e)}")
            raise PatternError(f"Evidence chain recording failed: {str(e)}")

    def _calculate_structural_confidence(
        self,
        relationships: Dict[str, List[Any]],
        temporal_context: Optional[TemporalContext] = None
    ) -> float:
        """Calculate structural confidence aligned with relationship metrics."""
        try:
            base_confidence = 0.5  # Start moderate

            # Adjust for relationship density
            if relationships["to"]:
                density = len(relationships["to"])
                base_confidence += min(0.3, density * 0.1)

            # Adjust for temporal context
            if temporal_context and temporal_context.start_time:
                base_confidence *= 0.9  # Slight reduction for temporal evolution

            return min(base_confidence, 1.0)

        except Exception as e:
            logger.error(f"Error calculating structural confidence: {str(e)}")
            return 0.1

    def _calculate_meaning_confidence(
        self,
        concept_changes: Dict[str, Dict[str, Any]],
        temporal_context: Optional[TemporalContext] = None
    ) -> float:
        """Calculate meaning confidence aligned with adaptive confidence."""
        try:
            base_confidence = 0.5  # Start moderate

            # Adjust for concept evolution
            if "to" in concept_changes:
                shift_rate = self._calculate_semantic_shift(concept_changes)
                base_confidence += min(0.3, 1.0 - shift_rate)

            # Adjust for temporal context
            if temporal_context and temporal_context.start_time:
                base_confidence *= 0.9  # Slight reduction for temporal evolution

            return min(base_confidence, 1.0)

        except Exception as e:
            logger.error(f"Error calculating meaning confidence: {str(e)}")
            return 0.1

    def _map_to_timeline(
        self,
        timestamp: str,
        structure_pattern: Dict[str, Any],
        meaning_pattern: Dict[str, Any],
        evidence_id: Optional[str] = None
    ) -> None:
        """Map patterns to timeline with temporal alignment."""
        try:
            timeline_entry = {
                "timestamp": timestamp,
                "patterns": {
                    "structure": structure_pattern,
                    "meaning": meaning_pattern
                },
                "evidence_id": evidence_id,
                "temporal_context": self._get_temporal_context({
                    "structure": structure_pattern,
                    "meaning": meaning_pattern
                })
            }
            
            self.temporal_maps[timestamp] = timeline_entry

        except Exception as e:
            logger.error(f"Error mapping to timeline: {str(e)}")

    def _update_pattern_metrics(
        self,
        evidence_id: str,
        structure_pattern: Dict[str, Any],
        meaning_pattern: Dict[str, Any]
    ) -> None:
        """Update pattern metrics aligned with uncertainty tracking."""
        try:
            evidence = self.evidence_chains.get(evidence_id)
            if not evidence:
                return

            metrics = UncertaintyMetrics(
                confidence_score=min(
                    structure_pattern["metrics"]["structural_confidence"],
                    meaning_pattern["metrics"]["meaning_confidence"]
                ),
                uncertainty_value=self._calculate_pattern_uncertainty(
                    structure_pattern,
                    meaning_pattern
                ),
                reliability_score=evidence.uncertainty_metrics.reliability_score,
                source_quality=evidence.uncertainty_metrics.source_quality
            )

            self.pattern_metrics[evidence_id] = metrics

        except Exception as e:
            logger.error(f"Error updating pattern metrics: {str(e)}")

    def _get_temporal_context(
        self,
        data: Dict[str, Any]
    ) -> Optional[TemporalContext]:
        """Extract temporal context aligned with relationship tracking."""
        try:
            if isinstance(data, dict):
                # Handle nested temporal data
                nested_context = next(
                    (
                        value.get("temporal_context")
                        for value in data.values()
                        if isinstance(value, dict) and "temporal_context" in value
                    ),
                    None
                )
                
                if nested_context:
                    return TemporalContext(**nested_context)

                # Check for direct temporal data
                if "start_time" in data or "end_time" in data:
                    return TemporalContext(
                        start_time=data.get("start_time"),
                        end_time=data.get("end_time"),
                        duration=data.get("duration")
                    )

            return None

        except Exception as e:
            logger.error(f"Error extracting temporal context: {str(e)}")
            return None

    def _get_uncertainty_metrics(
        self,
        evidence_id: str
    ) -> Optional[UncertaintyMetrics]:
        """Get uncertainty metrics matching relationship model."""
        return self.pattern_metrics.get(evidence_id)

    def _calculate_pattern_uncertainty(
        self,
        structure_pattern: Dict[str, Any],
        meaning_pattern: Dict[str, Any]
    ) -> float:
        """Calculate pattern uncertainty aligned with adaptive metrics."""
        try:
            structural_confidence = structure_pattern["metrics"]["structural_confidence"]
            meaning_confidence = meaning_pattern["metrics"]["meaning_confidence"]
            
            # Higher confidence = lower uncertainty
            uncertainty = 1.0 - ((structural_confidence + meaning_confidence) / 2)
            
            return max(0.1, min(uncertainty, 0.9))  # Keep between 0.1 and 0.9

        except Exception as e:
            logger.error(f"Error calculating pattern uncertainty: {str(e)}")
            return 0.5

    def get_latest_evidence(
        self,
        limit: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most recent evidence chains for evolution processing.
        """
        try:
            sorted_chains = sorted(
                self.evidence_chains.values(),
                key=lambda x: x.timestamp,
                reverse=True
            )
            return [chain.__dict__ for chain in sorted_chains[:limit]]

        except Exception as e:
            logger.error(f"Error retrieving evidence chains: {str(e)}")
            return []

    def get_pattern_history(
        self,
        pattern_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get pattern evolution history with optional type filter."""
        try:
            if pattern_type:
                return [
                    record for record in self.pattern_history
                    if pattern_type in record["patterns"]
                ]
            return self.pattern_history

        except Exception as e:
            logger.error(f"Error retrieving pattern history: {str(e)}")
            return []

    def get_temporal_patterns(
        self,
        timeframe: Optional[Tuple[str, str]] = None
    ) -> Dict[str, Any]:
        """Get temporal pattern mapping with optional timeframe filter."""
        try:
            if not timeframe:
                return self.temporal_maps

            start, end = timeframe
            return {
                timestamp: patterns
                for timestamp, patterns in self.temporal_maps.items()
                if start <= timestamp <= end
            }

        except Exception as e:
            logger.error(f"Error retrieving temporal patterns: {str(e)}")
            return {}

    def cleanup(self) -> None:
        """Clean up pattern tracking resources."""
        try:
            with self._lock:
                self.evidence_chains.clear()
                self.temporal_maps.clear()
                self.pattern_metrics.clear()
                self.pattern_versions.clear()
                self.temporal_contexts.clear()
                self.pattern_history.clear()
                self.latest_observations = {}
                
                logger.info("Cleaned up pattern tracking resources")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise PatternError(f"Cleanup failed: {str(e)}")
