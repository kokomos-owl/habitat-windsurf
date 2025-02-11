import uuid
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from dependency_injector.wiring import inject, Provide
from config import AppContainer

from utils.timestamp_service import TimestampService
from utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class TemporalContext:
    """Temporal context for relationships."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    sequence_order: Optional[int] = None

@dataclass
class SpatialContext:
    """Spatial context for relationships."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    geometry: Optional[Dict[str, Any]] = None
    scale: str = "local"

@dataclass
class UncertaintyMetrics:
    """Uncertainty metrics for relationships."""
    confidence_score: float = 1.0
    uncertainty_value: float = 0.0
    reliability_score: float = 1.0
    source_quality: float = 1.0

class RelationshipModel:
    """
    Represents a relationship between two AdaptiveIDs in the climate knowledge base.
    Supports temporal versioning, uncertainty propagation, and bidirectional updates.
    """
    
    @inject
    def __init__(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Dict[str, Any] = None,
        bidirectional: bool = False,
        timestamp_service: TimestampService = None,
        adaptive_learner: 'AdaptiveLearner' = None,
        uncertainty_propagator: 'UncertaintyPropagator' = None
    ):
        # Core attributes
        self.id: str = str(uuid.uuid4())
        self.source_id = source_id
        self.target_id = target_id
        self.relationship_type = relationship_type
        self.properties = properties or {}
        self.bidirectional = bidirectional

        # Version control
        self.version_history: List[Dict[str, Any]] = []
        self.current_version: str = "1.0.0"
        
        # Temporal context
        self.temporal_context = TemporalContext(
            start_time=datetime.utcnow(),
            end_time=None
        )
        
        # Spatial context
        self.spatial_context = SpatialContext()
        
        # Uncertainty handling
        self.uncertainty_metrics = UncertaintyMetrics()
        
        # Timestamps
        self.created_at = timestamp_service.get_timestamp() if timestamp_service else datetime.utcnow()
        self.updated_at = self.created_at
        
        # Services
        self.timestamp_service = timestamp_service
        self.adaptive_learner = adaptive_learner
        self.uncertainty_propagator = uncertainty_propagator
        
        # Initialize first version
        self._create_version("Initial creation")

    def update_relationship(
        self,
        properties: Optional[Dict[str, Any]] = None,
        temporal_context: Optional[Dict[str, Any]] = None,
        spatial_context: Optional[Dict[str, Any]] = None,
        uncertainty_update: Optional[Dict[str, float]] = None,
        reason: str = "Manual update"
    ) -> None:
        """
        Update relationship with new data and create a new version.
        
        Args:
            properties: New properties to update
            temporal_context: New temporal context
            spatial_context: New spatial context
            uncertainty_update: New uncertainty metrics
            reason: Reason for the update
        """
        try:
            # Update properties if provided
            if properties:
                self.properties.update(properties)

            # Update temporal context
            if temporal_context:
                self.temporal_context = TemporalContext(**temporal_context)

            # Update spatial context
            if spatial_context:
                self.spatial_context = SpatialContext(**spatial_context)

            # Update uncertainty metrics
            if uncertainty_update:
                self._update_uncertainty(uncertainty_update)

            # Create new version
            self._create_version(reason)
            
            # Update timestamp
            self.updated_at = self.timestamp_service.get_timestamp() if self.timestamp_service else datetime.utcnow()

            logger.info(f"Updated relationship {self.id} - {reason}")

        except Exception as e:
            logger.error(f"Error updating relationship {self.id}: {str(e)}")
            raise

    def _update_uncertainty(self, uncertainty_update: Dict[str, float]) -> None:
        """Update uncertainty metrics and propagate changes."""
        # Update local uncertainty metrics
        for key, value in uncertainty_update.items():
            if hasattr(self.uncertainty_metrics, key):
                setattr(self.uncertainty_metrics, key, value)

        # Propagate uncertainty changes
        if self.uncertainty_propagator:
            self.uncertainty_propagator.propagate(
                source_id=self.source_id,
                target_id=self.target_id,
                relationship_id=self.id,
                uncertainty_metrics=self.uncertainty_metrics
            )

    def _create_version(self, reason: str) -> None:
        """Create a new version of the relationship."""
        version = {
            "version_id": str(uuid.uuid4()),
            "timestamp": self.timestamp_service.get_timestamp() if self.timestamp_service else datetime.utcnow(),
            "reason": reason,
            "properties": self.properties.copy(),
            "temporal_context": vars(self.temporal_context),
            "spatial_context": vars(self.spatial_context),
            "uncertainty_metrics": vars(self.uncertainty_metrics),
            "version_number": self.current_version
        }
        self.version_history.append(version)
        self._increment_version()

    def _increment_version(self) -> None:
        """Increment the version number."""
        major, minor, patch = map(int, self.current_version.split('.'))
        patch += 1
        if patch > 9:
            patch = 0
            minor += 1
        if minor > 9:
            minor = 0
            major += 1
        self.current_version = f"{major}.{minor}.{patch}"

    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific version of the relationship."""
        for version in self.version_history:
            if version["version_id"] == version_id:
                return version
        return None

    def apply_bidirectional_update(self, update_data: Dict[str, Any]) -> None:
        """
        Apply updates from bidirectional learning process.
        
        Args:
            update_data: Learning-derived updates to apply
        """
        try:
            # Process through adaptive learner
            if self.adaptive_learner:
                adapted_data = self.adaptive_learner.adapt_relationship(
                    relationship_id=self.id,
                    source_id=self.source_id,
                    target_id=self.target_id,
                    update_data=update_data
                )

                # Apply adaptations
                self.update_relationship(
                    properties=adapted_data.get("properties"),
                    temporal_context=adapted_data.get("temporal_context"),
                    spatial_context=adapted_data.get("spatial_context"),
                    uncertainty_update=adapted_data.get("uncertainty_metrics"),
                    reason="Bidirectional learning update"
                )

                logger.info(f"Applied bidirectional update to relationship {self.id}")

        except Exception as e:
            logger.error(f"Error applying bidirectional update to relationship {self.id}: {str(e)}")
            raise

    def create_inverse(self) -> 'RelationshipModel':
        """Create inverse relationship for bidirectional relationships."""
        if not self.bidirectional:
            raise ValueError("Cannot create inverse for non-bidirectional relationship")

        inverse = RelationshipModel(
            source_id=self.target_id,
            target_id=self.source_id,
            relationship_type=f"inverse_{self.relationship_type}",
            properties=self.properties.copy(),
            bidirectional=True,
            timestamp_service=self.timestamp_service,
            adaptive_learner=self.adaptive_learner,
            uncertainty_propagator=self.uncertainty_propagator
        )

        # Copy contexts and uncertainty
        inverse.temporal_context = self.temporal_context
        inverse.spatial_context = self.spatial_context
        inverse.uncertainty_metrics = self.uncertainty_metrics

        return inverse

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "properties": self.properties,
            "bidirectional": self.bidirectional,
            "current_version": self.current_version,
            "temporal_context": vars(self.temporal_context),
            "spatial_context": vars(self.spatial_context),
            "uncertainty_metrics": vars(self.uncertainty_metrics),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RelationshipModel':
        """Create RelationshipModel instance from dictionary."""
        instance = cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relationship_type=data["relationship_type"],
            properties=data["properties"],
            bidirectional=data["bidirectional"]
        )
        
        # Restore contexts and uncertainty
        instance.temporal_context = TemporalContext(**data["temporal_context"])
        instance.spatial_context = SpatialContext(**data["spatial_context"])
        instance.uncertainty_metrics = UncertaintyMetrics(**data["uncertainty_metrics"])
        
        # Restore timestamps
        instance.created_at = data["created_at"]
        instance.updated_at = data["updated_at"]
        
        return instance

    def __str__(self) -> str:
        return f"RelationshipModel(id={self.id}, type={self.relationship_type}, version={self.current_version})"

    def validate_relationship(self) -> Tuple[bool, List[str]]:
        """
        Validate the relationship's current state.
        
        Returns:
            Tuple[bool, List[str]]: Validation success and list of any issues
        """
        issues = []
        
        # Check required fields
        if not self.source_id:
            issues.append("Missing source_id")
        if not self.target_id:
            issues.append("Missing target_id")
        if not self.relationship_type:
            issues.append("Missing relationship_type")
            
        # Check temporal consistency
        if self.temporal_context.end_time and self.temporal_context.start_time:
            if self.temporal_context.end_time < self.temporal_context.start_time:
                issues.append("End time precedes start time")
                
        # Check uncertainty bounds
        if not 0 <= self.uncertainty_metrics.confidence_score <= 1:
            issues.append("Confidence score out of bounds [0,1]")
        if not 0 <= self.uncertainty_metrics.uncertainty_value <= 1:
            issues.append("Uncertainty value out of bounds [0,1]")
            
        return len(issues) == 0, issues

    def merge_with(self, other: 'RelationshipModel') -> None:
        """
        Merge another relationship's properties into this one.
        
        Args:
            other: Another RelationshipModel to merge from
            
        Raises:
            ValueError: If relationships are incompatible
        """
        if self.source_id != other.source_id or self.target_id != other.target_id:
            raise ValueError("Cannot merge relationships with different endpoints")
            
        # Merge properties
        self.properties.update(other.properties)
        
        # Merge temporal context (take the wider time range)
        if other.temporal_context.start_time:
            if not self.temporal_context.start_time or other.temporal_context.start_time < self.temporal_context.start_time:
                self.temporal_context.start_time = other.temporal_context.start_time
                
        if other.temporal_context.end_time:
            if not self.temporal_context.end_time or other.temporal_context.end_time > self.temporal_context.end_time:
                self.temporal_context.end_time = other.temporal_context.end_time
                
        # Update uncertainty metrics
        self._merge_uncertainty_metrics(other.uncertainty_metrics)
        
        # Create new version for merge
        self._create_version(f"Merged with relationship {other.id}")

    def _merge_uncertainty_metrics(self, other_metrics: UncertaintyMetrics) -> None:
        """Merge uncertainty metrics using weighted averaging."""
        weight1 = self.uncertainty_metrics.reliability_score
        weight2 = other_metrics.reliability_score
        total_weight = weight1 + weight2
        
        if total_weight > 0:
            self.uncertainty_metrics.confidence_score = (
                (weight1 * self.uncertainty_metrics.confidence_score + 
                 weight2 * other_metrics.confidence_score) / total_weight
            )
            self.uncertainty_metrics.uncertainty_value = (
                (weight1 * self.uncertainty_metrics.uncertainty_value + 
                 weight2 * other_metrics.uncertainty_value) / total_weight
            )
            self.uncertainty_metrics.reliability_score = max(
                self.uncertainty_metrics.reliability_score,
                other_metrics.reliability_score
            )

    def apply_temporal_rules(self, rules: List[Dict[str, Any]]) -> None:
        """
        Apply temporal rules to the relationship.
        
        Args:
            rules: List of temporal rules to apply
        """
        for rule in rules:
            if rule["type"] == "duration":
                self._apply_duration_rule(rule)
            elif rule["type"] == "sequence":
                self._apply_sequence_rule(rule)
            elif rule["type"] == "overlap":
                self._apply_overlap_rule(rule)
                
        self._create_version("Applied temporal rules")

    def _apply_duration_rule(self, rule: Dict[str, Any]) -> None:
        """Apply duration-based temporal rule."""
        if "min_duration" in rule:
            if not self.temporal_context.duration:
                self.temporal_context.duration = rule["min_duration"]
            else:
                self.temporal_context.duration = max(
                    self.temporal_context.duration,
                    rule["min_duration"]
                )
                
        if "max_duration" in rule:
            if self.temporal_context.duration:
                self.temporal_context.duration = min(
                    self.temporal_context.duration,
                    rule["max_duration"]
                )

    def _apply_sequence_rule(self, rule: Dict[str, Any]) -> None:
        """Apply sequence-based temporal rule."""
        if "must_follow" in rule:
            if self.temporal_context.start_time:
                self.temporal_context.start_time = max(
                    self.temporal_context.start_time,
                    rule["must_follow"]
                )
                
        if "must_precede" in rule:
            if self.temporal_context.end_time:
                self.temporal_context.end_time = min(
                    self.temporal_context.end_time,
                    rule["must_precede"]
                )

    def _apply_overlap_rule(self, rule: Dict[str, Any]) -> None:
        """Apply overlap-based temporal rule."""
        if "overlap_start" in rule and "overlap_end" in rule:
            if self.temporal_context.start_time and self.temporal_context.end_time:
                # Ensure relationship overlaps with specified period
                self.temporal_context.start_time = max(
                    self.temporal_context.start_time,
                    rule["overlap_start"]
                )
                self.temporal_context.end_time = min(
                    self.temporal_context.end_time,
                    rule["overlap_end"]
                )

    def get_confidence_interval(self) -> Tuple[float, float]:
        """
        Calculate confidence interval based on uncertainty metrics.
        
        Returns:
            Tuple[float, float]: Lower and upper bounds of confidence interval
        """
        base = self.uncertainty_metrics.confidence_score
        spread = self.uncertainty_metrics.uncertainty_value
        
        lower_bound = max(0.0, base - spread)
        upper_bound = min(1.0, base + spread)
        
        return lower_bound, upper_bound

    def calculate_reliability(self) -> float:
        """
        Calculate overall reliability score.
        
        Returns:
            float: Computed reliability score
        """
        confidence = self.uncertainty_metrics.confidence_score
        uncertainty = self.uncertainty_metrics.uncertainty_value
        source_quality = self.uncertainty_metrics.source_quality
        
        # Weighted combination of metrics
        reliability = (
            0.4 * confidence +
            0.3 * (1 - uncertainty) +
            0.3 * source_quality
        )
        
        self.uncertainty_metrics.reliability_score = reliability
        return reliability

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        Compare two versions of the relationship.
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            Dict[str, Any]: Differences between versions
            
        Raises:
            ValueError: If either version not found
        """
        version1 = self.get_version(version_id1)
        version2 = self.get_version(version_id2)
        
        if not version1 or not version2:
            raise ValueError("Version not found")
            
        differences = {
            "properties": self._compare_dicts(version1["properties"], version2["properties"]),
            "temporal_context": self._compare_dicts(version1["temporal_context"], version2["temporal_context"]),
            "spatial_context": self._compare_dicts(version1["spatial_context"], version2["spatial_context"]),
            "uncertainty_metrics": self._compare_dicts(version1["uncertainty_metrics"], version2["uncertainty_metrics"])
        }
        
        return differences

    def _compare_dicts(self, dict1: Dict, dict2: Dict) -> Dict[str, Tuple[Any, Any]]:
        """Compare two dictionaries and return differences."""
        differences = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            value1 = dict1.get(key)
            value2 = dict2.get(key)
            if value1 != value2:
                differences[key] = (value1, value2)
                
        return differences

    def to_graph_format(self) -> Dict[str, Any]:
        """
        Convert relationship to graph database format.
        
        Returns:
            Dict[str, Any]: Graph-formatted relationship data
        """
        return {
            "relationship_id": self.id,
            "from_node": self.source_id,
            "to_node": self.target_id,
            "type": self.relationship_type,
            "properties": {
                **self.properties,
                "temporal_context": vars(self.temporal_context),
                "spatial_context": vars(self.spatial_context),
                "uncertainty_metrics": vars(self.uncertainty_metrics),
                "version": self.current_version,
                "bidirectional": self.bidirectional,
                "created_at": self.created_at,
                "updated_at": self.updated_at
            }
        }

    def to_vector(self) -> List[float]:
        """
        Convert relationship to vector representation for embedding.
        
        Returns:
            List[float]: Vector representation of relationship
        """
        # Basic vector representation - extend based on needs
        vector = [
            self.uncertainty_metrics.confidence_score,
            self.uncertainty_metrics.uncertainty_value,
            self.uncertainty_metrics.reliability_score,
            self.uncertainty_metrics.source_quality
        ]
        
        if self.temporal_context.start_time:
            vector.append(self.temporal_context.start_time.timestamp())
        if self.temporal_context.end_time:
            vector.append(self.temporal_context.end_time.timestamp())
            
        return vector

    def __eq__(self, other: object) -> bool:
        """Check equality with another RelationshipModel."""
        if not isinstance(other, RelationshipModel):
            return False
        return (self.id == other.id and 
                self.source_id == other.source_id and
                self.target_id == other.target_id and
                self.relationship_type == other.relationship_type)

    def __hash__(self) -> int:
        """Generate hash for RelationshipModel."""
        return hash((self.id, self.source_id, self.target_id, self.relationship_type))