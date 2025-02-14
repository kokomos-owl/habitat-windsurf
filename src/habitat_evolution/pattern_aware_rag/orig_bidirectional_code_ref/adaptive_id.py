# adaptive_id.py

import uuid
import json
import threading
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dependency_injector.wiring import inject, Provide
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # Insert at beginning to ensure our config is found first

# Import container from project root config
from config import AppContainer
from .relationship_model import RelationshipModel
from habitat_test.core.logging.logger import LoggingManager, LogContext
from utils.performance_monitor import performance_monitor
from utils.ethical_ai_checker import ethical_check
from utils.serialization import Serializable
from database.neo4j_client import Neo4jClient
from error_handling.error_handler import ErrorHandler
from event_manager import EventManager
from event_types import EventType
from .base_adaptive_id import BaseAdaptiveID

class AdaptiveIDException(Exception):
    """Base exception class for AdaptiveID-related errors."""
    pass

class VersionNotFoundException(AdaptiveIDException):
    """Raised when a specified version is not found."""
    pass

class InvalidValueError(AdaptiveIDException):
    """Raised when an invalid value is provided for an attribute."""
    pass

class Version:
    """Represents a version of a property in AdaptiveID."""
    def __init__(self, version_id: str, data: Dict[str, Any], timestamp: str, origin: str):
        self.version_id = version_id
        self.data = data
        self.timestamp = timestamp
        self.origin = origin

class AdaptiveID(Serializable):
    """
    Represents an adaptive concept in the climate knowledge base.

    This class allows for flexible, evolving, and traceable representation of concepts. It supports versioning,
    relationships, and integrates with the RelationshipRepository and ontology management.
    
    Thread-safety is ensured through the use of threading.Lock for critical operations.
    """

    @inject
    @performance_monitor.track
    @ethical_check
    def __init__(
        self,
        base_concept: str,
        weight: float = 1.0,
        confidence: float = 1.0,
        uncertainty: float = 0.0,
        config: Dict[str, Any] = Provide[AppContainer.config],
        timestamp_service: Any = Provide[AppContainer.timestamp_service],
        version_service: Any = Provide[AppContainer.version_service],
        relationship_repository: Any = Provide[AppContainer.relationship_repository],
        ontology_manager: Any = Provide[AppContainer.ontology_manager],
        bidirectional_learner: Any = Provide[AppContainer.bidirectional_learner],
        error_handler: ErrorHandler = Provide[AppContainer.error_handler],
        neo4j_client: Neo4jClient = Provide[AppContainer.neo4j_client],
        event_manager: EventManager = Provide[AppContainer.event_manager]
    ):
        """Initialize an AdaptiveID instance.
        
        Args:
            base_concept: The base concept this ID represents
            weight: Initial weight of the concept (default: 1.0)
            confidence: Initial confidence score (default: 1.0)
        """
        self.id = str(uuid.uuid4())
        self.base_concept = base_concept
        self.weight = weight
        self.confidence = confidence
        self.uncertainty = uncertainty
        
        # Initialize logging with context
        self.logger = self.initialize_logging()
        
        # Log initialization with metadata
        self.logger.log_process_start("adaptive_id_init", metadata={
            "base_concept": base_concept,
            "weight": weight,
            "confidence": confidence,
            "initialization_time": datetime.now().isoformat()
        })
        
        # Initialize core components
        self._lock = threading.Lock()
        self.versions: Dict[str, Version] = {}
        self.current_version = str(uuid.uuid4())
        self.temporal_context: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[str, RelationshipModel] = {}
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "version_count": 0,
            "relationship_count": 0
        }
        
        # Initialize first version
        self._create_initial_version()
        
        # Log successful initialization
        self.logger.log_process_end("adaptive_id_init", success=True, density_metrics={
            "base_concept_length": len(base_concept),
            "initial_weight": weight,
            "initial_confidence": confidence
        })
        
        self.version = config.get('ADAPTIVE_ID_VERSION', '1.0')
        self.max_versions: int = config.get('MAX_ADAPTIVE_ID_VERSIONS', 20)
        self.spatial_context: Dict[str, Any] = {
            "latitude": None,
            "longitude": None,
            "geometry": None  # Could be a GeoJSON object, WKT string, etc.
        }
        self.uncertainty: Dict[str, Any] = {}  # Flexible representation of uncertainty
        self.scale: str = None  # Or use an Enum for predefined scales
        self.source_attribution: List[str] = []  # Or a dictionary for more details

        self.relationships: List[str] = []
        self.properties: Dict[str, List[Version]] = {}
        
        self.creation_timestamp: str = timestamp_service.get_timestamp()
        self.last_updated_timestamp: str = self.creation_timestamp
        self.update_history: List[Dict[str, Any]] = []

        self.vector_representation: Optional[List[float]] = None
        self.usage_count: int = 0
        self.adaptation_score: float = 0.0

        self.config = config
        self.timestamp_service = timestamp_service
        self.version_service = version_service
        self.relationship_repository = relationship_repository
        self.ontology_manager = ontology_manager
        self.bidirectional_learner = bidirectional_learner
        self.error_handler = error_handler
        self.neo4j_client = neo4j_client
        self.event_manager = event_manager

    def initialize_logging(self) -> LoggingManager:
        """Initialize logging with proper context.
        
        Returns:
            LoggingManager: Configured logging manager instance
        """
        logger = LoggingManager(__name__)
        
        # Set consistent context
        logger.set_context(LogContext(
            process_id=self.id,
            component="adaptive_id",
            stage="initialization",
            system_component="concept_tracking"
        ))
        
        # Log initialization with metadata
        logger.log_process_start("adaptive_id_init", metadata={
            "base_concept": self.base_concept,
            "weight": self.weight,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "initialization_time": datetime.now().isoformat()
        })
        
        # Core initialization logging
        logger.info("Initializing core components...")
        logger.info(f"Created AdaptiveID instance with ID: {self.id}")
        
        # Log successful initialization
        logger.log_process_end("adaptive_id_init", success=True, density_metrics={
            "initial_weight": self.weight,
            "initial_confidence": self.confidence,
            "initial_uncertainty": self.uncertainty
        })
        
        return logger

    def _store_version(self, key: str, value: Any, origin: str) -> None:
        """Store a new version of a property value."""
        try:
            self.logger.set_context(LogContext(
                process_id=self.id,
                component="adaptive_id",
                stage="version_storage",
                system_component="concept_tracking"
            ))
            
            self.logger.log_process_start("version_storage", metadata={
                "key": key,
                "origin": origin,
                "current_version_count": len(self.versions)
            })
            
            version_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            new_version = Version(
                version_id=version_id,
                data={key: value},
                timestamp=timestamp,
                origin=origin
            )
            
            with self._lock:
                self.versions[version_id] = new_version
                self.current_version = version_id
                self.metadata["version_count"] += 1
                self.metadata["last_modified"] = timestamp
            
            self.logger.log_process_end("version_storage", success=True, density_metrics={
                "total_versions": len(self.versions),
                "property_key": key,
                "origin": origin
            })
            
        except Exception as e:
            self.logger.error("Version storage failed", metadata={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "key": key,
                "origin": origin
            })
            raise

    def update_temporal_context(self, key: str, value: Any, origin: str) -> None:
        """Update temporal context."""
        try:
            self.logger.set_context(LogContext(
                process_id=self.id,
                component="adaptive_id",
                stage="temporal_context_update",
                system_component="concept_tracking"
            ))
            
            self.logger.log_process_start("temporal_context_update", metadata={
                "key": key,
                "origin": origin
            })
            
            with self._lock:
                self.temporal_context[key] = value
                self._store_version('temporal_context', self.temporal_context, origin)
                self._update_attribute('temporal_context', self.temporal_context)
            
            self.logger.log_process_end("temporal_context_update", success=True, density_metrics={
                "context_size": len(self.temporal_context),
                "updated_key": key
            })
            
        except Exception as e:
            self.logger.error("Temporal context update failed", metadata={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "key": key,
                "origin": origin
            })
            raise

    def update_spatial_context(self, key: str, value: Any, origin: str) -> None:
        """Update spatial context."""
        try:
            self.logger.set_context(LogContext(
                process_id=self.id,
                component="adaptive_id",
                stage="spatial_context_update",
                system_component="concept_tracking"
            ))
            
            self.logger.log_process_start("spatial_context_update", metadata={
                "key": key,
                "origin": origin
            })
            
            with self._lock:
                self.spatial_context[key] = value
                self._store_version('spatial_context', self.spatial_context, origin)
                self._update_attribute('spatial_context', self.spatial_context)
            
            self.logger.log_process_end("spatial_context_update", success=True, density_metrics={
                "context_size": len(self.spatial_context),
                "updated_key": key
            })
            
        except Exception as e:
            self.logger.error("Spatial context update failed", metadata={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "key": key,
                "origin": origin
            })
            raise

    def get_state_at_time(self, timestamp: str) -> Dict[str, Any]:
        """
        Retrieve the state of the AdaptiveID at a specific timestamp.

        Args:
            timestamp (str): The timestamp to retrieve the state.

        Returns:
            Dict[str, Any]: The property values at the given timestamp.

        Raises:
            VersionNotFoundException: If no version matches the timestamp.
        """
        state_at_time = {}
        for prop, versions in self.properties.items():
            for version in versions:
                if version.timestamp <= timestamp:
                    state_at_time[prop] = version.data[prop]
        if not state_at_time:
            raise VersionNotFoundException(f"No state found for timestamp: {timestamp}")
        return state_at_time

    def compare_states(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """
        Compare two states of an AdaptiveID and return the differences.

        Args:
            state1 (Dict[str, Any]): The first state to compare.
            state2 (Dict[str, Any]): The second state to compare.

        Returns:
            Dict[str, Tuple[Any, Any]]: Differences between the two states.
        """
        differences = {}
        for key in state1:
            if key in state2 and state1[key] != state2[key]:
                differences[key] = (state1[key], state2[key])
        return differences

    def create_snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of the current state of the AdaptiveID instance.

        Returns:
            Dict[str, Any]: A snapshot representing the current state, including properties, relationships, and contexts.
        """
        with self._lock:
            snapshot = {
                "id": self.id,
                "base_concept": self.base_concept,
                "weight": self.weight,
                "confidence": self.confidence,
                "uncertainty": self.uncertainty,
                "temporal_context": self.temporal_context,
                "spatial_context": self.spatial_context,
                "relationships": self.relationships,
                "properties": {k: [v.__dict__ for v in versions] for k, versions in self.properties.items()},
                "creation_timestamp": self.creation_timestamp,
                "last_updated_timestamp": self.last_updated_timestamp,
                "update_history": self.update_history
            }
            self.logger.info(f"Created snapshot for AdaptiveID {self.id}")

            # Notify the event manager of a snapshot creation
            self.event_manager.publish(EventType.SNAPSHOT_CREATED, {
                "adaptive_id": self.id,
                "snapshot": snapshot
            })
            return snapshot

    def restore_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Restore the state of the AdaptiveID instance from a snapshot.

        Args:
            snapshot (Dict[str, Any]): A snapshot representing a previous state of the AdaptiveID instance.
        """
        with self._lock:
            self.id = snapshot["id"]
            self.base_concept = snapshot["base_concept"]
            self.weight = snapshot["weight"]
            self.confidence = snapshot["confidence"]
            self.uncertainty = snapshot["uncertainty"]
            self.temporal_context = snapshot["temporal_context"]
            self.spatial_context = snapshot["spatial_context"]
            self.relationships = snapshot["relationships"]
            self.properties = {k: [Version(**v) for v in versions] for k, versions in snapshot["properties"].items()}
            self.creation_timestamp = snapshot["creation_timestamp"]
            self.last_updated_timestamp = snapshot["last_updated_timestamp"]
            self.update_history = snapshot["update_history"]
            self.logger.info(f"Restored AdaptiveID {self.id} from snapshot")

            # Notify the event manager of a snapshot restoration
            self.event_manager.publish(EventType.SNAPSHOT_RESTORED, {
                "adaptive_id": self.id,
                "snapshot": snapshot
            })

    def __str__(self) -> str:
        """Return a string representation of the AdaptiveID."""
        return (f"AdaptiveID(id={self.id}, base_concept={self.base_concept}, "
                f"weight={self.weight}, confidence={self.confidence}, "
                f"uncertainty={self.uncertainty}, created={self.creation_timestamp})")

    def update_weight(self, new_weight: float, origin: str) -> None:
        """
        Update the weight of the concept.

        Args:
            new_weight (float): The new weight value.
            origin (str): The origin of the update.

        Raises:
            InvalidValueError: If the new weight is not between 0 and 1.
        """
        if 0 <= new_weight <= 1:
            with self._lock:
                self._store_version('weight', new_weight, origin)
                self._update_attribute('weight', new_weight)
        else:
            raise InvalidValueError("Weight must be between 0 and 1")

    def update_confidence(self, new_confidence: float, origin: str) -> None:
        """
        Update the confidence level of the concept.

        Args:
            new_confidence (float): The new confidence value.
            origin (str): The origin of the update.

        Raises:
            InvalidValueError: If the new confidence is not between 0 and 1.
        """
        if 0 <= new_confidence <= 1:
            with self._lock:
                self._store_version('confidence', new_confidence, origin)
                self._update_attribute('confidence', new_confidence)
        else:
            raise InvalidValueError("Confidence must be between 0 and 1")

    def update_uncertainty(self, new_uncertainty: float, origin: str) -> None:
        """
        Update the uncertainty level of the concept.

        Args:
            new_uncertainty (float): The new uncertainty value.
            origin (str): The origin of the update.

        Raises:
            InvalidValueError: If the new uncertainty is not between 0 and 1.
        """
        if 0 <= new_uncertainty <= 1:
            with self._lock:
                self._store_version('uncertainty', new_uncertainty, origin)
                self._update_attribute('uncertainty', new_uncertainty)
        else:
            raise InvalidValueError("Uncertainty must be between 0 and 1")

    def _update_attribute(self, attr_name: str, new_value: Any) -> None:
        """
        Helper method to update attributes and log changes.

        Args:
            attr_name (str): The name of the attribute to update.
            new_value (Any): The new value for the attribute.
        """
        old_value = getattr(self, attr_name)
        setattr(self, attr_name, new_value)
        self.last_updated_timestamp = self.timestamp_service.get_timestamp()
        self.update_history.append({
            'timestamp': self.last_updated_timestamp,
            'attribute': attr_name,
            'old_value': old_value,
            'new_value': new_value
        })
        self.logger.info(f"Updated {attr_name} for AdaptiveID {self.id} from {old_value} to {new_value}")

    def add_relationship(self, relationship: RelationshipModel, origin: str) -> None:
        """Add a relationship to this concept."""
        try:
            self.logger.set_context(LogContext(
                process_id=self.id,
                component="adaptive_id",
                stage="relationship_management",
                system_component="concept_tracking"
            ))
            
            self.logger.log_process_start("add_relationship", metadata={
                "relationship_id": relationship.id,
                "origin": origin
            })
            
            with self._lock:
                self.relationships.append(relationship.id)
                self._store_version('relationships', self.relationships, origin)
                self._update_attribute('relationships', self.relationships)
                self.relationship_repository.add_relationship(relationship)
            
            self.logger.log_process_end("add_relationship", success=True, density_metrics={
                "total_relationships": len(self.relationships)
            })
            
        except Exception as e:
            self.logger.error("Failed to add relationship", metadata={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "relationship_id": relationship.id
            })
            raise

    def remove_relationship(self, relationship_id: str, origin: str) -> None:
        """Remove a relationship from this concept."""
        try:
            self.logger.set_context(LogContext(
                process_id=self.id,
                component="adaptive_id",
                stage="relationship_management",
                system_component="concept_tracking"
            ))
            
            self.logger.log_process_start("remove_relationship", metadata={
                "relationship_id": relationship_id,
                "origin": origin
            })
            
            with self._lock:
                self.relationships.remove(relationship_id)
                self._store_version('relationships', self.relationships, origin)
                self._update_attribute('relationships', self.relationships)
                self.relationship_repository.remove_relationship(relationship_id)
            
            self.logger.log_process_end("remove_relationship", success=True, density_metrics={
                "total_relationships": len(self.relationships)
            })
            
        except Exception as e:
            self.logger.error("Failed to remove relationship", metadata={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "relationship_id": relationship_id
            })
            raise

    def update_property(self, key: str, value: Any, origin: str) -> None:
        """Update a property, creating a new version."""
        try:
            self.logger.set_context(LogContext(
                process_id=self.id,
                component="adaptive_id",
                stage="property_update",
                system_component="concept_tracking"
            ))
            
            self.logger.log_process_start("property_update", metadata={
                "key": key,
                "origin": origin,
                "current_version": self.current_version
            })
            
            with self._lock:
                version_id = self.version_service.get_new_version_id()
                new_version = Version(version_id, {key: value}, self.timestamp_service.get_timestamp(), origin)
                if key not in self.properties:
                    self.properties[key] = []
                self.properties[key].append(new_version)
                self._update_attribute(key, value)
                
                # Trigger bidirectional learning
                self.apply_bidirectional_learning({key: value})
            
            self.logger.log_process_end("property_update", success=True, density_metrics={
                "total_properties": len(self.properties),
                "property_versions": len(self.properties.get(key, []))
            })
            
        except Exception as e:
            self.logger.error("Property update failed", metadata={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "key": key,
                "origin": origin
            })
            raise AdaptiveIDException(f"Failed to update property: {str(e)}")

    def get_property_history(self, key: str) -> List[Version]:
        """
        Retrieve the version history of a property.

        Args:
            key (str): The key of the property to retrieve history for.

        Returns:
            List[Version]: A list of Version objects representing the property's history.
        """
        return self.properties.get(key, [])

    def revert_to_version(self, version_id: str) -> None:
        """Roll back to a previous version of all properties."""
        try:
            self.logger.set_context(LogContext(
                process_id=self.id,
                component="adaptive_id",
                stage="version_revert",
                system_component="concept_tracking"
            ))
            
            self.logger.log_process_start("version_revert", metadata={
                "target_version": version_id,
                "current_version": self.current_version
            })
            
            with self._lock:
                reverted = False
                for prop, versions in self.properties.items():
                    for version in versions:
                        if version.version_id == version_id:
                            self.update_property(prop, version.data[prop], origin="revert")
                            reverted = True
                if not reverted:
                    raise VersionNotFoundException(f"Version {version_id} not found")
            
            self.logger.log_process_end("version_revert", success=True, density_metrics={
                "total_properties": len(self.properties),
                "total_versions": sum(len(v) for v in self.properties.values())
            })
            
        except Exception as e:
            self.logger.error("Version revert failed", metadata={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "target_version": version_id
            })
            raise

    def get_relationship(self, relationship_id: str) -> Optional[RelationshipModel]:
        """
        Retrieve a full RelationshipModel object for a given relationship ID.

        This method interacts with the RelationshipRepository to retrieve the
        full RelationshipModel object.

        Args:
            relationship_id (str): The ID of the relationship to retrieve.

        Returns:
            Optional[RelationshipModel]: The RelationshipModel object if found, None otherwise.
        """
        return self.relationship_repository.get_relationship(relationship_id)

    def query_relationships(self, relationship_type: str = None) -> List[RelationshipModel]:
        """
        Query relationships based on type.

        This method interacts with the relationship repository to query relationships
        based on their type. If no type is specified, it returns all relationships.

        Args:
            relationship_type (str, optional): The type of relationship to query.

        Returns:
            List[RelationshipModel]: A list of RelationshipModel objects matching the query.
        """
        relationship_ids = self.relationship_repository.query_relationships(self.id, relationship_type)
        return [self.relationship_repository.get_relationship(rel_id) for rel_id in relationship_ids]
    
    def increment_usage(self, origin: str) -> None:
        """
        Increment the usage count of this concept. Each increment is versioned.

        Args:
            origin (str): The origin of the change (manual update, ontology influence, etc.).

        """
        with self._lock:
            self.usage_count += 1
            self._store_version('usage_count', self.usage_count, origin)
            self._update_attribute('usage_count', self.usage_count)
            
    def update_adaptation_score(self, new_score: float, origin: str) -> None:
        """
        Update the adaptation score of this concept. Each update is versioned.

        Args:
            new_score (float): The new adaptation score.
            origin (str): The origin of the change (manual update, ontology influence, etc.).

        """
        with self._lock:
            self._store_version('adaptation_score', new_score, origin)
            self._update_attribute('adaptation_score', new_score)
            
    def apply_bidirectional_learning(self, learning_data: Dict[str, Any]) -> None:
        """
        Apply bidirectional learning to update the concept based on new data.

        This method integrates with the bidirectional learning system to evolve
        the concept based on new information from both the data and ontology perspectives.

        Args:
            learning_data (Dict[str, Any]): New data used for bidirectional learning.

        Raises:
            AdaptiveIDException: If there's an error in the bidirectional learning process.
        """
        try:
            updated_properties = self.bidirectional_learner.learn(self.to_dict(), learning_data)
            for key, value in updated_properties.items():
                self.update_property(key, value, origin="bidirectional_learning")
            self.logger.info(f"Applied bidirectional learning to AdaptiveID {self.id}")
        except Exception as e:
            self.logger.exception(f"Error in bidirectional learning for AdaptiveID {self.id}: {e}")
            raise AdaptiveIDException(f"Failed to apply bidirectional learning: {str(e)}")
    
    def adjust_weight(self, adjustment_factor: float, origin: str) -> None:
        """
        Adjust the weight of the concept adaptively. Each adjustment is versioned.

        This method updates the weight based on an adjustment factor, which could be
        derived from usage patterns, relevance feedback, or other adaptive mechanisms.

        Args:
            adjustment_factor (float): Factor to adjust the weight by. Positive values
                increase the weight, negative values decrease it.
            origin (str): The origin of the change (manual update, ontology influence, etc.).

        Raises:
            InvalidValueError: If the resulting weight is not between 0 and 1.
        """
        new_weight = max(0, min(1, self.weight + adjustment_factor))
        self.update_weight(new_weight, origin)

    def update_property(self, key: str, value: Any, origin: str) -> None:
        """Update a property, creating a new version."""
        try:
            self.logger.set_context(LogContext(
                process_id=self.id,
                component="adaptive_id",
                stage="property_update",
                system_component="concept_tracking"
            ))
            
            self.logger.log_process_start("property_update", metadata={
                "key": key,
                "origin": origin,
                "current_version": self.current_version
            })
            
            with self._lock:
                version_id = self.version_service.get_new_version_id()
                new_version = Version(version_id, {key: value}, self.timestamp_service.get_timestamp(), origin)
                if key not in self.properties:
                    self.properties[key] = []
                self.properties[key].append(new_version)
                self._update_attribute(key, value)
                
                # Trigger bidirectional learning
                self.apply_bidirectional_learning({key: value})
            
            self.logger.log_process_end("property_update", success=True, density_metrics={
                "total_properties": len(self.properties),
                "property_versions": len(self.properties.get(key, []))
            })
            
        except Exception as e:
            self.logger.error("Property update failed", metadata={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "key": key,
                "origin": origin
            })
            raise AdaptiveIDException(f"Failed to update property: {str(e)}")

    def get_property_history(self, key: str) -> List[Version]:
        """
        Retrieve the version history of a property.

        Args:
            key (str): The key of the property to retrieve history for.

        Returns:
            List[Version]: A list of Version objects representing the property's history.
        """
        return self.properties.get(key, [])

    def revert_to_version(self, version_id: str) -> None:
        """Roll back to a previous version of all properties."""
        try:
            self.logger.set_context(LogContext(
                process_id=self.id,
                component="adaptive_id",
                stage="version_revert",
                system_component="concept_tracking"
            ))
            
            self.logger.log_process_start("version_revert", metadata={
                "target_version": version_id,
                "current_version": self.current_version
            })
            
            with self._lock:
                reverted = False
                for prop, versions in self.properties.items():
                    for version in versions:
                        if version.version_id == version_id:
                            self.update_property(prop, version.data[prop], origin="revert")
                            reverted = True
                if not reverted:
                    raise VersionNotFoundException(f"Version {version_id} not found")
            
            self.logger.log_process_end("version_revert", success=True, density_metrics={
                "total_properties": len(self.properties),
                "total_versions": sum(len(v) for v in self.properties.values())
            })
            
        except Exception as e:
            self.logger.error("Version revert failed", metadata={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "target_version": version_id
            })
            raise

    def influence_ontology(self) -> Dict[str, Any]:
        """
        Update the ontology based on the AdaptiveID's current state.

        This method interacts with the ontology management system to update the ontology
        based on the current state of this AdaptiveID.

        Returns:
            Dict[str, Any]: The updated state of the AdaptiveID in the ontology.
        """
        return self.ontology_manager.update_concept(self.to_dict())

    def update_from_ontology(self, ontology_data: Dict[str, Any], origin: str) -> None:
     """
     Update the AdaptiveID based on changes in the ontology. Each update is versioned.

     This method updates the AdaptiveID's attributes based on data from the ontology.

     Args:
         ontology_data (Dict[str, Any]): The updated data from the ontology.
         origin (str): The origin of the change (ontology update).
     """
     with self._lock:
         for key, value in ontology_data.items():
             if hasattr(self, key):
                 self.update_property(key, value, origin=origin)

    def change_propagation(self) -> List[str]:
        """
        Handle cascading updates in the ontology.

        This method interacts with the ontology management system to propagate changes
        to related concepts in the ontology.

        Returns:
            List[str]: A list of affected concept IDs.
        """
        return self.ontology_manager.propagate_changes(self.id)

    def conflict_resolution(self, conflicting_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle conflicting updates between AdaptiveID and ontology.

        This method implements a more advanced conflict resolution strategy that considers
        property weights, feedback, and other factors to determine the most reliable value.

        Args:
            conflicting_data (Dict[str, Any]): The conflicting data from the ontology.

        Returns:
            Dict[str, Any]: The resolved data after conflict resolution.
        """
        resolved_data = {}
        for key, value in conflicting_data.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if isinstance(current_value, (int, float)) and isinstance(value, (int, float)):
                    resolved_data[key] = max(current_value, value)  # Use higher value for numerical properties
                else:
                    # For non-numeric types, we'll keep the most recent value
                    resolved_data[key] = value
            else:
                resolved_data[key] = value
        self.logger.info(f"Conflict resolution completed for AdaptiveID {self.id} with resolved data: {resolved_data}")
        return resolved_data

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the AdaptiveID to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary representation of the AdaptiveID.
        """
        return {
            'id': self.id,
            'base_concept': self.base_concept,
            'weight': self.weight,
            'confidence': self.confidence,
            'uncertainty': self.uncertainty,
            'version': self.version,
            'temporal_context': self.temporal_context,
            'spatial_context': self.spatial_context,
            'relationships': self.relationships,
            'properties': {k: [v.__dict__ for v in versions] for k, versions in self.properties.items()},
            'creation_timestamp': self.creation_timestamp,
            'last_updated_timestamp': self.last_updated_timestamp,
            'update_history': self.update_history,
            'vector_representation': self.vector_representation,
            'source_references': self.source_references,
            'usage_count': self.usage_count,
            'adaptation_score': self.adaptation_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any],
                  config: Dict[str, Any] = Provide[AppContainer.config],
                  timestamp_service: Any = Provide[AppContainer.timestamp_service],
                  version_service: Any = Provide[AppContainer.version_service],
                  relationship_repository: Any = Provide[AppContainer.relationship_repository],
                  ontology_manager: Any = Provide[AppContainer.ontology_manager]) -> 'AdaptiveID':
        """
        Create an AdaptiveID instance from a dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary representation of an AdaptiveID.
            config (Dict[str, Any]): Configuration parameters injected from AppContainer.
            timestamp_service (Any): Service for generating consistent timestamps.
            version_service (Any): Service for managing versions.
            relationship_repository (Any): Repository for managing relationships.
            ontology_manager (Any): Service for managing the ontology.

        Returns:
            AdaptiveID: An instance of AdaptiveID.
        """
        instance = cls(data['base_concept'], data['weight'], data['confidence'],
                       data['uncertainty'], config, timestamp_service, version_service,
                       relationship_repository, ontology_manager)
        for key, value in data.items():
            if key not in ['base_concept', 'weight', 'confidence', 'uncertainty']:
                setattr(instance, key, value)
        instance.properties = {k: [Version(**v) for v in versions] for k, versions in data['properties'].items()}
        return instance

    def create_snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of the current state of the AdaptiveID.

        Returns:
            Dict[str, Any]: A dictionary representing the snapshot of the current state.
        """
        snapshot = self.to_dict()
        snapshot_id = str(uuid.uuid4())
        snapshot['snapshot_id'] = snapshot_id
        self.logger.info(f"Created snapshot for AdaptiveID {self.id} with snapshot ID: {snapshot_id}")
        return snapshot

    def restore_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Restore the state of the AdaptiveID from a given snapshot.

        Args:
            snapshot (Dict[str, Any]): The snapshot data to restore from.
        """
        with self._lock:
            for key, value in snapshot.items():
                if hasattr(self, key) and key != 'snapshot_id':
                    setattr(self, key, value)
            self.last_updated_timestamp = self.timestamp_service.get_timestamp()
        self.logger.info(f"Restored AdaptiveID {self.id} from snapshot ID: {snapshot['snapshot_id']}")

    def to_json(self) -> str:
        """
        Convert the AdaptiveID to a JSON string.

        Returns:
            str: JSON representation of the AdaptiveID.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str,
                  config: Dict[str, Any] = Provide[AppContainer.config],
                  timestamp_service: Any = Provide[AppContainer.timestamp_service],
                  version_service: Any = Provide[AppContainer.version_service],
                  relationship_repository: Any = Provide[AppContainer.relationship_repository],
                  ontology_manager: Any = Provide[AppContainer.ontology_manager]) -> 'AdaptiveID':
        """
        Create an AdaptiveID instance from a JSON string.

        Args:
            json_str (str): JSON string representation of an AdaptiveID.
            config (Dict[str, Any]): Configuration parameters injected from AppContainer.
            timestamp_service (Any): Service for generating consistent timestamps.
            version_service (Any): Service for managing versions.
            relationship_repository (Any): Repository for managing relationships.
            ontology_manager (Any): Service for managing the ontology.

        Returns:
            AdaptiveID: An instance of AdaptiveID.
        """
        data = json.loads(json_str)
        return cls.from_dict(data, config, timestamp_service, version_service,
                             relationship_repository, ontology_manager)

    def to_neo4j(self) -> Dict[str, Any]:
        """
        Convert the AdaptiveID to a Neo4j-friendly format.

        Returns:
            Dict[str, Any]: A dictionary representation of the AdaptiveID suitable for Neo4j storage.
        """
        neo4j_dict = self.to_dict()
        # Convert complex structures to JSON strings for Neo4j storage
        neo4j_dict['temporal_context'] = json.dumps(neo4j_dict['temporal_context'])
        neo4j_dict['spatial_context'] = json.dumps(neo4j_dict['spatial_context'])
        neo4j_dict['properties'] = json.dumps(neo4j_dict['properties'])
        neo4j_dict['update_history'] = json.dumps(neo4j_dict['update_history'])
        neo4j_dict['vector_representation'] = json.dumps(neo4j_dict['vector_representation'])
        return neo4j_dict

    @classmethod
    def from_neo4j(cls, neo4j_data: Dict[str, Any],
                   config: Dict[str, Any] = Provide[AppContainer.config],
                   timestamp_service: Any = Provide[AppContainer.timestamp_service],
                   version_service: Any = Provide[AppContainer.version_service],
                   relationship_repository: Any = Provide[AppContainer.relationship_repository],
                   ontology_manager: Any = Provide[AppContainer.ontology_manager]) -> 'AdaptiveID':
        """
        Create an AdaptiveID instance from Neo4j data.

        Args:
            neo4j_data (Dict[str, Any]): Data retrieved from Neo4j.
            config (Dict[str, Any]): Configuration parameters injected from AppContainer.
            timestamp_service (Any): Service for generating consistent timestamps.
            version_service (Any): Service for managing versions.
            relationship_repository (Any): Repository for managing relationships.
            ontology_manager (Any): Service for managing the ontology.

        Returns:
            AdaptiveID: An instance of AdaptiveID.
        """
        data = neo4j_data.copy()
        data['temporal_context'] = json.loads(data['temporal_context'])
        data['spatial_context'] = json.loads(data['spatial_context'])
        data['properties'] = json.loads(data['properties'])
        data['update_history'] = json.loads(data['update_history'])
        return cls.from_dict(data, config, timestamp_service, version_service,
                             relationship_repository, ontology_manager)

# End of adaptive_id.py