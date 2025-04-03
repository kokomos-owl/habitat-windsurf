"""
Vector-Tonic Persistence Connector.

This module connects the vector-tonic-window system's events with the persistence layer,
ensuring that patterns, field states, and relationships are properly captured and persisted
in ArangoDB. It serves as the bridge between the pattern detection system and the
persistence infrastructure.
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.habitat_evolution.adaptive_core.emergence.repository_factory import create_repositories

from src.habitat_evolution.core.services.event_bus import Event, LocalEventBus
from src.habitat_evolution.adaptive_core.emergence.persistence_integration import (
    VectorTonicPersistenceIntegration,
    PatternPersistenceService,
    FieldStatePersistenceService,
    RelationshipPersistenceService
)
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import VectorTonicWindowIntegrator
from src.habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager

# Import observer interfaces
from src.habitat_evolution.adaptive_core.emergence.interfaces.learning_window_observer import (
    LearningWindowObserverInterface,
    LearningWindowState
)
from src.habitat_evolution.adaptive_core.emergence.interfaces.pattern_observer import PatternObserverInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.field_observer import FieldObserverInterface

# Import repository interfaces
from src.habitat_evolution.adaptive_core.emergence.interfaces.field_state_repository import FieldStateRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.pattern_repository import PatternRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.relationship_repository import RelationshipRepositoryInterface
from src.habitat_evolution.adaptive_core.emergence.interfaces.topology_repository import TopologyRepositoryInterface

logger = logging.getLogger(__name__)

class VectorTonicPersistenceConnector(LearningWindowObserverInterface, PatternObserverInterface, FieldObserverInterface):
    """
    Connector between the vector-tonic-window system and the persistence layer.
    
    This class ensures that events from the vector-tonic-window system are properly
    captured and persisted in ArangoDB. It handles the bidirectional flow of information
    between the pattern detection system and the persistence infrastructure.
    
    It implements the following observer interfaces:
    - LearningWindowObserverInterface: For observing learning window events
    - PatternObserverInterface: For observing pattern events
    - FieldObserverInterface: For observing field state events
    """
    
    def __init__(self, event_bus=None, db=None,
                 field_state_repository: Optional[FieldStateRepositoryInterface] = None,
                 pattern_repository: Optional[PatternRepositoryInterface] = None,
                 relationship_repository: Optional[RelationshipRepositoryInterface] = None,
                 topology_repository: Optional[TopologyRepositoryInterface] = None):
        """Initialize the connector.
        
        Args:
            event_bus: Optional event bus. If not provided, a new event bus will be created.
            db: Optional database connection. If not provided, a new connection will be created.
            field_state_repository: Optional field state repository. If not provided, will use the one from persistence_integration.
            pattern_repository: Optional pattern repository. If not provided, will use the one from persistence_integration.
            relationship_repository: Optional relationship repository. If not provided, will use the one from persistence_integration.
            topology_repository: Optional topology repository. If not provided, will use the one from persistence_integration.
        """
        self.event_bus = event_bus or LocalEventBus()
        self.db = db or ArangoDBConnectionManager().get_db()
        
        # Create the persistence integration
        self.persistence_integration = VectorTonicPersistenceIntegration(self.event_bus, self.db)
        
        # Create direct references to persistence services for convenience
        self.pattern_service = self.persistence_integration.pattern_service
        self.field_state_service = self.persistence_integration.field_state_service
        self.relationship_service = self.persistence_integration.relationship_service
        
        # Store repository references
        self.field_state_repository = field_state_repository
        self.pattern_repository = pattern_repository
        self.relationship_repository = relationship_repository
        self.topology_repository = topology_repository
        
        # Track initialization state
        self.initialized = False
        
        # Track learning window state
        self.active_windows = {}
        
        # Track pattern cache
        self.pattern_cache = {}
    
    def initialize(self) -> None:
        """Initialize the connector.
        
        This method initializes the persistence integration and subscribes to events.
        It also initializes repositories if they haven't been provided.
        """
        if self.initialized:
            logger.info("VectorTonicPersistenceConnector already initialized")
            return
        
        # Initialize repositories if they haven't been provided
        if not all([self.field_state_repository, self.pattern_repository, 
                    self.relationship_repository, self.topology_repository]):
            logger.info("Creating repositories using factory methods")
            repositories = create_repositories(self.db)
            
            if not self.field_state_repository and "field_state_repository" in repositories:
                self.field_state_repository = repositories["field_state_repository"]
                
            if not self.pattern_repository and "pattern_repository" in repositories:
                self.pattern_repository = repositories["pattern_repository"]
                
            if not self.relationship_repository and "relationship_repository" in repositories:
                self.relationship_repository = repositories["relationship_repository"]
                
            if not self.topology_repository and "topology_repository" in repositories:
                self.topology_repository = repositories["topology_repository"]
        
        # Initialize the persistence integration
        if not hasattr(self, "persistence_integration") or self.persistence_integration is None:
            self.persistence_integration = VectorTonicPersistenceIntegration(self.db)
        
        self.persistence_integration.initialize()
        
        # Get services from the persistence integration
        self.pattern_service = self.persistence_integration.pattern_service
        self.field_state_service = self.persistence_integration.field_state_service
        self.relationship_service = self.persistence_integration.relationship_service
        
        # Subscribe to events
        # Pattern events
        self.event_bus.subscribe("pattern.detected", self.on_pattern_detected)
        self.event_bus.subscribe("pattern.evolved", self.on_pattern_evolution)
        self.event_bus.subscribe("pattern.quality.changed", self.on_pattern_quality_change)
        self.event_bus.subscribe("pattern.relationship.detected", self.on_pattern_relationship_detected)
        self.event_bus.subscribe("pattern.merged", self.on_pattern_merge)
        self.event_bus.subscribe("pattern.split", self.on_pattern_split)
        
        # Field state events
        self.event_bus.subscribe("field.state.changed", self.on_field_state_change)
        self.event_bus.subscribe("field.coherence.changed", self.on_field_coherence_change)
        self.event_bus.subscribe("field.stability.changed", self.on_field_stability_change)
        self.event_bus.subscribe("field.density.centers.shifted", self.on_density_center_shift)
        self.event_bus.subscribe("field.eigenspace.changed", self.on_eigenspace_change)
        self.event_bus.subscribe("field.topology.changed", self.on_topology_change)
        
        # Learning window events
        self.event_bus.subscribe("learning.window.state.changed", self.on_window_state_change)
        self.event_bus.subscribe("learning.window.opened", self.on_window_open)
        self.event_bus.subscribe("learning.window.closed", self.on_window_close)
        self.event_bus.subscribe("learning.window.back.pressure", self.on_back_pressure)
        
        # Legacy events
        self.event_bus.subscribe("document.processed", self._on_document_processed)
        self.event_bus.subscribe("vector.gradient.updated", self._on_vector_gradient_updated)
        
        # Initialize active windows dictionary
        self.active_windows = {}
        
        # Initialize pattern cache
        self.pattern_cache = {}
        
        self.initialized = True
        logger.info("VectorTonicPersistenceConnector initialized")
    
    def connect_to_integrator(self, integrator: VectorTonicWindowIntegrator):
        """Connect to a VectorTonicWindowIntegrator.
        
        This ensures that events from the integrator are properly captured and persisted.
        
        Args:
            integrator: The VectorTonicWindowIntegrator to connect to
        """
        if not self.initialized:
            self.initialize()
            
        # Ensure the integrator is initialized
        if not integrator.initialized:
            integrator.initialize()
            
        # Register as an observer for learning windows, patterns, and field states
        integrator.register_learning_window_observer(self)
        integrator.register_pattern_observer(self)
        integrator.register_field_observer(self)
            
        logger.info(f"Connected to VectorTonicWindowIntegrator {id(integrator)}")
        
        # Publish a connection event
        self.event_bus.publish(Event.create(
            "persistence.connected",
            {
                "integrator_id": id(integrator),
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    def process_document(self, document: Dict[str, Any]) -> str:
        """Process a document through the persistence integration.
        
        Args:
            document: The document to process
            
        Returns:
            The document ID
        """
        if not self.initialized:
            self.initialize()
            
        return self.persistence_integration.process_document(document)
    
    def _on_document_processed(self, event: Event):
        """Handle document processed events.
        
        Args:
            event: The document processed event
        """
        document_data = event.data
        if not document_data or "document_id" not in document_data:
            logger.warning("Invalid document processed data in event")
            return
            
        document_id = document_data["document_id"]
        logger.info(f"Document processed: {document_id}")
        
        # Extract entities and relationships if available
        if "entities" in document_data:
            for entity in document_data["entities"]:
                # Publish entity detected event
                self.event_bus.publish(Event.create(
                    "entity.detected",
                    {
                        "entity_id": entity.get("id"),
                        "entity_type": entity.get("type"),
                        "entity_text": entity.get("text"),
                        "document_id": document_id,
                        "confidence": entity.get("confidence", 0.5),
                        "timestamp": datetime.now().isoformat()
                    },
                    source="document_processor"
                ))
                
        if "relationships" in document_data:
            for relationship in document_data["relationships"]:
                # Publish relationship detected event
                self.event_bus.publish(Event.create(
                    "relationship.detected",
                    {
                        "source": relationship.get("source"),
                        "predicate": relationship.get("predicate"),
                        "target": relationship.get("target"),
                        "document_id": document_id,
                        "confidence": relationship.get("confidence", 0.5),
                        "timestamp": datetime.now().isoformat()
                    },
                    source="document_processor"
                ))
    
    def _on_vector_gradient_updated(self, event: Event):
        """Handle vector gradient updated events.
        
        Args:
            event: The vector gradient updated event
        """
        gradient_data = event.data.get("gradient", {})
        if not gradient_data:
            logger.warning("No gradient data found in event")
            return
            
        logger.info("Vector gradient updated")
        
        # Check if this gradient update should trigger a field state update
        if "field_state_id" in gradient_data:
            field_state_id = gradient_data["field_state_id"]
            
            # Get current field state metrics
            metrics = gradient_data.get("metrics", {})
            
            # Publish field state updated event if metrics are available
            if metrics:
                self.event_bus.publish(Event.create(
                    "field.state.updated",
                    {
                        "field_state": {
                            "id": field_state_id,
                            "metrics": metrics
                        },
                        "timestamp": datetime.now().isoformat()
                    },
                    source="gradient_monitor"
                ))
    
    # LearningWindowObserverInterface implementation
    def on_window_state_change(self, window_id: str, previous_state: LearningWindowState, 
                              new_state: LearningWindowState, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when a learning window changes state.
        
        Args:
            window_id: The ID of the learning window.
            previous_state: The previous state of the window.
            new_state: The new state of the window.
            metadata: Optional metadata about the state change.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Learning window {window_id} state changed: {previous_state.value} -> {new_state.value}")
        
        # Update active windows tracking
        self.active_windows[window_id] = {
            "state": new_state,
            "last_updated": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Publish window state change event
        self.event_bus.publish(Event.create(
            "learning.window.state.changed",
            {
                "window_id": window_id,
                "previous_state": previous_state.value,
                "new_state": new_state.value,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    def on_window_open(self, window_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when a learning window opens.
        
        Args:
            window_id: The ID of the learning window.
            metadata: Optional metadata about the window.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Learning window {window_id} opened")
        
        # Update active windows tracking
        self.active_windows[window_id] = {
            "state": LearningWindowState.OPEN,
            "opened_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Publish window opened event
        self.event_bus.publish(Event.create(
            "learning.window.opened",
            {
                "window_id": window_id,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    def on_window_close(self, window_id: str, patterns_detected: Dict[str, Any], 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when a learning window closes.
        
        Args:
            window_id: The ID of the learning window.
            patterns_detected: Patterns detected during the window.
            metadata: Optional metadata about the window.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Learning window {window_id} closed with {len(patterns_detected)} patterns")
        
        # Update active windows tracking
        if window_id in self.active_windows:
            self.active_windows[window_id]["state"] = LearningWindowState.CLOSED
            self.active_windows[window_id]["closed_at"] = datetime.now().isoformat()
            self.active_windows[window_id]["patterns_detected"] = patterns_detected
        
        # Process detected patterns
        for pattern_id, pattern_data in patterns_detected.items():
            # Cache the pattern
            self.pattern_cache[pattern_id] = pattern_data
            
            # Publish pattern detected event
            self.event_bus.publish(Event.create(
                "pattern.detected",
                {
                    "pattern_id": pattern_id,
                    "pattern_data": pattern_data,
                    "window_id": window_id,
                    "timestamp": datetime.now().isoformat()
                },
                source="learning_window"
            ))
        
        # Extract field state if available
        if metadata and "field_state" in metadata:
            field_state = metadata["field_state"]
            
            # Publish field state updated event
            self.event_bus.publish(Event.create(
                "field.state.updated",
                {
                    "field_state": field_state,
                    "window_id": window_id,
                    "timestamp": datetime.now().isoformat()
                },
                source="learning_window"
            ))
    
    def on_back_pressure(self, window_id: str, pressure_level: float, 
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when back pressure is detected in a learning window.
        
        Args:
            window_id: The ID of the learning window.
            pressure_level: The level of back pressure (0.0 to 1.0).
            metadata: Optional metadata about the back pressure.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Learning window {window_id} back pressure: {pressure_level}")
        
        # Update active windows tracking
        if window_id in self.active_windows:
            self.active_windows[window_id]["pressure_level"] = pressure_level
            self.active_windows[window_id]["pressure_detected_at"] = datetime.now().isoformat()
        
        # Publish back pressure event
        self.event_bus.publish(Event.create(
            "learning.window.back.pressure",
            {
                "window_id": window_id,
                "pressure_level": pressure_level,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    # PatternObserverInterface implementation
    def on_pattern_detected(self, pattern_id: str, pattern_data: Dict[str, Any], 
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when a new pattern is detected.
        
        Args:
            pattern_id: The ID of the detected pattern.
            pattern_data: The data of the detected pattern.
            metadata: Optional metadata about the detection.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Pattern detected: {pattern_id}")
        
        # Cache the pattern
        self.pattern_cache[pattern_id] = pattern_data
        
        # Persist the pattern if a repository is available
        if self.pattern_repository:
            try:
                # Save the pattern to the repository
                self.pattern_repository.save(pattern_data)
                logger.info(f"Pattern {pattern_id} persisted successfully")
            except Exception as e:
                logger.error(f"Failed to persist pattern {pattern_id}: {str(e)}")
        
        # Publish pattern detected event
        self.event_bus.publish(Event.create(
            "pattern.detected",
            {
                "pattern_id": pattern_id,
                "pattern_data": pattern_data,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    def on_pattern_evolution(self, pattern_id: str, previous_state: Dict[str, Any], 
                            new_state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when a pattern evolves.
        
        Args:
            pattern_id: The ID of the evolving pattern.
            previous_state: The previous state of the pattern.
            new_state: The new state of the pattern.
            metadata: Optional metadata about the evolution.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Pattern evolved: {pattern_id}")
        
        # Update pattern cache
        self.pattern_cache[pattern_id] = new_state
        
        # Persist the pattern evolution if a repository is available
        if self.pattern_repository:
            try:
                # Save the updated pattern to the repository
                self.pattern_repository.save(new_state)
                logger.info(f"Pattern evolution {pattern_id} persisted successfully")
            except Exception as e:
                logger.error(f"Failed to persist pattern evolution {pattern_id}: {str(e)}")
        
        # Publish pattern evolved event
        self.event_bus.publish(Event.create(
            "pattern.evolved",
            {
                "pattern_id": pattern_id,
                "previous_state": previous_state,
                "new_state": new_state,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    def on_pattern_quality_change(self, pattern_id: str, previous_quality: Dict[str, float], 
                                 new_quality: Dict[str, float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when a pattern's quality metrics change.
        
        Args:
            pattern_id: The ID of the pattern.
            previous_quality: The previous quality metrics.
            new_quality: The new quality metrics.
            metadata: Optional metadata about the quality change.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Pattern quality changed: {pattern_id}")
        
        # Update pattern cache if available
        if pattern_id in self.pattern_cache:
            self.pattern_cache[pattern_id]["quality"] = new_quality
            
            # Persist the updated pattern if a repository is available
            if self.pattern_repository:
                try:
                    # Save the updated pattern to the repository
                    self.pattern_repository.save(self.pattern_cache[pattern_id])
                    logger.info(f"Pattern quality change {pattern_id} persisted successfully")
                except Exception as e:
                    logger.error(f"Failed to persist pattern quality change {pattern_id}: {str(e)}")
        
        # Publish pattern quality changed event
        self.event_bus.publish(Event.create(
            "pattern.quality.changed",
            {
                "pattern_id": pattern_id,
                "previous_quality": previous_quality,
                "new_quality": new_quality,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    def on_pattern_relationship_detected(self, source_id: str, target_id: str, 
                                        relationship_data: Dict[str, Any], 
                                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when a relationship between patterns is detected.
        
        Args:
            source_id: The ID of the source pattern.
            target_id: The ID of the target pattern.
            relationship_data: The data of the relationship.
            metadata: Optional metadata about the relationship.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Pattern relationship detected: {source_id} -> {target_id}")
        
        # Persist the relationship if a repository is available
        if self.relationship_repository:
            try:
                # Ensure relationship data has source and target IDs
                relationship_data["source_id"] = source_id
                relationship_data["target_id"] = target_id
                
                # Save the relationship to the repository
                self.relationship_repository.save(relationship_data)
                logger.info(f"Relationship {source_id} -> {target_id} persisted successfully")
            except Exception as e:
                logger.error(f"Failed to persist relationship {source_id} -> {target_id}: {str(e)}")
        
        # Publish relationship detected event
        self.event_bus.publish(Event.create(
            "pattern.relationship.detected",
            {
                "source_id": source_id,
                "target_id": target_id,
                "relationship_data": relationship_data,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    def on_pattern_merge(self, merged_pattern_id: str, source_pattern_ids: List[str], 
                        merge_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when patterns are merged.
        
        Args:
            merged_pattern_id: The ID of the merged pattern.
            source_pattern_ids: The IDs of the source patterns.
            merge_data: Data about the merge.
            metadata: Optional metadata about the merge.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Patterns merged: {source_pattern_ids} -> {merged_pattern_id}")
        
        # Cache the merged pattern
        self.pattern_cache[merged_pattern_id] = merge_data.get("merged_pattern", {})
        
        # Persist the merged pattern if a repository is available
        if self.pattern_repository and "merged_pattern" in merge_data:
            try:
                # Save the merged pattern to the repository
                self.pattern_repository.save(merge_data["merged_pattern"])
                logger.info(f"Merged pattern {merged_pattern_id} persisted successfully")
            except Exception as e:
                logger.error(f"Failed to persist merged pattern {merged_pattern_id}: {str(e)}")
        
        # Publish pattern merged event
        self.event_bus.publish(Event.create(
            "pattern.merged",
            {
                "merged_pattern_id": merged_pattern_id,
                "source_pattern_ids": source_pattern_ids,
                "merge_data": merge_data,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    def on_pattern_split(self, source_pattern_id: str, result_pattern_ids: List[str], 
                        split_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when a pattern splits.
        
        Args:
            source_pattern_id: The ID of the source pattern.
            result_pattern_ids: The IDs of the resulting patterns.
            split_data: Data about the split.
            metadata: Optional metadata about the split.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Pattern split: {source_pattern_id} -> {result_pattern_ids}")
        
        # Persist the resulting patterns if a repository is available
        if self.pattern_repository and "result_patterns" in split_data:
            for pattern_id, pattern_data in split_data["result_patterns"].items():
                # Cache the pattern
                self.pattern_cache[pattern_id] = pattern_data
                
                try:
                    # Save the pattern to the repository
                    self.pattern_repository.save(pattern_data)
                    logger.info(f"Split result pattern {pattern_id} persisted successfully")
                except Exception as e:
                    logger.error(f"Failed to persist split result pattern {pattern_id}: {str(e)}")
        
        # Publish pattern split event
        self.event_bus.publish(Event.create(
            "pattern.split",
            {
                "source_pattern_id": source_pattern_id,
                "result_pattern_ids": result_pattern_ids,
                "split_data": split_data,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    # FieldObserverInterface implementation
    def on_field_state_change(self, field_id: str, previous_state: Dict[str, Any], 
                             new_state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when the field state changes.
        
        Args:
            field_id: The ID of the field.
            previous_state: The previous state of the field.
            new_state: The new state of the field.
            metadata: Optional metadata about the state change.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Field state changed: {field_id}")
        
        # Persist the field state if a repository is available
        if self.field_state_repository:
            try:
                # Ensure field state has an ID
                new_state["id"] = field_id
                
                # Save the field state to the repository
                self.field_state_repository.save(new_state)
                logger.info(f"Field state {field_id} persisted successfully")
            except Exception as e:
                logger.error(f"Failed to persist field state {field_id}: {str(e)}")
        
        # Publish field state changed event
        self.event_bus.publish(Event.create(
            "field.state.changed",
            {
                "field_id": field_id,
                "previous_state": previous_state,
                "new_state": new_state,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    def on_field_coherence_change(self, field_id: str, previous_coherence: float, 
                                 new_coherence: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when the field coherence changes.
        
        Args:
            field_id: The ID of the field.
            previous_coherence: The previous coherence value.
            new_coherence: The new coherence value.
            metadata: Optional metadata about the coherence change.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Field coherence changed: {field_id} ({previous_coherence} -> {new_coherence})")
        
        # Publish field coherence changed event
        self.event_bus.publish(Event.create(
            "field.coherence.changed",
            {
                "field_id": field_id,
                "previous_coherence": previous_coherence,
                "new_coherence": new_coherence,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    def on_field_stability_change(self, field_id: str, previous_stability: float, 
                                 new_stability: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when the field stability changes.
        
        Args:
            field_id: The ID of the field.
            previous_stability: The previous stability value.
            new_stability: The new stability value.
            metadata: Optional metadata about the stability change.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Field stability changed: {field_id} ({previous_stability} -> {new_stability})")
        
        # Publish field stability changed event
        self.event_bus.publish(Event.create(
            "field.stability.changed",
            {
                "field_id": field_id,
                "previous_stability": previous_stability,
                "new_stability": new_stability,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    def on_density_center_shift(self, field_id: str, previous_centers: List[Dict[str, Any]], 
                               new_centers: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when density centers shift.
        
        Args:
            field_id: The ID of the field.
            previous_centers: The previous density centers.
            new_centers: The new density centers.
            metadata: Optional metadata about the shift.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Density centers shifted: {field_id} ({len(previous_centers)} -> {len(new_centers)})")
        
        # If we have a field state repository and the field state is available, update it
        if self.field_state_repository:
            try:
                # Get current field state
                field_state = self.field_state_repository.find_by_id(field_id)
                if field_state:
                    # Update density centers
                    field_state["density_centers"] = new_centers
                    
                    # Save updated field state
                    self.field_state_repository.save(field_state)
                    logger.info(f"Updated density centers for field {field_id}")
            except Exception as e:
                logger.error(f"Failed to update density centers for field {field_id}: {str(e)}")
        
        # Publish density centers shifted event
        self.event_bus.publish(Event.create(
            "field.density.centers.shifted",
            {
                "field_id": field_id,
                "previous_centers": previous_centers,
                "new_centers": new_centers,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    def on_eigenspace_change(self, field_id: str, previous_eigenspace: Dict[str, Any], 
                            new_eigenspace: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when the eigenspace changes.
        
        Args:
            field_id: The ID of the field.
            previous_eigenspace: The previous eigenspace properties.
            new_eigenspace: The new eigenspace properties.
            metadata: Optional metadata about the change.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Eigenspace changed: {field_id}")
        
        # If we have a field state repository and the field state is available, update it
        if self.field_state_repository:
            try:
                # Get current field state
                field_state = self.field_state_repository.find_by_id(field_id)
                if field_state:
                    # Update eigenspace properties
                    field_state["eigenvalues"] = new_eigenspace.get("eigenvalues", [])
                    field_state["eigenvectors"] = new_eigenspace.get("eigenvectors", [])
                    field_state["effective_dimensionality"] = new_eigenspace.get("effective_dimensionality", 0)
                    
                    # Save updated field state
                    self.field_state_repository.save(field_state)
                    logger.info(f"Updated eigenspace for field {field_id}")
            except Exception as e:
                logger.error(f"Failed to update eigenspace for field {field_id}: {str(e)}")
        
        # Publish eigenspace changed event
        self.event_bus.publish(Event.create(
            "field.eigenspace.changed",
            {
                "field_id": field_id,
                "previous_eigenspace": previous_eigenspace,
                "new_eigenspace": new_eigenspace,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    def on_topology_change(self, field_id: str, previous_topology: Dict[str, Any], 
                          new_topology: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when the field topology changes.
        
        Args:
            field_id: The ID of the field.
            previous_topology: The previous topology.
            new_topology: The new topology.
            metadata: Optional metadata about the change.
        """
        if not self.initialized:
            self.initialize()
            
        logger.info(f"Topology changed: {field_id}")
        
        # Persist the topology if a repository is available
        if self.topology_repository:
            try:
                # Ensure topology has an ID
                new_topology["id"] = new_topology.get("id", str(uuid.uuid4()))
                new_topology["field_id"] = field_id
                
                # Save the topology to the repository
                self.topology_repository.save(new_topology)
                logger.info(f"Topology for field {field_id} persisted successfully")
            except Exception as e:
                logger.error(f"Failed to persist topology for field {field_id}: {str(e)}")
        
        # Publish topology changed event
        self.event_bus.publish(Event.create(
            "field.topology.changed",
            {
                "field_id": field_id,
                "previous_topology": previous_topology,
                "new_topology": new_topology,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_connector"
        ))
    
    # Legacy event handler for compatibility
    def _on_learning_window_closed(self, event: Event):
        """Handle learning window closed events.
        
        Args:
            event: The learning window closed event
        """
        window_data = event.data
        if not window_data or "window_id" not in window_data:
            logger.warning("Invalid learning window data in event")
            return
            
        window_id = window_data["window_id"]
        patterns = window_data.get("patterns", [])
        
        # Convert to the format expected by on_window_close
        patterns_dict = {}
        for pattern in patterns:
            if "id" in pattern:
                patterns_dict[pattern["id"]] = pattern

def on_field_stability_change(self, field_id: str, previous_stability: float, 
                             new_stability: float, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Called when the field stability changes.
    
    Args:
        field_id: The ID of the field.
        previous_stability: The previous stability value.
        new_stability: The new stability value.
        metadata: Optional metadata about the stability change.
    """
    if not self.initialized:
        self.initialize()
            
    logger.info(f"Field stability changed: {field_id} ({previous_stability} -> {new_stability})")
        
    # Publish field stability changed event
    self.event_bus.publish(Event.create(
        "field.stability.changed",
        {
            "field_id": field_id,
            "previous_stability": previous_stability,
            "new_stability": new_stability,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        },
        source="persistence_connector"
    ))

def on_density_center_shift(self, field_id: str, previous_centers: List[Dict[str, Any]], 
                           new_centers: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> None:
    """Called when density centers shift.
    
    Args:
        field_id: The ID of the field.
        previous_centers: The previous density centers.
        new_centers: The new density centers.
        metadata: Optional metadata about the shift.
    """
    if not self.initialized:
        self.initialize()
            
    logger.info(f"Density centers shifted: {field_id} ({len(previous_centers)} -> {len(new_centers)})")
        
    # If we have a field state repository and the field state is available, update it
    if self.field_state_repository:
        try:
            # Get current field state
            field_state = self.field_state_repository.find_by_id(field_id)
            if field_state:
                # Update density centers
                field_state["density_centers"] = new_centers
                
                # Save updated field state
                self.field_state_repository.save(field_state)
                logger.info(f"Updated density centers for field {field_id}")
        except Exception as e:
            logger.error(f"Failed to update density centers for field {field_id}: {str(e)}")
        
    # Publish density centers shifted event
    self.event_bus.publish(Event.create(
        "field.density.centers.shifted",
        {
            "field_id": field_id,
            "previous_centers": previous_centers,
            "new_centers": new_centers,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        },
        source="persistence_connector"
    ))

def on_eigenspace_change(self, field_id: str, previous_eigenspace: Dict[str, Any], 
                        new_eigenspace: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
    """Called when the eigenspace changes.
    
    Args:
        field_id: The ID of the field.
        previous_eigenspace: The previous eigenspace properties.
        new_eigenspace: The new eigenspace properties.
        metadata: Optional metadata about the change.
    """
    if not self.initialized:
        self.initialize()
            
    logger.info(f"Eigenspace changed: {field_id}")
        
    # If we have a field state repository and the field state is available, update it
    if self.field_state_repository:
        try:
            # Get current field state
            field_state = self.field_state_repository.find_by_id(field_id)
            if field_state:
                # Update eigenspace properties
                field_state["eigenvalues"] = new_eigenspace.get("eigenvalues", [])
                field_state["eigenvectors"] = new_eigenspace.get("eigenvectors", [])
                field_state["effective_dimensionality"] = new_eigenspace.get("effective_dimensionality", 0)
                
                # Save updated field state
                self.field_state_repository.save(field_state)
                logger.info(f"Updated eigenspace for field {field_id}")
        except Exception as e:
            logger.error(f"Failed to update eigenspace for field {field_id}: {str(e)}")
        
    # Publish eigenspace changed event
    self.event_bus.publish(Event.create(
        "field.eigenspace.changed",
        {
            "field_id": field_id,
            "previous_eigenspace": previous_eigenspace,
            "new_eigenspace": new_eigenspace,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        },
        source="persistence_connector"
    ))

def on_topology_change(self, field_id: str, previous_topology: Dict[str, Any], 
                      new_topology: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
    """Called when the field topology changes.
    
    Args:
        field_id: The ID of the field.
        previous_topology: The previous topology.
        new_topology: The new topology.
        metadata: Optional metadata about the change.
    """
    if not self.initialized:
        self.initialize()
            
    logger.info(f"Topology changed: {field_id}")
        
    # Persist the topology if a repository is available
    if self.topology_repository:
        try:
            # Ensure topology has an ID
            new_topology["id"] = new_topology.get("id", str(uuid.uuid4()))
            new_topology["field_id"] = field_id
            
            # Save the topology to the repository
            self.topology_repository.save(new_topology)
            logger.info(f"Topology for field {field_id} persisted successfully")
        except Exception as e:
            logger.error(f"Failed to persist topology for field {field_id}: {str(e)}")
        
    # Publish topology changed event
    self.event_bus.publish(Event.create(
        "field.topology.changed",
        {
            "field_id": field_id,
            "previous_topology": previous_topology,
            "new_topology": new_topology,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        },
        source="persistence_connector"
    ))

# Legacy event handler for compatibility
def _on_learning_window_closed(self, event: Event):
    """Handle learning window closed events.
    
    Args:
        event: The learning window closed event
    """
    window_data = event.data
    if not window_data or "window_id" not in window_data:
        logger.warning("Invalid learning window data in event")
        return
            
    window_id = window_data["window_id"]
    patterns = window_data.get("patterns", [])
        
    # Convert to the format expected by on_window_close
    patterns_dict = {}
    for pattern in patterns:
        if "id" in pattern:
            patterns_dict[pattern["id"]] = pattern
        
    # Call the observer method
    self.on_window_close(window_id, patterns_dict, window_data)


def create_connector(event_bus=None, db=None, **kwargs):
    """Create a VectorTonicPersistenceConnector instance.
    
    Args:
        event_bus: The event bus to use. If None, a new LocalEventBus will be created.
        db: The database connection to use.
        **kwargs: Additional keyword arguments to pass to the connector constructor.
            These can include:
            - field_state_repository: A field state repository instance.
            - pattern_repository: A pattern repository instance.
            - relationship_repository: A relationship repository instance.
            - topology_repository: A topology repository instance.
        
    Returns:
        A VectorTonicPersistenceConnector instance.
    """
    if event_bus is None:
        event_bus = LocalEventBus()
    
    # Create repositories if not provided
    repositories = {}
    if db is not None and not any(repo in kwargs for repo in [
        'field_state_repository', 'pattern_repository', 
        'relationship_repository', 'topology_repository'
    ]):
        repositories = create_repositories(db)
    
    # Create connector with repositories
    connector = VectorTonicPersistenceConnector(
        event_bus=event_bus, 
        db=db,
        field_state_repository=kwargs.get('field_state_repository', repositories.get('field_state_repository')),
        pattern_repository=kwargs.get('pattern_repository', repositories.get('pattern_repository')),
        relationship_repository=kwargs.get('relationship_repository', repositories.get('relationship_repository')),
        topology_repository=kwargs.get('topology_repository', repositories.get('topology_repository'))
    )
    
    # Initialize the connector
    connector.initialize()
    
    return connector
