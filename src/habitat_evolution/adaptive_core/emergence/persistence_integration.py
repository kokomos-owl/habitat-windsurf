"""
Persistence Integration for Vector-Tonic Window System.

This module provides integration between the vector-tonic-window system and 
the ArangoDB persistence layer, ensuring that semantic relationships and patterns
are correctly processed, persisted, and made queryable.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from src.habitat_evolution.core.services.event_bus import Event, LocalEventBus
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.pattern_repository import PatternRepository
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.field_state_repository import TonicHarmonicFieldStateRepository
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.predicate_relationship_repository import PredicateRelationshipRepository
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.boundary_repository import BoundaryRepository
from src.habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager

logger = logging.getLogger(__name__)

class AdaptiveIDRepository:
    """Repository for persisting AdaptiveID objects in ArangoDB."""
    
    def __init__(self, db=None):
        """Initialize the repository with a database connection.
        
        Args:
            db: Optional database connection. If not provided, a new connection will be created.
        """
        self.db = db or ArangoDBConnectionManager().get_db()
        self.collection_name = "AdaptiveID"
        self.version_collection_name = "AdaptiveIDVersion"
        
    def initialize(self):
        """Initialize collections needed for AdaptiveID persistence."""
        # Create AdaptiveID collection if it doesn't exist
        if not self.db.has_collection(self.collection_name):
            self.db.create_collection(self.collection_name)
            
        # Create version collection if it doesn't exist
        if not self.db.has_collection(self.version_collection_name):
            self.db.create_collection(self.version_collection_name)
    
    def save(self, adaptive_id: AdaptiveID) -> str:
        """Save an AdaptiveID object to the database.
        
        Args:
            adaptive_id: The AdaptiveID object to save
            
        Returns:
            The database ID of the saved object
        """
        # Create document for the AdaptiveID
        doc = {
            "_key": adaptive_id.id,
            "base_concept": adaptive_id.base_concept,
            "creator_id": adaptive_id.creator_id,
            "weight": adaptive_id.weight,
            "confidence": adaptive_id.confidence,
            "uncertainty": adaptive_id.uncertainty,
            "user_interactions": adaptive_id.user_interactions,
            "current_version": adaptive_id.current_version,
            "temporal_context": adaptive_id.temporal_context,
            "spatial_context": adaptive_id.spatial_context,
            "metadata": adaptive_id.metadata
        }
        
        # Save the AdaptiveID document
        collection = self.db.collection(self.collection_name)
        
        # Check if document already exists
        if collection.has(adaptive_id.id):
            collection.replace(adaptive_id.id, doc)
            result = {"_id": f"{self.collection_name}/{adaptive_id.id}"}
        else:
            result = collection.insert(doc, return_new=True)
        
        # Save all versions
        self._save_versions(adaptive_id)
        
        return result["_id"]
    
    def _save_versions(self, adaptive_id: AdaptiveID):
        """Save all versions of an AdaptiveID.
        
        Args:
            adaptive_id: The AdaptiveID object whose versions to save
        """
        version_collection = self.db.collection(self.version_collection_name)
        
        for version_id, version in adaptive_id.versions.items():
            version_doc = {
                "_key": version_id,
                "adaptive_id": adaptive_id.id,
                "data": version.data,
                "timestamp": version.timestamp,
                "origin": version.origin
            }
            
            # Insert or replace the version
            if version_collection.has(version_id):
                version_collection.replace(version_id, version_doc)
            else:
                version_collection.insert(version_doc)
    
    def find_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Find an AdaptiveID by its ID.
        
        Args:
            id: The ID of the AdaptiveID to find
            
        Returns:
            The AdaptiveID document or None if not found
        """
        collection = self.db.collection(self.collection_name)
        try:
            return collection.get(id)
        except:
            return None
    
    def find_by_base_concept(self, base_concept: str) -> List[Dict[str, Any]]:
        """Find AdaptiveIDs by base concept.
        
        Args:
            base_concept: The base concept to search for
            
        Returns:
            List of matching AdaptiveID documents
        """
        aql = "FOR doc IN AdaptiveID FILTER doc.base_concept == @base_concept RETURN doc"
        cursor = self.db.aql.execute(aql, bind_vars={"base_concept": base_concept})
        return list(cursor)


class PatternPersistenceService:
    """Service for persisting patterns detected by the vector-tonic-window system."""
    
    def __init__(self, event_bus, db=None):
        """Initialize the pattern persistence service.
        
        Args:
            event_bus: Event bus for subscribing to events
            db: Optional database connection. If not provided, a new connection will be created.
        """
        self.event_bus = event_bus
        self.db = db or ArangoDBConnectionManager().get_db()
        self.adaptive_id_repo = AdaptiveIDRepository(self.db)
        self.pattern_repo = PatternRepository()
        
    def initialize(self):
        """Initialize the service and subscribe to events."""
        # Initialize repositories
        self.adaptive_id_repo.initialize()
        
        # Subscribe to pattern events
        self.event_bus.subscribe("pattern.detected", self._on_pattern_detected)
        self.event_bus.subscribe("pattern.evolved", self._on_pattern_evolved)
        self.event_bus.subscribe("pattern.semantic_boundary", self._on_semantic_boundary)
        
        logger.info("Pattern persistence service initialized")
        
    def _on_pattern_detected(self, event):
        """Handle pattern detected events.
        
        Args:
            event: The pattern detected event
        """
        pattern_data = event.data
        if not pattern_data or "pattern_id" not in pattern_data:
            logger.warning("Invalid pattern data in event")
            return
            
        logger.info(f"Processing pattern detection: {pattern_data['pattern_id']}")
            
        # Create an AdaptiveID for the pattern
        pattern_id = pattern_data["pattern_id"]
        base_concept = pattern_id.replace("pattern_", "")
        creator_id = pattern_data.get("source", "pattern_detector")
        confidence = pattern_data.get("confidence", 0.5)
        
        # Create the AdaptiveID
        adaptive_id = AdaptiveID(
            base_concept=base_concept,
            creator_id=creator_id,
            confidence=confidence
        )
        
        # Add pattern data to the AdaptiveID
        if "pattern_data" in pattern_data:
            for key, value in pattern_data["pattern_data"].items():
                adaptive_id.set_property(key, value, "pattern_detection")
        
        # Add temporal context if available
        if "timestamp" in pattern_data:
            adaptive_id.set_temporal_context(
                "detection_time", 
                pattern_data["timestamp"], 
                "pattern_detection"
            )
        
        # Save the AdaptiveID
        adaptive_id_id = self.adaptive_id_repo.save(adaptive_id)
        
        # Log the persistence
        logger.info(f"Persisted pattern {pattern_id} as AdaptiveID {adaptive_id.id}")
        
    def _on_pattern_evolved(self, event):
        """Handle pattern evolved events.
        
        Args:
            event: The pattern evolved event
        """
        evolution_data = event.data
        if not evolution_data or "pattern_id" not in evolution_data:
            logger.warning("Invalid pattern evolution data in event")
            return
            
        pattern_id = evolution_data["pattern_id"]
        logger.info(f"Processing pattern evolution: {pattern_id}")
        
        # Find existing AdaptiveID documents for this pattern
        existing_ids = self.adaptive_id_repo.find_by_base_concept(
            pattern_id.replace("pattern_", "")
        )
        
        if not existing_ids:
            logger.warning(f"Cannot evolve pattern {pattern_id}: AdaptiveID not found")
            return
            
        # Get the first matching AdaptiveID
        existing_id_doc = existing_ids[0]
        
        # Create a new version directly in the database
        version_collection = self.db.collection("AdaptiveIDVersion")
        version_id = str(uuid.uuid4())
        
        version_doc = {
            "_key": version_id,
            "adaptive_id": existing_id_doc["_id"],
            "data": evolution_data.get("new_state", {}),
            "timestamp": datetime.now().isoformat(),
            "origin": "pattern_evolution"
        }
        
        version_collection.insert(version_doc)
        
        # Update the current version in the AdaptiveID document
        adaptive_id_collection = self.db.collection("AdaptiveID")
        adaptive_id_collection.update(
            existing_id_doc["_key"],
            {
                "current_version": version_id,
                "metadata.last_modified": datetime.now().isoformat(),
                "metadata.version_count": existing_id_doc["metadata"]["version_count"] + 1
            }
        )
        
        logger.info(f"Updated pattern {pattern_id} with new version {version_id}")
        
    def _on_semantic_boundary(self, event):
        """Handle semantic boundary events.
        
        Args:
            event: The semantic boundary event
        """
        boundary_data = event.data
        if not boundary_data or "pattern_id" not in boundary_data:
            logger.warning("Invalid semantic boundary data in event")
            return
            
        pattern_id = boundary_data["pattern_id"]
        logger.info(f"Processing semantic boundary crossing: {pattern_id}")
        
        # Find existing AdaptiveID documents for this pattern
        existing_ids = self.adaptive_id_repo.find_by_base_concept(
            pattern_id.replace("pattern_", "")
        )
        
        if not existing_ids:
            logger.warning(f"Cannot record boundary for pattern {pattern_id}: AdaptiveID not found")
            return
            
        # Get the first matching AdaptiveID
        existing_id_doc = existing_ids[0]
        
        # Add boundary crossing as temporal context
        boundary_repo = BoundaryRepository()
        boundary_id = str(uuid.uuid4())
        
        boundary_doc = {
            "_key": boundary_id,
            "pattern_id": pattern_id,
            "adaptive_id": existing_id_doc["_id"],
            "from_state": boundary_data.get("from_state", {}),
            "to_state": boundary_data.get("to_state", {}),
            "boundary_type": boundary_data.get("boundary_type", "unknown"),
            "field_state_id": boundary_data.get("field_state_id", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save boundary crossing
        boundary_repo.save(boundary_doc)
        
        logger.info(f"Recorded semantic boundary crossing for pattern {pattern_id}")


class FieldStatePersistenceService:
    """Service for persisting field state metrics."""
    
    def __init__(self, event_bus, db=None):
        """Initialize the field state persistence service.
        
        Args:
            event_bus: Event bus for subscribing to events
            db: Optional database connection. If not provided, a new connection will be created.
        """
        self.event_bus = event_bus
        self.db = db or ArangoDBConnectionManager().get_db()
        self.field_state_repo = TonicHarmonicFieldStateRepository()
        
    def initialize(self):
        """Initialize the service and subscribe to events."""
        # Subscribe to field state events
        self.event_bus.subscribe("field.state.updated", self._on_field_state_updated)
        
        logger.info("Field state persistence service initialized")
        
    def _on_field_state_updated(self, event):
        """Handle field state updated events.
        
        Args:
            event: The field state updated event
        """
        field_state_data = event.data.get("field_state", {})
        if not field_state_data:
            logger.warning("No field state data found in event")
            return
            
        logger.info(f"Processing field state update: {field_state_data.get('id', 'unknown')}")
            
        # Create field state document
        field_state_doc = {
            "id": field_state_data.get("id", str(uuid.uuid4())),
            "timestamp": datetime.now().isoformat(),
            "density": field_state_data.get("density", 0.5),
            "turbulence": field_state_data.get("turbulence", 0.3),
            "coherence": field_state_data.get("coherence", 0.7),
            "stability": field_state_data.get("stability", 0.7)
        }
        
        # Extract additional metrics if available
        if "metrics" in field_state_data:
            metrics = field_state_data["metrics"]
            field_state_doc.update(metrics)
            
        # Save to database
        self.field_state_repo.save(field_state_doc)
        
        logger.info(f"Persisted field state update: {field_state_doc['id']}")


class RelationshipPersistenceService:
    """Service for persisting relationships between AdaptiveIDs."""
    
    def __init__(self, event_bus, db=None):
        """Initialize the relationship persistence service.
        
        Args:
            event_bus: Event bus for subscribing to events
            db: Optional database connection. If not provided, a new connection will be created.
        """
        self.event_bus = event_bus
        self.db = db or ArangoDBConnectionManager().get_db()
        self.adaptive_id_repo = AdaptiveIDRepository(self.db)
        self.predicate_repo = PredicateRelationshipRepository(self.db)
        
    def initialize(self):
        """Initialize the service and subscribe to events."""
        # Initialize repositories
        self.adaptive_id_repo.initialize()
        
        # Subscribe to relationship events
        self.event_bus.subscribe("relationship.detected", self._on_relationship_detected)
        
        logger.info("Relationship persistence service initialized")
        
    def _on_relationship_detected(self, event):
        """Handle relationship detected events.
        
        Args:
            event: The relationship detected event
        """
        rel_data = event.data
        if not rel_data or "source" not in rel_data or "target" not in rel_data or "predicate" not in rel_data:
            logger.warning("Invalid relationship data in event")
            return
            
        logger.info(f"Processing relationship: {rel_data['source']} -{rel_data['predicate']}-> {rel_data['target']}")
            
        # Find or create AdaptiveIDs for source and target
        source_id = self._find_or_create_adaptive_id(rel_data["source"])
        target_id = self._find_or_create_adaptive_id(rel_data["target"])
        
        # Extract properties
        properties = {
            "confidence": rel_data.get("confidence", 0.5),
            "observation_count": rel_data.get("observation_count", 1)
        }
        
        # Add harmonic properties if available
        if "harmonic_properties" in rel_data:
            properties["harmonic_properties"] = rel_data["harmonic_properties"]
            
        # Save relationship using the PredicateRelationshipRepository
        edge_id = self.predicate_repo.save_relationship(
            f"AdaptiveID/{source_id}",
            rel_data["predicate"],
            f"AdaptiveID/{target_id}",
            properties
        )
        
        logger.info(f"Persisted relationship: {source_id} -{rel_data['predicate']}-> {target_id}")
        
    def _find_or_create_adaptive_id(self, concept: str) -> str:
        """Find an existing AdaptiveID for a concept or create a new one.
        
        Args:
            concept: The concept to find or create an AdaptiveID for
            
        Returns:
            The ID of the AdaptiveID
        """
        # Try to find existing AdaptiveID
        existing = self.adaptive_id_repo.find_by_base_concept(concept)
        
        if existing:
            return existing[0]["_id"].split("/")[1]
            
        # Create new AdaptiveID
        adaptive_id = AdaptiveID(
            base_concept=concept,
            creator_id="relationship_detector"
        )
        
        # Save and return ID
        saved_id = self.adaptive_id_repo.save(adaptive_id)
        return saved_id.split("/")[1]


class VectorTonicPersistenceIntegration:
    """Integration service for vector-tonic-window system and ArangoDB persistence."""
    
    def __init__(self, event_bus=None, db=None):
        """Initialize the integration service.
        
        Args:
            event_bus: Optional event bus. If not provided, a new event bus will be created.
            db: Optional database connection. If not provided, a new connection will be created.
        """
        self.event_bus = event_bus or LocalEventBus()
        self.db = db or ArangoDBConnectionManager().get_db()
        
        # Create persistence services
        self.pattern_service = PatternPersistenceService(self.event_bus, self.db)
        self.relationship_service = RelationshipPersistenceService(self.event_bus, self.db)
        self.field_state_service = FieldStatePersistenceService(self.event_bus, self.db)
        
    def initialize(self):
        """Initialize all persistence services."""
        self.pattern_service.initialize()
        self.relationship_service.initialize()
        self.field_state_service.initialize()
        
        logger.info("Vector-tonic persistence integration initialized")
        
    def process_document(self, document):
        """Process a document through the vector-tonic-window system.
        
        Args:
            document: The document to process
        """
        # Create an AdaptiveID for the document
        doc_id = AdaptiveID(
            base_concept=document.get("id", "document"),
            creator_id="document_processor"
        )
        
        # Add document content as a property
        if "content" in document:
            doc_id.set_property("content", document["content"], "document_processing")
            
        # Save the document AdaptiveID
        adaptive_id_repo = AdaptiveIDRepository(self.db)
        adaptive_id_repo.initialize()
        adaptive_id_repo.save(doc_id)
        
        # Publish document processing event
        self.event_bus.publish(Event(
            "document.processing",
            {
                "document_id": doc_id.id,
                "content": document.get("content", ""),
                "timestamp": datetime.now().isoformat()
            },
            source="persistence_integration"
        ))
        
        logger.info(f"Published document processing event for {doc_id.id}")
        
        return doc_id.id
