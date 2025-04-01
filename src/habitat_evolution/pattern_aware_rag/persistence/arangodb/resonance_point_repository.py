"""
ResonancePoint Repository for ArangoDB.

Handles persistence of ResonancePoint objects to ArangoDB, including
strength, stability, and contributing patterns.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime
import uuid
import logging

from src.habitat_evolution.adaptive_core.persistence.arangodb.base_repository import ArangoDBBaseRepository
from src.habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager
from src.habitat_evolution.pattern_aware_rag.topology.resonance import ResonancePoint

logger = logging.getLogger(__name__)

class ResonancePointRepository(ArangoDBBaseRepository):
    """
    Repository for persisting ResonancePoint objects to ArangoDB.
    
    This repository handles the serialization and deserialization of resonance points,
    including their strength, stability, and contributing patterns.
    """
    
    def __init__(self):
        """Initialize the repository with the appropriate collection name."""
        super().__init__()
        self.collection_name = "ResonancePoint"
        self.connection_manager = ArangoDBConnectionManager()
    
    def _to_document_properties(self, point: ResonancePoint) -> Dict[str, Any]:
        """
        Convert a ResonancePoint to ArangoDB document properties.
        
        Args:
            point: The resonance point to convert
            
        Returns:
            A dictionary of document properties
        """
        # Extract the core properties
        properties = {
            "_key": point.id,
            "id": point.id,
            "created_at": point.created_at.isoformat() if hasattr(point, 'created_at') else datetime.now().isoformat(),
            "updated_at": point.updated_at.isoformat() if hasattr(point, 'updated_at') else datetime.now().isoformat(),
            "strength": point.strength if hasattr(point, 'strength') else 0.0,
            "stability": point.stability if hasattr(point, 'stability') else 0.0,
            "attractor_radius": point.attractor_radius if hasattr(point, 'attractor_radius') else 0.0,
            "last_updated": point.last_updated.isoformat() if hasattr(point, 'last_updated') else datetime.now().isoformat(),
        }
        
        # Handle contributing pattern IDs
        if hasattr(point, 'contributing_pattern_ids') and point.contributing_pattern_ids:
            properties['contributing_pattern_ids'] = json.dumps(point.contributing_pattern_ids)
        else:
            properties['contributing_pattern_ids'] = json.dumps({})
        
        # Handle dimensional coordinates
        if hasattr(point, 'dimensional_coordinates') and point.dimensional_coordinates:
            properties['dimensional_coordinates'] = json.dumps(point.dimensional_coordinates)
        else:
            properties['dimensional_coordinates'] = json.dumps([])
        
        # Handle oscillatory properties
        if hasattr(point, 'oscillatory_properties') and point.oscillatory_properties:
            properties['oscillatory_properties'] = json.dumps(point.oscillatory_properties)
        else:
            properties['oscillatory_properties'] = json.dumps({})
        
        return properties
    
    def _dict_to_entity(self, properties: Dict[str, Any]) -> ResonancePoint:
        """
        Convert ArangoDB document properties to a ResonancePoint.
        
        Args:
            properties: The document properties to convert
            
        Returns:
            A ResonancePoint object
        """
        # Create a minimal resonance point
        point = ResonancePoint(
            id=properties.get('id'),
            strength=properties.get('strength', 0.0),
            stability=properties.get('stability', 0.0),
            attractor_radius=properties.get('attractor_radius', 0.0)
        )
        
        # Set timestamps
        if 'created_at' in properties:
            point.created_at = datetime.fromisoformat(properties['created_at'])
        if 'updated_at' in properties:
            point.updated_at = datetime.fromisoformat(properties['updated_at'])
        if 'last_updated' in properties:
            point.last_updated = datetime.fromisoformat(properties['last_updated'])
        
        # Set contributing pattern IDs
        point.contributing_pattern_ids = json.loads(properties.get('contributing_pattern_ids', '{}'))
        
        # Set dimensional coordinates
        point.dimensional_coordinates = json.loads(properties.get('dimensional_coordinates', '[]'))
        
        # Set oscillatory properties
        point.oscillatory_properties = json.loads(properties.get('oscillatory_properties', '{}'))
        
        return point
    
    def save(self, point: ResonancePoint) -> str:
        """
        Save a ResonancePoint to ArangoDB.
        
        Args:
            point: The resonance point to save
            
        Returns:
            The ID of the saved resonance point
        """
        # Convert to document properties
        doc_properties = self._to_document_properties(point)
        
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Check if the document already exists
        existing_doc = None
        try:
            existing_doc = collection.get(point.id)
        except:
            pass
        
        if existing_doc:
            # Update the existing document
            collection.update(doc_properties)
        else:
            # Insert a new document
            collection.insert(doc_properties)
        
        # Return the ID
        return point.id
    
    def find_by_id(self, point_id: str) -> Optional[ResonancePoint]:
        """
        Find a ResonancePoint by ID.
        
        Args:
            point_id: The ID of the resonance point to find
            
        Returns:
            The resonance point if found, None otherwise
        """
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Get the document
        doc = collection.get(point_id)
        if not doc:
            return None
        
        # Convert to entity
        return self._dict_to_entity(doc)
    
    def find_by_pattern_id(self, pattern_id: str) -> List[ResonancePoint]:
        """
        Find ResonancePoint objects that include a specific pattern.
        
        Args:
            pattern_id: The ID of the pattern to search for
            
        Returns:
            A list of matching resonance points
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for resonance points that include the pattern
        query = """
        FOR rp IN ResonancePoint
            FILTER HAS(rp.contributing_pattern_ids, @pattern_id)
            RETURN rp
        """
        
        cursor = db.aql.execute(query, bind_vars={'pattern_id': pattern_id})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_by_strength_threshold(self, threshold: float) -> List[ResonancePoint]:
        """
        Find ResonancePoint objects with strength above a threshold.
        
        Args:
            threshold: The strength threshold
            
        Returns:
            A list of matching resonance points
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for resonance points with strength above the threshold
        query = """
        FOR rp IN ResonancePoint
            FILTER rp.strength >= @threshold
            SORT rp.strength DESC
            RETURN rp
        """
        
        cursor = db.aql.execute(query, bind_vars={'threshold': threshold})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_by_stability_threshold(self, threshold: float) -> List[ResonancePoint]:
        """
        Find ResonancePoint objects with stability above a threshold.
        
        Args:
            threshold: The stability threshold
            
        Returns:
            A list of matching resonance points
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for resonance points with stability above the threshold
        query = """
        FOR rp IN ResonancePoint
            FILTER rp.stability >= @threshold
            SORT rp.stability DESC
            RETURN rp
        """
        
        cursor = db.aql.execute(query, bind_vars={'threshold': threshold})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_all(self) -> List[ResonancePoint]:
        """
        Find all ResonancePoint objects.
        
        Returns:
            A list of all resonance points
        """
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Get all documents
        docs = collection.all()
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in docs]
