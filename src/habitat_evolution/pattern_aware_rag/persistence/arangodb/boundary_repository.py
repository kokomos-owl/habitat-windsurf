"""
Boundary Repository for ArangoDB.

Handles persistence of Boundary objects to ArangoDB, including
permeability and oscillatory properties.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime
import uuid
import logging

from src.habitat_evolution.adaptive_core.persistence.arangodb.base_repository import ArangoDBBaseRepository
from src.habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager
from src.habitat_evolution.pattern_aware_rag.topology.boundary import Boundary

logger = logging.getLogger(__name__)

class BoundaryRepository(ArangoDBBaseRepository):
    """
    Repository for persisting Boundary objects to ArangoDB.
    
    This repository handles the serialization and deserialization of boundaries,
    including their permeability and oscillatory properties.
    """
    
    def __init__(self):
        """Initialize the repository with the appropriate collection name."""
        super().__init__()
        self.collection_name = "Boundary"
        self.connection_manager = ArangoDBConnectionManager()
    
    def _to_document_properties(self, boundary: Boundary) -> Dict[str, Any]:
        """
        Convert a Boundary to ArangoDB document properties.
        
        Args:
            boundary: The boundary to convert
            
        Returns:
            A dictionary of document properties
        """
        # Extract the core properties
        properties = {
            "_key": boundary.id,
            "id": boundary.id,
            "created_at": boundary.created_at.isoformat() if hasattr(boundary, 'created_at') else datetime.now().isoformat(),
            "updated_at": boundary.updated_at.isoformat() if hasattr(boundary, 'updated_at') else datetime.now().isoformat(),
            "permeability": boundary.permeability if hasattr(boundary, 'permeability') else 0.5,
            "stability": boundary.stability if hasattr(boundary, 'stability') else 0.5,
        }
        
        # Handle domain IDs
        if hasattr(boundary, 'domain_ids') and boundary.domain_ids:
            properties['domain_ids'] = json.dumps(boundary.domain_ids)
        else:
            properties['domain_ids'] = json.dumps([])
        
        # Handle oscillatory properties
        if hasattr(boundary, 'oscillatory_properties') and boundary.oscillatory_properties:
            properties['oscillatory_properties'] = json.dumps(boundary.oscillatory_properties)
        else:
            properties['oscillatory_properties'] = json.dumps({})
        
        # Handle emergent forms
        if hasattr(boundary, 'emergent_forms') and boundary.emergent_forms:
            properties['emergent_forms'] = json.dumps(boundary.emergent_forms)
        else:
            properties['emergent_forms'] = json.dumps([])
        
        # Handle dimensional coordinates
        if hasattr(boundary, 'dimensional_coordinates') and boundary.dimensional_coordinates:
            properties['dimensional_coordinates'] = json.dumps(boundary.dimensional_coordinates)
        else:
            properties['dimensional_coordinates'] = json.dumps([])
        
        return properties
    
    def _dict_to_entity(self, properties: Dict[str, Any]) -> Boundary:
        """
        Convert ArangoDB document properties to a Boundary.
        
        Args:
            properties: The document properties to convert
            
        Returns:
            A Boundary object
        """
        # Create a minimal boundary
        boundary = Boundary(id=properties.get('id'))
        
        # Set timestamps
        if 'created_at' in properties:
            boundary.created_at = datetime.fromisoformat(properties['created_at'])
        if 'updated_at' in properties:
            boundary.updated_at = datetime.fromisoformat(properties['updated_at'])
        
        # Set core properties
        boundary.permeability = properties.get('permeability', 0.5)
        boundary.stability = properties.get('stability', 0.5)
        
        # Set domain IDs
        boundary.domain_ids = json.loads(properties.get('domain_ids', '[]'))
        
        # Set oscillatory properties
        boundary.oscillatory_properties = json.loads(properties.get('oscillatory_properties', '{}'))
        
        # Set emergent forms
        boundary.emergent_forms = json.loads(properties.get('emergent_forms', '[]'))
        
        # Set dimensional coordinates
        boundary.dimensional_coordinates = json.loads(properties.get('dimensional_coordinates', '[]'))
        
        return boundary
    
    def save(self, boundary: Boundary) -> str:
        """
        Save a Boundary to ArangoDB.
        
        Args:
            boundary: The boundary to save
            
        Returns:
            The ID of the saved boundary
        """
        # Convert to document properties
        doc_properties = self._to_document_properties(boundary)
        
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Check if the document already exists
        existing_doc = None
        try:
            existing_doc = collection.get(boundary.id)
        except:
            pass
        
        if existing_doc:
            # Update the existing document
            collection.update(doc_properties)
        else:
            # Insert a new document
            collection.insert(doc_properties)
        
        # Return the ID
        return boundary.id
    
    def find_by_id(self, boundary_id: str) -> Optional[Boundary]:
        """
        Find a Boundary by ID.
        
        Args:
            boundary_id: The ID of the boundary to find
            
        Returns:
            The boundary if found, None otherwise
        """
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Get the document
        doc = collection.get(boundary_id)
        if not doc:
            return None
        
        # Convert to entity
        return self._dict_to_entity(doc)
    
    def find_by_domain_id(self, domain_id: str) -> List[Boundary]:
        """
        Find Boundary objects that connect to a specific domain.
        
        Args:
            domain_id: The ID of the domain to search for
            
        Returns:
            A list of matching boundaries
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for boundaries connected to the domain
        query = """
        FOR b IN Boundary
            FILTER @domain_id IN b.domain_ids
            RETURN b
        """
        
        cursor = db.aql.execute(query, bind_vars={'domain_id': domain_id})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_by_permeability_threshold(self, threshold: float) -> List[Boundary]:
        """
        Find Boundary objects with permeability above a threshold.
        
        Args:
            threshold: The permeability threshold
            
        Returns:
            A list of matching boundaries
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for boundaries with permeability above the threshold
        query = """
        FOR b IN Boundary
            FILTER b.permeability >= @threshold
            SORT b.permeability DESC
            RETURN b
        """
        
        cursor = db.aql.execute(query, bind_vars={'threshold': threshold})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_boundaries_between_domains(self, domain_id1: str, domain_id2: str) -> List[Boundary]:
        """
        Find Boundary objects that connect two specific domains.
        
        Args:
            domain_id1: The ID of the first domain
            domain_id2: The ID of the second domain
            
        Returns:
            A list of matching boundaries
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for boundaries connecting the two domains
        query = """
        FOR b IN Boundary
            FILTER @domain_id1 IN b.domain_ids AND @domain_id2 IN b.domain_ids
            RETURN b
        """
        
        cursor = db.aql.execute(query, bind_vars={'domain_id1': domain_id1, 'domain_id2': domain_id2})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_all(self) -> List[Boundary]:
        """
        Find all Boundary objects.
        
        Returns:
            A list of all boundaries
        """
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Get all documents
        docs = collection.all()
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in docs]
