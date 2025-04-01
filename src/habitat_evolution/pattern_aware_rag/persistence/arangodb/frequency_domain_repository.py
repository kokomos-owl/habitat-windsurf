"""
FrequencyDomain Repository for ArangoDB.

Handles persistence of FrequencyDomain objects to ArangoDB, including
dimensional properties and resonance characteristics.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime
import uuid
import logging

from src.habitat_evolution.adaptive_core.persistence.arangodb.base_repository import ArangoDBBaseRepository
from src.habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager
from src.habitat_evolution.pattern_aware_rag.topology.domain import FrequencyDomain

logger = logging.getLogger(__name__)

class FrequencyDomainRepository(ArangoDBBaseRepository):
    """
    Repository for persisting FrequencyDomain objects to ArangoDB.
    
    This repository handles the serialization and deserialization of frequency domains,
    including their dimensional properties and resonance characteristics.
    """
    
    def __init__(self):
        """Initialize the repository with the appropriate collection name."""
        super().__init__()
        self.collection_name = "FrequencyDomain"
        self.connection_manager = ArangoDBConnectionManager()
    
    def _to_document_properties(self, domain: FrequencyDomain) -> Dict[str, Any]:
        """
        Convert a FrequencyDomain to ArangoDB document properties.
        
        Args:
            domain: The frequency domain to convert
            
        Returns:
            A dictionary of document properties
        """
        # Extract the core properties
        properties = {
            "_key": domain.id,
            "id": domain.id,
            "name": domain.name if hasattr(domain, 'name') else "",
            "created_at": domain.created_at.isoformat() if hasattr(domain, 'created_at') else datetime.now().isoformat(),
            "updated_at": domain.updated_at.isoformat() if hasattr(domain, 'updated_at') else datetime.now().isoformat(),
            "coherence": domain.coherence if hasattr(domain, 'coherence') else 0.0,
            "stability": domain.stability if hasattr(domain, 'stability') else 0.0,
            "energy": domain.energy if hasattr(domain, 'energy') else 0.0,
        }
        
        # Handle dimensional properties
        if hasattr(domain, 'dimensional_properties') and domain.dimensional_properties:
            properties['dimensional_properties'] = json.dumps(domain.dimensional_properties)
        else:
            properties['dimensional_properties'] = json.dumps({})
        
        # Handle center coordinates
        if hasattr(domain, 'center_coordinates') and domain.center_coordinates:
            properties['center_coordinates'] = json.dumps(domain.center_coordinates)
        else:
            properties['center_coordinates'] = json.dumps([])
        
        # Handle resonance characteristics
        if hasattr(domain, 'resonance_characteristics') and domain.resonance_characteristics:
            properties['resonance_characteristics'] = json.dumps(domain.resonance_characteristics)
        else:
            properties['resonance_characteristics'] = json.dumps({})
        
        # Handle oscillatory properties
        if hasattr(domain, 'oscillatory_properties') and domain.oscillatory_properties:
            properties['oscillatory_properties'] = json.dumps(domain.oscillatory_properties)
        else:
            properties['oscillatory_properties'] = json.dumps({})
        
        # Handle pattern IDs
        if hasattr(domain, 'pattern_ids') and domain.pattern_ids:
            properties['pattern_ids'] = json.dumps(domain.pattern_ids)
        else:
            properties['pattern_ids'] = json.dumps([])
        
        return properties
    
    def _dict_to_entity(self, properties: Dict[str, Any]) -> FrequencyDomain:
        """
        Convert ArangoDB document properties to a FrequencyDomain.
        
        Args:
            properties: The document properties to convert
            
        Returns:
            A FrequencyDomain object
        """
        # Create a minimal frequency domain
        domain = FrequencyDomain(
            id=properties.get('id'),
            name=properties.get('name', "")
        )
        
        # Set timestamps
        if 'created_at' in properties:
            domain.created_at = datetime.fromisoformat(properties['created_at'])
        if 'updated_at' in properties:
            domain.updated_at = datetime.fromisoformat(properties['updated_at'])
        
        # Set core properties
        domain.coherence = properties.get('coherence', 0.0)
        domain.stability = properties.get('stability', 0.0)
        domain.energy = properties.get('energy', 0.0)
        
        # Set dimensional properties
        domain.dimensional_properties = json.loads(properties.get('dimensional_properties', '{}'))
        
        # Set center coordinates
        domain.center_coordinates = json.loads(properties.get('center_coordinates', '[]'))
        
        # Set resonance characteristics
        domain.resonance_characteristics = json.loads(properties.get('resonance_characteristics', '{}'))
        
        # Set oscillatory properties
        domain.oscillatory_properties = json.loads(properties.get('oscillatory_properties', '{}'))
        
        # Set pattern IDs
        domain.pattern_ids = json.loads(properties.get('pattern_ids', '[]'))
        
        return domain
    
    def save(self, domain: FrequencyDomain) -> str:
        """
        Save a FrequencyDomain to ArangoDB.
        
        Args:
            domain: The frequency domain to save
            
        Returns:
            The ID of the saved frequency domain
        """
        # Convert to document properties
        doc_properties = self._to_document_properties(domain)
        
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Check if the document already exists
        existing_doc = None
        try:
            existing_doc = collection.get(domain.id)
        except:
            pass
        
        if existing_doc:
            # Update the existing document
            collection.update(doc_properties)
        else:
            # Insert a new document
            collection.insert(doc_properties)
        
        # Return the ID
        return domain.id
    
    def find_by_id(self, domain_id: str) -> Optional[FrequencyDomain]:
        """
        Find a FrequencyDomain by ID.
        
        Args:
            domain_id: The ID of the frequency domain to find
            
        Returns:
            The frequency domain if found, None otherwise
        """
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Get the document
        doc = collection.get(domain_id)
        if not doc:
            return None
        
        # Convert to entity
        return self._dict_to_entity(doc)
    
    def find_by_name(self, name: str) -> List[FrequencyDomain]:
        """
        Find FrequencyDomain objects by name.
        
        Args:
            name: The name to search for
            
        Returns:
            A list of matching frequency domains
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for domains with the given name
        query = """
        FOR fd IN FrequencyDomain
            FILTER fd.name == @name
            RETURN fd
        """
        
        cursor = db.aql.execute(query, bind_vars={'name': name})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_by_coherence_threshold(self, threshold: float) -> List[FrequencyDomain]:
        """
        Find FrequencyDomain objects with coherence above a threshold.
        
        Args:
            threshold: The coherence threshold
            
        Returns:
            A list of matching frequency domains
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for domains with coherence above the threshold
        query = """
        FOR fd IN FrequencyDomain
            FILTER fd.coherence >= @threshold
            SORT fd.coherence DESC
            RETURN fd
        """
        
        cursor = db.aql.execute(query, bind_vars={'threshold': threshold})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_domains_with_pattern(self, pattern_id: str) -> List[FrequencyDomain]:
        """
        Find FrequencyDomain objects that contain a specific pattern.
        
        Args:
            pattern_id: The ID of the pattern to search for
            
        Returns:
            A list of matching frequency domains
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for domains containing the pattern
        query = """
        FOR fd IN FrequencyDomain
            FILTER @pattern_id IN fd.pattern_ids
            RETURN fd
        """
        
        cursor = db.aql.execute(query, bind_vars={'pattern_id': pattern_id})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_all(self) -> List[FrequencyDomain]:
        """
        Find all FrequencyDomain objects.
        
        Returns:
            A list of all frequency domains
        """
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Get all documents
        docs = collection.all()
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in docs]
