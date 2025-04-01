"""
TopologyState Repository for ArangoDB.

Handles persistence of TopologyState objects to ArangoDB, including
frequency domains, boundaries, and resonance points.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime
import uuid
import logging

from src.habitat_evolution.adaptive_core.persistence.arangodb.base_repository import ArangoDBBaseRepository
from src.habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager
from src.habitat_evolution.pattern_aware_rag.topology.manager import TopologyState

logger = logging.getLogger(__name__)

class TopologyStateRepository(ArangoDBBaseRepository):
    """
    Repository for persisting TopologyState objects to ArangoDB.
    
    This repository handles the serialization and deserialization of topology states,
    including frequency domains, boundaries, and resonance points. It supports
    bidirectional relationships between topology and temporality.
    """
    
    def __init__(self):
        """Initialize the repository with the appropriate collection name."""
        super().__init__()
        self.collection_name = "TopologyState"
        self.connection_manager = ArangoDBConnectionManager()
    
    def _to_document_properties(self, topology_state: TopologyState) -> Dict[str, Any]:
        """
        Convert a TopologyState to ArangoDB document properties.
        
        Args:
            topology_state: The topology state to convert
            
        Returns:
            A dictionary of document properties
        """
        # Extract the core properties
        properties = {
            "_key": topology_state.id,
            "id": topology_state.id,
            "created_at": topology_state.created_at.isoformat() if hasattr(topology_state, 'created_at') else datetime.now().isoformat(),
            "updated_at": topology_state.updated_at.isoformat() if hasattr(topology_state, 'updated_at') else datetime.now().isoformat(),
            "is_test_state": topology_state.is_test_state if hasattr(topology_state, 'is_test_state') else False,
            "coherence_score": topology_state.coherence_score if hasattr(topology_state, 'coherence_score') else 0.0,
            "stability_score": topology_state.stability_score if hasattr(topology_state, 'stability_score') else 0.0,
            "field_state_id": topology_state.field_state_id if hasattr(topology_state, 'field_state_id') else None,
        }
        
        # Store domain IDs only - the domains themselves are stored in the FrequencyDomain collection
        if hasattr(topology_state, 'frequency_domains') and topology_state.frequency_domains:
            properties['domain_ids'] = json.dumps(list(topology_state.frequency_domains.keys()))
        else:
            properties['domain_ids'] = json.dumps([])
        
        # Store boundary IDs only - the boundaries themselves are stored in the Boundary collection
        if hasattr(topology_state, 'boundaries') and topology_state.boundaries:
            properties['boundary_ids'] = json.dumps(list(topology_state.boundaries.keys()))
        else:
            properties['boundary_ids'] = json.dumps([])
        
        # Store resonance point IDs only - the points themselves are stored in the ResonancePoint collection
        if hasattr(topology_state, 'resonance_points') and topology_state.resonance_points:
            properties['resonance_point_ids'] = json.dumps(list(topology_state.resonance_points.keys()))
        else:
            properties['resonance_point_ids'] = json.dumps([])
        
        # Store pattern eigenspace properties if available
        if hasattr(topology_state, 'pattern_eigenspace_properties') and topology_state.pattern_eigenspace_properties:
            # Store a simplified version to avoid excessive storage
            simplified_properties = {}
            for pattern_id, props in topology_state.pattern_eigenspace_properties.items():
                simplified_properties[pattern_id] = {
                    'eigenspace_centrality': props.get('eigenspace_centrality', 0.0),
                    'eigenspace_stability': props.get('eigenspace_stability', 0.0),
                    'frequency': props.get('frequency', 0.0),
                    'temporal_coherence': props.get('temporal_coherence', 0.0)
                }
            properties['pattern_eigenspace_properties'] = json.dumps(simplified_properties)
        else:
            properties['pattern_eigenspace_properties'] = json.dumps({})
        
        # Store learning window information if available
        if hasattr(topology_state, 'learning_windows') and topology_state.learning_windows:
            properties['learning_windows'] = json.dumps(topology_state.learning_windows)
        else:
            properties['learning_windows'] = json.dumps({})
        
        return properties
    
    def _dict_to_entity(self, properties: Dict[str, Any]) -> TopologyState:
        """
        Convert ArangoDB document properties to a TopologyState.
        
        Args:
            properties: The document properties to convert
            
        Returns:
            A TopologyState object
        """
        # Create a minimal topology state
        topology_state = TopologyState(
            id=properties.get('id'),
            is_test_state=properties.get('is_test_state', False)
        )
        
        # Set timestamps
        if 'created_at' in properties:
            topology_state.created_at = datetime.fromisoformat(properties['created_at'])
        if 'updated_at' in properties:
            topology_state.updated_at = datetime.fromisoformat(properties['updated_at'])
        
        # Set scores
        topology_state.coherence_score = properties.get('coherence_score', 0.0)
        topology_state.stability_score = properties.get('stability_score', 0.0)
        
        # Set field state ID
        topology_state.field_state_id = properties.get('field_state_id')
        
        # Set learning windows
        topology_state.learning_windows = json.loads(properties.get('learning_windows', '{}'))
        
        # Set pattern eigenspace properties
        topology_state.pattern_eigenspace_properties = json.loads(properties.get('pattern_eigenspace_properties', '{}'))
        
        # Note: frequency domains, boundaries, and resonance points would be loaded separately
        # through their respective repositories to avoid excessive data loading
        
        return topology_state
    
    def save(self, topology_state: TopologyState) -> str:
        """
        Save a TopologyState to ArangoDB.
        
        Args:
            topology_state: The topology state to save
            
        Returns:
            The ID of the saved topology state
        """
        # Convert to document properties
        doc_properties = self._to_document_properties(topology_state)
        
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Check if the document already exists
        existing_doc = None
        try:
            existing_doc = collection.get(topology_state.id)
        except:
            pass
        
        if existing_doc:
            # Update the existing document
            collection.update(doc_properties)
        else:
            # Insert a new document
            collection.insert(doc_properties)
        
        # Return the ID
        return topology_state.id
    
    def find_by_id(self, topology_state_id: str) -> Optional[TopologyState]:
        """
        Find a TopologyState by ID.
        
        Args:
            topology_state_id: The ID of the topology state to find
            
        Returns:
            The topology state if found, None otherwise
        """
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Get the document
        doc = collection.get(topology_state_id)
        if not doc:
            return None
        
        # Convert to entity
        return self._dict_to_entity(doc)
    
    def find_latest(self) -> Optional[TopologyState]:
        """
        Find the latest TopologyState.
        
        Returns:
            The latest topology state if found, None otherwise
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for the latest topology state
        query = """
        FOR ts IN TopologyState
            SORT ts.created_at DESC
            LIMIT 1
            RETURN ts
        """
        
        cursor = db.aql.execute(query)
        results = [doc for doc in cursor]
        
        if not results:
            return None
        
        # Convert to entity
        return self._dict_to_entity(results[0])
    
    def find_by_field_state_id(self, field_state_id: str) -> List[TopologyState]:
        """
        Find all TopologyState objects associated with a field state.
        
        Args:
            field_state_id: The ID of the field state
            
        Returns:
            A list of topology states
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for topology states with the given field state ID
        query = """
        FOR ts IN TopologyState
            FILTER ts.field_state_id == @field_state_id
            SORT ts.created_at DESC
            RETURN ts
        """
        
        cursor = db.aql.execute(query, bind_vars={'field_state_id': field_state_id})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def load_frequency_domains(self, topology_state: TopologyState) -> None:
        """
        Load frequency domains for a topology state.
        
        Args:
            topology_state: The topology state to load domains for
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Get domain IDs
        domain_ids = json.loads(topology_state.get('domain_ids', '[]'))
        if not domain_ids:
            return
        
        # Query for domains
        query = """
        FOR fd IN FrequencyDomain
            FILTER fd.id IN @domain_ids
            RETURN fd
        """
        
        cursor = db.aql.execute(query, bind_vars={'domain_ids': domain_ids})
        domains = {doc['id']: doc for doc in cursor}
        
        # Set domains on the topology state
        topology_state.frequency_domains = domains
    
    def load_boundaries(self, topology_state: TopologyState) -> None:
        """
        Load boundaries for a topology state.
        
        Args:
            topology_state: The topology state to load boundaries for
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Get boundary IDs
        boundary_ids = json.loads(topology_state.get('boundary_ids', '[]'))
        if not boundary_ids:
            return
        
        # Query for boundaries
        query = """
        FOR b IN Boundary
            FILTER b.id IN @boundary_ids
            RETURN b
        """
        
        cursor = db.aql.execute(query, bind_vars={'boundary_ids': boundary_ids})
        boundaries = {doc['id']: doc for doc in cursor}
        
        # Set boundaries on the topology state
        topology_state.boundaries = boundaries
    
    def load_resonance_points(self, topology_state: TopologyState) -> None:
        """
        Load resonance points for a topology state.
        
        Args:
            topology_state: The topology state to load resonance points for
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Get resonance point IDs
        point_ids = json.loads(topology_state.get('resonance_point_ids', '[]'))
        if not point_ids:
            return
        
        # Query for resonance points
        query = """
        FOR rp IN ResonancePoint
            FILTER rp.id IN @point_ids
            RETURN rp
        """
        
        cursor = db.aql.execute(query, bind_vars={'point_ids': point_ids})
        points = {doc['id']: doc for doc in cursor}
        
        # Set resonance points on the topology state
        topology_state.resonance_points = points
