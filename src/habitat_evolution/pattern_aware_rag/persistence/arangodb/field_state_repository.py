"""
TonicHarmonicFieldState Repository for ArangoDB.

Handles persistence of TonicHarmonicFieldState objects to ArangoDB, including
eigenspace properties, resonance relationships, and metrics.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime
import uuid
import numpy as np

from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.adaptive_core.persistence.arangodb.base_repository import ArangoDBBaseRepository
from src.habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager

class TonicHarmonicFieldStateRepository(ArangoDBBaseRepository):
    """
    Repository for persisting TonicHarmonicFieldState objects to ArangoDB.
    
    This repository handles the serialization and deserialization of field states,
    including their eigenspace properties, resonance relationships, and metrics.
    It supports versioning and context tracking for field states.
    """
    
    def __init__(self):
        """Initialize the repository with the appropriate collection name."""
        super().__init__()
        self.collection_name = "TonicHarmonicFieldState"
        self.connection_manager = ArangoDBConnectionManager()
    
    def _to_document_properties(self, field_state: TonicHarmonicFieldState) -> Dict[str, Any]:
        """
        Convert a TonicHarmonicFieldState to ArangoDB document properties.
        
        Args:
            field_state: The field state to convert
            
        Returns:
            A dictionary of document properties
        """
        # Extract the core properties
        properties = {
            "_key": field_state.id,
            "id": field_state.id,
            "version_id": field_state.version_id,
            "created_at": field_state.created_at.isoformat() if hasattr(field_state, 'created_at') else datetime.now().isoformat(),
            "updated_at": field_state.updated_at.isoformat() if hasattr(field_state, 'updated_at') else datetime.now().isoformat(),
            "version_history": json.dumps(field_state.version_history) if hasattr(field_state, 'version_history') else json.dumps([]),
            "field_metrics": json.dumps(field_state.field_metrics) if hasattr(field_state, 'field_metrics') else json.dumps({}),
        }
        
        # Extract field analysis properties
        if hasattr(field_state, 'field_analysis'):
            # Handle topology
            if 'topology' in field_state.field_analysis:
                topology = field_state.field_analysis['topology']
                properties['effective_dimensionality'] = topology.get('effective_dimensionality', 0)
                properties['principal_dimensions'] = json.dumps(topology.get('principal_dimensions', []))
                
                # Convert numpy arrays to lists for JSON serialization
                eigenvalues = topology.get('eigenvalues', [])
                if isinstance(eigenvalues, np.ndarray):
                    eigenvalues = eigenvalues.tolist()
                properties['eigenvalues'] = json.dumps(eigenvalues)
                
                eigenvectors = topology.get('eigenvectors', [])
                if isinstance(eigenvectors, np.ndarray):
                    eigenvectors = eigenvectors.tolist()
                properties['eigenvectors'] = json.dumps(eigenvectors)
            
            # Handle density
            if 'density' in field_state.field_analysis:
                density = field_state.field_analysis['density']
                properties['density_centers'] = json.dumps(density.get('density_centers', []))
                
                # Handle density map (potentially large)
                density_map = density.get('density_map', {})
                if isinstance(density_map, dict) and 'values' in density_map:
                    # Store only metadata about the density map to avoid excessive storage
                    properties['density_map_metadata'] = json.dumps({
                        'resolution': density_map.get('resolution', []),
                        'dimensions': len(density_map.get('resolution', [])),
                        'has_values': True
                    })
                else:
                    properties['density_map_metadata'] = json.dumps({
                        'has_values': False
                    })
            
            # Handle field properties
            if 'field_properties' in field_state.field_analysis:
                field_props = field_state.field_analysis['field_properties']
                properties['coherence'] = field_props.get('coherence', 0.0)
                properties['navigability_score'] = field_props.get('navigability_score', 0.0)
                properties['stability'] = field_props.get('stability', 0.0)
                properties['resonance_patterns'] = json.dumps(field_props.get('resonance_patterns', []))
        
        # Handle patterns
        if hasattr(field_state, 'patterns') and field_state.patterns:
            # Store pattern IDs only - the patterns themselves are stored in the Pattern collection
            properties['pattern_ids'] = json.dumps(list(field_state.patterns.keys()))
        else:
            properties['pattern_ids'] = json.dumps([])
        
        # Handle resonance relationships
        if hasattr(field_state, 'resonance_relationships') and field_state.resonance_relationships:
            # Store a simplified version of resonance relationships
            simplified_relationships = {}
            for pattern_id, related_patterns in field_state.resonance_relationships.items():
                simplified_relationships[pattern_id] = list(related_patterns.keys())
            properties['resonance_relationships'] = json.dumps(simplified_relationships)
        else:
            properties['resonance_relationships'] = json.dumps({})
        
        return properties
    
    def _dict_to_entity(self, properties: Dict[str, Any]) -> TonicHarmonicFieldState:
        """
        Convert ArangoDB document properties to a TonicHarmonicFieldState.
        
        Args:
            properties: The document properties to convert
            
        Returns:
            A TonicHarmonicFieldState object
        """
        # Reconstruct field analysis
        field_analysis = {
            'topology': {
                'effective_dimensionality': properties.get('effective_dimensionality', 0),
                'principal_dimensions': json.loads(properties.get('principal_dimensions', '[]')),
                'eigenvalues': json.loads(properties.get('eigenvalues', '[]')),
                'eigenvectors': json.loads(properties.get('eigenvectors', '[]'))
            },
            'density': {
                'density_centers': json.loads(properties.get('density_centers', '[]')),
                'density_map': {
                    'resolution': json.loads(properties.get('density_map_metadata', '{}')).get('resolution', []),
                    'values': []  # Empty placeholder - full values would be loaded separately if needed
                }
            },
            'field_properties': {
                'coherence': properties.get('coherence', 0.0),
                'navigability_score': properties.get('navigability_score', 0.0),
                'stability': properties.get('stability', 0.0),
                'resonance_patterns': json.loads(properties.get('resonance_patterns', '[]'))
            }
        }
        
        # Create a minimal field state
        field_state = TonicHarmonicFieldState(field_analysis)
        
        # Set the core properties
        field_state.id = properties.get('id')
        field_state.version_id = properties.get('version_id')
        
        # Set timestamps
        if 'created_at' in properties:
            field_state.created_at = datetime.fromisoformat(properties['created_at'])
        if 'updated_at' in properties:
            field_state.updated_at = datetime.fromisoformat(properties['updated_at'])
        
        # Set version history
        field_state.version_history = json.loads(properties.get('version_history', '[]'))
        
        # Set field metrics
        field_state.field_metrics = json.loads(properties.get('field_metrics', '{}'))
        
        # Note: patterns and resonance relationships would be loaded separately
        # through their respective repositories to avoid excessive data loading
        
        return field_state
    
    def save(self, field_state: TonicHarmonicFieldState) -> str:
        """
        Save a TonicHarmonicFieldState to ArangoDB.
        
        Args:
            field_state: The field state to save
            
        Returns:
            The ID of the saved field state
        """
        # Convert to document properties
        doc_properties = self._to_document_properties(field_state)
        
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Check if the document already exists
        existing_doc = None
        try:
            existing_doc = collection.get(field_state.id)
        except:
            pass
        
        if existing_doc:
            # Update the existing document
            collection.update(doc_properties)
        else:
            # Insert a new document
            collection.insert(doc_properties)
        
        # Return the ID
        return field_state.id
    
    def find_by_id(self, field_state_id: str) -> Optional[TonicHarmonicFieldState]:
        """
        Find a TonicHarmonicFieldState by ID.
        
        Args:
            field_state_id: The ID of the field state to find
            
        Returns:
            The field state if found, None otherwise
        """
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Get the document
        doc = collection.get(field_state_id)
        if not doc:
            return None
        
        # Convert to entity
        return self._dict_to_entity(doc)
    
    def find_latest(self) -> Optional[TonicHarmonicFieldState]:
        """
        Find the latest TonicHarmonicFieldState.
        
        Returns:
            The latest field state if found, None otherwise
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for the latest field state
        query = """
        FOR fs IN TonicHarmonicFieldState
            SORT fs.created_at DESC
            LIMIT 1
            RETURN fs
        """
        
        cursor = db.aql.execute(query)
        results = [doc for doc in cursor]
        
        if not results:
            return None
        
        # Convert to entity
        return self._dict_to_entity(results[0])
    
    def find_version_history(self, field_state_id: str) -> List[Dict[str, Any]]:
        """
        Find the version history of a TonicHarmonicFieldState.
        
        Args:
            field_state_id: The ID of the field state
            
        Returns:
            A list of version history entries
        """
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Get the document
        doc = collection.get(field_state_id)
        if not doc:
            return []
        
        # Return the version history
        return json.loads(doc.get('version_history', '[]'))
    
    def find_by_version_id(self, version_id: str) -> Optional[TonicHarmonicFieldState]:
        """
        Find a TonicHarmonicFieldState by version ID.
        
        Args:
            version_id: The version ID to find
            
        Returns:
            The field state if found, None otherwise
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for the field state with the given version ID
        query = """
        FOR fs IN TonicHarmonicFieldState
            FILTER fs.version_id == @version_id
            RETURN fs
        """
        
        cursor = db.aql.execute(query, bind_vars={'version_id': version_id})
        results = [doc for doc in cursor]
        
        if not results:
            return None
        
        # Convert to entity
        return self._dict_to_entity(results[0])
