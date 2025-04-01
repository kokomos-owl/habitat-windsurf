"""
Pattern Repository for ArangoDB.

Handles persistence of Pattern objects to ArangoDB, including
eigenspace properties, temporal and oscillatory characteristics.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime
import uuid
import logging

from src.habitat_evolution.adaptive_core.persistence.arangodb.base_repository import ArangoDBBaseRepository
from src.habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager
from src.habitat_evolution.adaptive_core.models import Pattern

logger = logging.getLogger(__name__)

class PatternRepository(ArangoDBBaseRepository):
    """
    Repository for persisting Pattern objects to ArangoDB.
    
    This repository handles the serialization and deserialization of patterns,
    including their eigenspace properties, temporal and oscillatory characteristics.
    """
    
    def __init__(self):
        """Initialize the repository with the appropriate collection name."""
        super().__init__()
        self.collection_name = "Pattern"
        self.connection_manager = ArangoDBConnectionManager()
    
    def _to_document_properties(self, pattern: Pattern) -> Dict[str, Any]:
        """
        Convert a Pattern to ArangoDB document properties.
        
        Args:
            pattern: The pattern to convert
            
        Returns:
            A dictionary of document properties
        """
        # Extract the core properties
        properties = {
            "_key": pattern.id,
            "id": pattern.id,
            "created_at": pattern.created_at.isoformat() if hasattr(pattern, 'created_at') else datetime.now().isoformat(),
            "updated_at": pattern.updated_at.isoformat() if hasattr(pattern, 'updated_at') else datetime.now().isoformat(),
            "pattern_type": pattern.pattern_type if hasattr(pattern, 'pattern_type') else "unknown",
            "source": pattern.source if hasattr(pattern, 'source') else "",
            "predicate": pattern.predicate if hasattr(pattern, 'predicate') else "",
            "target": pattern.target if hasattr(pattern, 'target') else "",
            "confidence": pattern.confidence if hasattr(pattern, 'confidence') else 0.0,
            "timestamp_ms": pattern.timestamp_ms if hasattr(pattern, 'timestamp_ms') else int(datetime.now().timestamp() * 1000),
        }
        
        # Handle eigenspace properties
        if hasattr(pattern, 'eigenspace_properties') and pattern.eigenspace_properties:
            # Extract key eigenspace properties
            eigenspace_props = pattern.eigenspace_properties
            properties['primary_dimensions'] = json.dumps(eigenspace_props.get('primary_dimensions', []))
            properties['dimensional_coordinates'] = json.dumps(eigenspace_props.get('dimensional_coordinates', []))
            properties['eigenspace_centrality'] = eigenspace_props.get('eigenspace_centrality', 0.0)
            properties['eigenspace_stability'] = eigenspace_props.get('eigenspace_stability', 0.0)
            properties['dimensional_variance'] = eigenspace_props.get('dimensional_variance', 0.0)
            properties['resonance_groups'] = json.dumps(eigenspace_props.get('resonance_groups', []))
        else:
            properties['primary_dimensions'] = json.dumps([])
            properties['dimensional_coordinates'] = json.dumps([])
            properties['eigenspace_centrality'] = 0.0
            properties['eigenspace_stability'] = 0.0
            properties['dimensional_variance'] = 0.0
            properties['resonance_groups'] = json.dumps([])
        
        # Handle temporal properties
        if hasattr(pattern, 'temporal_properties') and pattern.temporal_properties:
            temporal_props = pattern.temporal_properties
            properties['frequency'] = temporal_props.get('frequency', 0.0)
            properties['phase_position'] = temporal_props.get('phase_position', 0.0)
            properties['temporal_coherence'] = temporal_props.get('temporal_coherence', 0.0)
            properties['harmonic_ratio'] = temporal_props.get('harmonic_ratio', 0.0)
        else:
            properties['frequency'] = 0.0
            properties['phase_position'] = 0.0
            properties['temporal_coherence'] = 0.0
            properties['harmonic_ratio'] = 0.0
        
        # Handle oscillatory properties
        if hasattr(pattern, 'oscillatory_properties') and pattern.oscillatory_properties:
            oscillatory_props = pattern.oscillatory_properties
            properties['tonic_value'] = oscillatory_props.get('tonic_value', 0.5)
            properties['harmonic_value'] = oscillatory_props.get('harmonic_value', 0.5)
            properties['pattern_energy'] = oscillatory_props.get('energy', 0.0)
        else:
            properties['tonic_value'] = 0.5
            properties['harmonic_value'] = 0.5
            properties['pattern_energy'] = 0.0
        
        # Handle context
        if hasattr(pattern, 'context') and pattern.context:
            properties['context'] = json.dumps(pattern.context)
        else:
            properties['context'] = json.dumps({})
        
        return properties
    
    def _dict_to_entity(self, properties: Dict[str, Any]) -> Pattern:
        """
        Convert ArangoDB document properties to a Pattern.
        
        Args:
            properties: The document properties to convert
            
        Returns:
            A Pattern object
        """
        # Create a minimal pattern
        pattern = Pattern(
            id=properties.get('id'),
            pattern_type=properties.get('pattern_type', 'unknown'),
            source=properties.get('source', ''),
            predicate=properties.get('predicate', ''),
            target=properties.get('target', ''),
            confidence=properties.get('confidence', 0.0)
        )
        
        # Set timestamps
        if 'created_at' in properties:
            pattern.created_at = datetime.fromisoformat(properties['created_at'])
        if 'updated_at' in properties:
            pattern.updated_at = datetime.fromisoformat(properties['updated_at'])
        
        pattern.timestamp_ms = properties.get('timestamp_ms', int(datetime.now().timestamp() * 1000))
        
        # Set eigenspace properties
        pattern.eigenspace_properties = {
            'primary_dimensions': json.loads(properties.get('primary_dimensions', '[]')),
            'dimensional_coordinates': json.loads(properties.get('dimensional_coordinates', '[]')),
            'eigenspace_centrality': properties.get('eigenspace_centrality', 0.0),
            'eigenspace_stability': properties.get('eigenspace_stability', 0.0),
            'dimensional_variance': properties.get('dimensional_variance', 0.0),
            'resonance_groups': json.loads(properties.get('resonance_groups', '[]'))
        }
        
        # Set temporal properties
        pattern.temporal_properties = {
            'frequency': properties.get('frequency', 0.0),
            'phase_position': properties.get('phase_position', 0.0),
            'temporal_coherence': properties.get('temporal_coherence', 0.0),
            'harmonic_ratio': properties.get('harmonic_ratio', 0.0)
        }
        
        # Set oscillatory properties
        pattern.oscillatory_properties = {
            'tonic_value': properties.get('tonic_value', 0.5),
            'harmonic_value': properties.get('harmonic_value', 0.5),
            'energy': properties.get('pattern_energy', 0.0)
        }
        
        # Set context
        pattern.context = json.loads(properties.get('context', '{}'))
        
        return pattern
    
    def save(self, pattern: Pattern) -> str:
        """
        Save a Pattern to ArangoDB.
        
        Args:
            pattern: The pattern to save
            
        Returns:
            The ID of the saved pattern
        """
        # Convert to document properties
        doc_properties = self._to_document_properties(pattern)
        
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Check if the document already exists
        existing_doc = None
        try:
            existing_doc = collection.get(pattern.id)
        except:
            pass
        
        if existing_doc:
            # Update the existing document
            collection.update(doc_properties)
        else:
            # Insert a new document
            collection.insert(doc_properties)
        
        # Return the ID
        return pattern.id
    
    def find_by_id(self, pattern_id: str) -> Optional[Pattern]:
        """
        Find a Pattern by ID.
        
        Args:
            pattern_id: The ID of the pattern to find
            
        Returns:
            The pattern if found, None otherwise
        """
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Get the document
        doc = collection.get(pattern_id)
        if not doc:
            return None
        
        # Convert to entity
        return self._dict_to_entity(doc)
    
    def find_by_predicate(self, predicate: str) -> List[Pattern]:
        """
        Find Pattern objects with a specific predicate.
        
        Args:
            predicate: The predicate to search for
            
        Returns:
            A list of matching patterns
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for patterns with the given predicate
        query = """
        FOR p IN Pattern
            FILTER p.predicate == @predicate
            RETURN p
        """
        
        cursor = db.aql.execute(query, bind_vars={'predicate': predicate})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_by_source_target(self, source: str, target: str) -> List[Pattern]:
        """
        Find Pattern objects with specific source and target.
        
        Args:
            source: The source to search for
            target: The target to search for
            
        Returns:
            A list of matching patterns
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for patterns with the given source and target
        query = """
        FOR p IN Pattern
            FILTER p.source == @source AND p.target == @target
            RETURN p
        """
        
        cursor = db.aql.execute(query, bind_vars={'source': source, 'target': target})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_by_temporal_coherence_threshold(self, threshold: float) -> List[Pattern]:
        """
        Find Pattern objects with temporal coherence above a threshold.
        
        Args:
            threshold: The temporal coherence threshold
            
        Returns:
            A list of matching patterns
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for patterns with temporal coherence above the threshold
        query = """
        FOR p IN Pattern
            FILTER p.temporal_coherence >= @threshold
            SORT p.temporal_coherence DESC
            RETURN p
        """
        
        cursor = db.aql.execute(query, bind_vars={'threshold': threshold})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_by_eigenspace_centrality_threshold(self, threshold: float) -> List[Pattern]:
        """
        Find Pattern objects with eigenspace centrality above a threshold.
        
        Args:
            threshold: The eigenspace centrality threshold
            
        Returns:
            A list of matching patterns
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for patterns with eigenspace centrality above the threshold
        query = """
        FOR p IN Pattern
            FILTER p.eigenspace_centrality >= @threshold
            SORT p.eigenspace_centrality DESC
            RETURN p
        """
        
        cursor = db.aql.execute(query, bind_vars={'threshold': threshold})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_patterns_in_resonance_group(self, group_id: str) -> List[Pattern]:
        """
        Find Pattern objects that belong to a specific resonance group.
        
        Args:
            group_id: The ID of the resonance group
            
        Returns:
            A list of matching patterns
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for patterns in the resonance group
        query = """
        FOR p IN Pattern
            FILTER @group_id IN p.resonance_groups
            RETURN p
        """
        
        cursor = db.aql.execute(query, bind_vars={'group_id': group_id})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_all(self) -> List[Pattern]:
        """
        Find all Pattern objects.
        
        Returns:
            A list of all patterns
        """
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Get all documents
        docs = collection.all()
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in docs]
