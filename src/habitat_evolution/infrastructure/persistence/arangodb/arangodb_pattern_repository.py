"""
ArangoDB pattern repository implementation for Habitat Evolution.

This module provides a concrete implementation of pattern repository functionality
using ArangoDB as the persistence layer, supporting the pattern evolution and
co-evolution principles of Habitat Evolution.
"""

import logging
from typing import Dict, List, Any, Optional, Type, Union
from datetime import datetime
import uuid

from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_graph_repository import ArangoDBGraphRepository

logger = logging.getLogger(__name__)


class Pattern:
    """
    Pattern entity for Habitat Evolution.
    
    This class represents a pattern in the Habitat Evolution system,
    with support for metadata, quality metrics, and relationships.
    """
    
    def __init__(self, 
                 id: Optional[str] = None,
                 name: Optional[str] = None,
                 pattern_type: Optional[str] = None,
                 description: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 created_at: Optional[str] = None,
                 updated_at: Optional[str] = None):
        """
        Initialize a new pattern.
        
        Args:
            id: Optional ID for the pattern
            name: Optional name for the pattern
            pattern_type: Optional type for the pattern
            description: Optional description for the pattern
            metadata: Optional metadata for the pattern
            created_at: Optional creation timestamp
            updated_at: Optional update timestamp
        """
        self.id = id
        self.name = name
        self.type = pattern_type
        self.description = description
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.updated_at = updated_at or datetime.utcnow().isoformat()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pattern':
        """
        Create a pattern from a dictionary.
        
        Args:
            data: Dictionary containing pattern data
            
        Returns:
            A new Pattern instance
        """
        pattern_id = data.get('_id', None)
        pattern_type = data.get('type', None)
        metadata = data.get('metadata', {})
        
        # Ensure metadata is a dictionary
        if metadata is None:
            metadata = {}
            
        # Ensure coherence is in metadata if quality is present
        if 'quality' in metadata and 'coherence' not in metadata:
            metadata['coherence'] = metadata['quality']
            
        return cls(
            id=pattern_id,
            name=data.get('name', None),
            pattern_type=pattern_type,
            description=data.get('description', None),
            metadata=metadata,
            created_at=data.get('created_at', None),
            updated_at=data.get('updated_at', None)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the pattern to a dictionary.
        
        Returns:
            Dictionary representation of the pattern
        """
        result = {
            'name': self.name,
            'type': self.type,
            'description': self.description,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
        
        # Filter out None values
        return {k: v for k, v in result.items() if v is not None}


class ArangoDBPatternRepository(ArangoDBGraphRepository[Pattern]):
    """
    ArangoDB implementation of a pattern repository.
    
    This repository provides pattern-specific functionality using ArangoDB
    as the persistence layer, supporting the pattern evolution and co-evolution
    principles of Habitat Evolution.
    """
    
    def __init__(self, 
                 db_connection: ArangoDBConnectionInterface,
                 event_service: EventServiceInterface):
        """
        Initialize a new ArangoDB pattern repository.
        
        Args:
            db_connection: The ArangoDB connection to use
            event_service: The event service for publishing events
        """
        super().__init__(
            node_collection_name="patterns",
            edge_collection_name="pattern_relationships",
            graph_name="pattern_graph",
            db_connection=db_connection,
            event_service=event_service,
            entity_class=Pattern
        )
        logger.debug("ArangoDBPatternRepository created")
    
    def find_by_name(self, name: str, exact_match: bool = True) -> List[Pattern]:
        """
        Find patterns by name.
        
        Args:
            name: The name to search for
            exact_match: Whether to require an exact match
            
        Returns:
            A list of matching patterns
        """
        if not self._initialized:
            self.initialize()
            
        query = "FOR p IN patterns"
        
        if exact_match:
            query += " FILTER p.name == @name"
        else:
            query += " FILTER LIKE(p.name, @name, true)"
            name = f"%{name}%"
            
        query += " RETURN p"
        
        documents = self._db_connection.execute_query(query, {"name": name})
        return [self._to_entity(doc) for doc in documents]
    
    def find_by_quality(self, min_quality: float = 0.0, 
                       max_quality: float = 1.0) -> List[Pattern]:
        """
        Find patterns by quality.
        
        Args:
            min_quality: The minimum quality threshold
            max_quality: The maximum quality threshold
            
        Returns:
            A list of matching patterns
        """
        if not self._initialized:
            self.initialize()
            
        query = """
        FOR p IN patterns
        FILTER (
            (p.metadata.quality >= @min_quality AND p.metadata.quality <= @max_quality) OR
            (p.metadata.coherence >= @min_quality AND p.metadata.coherence <= @max_quality)
        )
        RETURN p
        """
        
        documents = self._db_connection.execute_query(query, {
            "min_quality": min_quality,
            "max_quality": max_quality
        })
        
        return [self._to_entity(doc) for doc in documents]
    
    def find_by_creation_time(self, start_time: datetime, 
                             end_time: Optional[datetime] = None) -> List[Pattern]:
        """
        Find patterns by creation time.
        
        Args:
            start_time: The start of the time range
            end_time: The end of the time range (defaults to now)
            
        Returns:
            A list of patterns created in the time range
        """
        if not self._initialized:
            self.initialize()
            
        if end_time is None:
            end_time = datetime.utcnow()
            
        start_str = start_time.isoformat()
        end_str = end_time.isoformat()
        
        query = """
        FOR p IN patterns
        FILTER p.created_at >= @start_time AND p.created_at <= @end_time
        RETURN p
        """
        
        documents = self._db_connection.execute_query(query, {
            "start_time": start_str,
            "end_time": end_str
        })
        
        return [self._to_entity(doc) for doc in documents]
    
    def find_related_patterns(self, pattern_id: str, 
                             relationship_type: Optional[str] = None,
                             max_depth: int = 1) -> List[Pattern]:
        """
        Find patterns related to the specified pattern.
        
        Args:
            pattern_id: The ID of the pattern
            relationship_type: Optional type of relationship
            max_depth: The maximum relationship depth
            
        Returns:
            A list of related patterns
        """
        if not self._initialized:
            self.initialize()
            
        query = f"""
        FOR v, e, p IN 1..@max_depth ANY @pattern_id GRAPH @graph_name
        """
        
        bind_vars = {
            "pattern_id": pattern_id,
            "max_depth": max_depth,
            "graph_name": self._graph_name
        }
        
        if relationship_type:
            query += " FILTER e.type == @relationship_type"
            bind_vars["relationship_type"] = relationship_type
            
        query += " RETURN v"
        
        documents = self._db_connection.execute_query(query, bind_vars)
        return [self._to_entity(doc) for doc in documents]
    
    def save_pattern_relationship(self, source_id: str, target_id: str,
                                 relationship_type: str,
                                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Save a relationship between two patterns.
        
        Args:
            source_id: The ID of the source pattern
            target_id: The ID of the target pattern
            relationship_type: The type of the relationship
            metadata: Optional metadata for the relationship
            
        Returns:
            The created relationship
        """
        if not self._initialized:
            self.initialize()
            
        return self.create_edge(source_id, target_id, relationship_type, metadata)
    
    def update_pattern_quality(self, pattern_id: str, 
                              quality: float,
                              coherence: Optional[float] = None,
                              stability: Optional[float] = None) -> Optional[Pattern]:
        """
        Update the quality of a pattern.
        
        Args:
            pattern_id: The ID of the pattern
            quality: The new quality value (0.0-1.0)
            coherence: Optional coherence value (0.0-1.0)
            stability: Optional stability value (0.0-1.0)
            
        Returns:
            The updated pattern, or None if not found
        """
        if not self._initialized:
            self.initialize()
            
        # Get current pattern
        pattern = self.find_by_id(pattern_id)
        if not pattern:
            return None
            
        # Update metadata
        metadata = pattern.metadata.copy() if pattern.metadata else {}
        metadata['quality'] = quality
        
        if coherence is not None:
            metadata['coherence'] = coherence
            
        if stability is not None:
            metadata['stability'] = stability
            
        # Update pattern
        updates = {
            'metadata': metadata,
            'updated_at': datetime.utcnow().isoformat()
        }
        
        result = self.update_node(pattern_id, updates)
        if result:
            return self._to_entity(result)
        return None
    
    def create_pattern(self, name: str, pattern_type: str,
                      description: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Pattern:
        """
        Create a new pattern.
        
        Args:
            name: The name of the pattern
            pattern_type: The type of the pattern
            description: Optional description of the pattern
            metadata: Optional metadata for the pattern
            
        Returns:
            The created pattern
        """
        if not self._initialized:
            self.initialize()
            
        # Create pattern entity
        pattern = Pattern(
            name=name,
            pattern_type=pattern_type,
            description=description,
            metadata=metadata or {}
        )
        
        # Convert to document and save
        document = self._to_document(pattern)
        result = self.create_node(document)
        
        # Convert result back to entity
        return self._to_entity(result)
    
    def _to_entity(self, document: Dict[str, Any]) -> Pattern:
        """
        Convert a document to a pattern entity.
        
        Args:
            document: The document to convert
            
        Returns:
            The pattern entity
        """
        return Pattern.from_dict(document)
    
    def _to_document(self, entity: Pattern) -> Dict[str, Any]:
        """
        Convert a pattern entity to a document.
        
        Args:
            entity: The entity to convert
            
        Returns:
            The document
        """
        return entity.to_dict()
