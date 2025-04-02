"""
Base Neo4j repository implementation for the Adaptive Core system.
"""

from typing import Dict, Any, List, Optional, TypeVar, Generic
from abc import ABC
import json
from datetime import datetime

# This is a mock implementation since we're not using Neo4j anymore
from abc import ABC
from typing import TypeVar, Generic, Dict, Any, List, Optional

T = TypeVar('T')

class Repository(Generic[T], ABC):
    """Mock Repository interface for backward compatibility."""
    pass
from .connection import Neo4jConnectionManager

class Neo4jBaseRepository(Repository[T], ABC):
    """
    Base Neo4j repository implementation providing common CRUD operations.
    """
    
    def __init__(self):
        self.connection_manager = Neo4jConnectionManager()
        # Node label should be set by concrete implementations
        self.node_label = self.__class__.__name__.replace('Repository', '')
    
    def _to_node_properties(self, entity: T) -> Dict[str, Any]:
        """Convert entity to Neo4j node properties"""
        if hasattr(entity, 'to_dict'):
            properties = entity.to_dict()
        else:
            properties = entity.__dict__
            
        # Convert any non-primitive types to JSON strings
        for key, value in properties.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                properties[key] = json.dumps(value)
                
        return properties
    
    def _from_node_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Neo4j node properties back to Python types"""
        result = {}
        for key, value in properties.items():
            if isinstance(value, str):
                try:
                    # Attempt to parse JSON strings back to Python objects
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    result[key] = value
            else:
                result[key] = value
        return result

    def create(self, entity: T) -> str:
        """Create a new entity node in Neo4j"""
        properties = self._to_node_properties(entity)
        
        # Add metadata
        properties['created_at'] = datetime.now().isoformat()
        properties['last_modified'] = properties['created_at']
        
        query = (
            f"CREATE (n:{self.node_label} $properties) "
            "RETURN n.id as id"
        )
        
        with self.connection_manager.get_session() as session:
            result = session.run(query, properties=properties)
            record = result.single()
            if not record:
                raise ValueError("Failed to create entity")
            return record["id"]

    def read(self, entity_id: str) -> Optional[T]:
        """Read an entity node from Neo4j"""
        query = (
            f"MATCH (n:{self.node_label} {{id: $id}}) "
            "RETURN n"
        )
        
        with self.connection_manager.get_session() as session:
            result = session.run(query, id=entity_id)
            record = result.single()
            if not record:
                return None
                
            properties = self._from_node_properties(dict(record["n"]))
            return self._create_entity(properties)

    def update(self, entity: T) -> None:
        """Update an entity node in Neo4j"""
        properties = self._to_node_properties(entity)
        properties['last_modified'] = datetime.now().isoformat()
        
        query = (
            f"MATCH (n:{self.node_label} {{id: $id}}) "
            "SET n += $properties"
        )
        
        with self.connection_manager.get_session() as session:
            result = session.run(
                query,
                id=properties['id'],
                properties=properties
            )
            if result.consume().counters.nodes_created == 0:
                raise ValueError(f"Entity with id {properties['id']} not found")

    def delete(self, entity_id: str) -> None:
        """Delete an entity node from Neo4j"""
        query = (
            f"MATCH (n:{self.node_label} {{id: $id}}) "
            "DETACH DELETE n"
        )
        
        with self.connection_manager.get_session() as session:
            result = session.run(query, id=entity_id)
            if result.consume().counters.nodes_deleted == 0:
                raise ValueError(f"Entity with id {entity_id} not found")

    def list(self, filter_params: Optional[Dict[str, Any]] = None) -> List[T]:
        """List entity nodes from Neo4j with optional filtering"""
        where_clause = ""
        if filter_params:
            conditions = []
            for key, value in filter_params.items():
                conditions.append(f"n.{key} = ${key}")
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
        
        query = (
            f"MATCH (n:{self.node_label}) "
            f"{where_clause} "
            "RETURN n"
        )
        
        with self.connection_manager.get_session() as session:
            result = session.run(query, **(filter_params or {}))
            entities = []
            for record in result:
                properties = self._from_node_properties(dict(record["n"]))
                entity = self._create_entity(properties)
                entities.append(entity)
            return entities

    def _create_entity(self, properties: Dict[str, Any]) -> T:
        """
        Create an entity instance from properties.
        Must be implemented by concrete repository classes.
        """
        raise NotImplementedError("Concrete repositories must implement _create_entity")
