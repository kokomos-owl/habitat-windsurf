"""
Repository interface and implementation for field state storage.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from datetime import datetime

from ..services.field.interfaces import FieldState
from .neo4j.base_repository import Neo4jBaseRepository

class FieldRepository(ABC):
    """Repository interface for field state storage"""
    
    @abstractmethod
    async def get_field_state(self, field_id: str) -> Optional[FieldState]:
        """Get the current state of a field"""
        pass
        
    @abstractmethod
    async def update_field_state(self, field_id: str, state: FieldState) -> None:
        """Update the state of a field"""
        pass
        
    @abstractmethod
    async def delete_field_state(self, field_id: str) -> None:
        """Delete a field's state"""
        pass

class Neo4jFieldRepository(Neo4jBaseRepository[FieldState], FieldRepository):
    """Neo4j implementation of field repository"""
    
    def __init__(self):
        super().__init__()
        self.node_label = "FieldState"
        
    def _create_entity(self, properties: Dict) -> FieldState:
        """Create a FieldState instance from Neo4j properties"""
        return FieldState(
            field_id=properties["field_id"],
            timestamp=datetime.fromisoformat(properties["timestamp"]),
            potential=float(properties["potential"]),
            gradient=properties["gradient"],
            stability=float(properties["stability"]),
            metadata=properties["metadata"]
        )
        
    async def get_field_state(self, field_id: str) -> Optional[FieldState]:
        """Get the current state of a field"""
        query = (
            f"MATCH (n:{self.node_label} {{field_id: $field_id}}) "
            "RETURN n"
        )
        
        with self.connection_manager.get_session() as session:
            result = session.run(query, field_id=field_id)
            record = result.single()
            if not record:
                return None
                
            properties = self._from_node_properties(dict(record["n"]))
            return self._create_entity(properties)
            
    async def update_field_state(self, field_id: str, state: FieldState) -> None:
        """Update the state of a field"""
        properties = self._to_node_properties(state)
        properties["last_modified"] = datetime.now().isoformat()
        
        query = (
            f"MERGE (n:{self.node_label} {{field_id: $field_id}}) "
            "SET n += $properties"
        )
        
        with self.connection_manager.get_session() as session:
            session.run(
                query,
                field_id=field_id,
                properties=properties
            )
            
    async def delete_field_state(self, field_id: str) -> None:
        """Delete a field's state"""
        query = (
            f"MATCH (n:{self.node_label} {{field_id: $field_id}}) "
            "DETACH DELETE n"
        )
        
        with self.connection_manager.get_session() as session:
            result = session.run(query, field_id=field_id)
            if result.consume().counters.nodes_deleted == 0:
                raise ValueError(f"Field state with id {field_id} not found")
