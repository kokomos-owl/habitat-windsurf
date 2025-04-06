"""
Repository interface and implementation for field state storage.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from datetime import datetime

from ..services.field.interfaces import FieldState
from ...infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection

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

class ArangoFieldRepository(FieldRepository):
    """ArangoDB implementation of field repository"""
    
    def __init__(self, db_connection: ArangoDBConnection):
        self.db_connection = db_connection
        self.collection_name = "field_states"
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure the field_states collection exists"""
        if not self.db_connection.collection_exists(self.collection_name):
            self.db_connection.create_collection(self.collection_name)
    
    def _create_entity(self, properties: Dict) -> FieldState:
        """Create a FieldState instance from ArangoDB properties"""
        return FieldState(
            field_id=properties.get("field_id"),
            state_vector=properties.get("state_vector", {}),
            pressure=properties.get("pressure", 0.0),
            stability=properties.get("stability", 0.0),
            timestamp=properties.get("timestamp", datetime.now())
        )
    
    async def get_field_state(self, field_id: str) -> Optional[FieldState]:
        """Get the current state of a field"""
        query = f"FOR doc IN {self.collection_name} FILTER doc.field_id == @field_id RETURN doc"
        
        result = await self.db_connection.execute_query(query, bind_vars={"field_id": field_id})
        
        if not result or len(result) == 0:
            return None
            
        properties = result[0]
        return self._create_entity(properties)
    
    async def update_field_state(self, field_id: str, state: FieldState) -> None:
        """Update the state of a field"""
        query = f"""
        UPSERT {{ field_id: @field_id }}
        INSERT {{
            field_id: @field_id,
            state_vector: @state_vector,
            pressure: @pressure,
            stability: @stability,
            timestamp: @timestamp
        }}
        UPDATE {{
            state_vector: @state_vector,
            pressure: @pressure,
            stability: @stability,
            timestamp: @timestamp
        }}
        IN {self.collection_name}
        """
        
        params = {
            "field_id": field_id,
            "state_vector": state.state_vector,
            "pressure": state.pressure,
            "stability": state.stability,
            "timestamp": state.timestamp
        }
        
        await self.db_connection.execute_query(query, bind_vars=params)
    
    async def delete_field_state(self, field_id: str) -> None:
        """Delete a field's state"""
        query = f"FOR doc IN {self.collection_name} FILTER doc.field_id == @field_id REMOVE doc IN {self.collection_name}"
        
        await self.db_connection.execute_query(query, bind_vars={"field_id": field_id})
