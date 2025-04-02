"""
Base ArangoDB repository implementation for the Adaptive Core system.
"""

from typing import Dict, Any, List, Optional, TypeVar, Generic, Union
from abc import ABC
import json
from datetime import datetime
import uuid

T = TypeVar('T')

class Repository(Generic[T], ABC):
    """Base Repository interface."""
    pass
from .connection import ArangoDBConnectionManager

# T is already defined above

class ArangoDBBaseRepository(Repository[T], ABC):
    """
    Base ArangoDB repository implementation providing common CRUD operations.
    Designed to support domain-predicate tracking and evolutionary pattern detection.
    """
    
    def __init__(self):
        self.connection_manager = ArangoDBConnectionManager()
        # Collection name should be set by concrete implementations
        self.collection_name = self.__class__.__name__.replace('Repository', '')
        self.is_edge_collection = False  # Should be overridden by edge repositories
    
    def _to_document_properties(self, entity: T) -> Dict[str, Any]:
        """Convert entity to ArangoDB document properties"""
        if hasattr(entity, 'to_dict'):
            properties = entity.to_dict()
        else:
            properties = entity.__dict__.copy()
            
        # Ensure _key exists (ArangoDB's document identifier)
        if 'id' in properties and '_key' not in properties:
            properties['_key'] = str(properties['id'])
        elif '_key' not in properties:
            properties['_key'] = str(uuid.uuid4())
            
        # Convert datetime objects to ISO format strings
        for key, value in properties.items():
            if isinstance(value, datetime):
                properties[key] = value.isoformat()
                
        return properties
    
    def _from_document_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ArangoDB document properties back to Python types"""
        result = properties.copy()
        
        # Map _key to id if id doesn't exist
        if '_key' in result and 'id' not in result:
            result['id'] = result['_key']
            
        # Handle special ArangoDB fields
        for special_field in ['_id', '_rev', '_key']:
            if special_field in result:
                # Keep these fields but also ensure regular id exists
                pass
                
        return result
    
    def create(self, entity: T) -> str:
        """Create a new document in ArangoDB"""
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Convert entity to document properties
        doc_properties = self._to_document_properties(entity)
        
        # Insert document
        result = collection.insert(doc_properties, return_new=True)
        
        # Return the document key
        return result['_key']
    
    def read(self, entity_id: str) -> Optional[T]:
        """Read a document from ArangoDB"""
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        try:
            # Get document by key
            doc = collection.get(entity_id)
            if not doc:
                return None
                
            # Convert document properties to entity
            entity_dict = self._from_document_properties(doc)
            return self._dict_to_entity(entity_dict)
            
        except Exception as e:
            print(f"Error reading document: {str(e)}")
            return None
    
    def update(self, entity_id: str, entity: T) -> bool:
        """Update a document in ArangoDB"""
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Convert entity to document properties
        doc_properties = self._to_document_properties(entity)
        
        try:
            # Update document
            result = collection.update(entity_id, doc_properties, return_new=True)
            return True
        except Exception as e:
            print(f"Error updating document: {str(e)}")
            return False
    
    def delete(self, entity_id: str) -> bool:
        """Delete a document from ArangoDB"""
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        try:
            # Delete document
            collection.delete(entity_id)
            return True
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False
    
    def list(self, filters: Optional[Dict[str, Any]] = None) -> List[T]:
        """List documents from ArangoDB with optional filtering"""
        db = self.connection_manager.get_db()
        
        # Build AQL query
        query = f"FOR doc IN {self.collection_name}"
        
        # Add filters if provided
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                filter_conditions.append(f"doc.{key} == @{key}")
            
            if filter_conditions:
                query += " FILTER " + " AND ".join(filter_conditions)
        
        query += " RETURN doc"
        
        # Execute query
        cursor = db.aql.execute(query, bind_vars=filters or {})
        
        # Convert results to entities
        entities = []
        for doc in cursor:
            entity_dict = self._from_document_properties(doc)
            entity = self._dict_to_entity(entity_dict)
            entities.append(entity)
            
        return entities
    
    def _dict_to_entity(self, entity_dict: Dict[str, Any]) -> T:
        """Convert a dictionary to an entity - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _dict_to_entity method")
        
    def create_relationship(self, from_id: str, to_id: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """Create a relationship between two documents (for edge collections)"""
        if not self.is_edge_collection:
            raise ValueError(f"Collection {self.collection_name} is not an edge collection")
            
        db = self.connection_manager.get_db()
        edge_collection = db.collection(self.collection_name)
        
        # Prepare edge document
        edge_doc = properties or {}
        edge_doc['_from'] = from_id
        edge_doc['_to'] = to_id
        
        # Insert edge
        result = edge_collection.insert(edge_doc)
        
        return result['_key']
        
    def get_connected(self, doc_id: str, direction: str = 'outbound') -> List[Dict[str, Any]]:
        """Get connected documents (for edge collections)"""
        if not self.is_edge_collection:
            raise ValueError(f"Collection {self.collection_name} is not an edge collection")
            
        db = self.connection_manager.get_db()
        
        # Build AQL query based on direction
        if direction == 'outbound':
            query = f"""
            FOR v, e IN 1..1 OUTBOUND @doc_id {self.collection_name}
                RETURN {{document: v, edge: e}}
            """
        elif direction == 'inbound':
            query = f"""
            FOR v, e IN 1..1 INBOUND @doc_id {self.collection_name}
                RETURN {{document: v, edge: e}}
            """
        else:  # any
            query = f"""
            FOR v, e IN 1..1 ANY @doc_id {self.collection_name}
                RETURN {{document: v, edge: e}}
            """
            
        # Execute query
        cursor = db.aql.execute(query, bind_vars={'doc_id': doc_id})
        
        # Return results
        return list(cursor)
