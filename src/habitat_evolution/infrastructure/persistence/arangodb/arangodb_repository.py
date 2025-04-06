"""
ArangoDB repository implementation for Habitat Evolution.

This module provides a concrete implementation of the RepositoryInterface
using ArangoDB as the persistence layer, supporting the pattern evolution and
co-evolution principles of Habitat Evolution.
"""

import logging
from typing import Dict, List, Any, Optional, TypeVar, Generic, Type
from datetime import datetime
import uuid

from src.habitat_evolution.infrastructure.interfaces.repositories.repository_interface import RepositoryInterface
from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface

T = TypeVar('T')
logger = logging.getLogger(__name__)


class ArangoDBRepository(RepositoryInterface[T], Generic[T]):
    """
    ArangoDB implementation of the RepositoryInterface.
    
    This repository provides a consistent approach to data access using ArangoDB
    as the persistence layer, supporting the pattern evolution and co-evolution
    principles of Habitat Evolution.
    """
    
    def __init__(self, 
                 collection_name: str,
                 db_connection: ArangoDBConnectionInterface,
                 event_service: EventServiceInterface,
                 entity_class: Type[T]):
        """
        Initialize a new ArangoDB repository.
        
        Args:
            collection_name: The name of the collection
            db_connection: The ArangoDB connection to use
            event_service: The event service for publishing events
            entity_class: The class to use for entity instantiation
        """
        self._collection_name = collection_name
        self._db_connection = db_connection
        self._event_service = event_service
        self._entity_class = entity_class
        self._initialized = False
        logger.debug(f"ArangoDBRepository created for collection: {collection_name}")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the repository with the specified configuration.
        
        Args:
            config: Optional configuration for the repository
        """
        if self._initialized:
            logger.warning(f"ArangoDBRepository for {self._collection_name} already initialized")
            return
            
        logger.info(f"Initializing ArangoDBRepository for {self._collection_name}")
        
        # Ensure collection exists
        self._db_connection.ensure_collection(self._collection_name)
        
        self._initialized = True
        logger.info(f"ArangoDBRepository for {self._collection_name} initialized")
        
        # Publish initialization event
        self._event_service.publish("repository.initialized", {
            "collection": self._collection_name,
            "repository_type": "ArangoDBRepository"
        })
    
    def shutdown(self) -> None:
        """
        Release resources when shutting down the repository.
        """
        if not self._initialized:
            logger.warning(f"ArangoDBRepository for {self._collection_name} not initialized")
            return
            
        logger.info(f"Shutting down ArangoDBRepository for {self._collection_name}")
        self._initialized = False
        logger.info(f"ArangoDBRepository for {self._collection_name} shut down")
        
        # Publish shutdown event
        self._event_service.publish("repository.shutdown", {
            "collection": self._collection_name,
            "repository_type": "ArangoDBRepository"
        })
    
    def find_by_id(self, entity_id: str) -> Optional[T]:
        """
        Find an entity by ID.
        
        Args:
            entity_id: The ID of the entity to find
            
        Returns:
            The entity, or None if not found
        """
        if not self._initialized:
            self.initialize()
            
        try:
            document = self._db_connection.get_document(self._collection_name, entity_id)
            return self._to_entity(document)
        except Exception as e:
            logger.error(f"Error finding entity {entity_id} in {self._collection_name}: {str(e)}")
            return None
    
    def find_all(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[T]:
        """
        Find all entities matching the filter criteria.
        
        Args:
            filter_criteria: Optional criteria to filter entities by
            
        Returns:
            A list of matching entities
        """
        if not self._initialized:
            self.initialize()
            
        query = f"FOR doc IN {self._collection_name}"
        bind_vars = {}
        
        if filter_criteria:
            filters = []
            for key, value in filter_criteria.items():
                if key == "metadata":
                    for meta_key, meta_value in value.items():
                        filters.append(f"doc.metadata.{meta_key} == @metadata_{meta_key}")
                        bind_vars[f"metadata_{meta_key}"] = meta_value
                else:
                    filters.append(f"doc.{key} == @{key}")
                    bind_vars[key] = value
                    
            if filters:
                query += " FILTER " + " AND ".join(filters)
                
        query += " RETURN doc"
        
        documents = self._db_connection.execute_query(query, bind_vars)
        return [self._to_entity(doc) for doc in documents]
    
    def save(self, entity: T) -> T:
        """
        Save an entity.
        
        Args:
            entity: The entity to save
            
        Returns:
            The saved entity
        """
        if not self._initialized:
            self.initialize()
            
        document = self._to_document(entity)
        
        # Check if this is an update or insert
        is_update = hasattr(entity, 'id') and getattr(entity, 'id') is not None
        
        if is_update:
            # Update existing document
            entity_id = getattr(entity, 'id')
            result = self._db_connection.update_document(self._collection_name, entity_id, document)
            
            # Publish entity updated event
            self._event_service.publish("repository.entity_updated", {
                "collection": self._collection_name,
                "entity_id": entity_id,
                "entity_type": self._entity_class.__name__
            })
        else:
            # Insert new document
            if '_key' not in document:
                document['_key'] = str(uuid.uuid4())
                
            result = self._db_connection.insert(self._collection_name, document)
            
            # Set the ID on the entity
            if hasattr(entity, 'id'):
                setattr(entity, 'id', result['_id'])
                
            # Publish entity created event
            self._event_service.publish("repository.entity_created", {
                "collection": self._collection_name,
                "entity_id": result['_id'],
                "entity_type": self._entity_class.__name__
            })
            
        return self._to_entity(result)
    
    def delete(self, entity_id: str) -> bool:
        """
        Delete an entity.
        
        Args:
            entity_id: The ID of the entity to delete
            
        Returns:
            True if the entity was deleted, False otherwise
        """
        if not self._initialized:
            self.initialize()
            
        try:
            self._db_connection.delete_document(self._collection_name, entity_id)
            
            # Publish entity deleted event
            self._event_service.publish("repository.entity_deleted", {
                "collection": self._collection_name,
                "entity_id": entity_id,
                "entity_type": self._entity_class.__name__
            })
            
            return True
        except Exception as e:
            logger.error(f"Error deleting entity {entity_id} from {self._collection_name}: {str(e)}")
            return False
    
    def count(self, filter_criteria: Optional[Dict[str, Any]] = None) -> int:
        """
        Count entities matching the filter criteria.
        
        Args:
            filter_criteria: Optional criteria to filter entities by
            
        Returns:
            The number of matching entities
        """
        if not self._initialized:
            self.initialize()
            
        query = f"RETURN LENGTH({self._collection_name})"
        
        if filter_criteria:
            query = f"RETURN LENGTH(FOR doc IN {self._collection_name}"
            bind_vars = {}
            
            filters = []
            for key, value in filter_criteria.items():
                if key == "metadata":
                    for meta_key, meta_value in value.items():
                        filters.append(f"doc.metadata.{meta_key} == @metadata_{meta_key}")
                        bind_vars[f"metadata_{meta_key}"] = meta_value
                else:
                    filters.append(f"doc.{key} == @{key}")
                    bind_vars[key] = value
                    
            if filters:
                query += " FILTER " + " AND ".join(filters)
                
            query += " RETURN doc)"
            
            results = self._db_connection.execute_query(query, bind_vars)
        else:
            results = self._db_connection.execute_query(query)
            
        return results[0] if results else 0
    
    def _to_entity(self, document: Dict[str, Any]) -> T:
        """
        Convert a document to an entity.
        
        Args:
            document: The document to convert
            
        Returns:
            The entity
        """
        # This is a base implementation that needs to be overridden by subclasses
        # for proper entity conversion
        if hasattr(self._entity_class, 'from_dict'):
            return self._entity_class.from_dict(document)
        else:
            # Simple conversion for basic entities
            entity = self._entity_class()
            
            # Map document fields to entity attributes
            for key, value in document.items():
                if key.startswith('_'):
                    # Handle special ArangoDB fields
                    if key == '_id':
                        if hasattr(entity, 'id'):
                            setattr(entity, 'id', value)
                    elif key == '_key':
                        if hasattr(entity, 'key'):
                            setattr(entity, 'key', value)
                else:
                    # Map regular fields
                    if hasattr(entity, key):
                        setattr(entity, key, value)
                        
            return entity
    
    def _to_document(self, entity: T) -> Dict[str, Any]:
        """
        Convert an entity to a document.
        
        Args:
            entity: The entity to convert
            
        Returns:
            The document
        """
        # This is a base implementation that needs to be overridden by subclasses
        # for proper document conversion
        if hasattr(entity, 'to_dict'):
            return entity.to_dict()
        else:
            # Simple conversion for basic entities
            document = {}
            
            # Map entity attributes to document fields
            for attr in dir(entity):
                # Skip private and special attributes
                if attr.startswith('_') or callable(getattr(entity, attr)):
                    continue
                    
                value = getattr(entity, attr)
                
                # Handle special attributes
                if attr == 'id':
                    if value is not None:
                        if '/' in value:
                            document['_id'] = value
                            document['_key'] = value.split('/')[1]
                        else:
                            document['_key'] = value
                elif attr == 'key':
                    if value is not None:
                        document['_key'] = value
                else:
                    # Map regular attributes
                    document[attr] = value
                    
            return document
