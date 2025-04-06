"""
ArangoDB implementation of the DocumentServiceInterface for Habitat Evolution.

This module provides a concrete implementation of the DocumentServiceInterface
using ArangoDB as the persistence layer, supporting the pattern evolution and
co-evolution principles of Habitat Evolution.
"""

import logging
from typing import Dict, List, Any, Optional, Union, BinaryIO
from datetime import datetime
import uuid
import json
import os
import hashlib

from src.habitat_evolution.infrastructure.interfaces.services.document_service_interface import DocumentServiceInterface
from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface

logger = logging.getLogger(__name__)


class ArangoDBDocumentService(DocumentServiceInterface):
    """
    ArangoDB implementation of the DocumentServiceInterface.
    
    This service provides a consistent approach to document management using ArangoDB
    as the persistence layer, supporting the pattern evolution and co-evolution
    principles of Habitat Evolution.
    """
    
    def __init__(self, 
                 db_connection: ArangoDBConnectionInterface,
                 event_service: EventServiceInterface):
        """
        Initialize a new ArangoDB document service.
        
        Args:
            db_connection: The ArangoDB connection to use
            event_service: The event service for publishing events
        """
        self._db_connection = db_connection
        self._event_service = event_service
        self._initialized = False
        logger.debug("ArangoDBDocumentService created")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the document service with the specified configuration.
        
        Args:
            config: Optional configuration for the document service
        """
        if self._initialized:
            logger.warning("ArangoDBDocumentService already initialized")
            return
            
        logger.info("Initializing ArangoDBDocumentService")
        
        # Ensure required collections exist
        self._db_connection.ensure_collection("documents")
        self._db_connection.ensure_collection("patterns")
        self._db_connection.ensure_edge_collection("document_patterns")
        
        # Create graph if it doesn't exist
        self._db_connection.ensure_graph(
            "document_pattern_graph",
            edge_definitions=[
                {
                    "collection": "document_patterns",
                    "from": ["documents"],
                    "to": ["patterns"]
                }
            ]
        )
        
        self._initialized = True
        logger.info("ArangoDBDocumentService initialized")
        
        # Publish initialization event
        self._event_service.publish("document_service.initialized", {
            "service": "ArangoDBDocumentService"
        })
    
    def shutdown(self) -> None:
        """
        Release resources when shutting down the document service.
        """
        if not self._initialized:
            logger.warning("ArangoDBDocumentService not initialized")
            return
            
        logger.info("Shutting down ArangoDBDocumentService")
        self._initialized = False
        logger.info("ArangoDBDocumentService shut down")
        
        # Publish shutdown event
        self._event_service.publish("document_service.shutdown", {
            "service": "ArangoDBDocumentService"
        })
    
    def process_document(self, content: Union[str, BinaryIO], 
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document.
        
        Args:
            content: The document content as text or file-like object
            metadata: Optional metadata for the document
            
        Returns:
            Processing results including document ID and extracted information
        """
        if not self._initialized:
            self.initialize()
            
        # Convert file-like object to string if necessary
        if hasattr(content, 'read'):
            content = content.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
        
        # Generate document ID and hash
        document_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Create document object
        document = {
            "_key": document_id,
            "content": content,
            "content_hash": content_hash,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "processed": True
        }
        
        # Insert document
        result = self._db_connection.insert("documents", document)
        
        # Extract patterns (placeholder implementation)
        patterns = self._extract_patterns_from_content(content)
        
        # Link patterns to document
        for pattern in patterns:
            self.add_pattern_to_document(result["_id"], pattern["_id"], {
                "confidence": pattern["confidence"]
            })
        
        # Publish document processed event
        self._event_service.publish("document.processed", {
            "document_id": result["_id"],
            "pattern_count": len(patterns)
        })
        
        return {
            "document_id": result["_id"],
            "content_hash": content_hash,
            "patterns": patterns,
            "metadata": metadata or {}
        }
    
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Get a document by ID.
        
        Args:
            document_id: The ID of the document to get
            
        Returns:
            The document
        """
        if not self._initialized:
            self.initialize()
            
        try:
            return self._db_connection.get_document("documents", document_id)
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {str(e)}")
            raise ValueError(f"Document not found: {document_id}")
    
    def find_documents(self, filter_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find documents matching the filter criteria.
        
        Args:
            filter_criteria: Criteria to filter documents by
            
        Returns:
            A list of matching documents
        """
        if not self._initialized:
            self.initialize()
            
        query = "FOR d IN documents"
        filters = []
        bind_vars = {}
        
        # Process filter criteria
        for key, value in filter_criteria.items():
            if key == "metadata":
                for meta_key, meta_value in value.items():
                    filters.append(f"d.metadata.{meta_key} == @metadata_{meta_key}")
                    bind_vars[f"metadata_{meta_key}"] = meta_value
            elif key == "created_after":
                filters.append("d.created_at >= @created_after")
                bind_vars["created_after"] = value
            elif key == "created_before":
                filters.append("d.created_at <= @created_before")
                bind_vars["created_before"] = value
            elif key == "pattern_id":
                # This requires a graph traversal, handled separately
                pass
            else:
                filters.append(f"d.{key} == @{key}")
                bind_vars[key] = value
        
        # Add filters to query
        if filters:
            query += " FILTER " + " AND ".join(filters)
            
        # Handle pattern_id filter separately
        if "pattern_id" in filter_criteria:
            pattern_query = """
            FOR d IN INBOUND @pattern_id document_patterns
            RETURN d
            """
            return self._db_connection.execute_query(
                pattern_query, 
                {"pattern_id": filter_criteria["pattern_id"]}
            )
        
        # Complete and execute query
        query += " RETURN d"
        return self._db_connection.execute_query(query, bind_vars)
    
    def search_documents(self, query: str, 
                        filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search documents by content.
        
        Args:
            query: The search query
            filter_criteria: Optional criteria to filter documents by
            
        Returns:
            A list of matching documents
        """
        if not self._initialized:
            self.initialize()
            
        # Simple content search implementation
        # In a real implementation, this would use full-text search capabilities
        aql_query = """
        FOR d IN documents
        FILTER CONTAINS(d.content, @query, true)
        """
        
        bind_vars = {"query": query}
        
        # Add filter criteria if provided
        if filter_criteria:
            filters = []
            for key, value in filter_criteria.items():
                if key == "metadata":
                    for meta_key, meta_value in value.items():
                        filters.append(f"d.metadata.{meta_key} == @metadata_{meta_key}")
                        bind_vars[f"metadata_{meta_key}"] = meta_value
                else:
                    filters.append(f"d.{key} == @{key}")
                    bind_vars[key] = value
                    
            if filters:
                aql_query += " AND " + " AND ".join(filters)
        
        aql_query += " RETURN d"
        return self._db_connection.execute_query(aql_query, bind_vars)
    
    def update_document(self, document_id: str, 
                       updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a document.
        
        Args:
            document_id: The ID of the document to update
            updates: The updates to apply to the document
            
        Returns:
            The updated document
        """
        if not self._initialized:
            self.initialize()
            
        # Get current document
        document = self.get_document(document_id)
        
        # Apply updates
        for key, value in updates.items():
            if key == "metadata":
                # Merge metadata
                document["metadata"].update(value)
            else:
                document[key] = value
                
        # Update timestamp
        document["updated_at"] = datetime.utcnow().isoformat()
        
        # Update document
        result = self._db_connection.update_document("documents", document_id, document)
        
        # Publish document updated event
        self._event_service.publish("document.updated", {
            "document_id": document_id,
            "updates": list(updates.keys())
        })
        
        return result
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            document_id: The ID of the document to delete
            
        Returns:
            True if the document was deleted, False otherwise
        """
        if not self._initialized:
            self.initialize()
            
        try:
            # Delete document-pattern edges first
            query = """
            FOR edge IN document_patterns
            FILTER edge._from == @document_id
            REMOVE edge IN document_patterns
            """
            self._db_connection.execute_query(query, {"document_id": document_id})
            
            # Delete document
            self._db_connection.delete_document("documents", document_id)
            
            # Publish document deleted event
            self._event_service.publish("document.deleted", {
                "document_id": document_id
            })
            
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    def extract_patterns(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Extract patterns from a document.
        
        Args:
            document_id: The ID of the document to extract patterns from
            
        Returns:
            A list of extracted patterns
        """
        if not self._initialized:
            self.initialize()
            
        # Get document
        document = self.get_document(document_id)
        
        # Extract patterns
        patterns = self._extract_patterns_from_content(document["content"])
        
        # Link patterns to document
        for pattern in patterns:
            self.add_pattern_to_document(document_id, pattern["_id"], {
                "confidence": pattern["confidence"]
            })
        
        # Publish patterns extracted event
        self._event_service.publish("document.patterns_extracted", {
            "document_id": document_id,
            "pattern_count": len(patterns)
        })
        
        return patterns
    
    def get_document_patterns(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get patterns associated with a document.
        
        Args:
            document_id: The ID of the document to get patterns for
            
        Returns:
            A list of patterns associated with the document
        """
        if not self._initialized:
            self.initialize()
            
        query = """
        FOR pattern, edge IN OUTBOUND @document_id document_patterns
        RETURN {
            "pattern": pattern,
            "metadata": edge
        }
        """
        
        results = self._db_connection.execute_query(query, {"document_id": document_id})
        return [result["pattern"] for result in results]
    
    def get_document_metrics(self, document_id: str) -> Dict[str, Any]:
        """
        Get metrics for a document.
        
        Args:
            document_id: The ID of the document to get metrics for
            
        Returns:
            Metrics for the document
        """
        if not self._initialized:
            self.initialize()
            
        # Get document
        document = self.get_document(document_id)
        
        # Get patterns
        patterns = self.get_document_patterns(document_id)
        
        # Calculate metrics
        metrics = {
            "document_id": document_id,
            "content_length": len(document["content"]),
            "pattern_count": len(patterns),
            "created_at": document["created_at"],
            "updated_at": document["updated_at"]
        }
        
        return metrics
    
    def add_pattern_to_document(self, document_id: str, pattern_id: str, 
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Associate a pattern with a document.
        
        Args:
            document_id: The ID of the document
            pattern_id: The ID of the pattern
            metadata: Optional metadata for the association
            
        Returns:
            True if the association was created, False otherwise
        """
        if not self._initialized:
            self.initialize()
            
        edge = {
            "_from": document_id,
            "_to": pattern_id,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        try:
            self._db_connection.insert("document_patterns", edge)
            return True
        except Exception as e:
            logger.error(f"Error adding pattern {pattern_id} to document {document_id}: {str(e)}")
            return False
    
    def _extract_patterns_from_content(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract patterns from document content.
        
        This is a placeholder implementation. In a real implementation,
        this would use NLP, ML, or other pattern extraction techniques.
        
        Args:
            content: The document content
            
        Returns:
            A list of extracted patterns
        """
        # Placeholder implementation
        # In a real implementation, this would use more sophisticated techniques
        patterns = []
        
        # Simple keyword-based pattern extraction
        keywords = ["pattern", "evolution", "habitat", "emergence", "coherence"]
        for keyword in keywords:
            if keyword in content.lower():
                pattern_id = f"pattern/{str(uuid.uuid4())}"
                pattern = {
                    "_key": pattern_id.split("/")[1],
                    "_id": pattern_id,
                    "name": f"{keyword.capitalize()} Pattern",
                    "type": "keyword",
                    "confidence": 0.7,
                    "created_at": datetime.utcnow().isoformat()
                }
                patterns.append(pattern)
                
                # Insert pattern into database
                try:
                    self._db_connection.insert("patterns", {
                        "_key": pattern_id.split("/")[1],
                        "name": pattern["name"],
                        "type": pattern["type"],
                        "created_at": pattern["created_at"]
                    })
                except Exception as e:
                    logger.error(f"Error inserting pattern: {str(e)}")
        
        return patterns
