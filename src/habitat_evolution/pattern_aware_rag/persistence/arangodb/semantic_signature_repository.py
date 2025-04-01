"""
SemanticSignature Repository for ArangoDB.

Handles persistence of SemanticSignature objects to ArangoDB, including
entity identity across time periods and basic pattern matching for RAG.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime
import uuid
import logging
import numpy as np

from src.habitat_evolution.adaptive_core.persistence.arangodb.base_repository import ArangoDBBaseRepository
from src.habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager

logger = logging.getLogger(__name__)

class SemanticSignature:
    """
    Represents a semantic signature for an entity.
    
    A semantic signature captures essential identity information across time periods
    and supports basic pattern matching for RAG.
    """
    
    def __init__(
        self,
        id: str = None,
        entity_id: str = None,
        signature_vector: List[float] = None,
        temporal_context: Dict[str, Any] = None,
        version: int = 1
    ):
        """
        Initialize a semantic signature.
        
        Args:
            id: The ID of the signature
            entity_id: The ID of the entity this signature represents
            signature_vector: The vector representation of the signature
            temporal_context: Temporal context information
            version: The version of the signature
        """
        self.id = id or str(uuid.uuid4())
        self.entity_id = entity_id
        self.signature_vector = signature_vector or []
        self.temporal_context = temporal_context or {}
        self.version = version
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the semantic signature to a dictionary.
        
        Returns:
            A dictionary representation of the signature
        """
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "signature_vector": self.signature_vector,
            "temporal_context": self.temporal_context,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticSignature':
        """
        Create a semantic signature from a dictionary.
        
        Args:
            data: The dictionary to create the signature from
            
        Returns:
            A SemanticSignature object
        """
        signature = cls(
            id=data.get('id'),
            entity_id=data.get('entity_id'),
            signature_vector=data.get('signature_vector', []),
            temporal_context=data.get('temporal_context', {}),
            version=data.get('version', 1)
        )
        
        if 'created_at' in data:
            signature.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            signature.updated_at = datetime.fromisoformat(data['updated_at'])
        
        return signature
    
    def similarity(self, other: 'SemanticSignature') -> float:
        """
        Calculate the cosine similarity between this signature and another.
        
        Args:
            other: The other signature to compare with
            
        Returns:
            The cosine similarity between the signatures
        """
        if not self.signature_vector or not other.signature_vector:
            return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(self.signature_vector)
        vec2 = np.array(other.signature_vector)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class SemanticSignatureRepository(ArangoDBBaseRepository):
    """
    Repository for persisting SemanticSignature objects to ArangoDB.
    
    This repository handles the serialization and deserialization of semantic signatures,
    including entity identity across time periods and basic pattern matching for RAG.
    """
    
    def __init__(self):
        """Initialize the repository with the appropriate collection name."""
        super().__init__()
        self.collection_name = "SemanticSignature"
        self.connection_manager = ArangoDBConnectionManager()
    
    def _to_document_properties(self, signature: SemanticSignature) -> Dict[str, Any]:
        """
        Convert a SemanticSignature to ArangoDB document properties.
        
        Args:
            signature: The semantic signature to convert
            
        Returns:
            A dictionary of document properties
        """
        # Convert to dictionary
        properties = signature.to_dict()
        
        # Ensure _key is set
        properties["_key"] = signature.id
        
        # Convert signature vector to JSON
        properties["signature_vector"] = json.dumps(signature.signature_vector)
        
        # Convert temporal context to JSON
        properties["temporal_context"] = json.dumps(signature.temporal_context)
        
        return properties
    
    def _dict_to_entity(self, properties: Dict[str, Any]) -> SemanticSignature:
        """
        Convert ArangoDB document properties to a SemanticSignature.
        
        Args:
            properties: The document properties to convert
            
        Returns:
            A SemanticSignature object
        """
        # Convert JSON properties
        if 'signature_vector' in properties and isinstance(properties['signature_vector'], str):
            properties['signature_vector'] = json.loads(properties['signature_vector'])
        
        if 'temporal_context' in properties and isinstance(properties['temporal_context'], str):
            properties['temporal_context'] = json.loads(properties['temporal_context'])
        
        # Create signature from dictionary
        return SemanticSignature.from_dict(properties)
    
    def save(self, signature: SemanticSignature) -> str:
        """
        Save a SemanticSignature to ArangoDB.
        
        Args:
            signature: The semantic signature to save
            
        Returns:
            The ID of the saved signature
        """
        # Update the updated_at timestamp
        signature.updated_at = datetime.now()
        
        # Convert to document properties
        doc_properties = self._to_document_properties(signature)
        
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Check if the document already exists
        existing_doc = None
        try:
            existing_doc = collection.get(signature.id)
        except:
            pass
        
        if existing_doc:
            # Update the existing document
            collection.update(doc_properties)
        else:
            # Insert a new document
            collection.insert(doc_properties)
        
        # Return the ID
        return signature.id
    
    def find_by_id(self, signature_id: str) -> Optional[SemanticSignature]:
        """
        Find a SemanticSignature by ID.
        
        Args:
            signature_id: The ID of the signature to find
            
        Returns:
            The signature if found, None otherwise
        """
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Get the document
        doc = collection.get(signature_id)
        if not doc:
            return None
        
        # Convert to entity
        return self._dict_to_entity(doc)
    
    def find_by_entity_id(self, entity_id: str) -> List[SemanticSignature]:
        """
        Find SemanticSignature objects for a specific entity.
        
        Args:
            entity_id: The ID of the entity to search for
            
        Returns:
            A list of matching signatures
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for signatures with the given entity ID
        query = """
        FOR s IN SemanticSignature
            FILTER s.entity_id == @entity_id
            SORT s.version DESC
            RETURN s
        """
        
        cursor = db.aql.execute(query, bind_vars={'entity_id': entity_id})
        results = [doc for doc in cursor]
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in results]
    
    def find_latest_by_entity_id(self, entity_id: str) -> Optional[SemanticSignature]:
        """
        Find the latest SemanticSignature for a specific entity.
        
        Args:
            entity_id: The ID of the entity to search for
            
        Returns:
            The latest signature if found, None otherwise
        """
        # Get the database
        db = self.connection_manager.get_db()
        
        # Query for the latest signature with the given entity ID
        query = """
        FOR s IN SemanticSignature
            FILTER s.entity_id == @entity_id
            SORT s.version DESC
            LIMIT 1
            RETURN s
        """
        
        cursor = db.aql.execute(query, bind_vars={'entity_id': entity_id})
        results = [doc for doc in cursor]
        
        if not results:
            return None
        
        # Convert to entity
        return self._dict_to_entity(results[0])
    
    def find_similar(self, signature: SemanticSignature, threshold: float = 0.7, limit: int = 10) -> List[Tuple[SemanticSignature, float]]:
        """
        Find SemanticSignature objects similar to the given signature.
        
        Args:
            signature: The signature to compare with
            threshold: The similarity threshold
            limit: The maximum number of results to return
            
        Returns:
            A list of tuples containing matching signatures and their similarity scores
        """
        # Get all signatures
        all_signatures = self.find_all()
        
        # Calculate similarity scores
        similarities = []
        for other in all_signatures:
            if other.id != signature.id:
                similarity = signature.similarity(other)
                if similarity >= threshold:
                    similarities.append((other, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top results
        return similarities[:limit]
    
    def find_all(self) -> List[SemanticSignature]:
        """
        Find all SemanticSignature objects.
        
        Returns:
            A list of all signatures
        """
        # Get the database
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        # Get all documents
        docs = collection.all()
        
        # Convert to entities
        return [self._dict_to_entity(doc) for doc in docs]
