"""
Domain repository implementation for ArangoDB.
Manages semantic domains identified within documents.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import uuid
from datetime import datetime

from .base_repository import ArangoDBBaseRepository

@dataclass
class Domain:
    """Represents a semantic domain identified within a document."""
    id: str
    document_id: str
    text: str
    vector: List[float]
    start_position: int
    end_position: int
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, document_id: str, text: str, vector: List[float], 
               start_position: int, end_position: int, metadata: Optional[Dict[str, Any]] = None):
        """Factory method to create a new Domain instance."""
        return cls(
            id=str(uuid.uuid4()),
            document_id=document_id,
            text=text,
            vector=vector,
            start_position=start_position,
            end_position=end_position,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert domain to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "vector": self.vector,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "metadata": self.metadata
        }


class DomainRepository(ArangoDBBaseRepository[Domain]):
    """Repository for managing Domain entities in ArangoDB."""
    
    def __init__(self):
        super().__init__()
        self.collection_name = "Domain"
    
    def _dict_to_entity(self, entity_dict: Dict[str, Any]) -> Domain:
        """Convert a dictionary to a Domain entity."""
        # Convert ISO format string back to datetime if needed
        created_at = entity_dict.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now()
        elif created_at is None:
            created_at = datetime.now()
            
        return Domain(
            id=entity_dict.get("id"),
            document_id=entity_dict.get("document_id"),
            text=entity_dict.get("text"),
            vector=entity_dict.get("vector", []),
            start_position=entity_dict.get("start_position", 0),
            end_position=entity_dict.get("end_position", 0),
            created_at=created_at,
            metadata=entity_dict.get("metadata", {})
        )
    
    def find_by_document_id(self, document_id: str) -> List[Domain]:
        """Find all domains within a specific document."""
        return self.list({"document_id": document_id})
    
    def find_similar_domains(self, vector: List[float], threshold: float = 0.8, limit: int = 10) -> List[Domain]:
        """
        Find semantically similar domains using vector similarity.
        
        Args:
            vector: The query vector to compare against
            threshold: Similarity threshold (0-1)
            limit: Maximum number of results to return
            
        Returns:
            List of similar domains
        """
        db = self.connection_manager.get_db()
        
        # AQL query using vector similarity
        query = f"""
        FOR doc IN {self.collection_name}
            LET similarity = LENGTH(doc.vector) == LENGTH(@vector) ? 
                1 - SQRT(SUM(
                    FOR i IN 0..LENGTH(@vector)-1
                        RETURN POW(doc.vector[i] - @vector[i], 2)
                ) / LENGTH(@vector)) : 0
            FILTER similarity >= @threshold
            SORT similarity DESC
            LIMIT @limit
            RETURN doc
        """
        
        # Execute query
        cursor = db.aql.execute(
            query, 
            bind_vars={
                "vector": vector,
                "threshold": threshold,
                "limit": limit
            }
        )
        
        # Convert results to Domain entities
        domains = []
        for doc in cursor:
            entity_dict = self._from_document_properties(doc)
            domain = self._dict_to_entity(entity_dict)
            domains.append(domain)
            
        return domains
    
    def get_domains_with_predicates(self) -> List[Dict[str, Any]]:
        """Get domains with their associated predicates."""
        db = self.connection_manager.get_db()
        
        query = """
        FOR domain IN Domain
            LET predicates = (
                FOR pred, edge IN 1..1 OUTBOUND domain._id DomainContainsPredicate
                    RETURN pred
            )
            FILTER LENGTH(predicates) > 0
            RETURN {
                domain: domain,
                predicates: predicates
            }
        """
        
        cursor = db.aql.execute(query)
        return list(cursor)
