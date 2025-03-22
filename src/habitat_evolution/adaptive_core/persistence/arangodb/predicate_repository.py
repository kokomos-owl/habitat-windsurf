"""
Predicate repository implementation for ArangoDB.
Manages subject-verb-object structures extracted from text.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import uuid
from datetime import datetime

from .base_repository import ArangoDBBaseRepository

@dataclass
class Predicate:
    """Represents a subject-verb-object structure extracted from text."""
    id: str
    domain_id: str
    subject: str
    verb: str
    object: str
    text: str
    position: int
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, domain_id: str, subject: str, verb: str, object: str, 
               text: str, position: int, metadata: Optional[Dict[str, Any]] = None):
        """Factory method to create a new Predicate instance."""
        return cls(
            id=str(uuid.uuid4()),
            domain_id=domain_id,
            subject=subject,
            verb=verb,
            object=object,
            text=text,
            position=position,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert predicate to dictionary."""
        return {
            "id": self.id,
            "domain_id": self.domain_id,
            "subject": self.subject,
            "verb": self.verb,
            "object": self.object,
            "text": self.text,
            "position": self.position,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "metadata": self.metadata
        }


class PredicateRepository(ArangoDBBaseRepository[Predicate]):
    """Repository for managing Predicate entities in ArangoDB."""
    
    def __init__(self):
        super().__init__()
        self.collection_name = "Predicate"
    
    def _dict_to_entity(self, entity_dict: Dict[str, Any]) -> Predicate:
        """Convert a dictionary to a Predicate entity."""
        # Convert ISO format string back to datetime if needed
        created_at = entity_dict.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now()
        elif created_at is None:
            created_at = datetime.now()
            
        return Predicate(
            id=entity_dict.get("id"),
            domain_id=entity_dict.get("domain_id"),
            subject=entity_dict.get("subject", ""),
            verb=entity_dict.get("verb", ""),
            object=entity_dict.get("object", ""),
            text=entity_dict.get("text", ""),
            position=entity_dict.get("position", 0),
            created_at=created_at,
            metadata=entity_dict.get("metadata", {})
        )
    
    def find_by_domain_id(self, domain_id: str) -> List[Predicate]:
        """Find all predicates within a specific domain."""
        return self.list({"domain_id": domain_id})
    
    def find_by_verb(self, verb: str) -> List[Predicate]:
        """Find all predicates with a specific verb."""
        db = self.connection_manager.get_db()
        
        # AQL query to find predicates with specific verb
        query = """
        FOR p IN Predicate
            FILTER p.verb == @verb
            RETURN p
        """
        
        cursor = db.aql.execute(query, bind_vars={"verb": verb})
        
        # Convert results to Predicate entities
        predicates = []
        for doc in cursor:
            entity_dict = self._from_document_properties(doc)
            predicate = self._dict_to_entity(entity_dict)
            predicates.append(predicate)
            
        return predicates
    
    def find_by_subject_or_object(self, term: str) -> List[Predicate]:
        """Find all predicates where a term appears as subject or object."""
        db = self.connection_manager.get_db()
        
        # AQL query to find predicates with specific subject or object
        query = """
        FOR p IN Predicate
            FILTER p.subject == @term OR p.object == @term
            RETURN p
        """
        
        cursor = db.aql.execute(query, bind_vars={"term": term})
        
        # Convert results to Predicate entities
        predicates = []
        for doc in cursor:
            entity_dict = self._from_document_properties(doc)
            predicate = self._dict_to_entity(entity_dict)
            predicates.append(predicate)
            
        return predicates
    
    def link_to_domain(self, predicate_id: str, domain_id: str) -> str:
        """Link a predicate to a domain."""
        db = self.connection_manager.get_db()
        
        # Check if the edge collection exists
        if not db.has_collection("DomainContainsPredicate"):
            raise ValueError("Edge collection DomainContainsPredicate does not exist")
            
        edge_collection = db.collection("DomainContainsPredicate")
        
        # Create the edge
        edge_doc = {
            "_from": f"Domain/{domain_id}",
            "_to": f"Predicate/{predicate_id}"
        }
        
        result = edge_collection.insert(edge_doc)
        return result["_key"]
