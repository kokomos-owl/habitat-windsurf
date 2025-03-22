"""
Document repository implementation for ArangoDB.
Manages documents processed by Habitat for domain-predicate tracking.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import uuid
from datetime import datetime

from .base_repository import ArangoDBBaseRepository

@dataclass
class Document:
    """Represents a document processed by Habitat."""
    id: str
    title: str
    content: str
    source: str
    created_at: datetime
    processed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, title: str, content: str, source: str, 
               metadata: Optional[Dict[str, Any]] = None):
        """Factory method to create a new Document instance."""
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            content=content,
            source=source,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "processed_at": self.processed_at.isoformat() if isinstance(self.processed_at, datetime) else self.processed_at,
            "metadata": self.metadata
        }


class DocumentRepository(ArangoDBBaseRepository[Document]):
    """Repository for managing Document entities in ArangoDB."""
    
    def __init__(self):
        super().__init__()
        self.collection_name = "Document"
    
    def _dict_to_entity(self, entity_dict: Dict[str, Any]) -> Document:
        """Convert a dictionary to a Document entity."""
        # Convert ISO format strings back to datetime if needed
        created_at = entity_dict.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now()
        elif created_at is None:
            created_at = datetime.now()
            
        processed_at = entity_dict.get("processed_at")
        if isinstance(processed_at, str):
            try:
                processed_at = datetime.fromisoformat(processed_at)
            except ValueError:
                processed_at = None
            
        return Document(
            id=entity_dict.get("id"),
            title=entity_dict.get("title", ""),
            content=entity_dict.get("content", ""),
            source=entity_dict.get("source", ""),
            created_at=created_at,
            processed_at=processed_at,
            metadata=entity_dict.get("metadata", {})
        )
    
    def find_by_title(self, title: str) -> List[Document]:
        """Find documents by title (partial match)."""
        db = self.connection_manager.get_db()
        
        # AQL query to find documents by title
        query = """
        FOR d IN Document
            FILTER CONTAINS(d.title, @title, true)
            RETURN d
        """
        
        cursor = db.aql.execute(query, bind_vars={"title": title})
        
        # Convert results to Document entities
        documents = []
        for doc in cursor:
            entity_dict = self._from_document_properties(doc)
            document = self._dict_to_entity(entity_dict)
            documents.append(document)
            
        return documents
    
    def find_by_source(self, source: str) -> List[Document]:
        """Find documents by source."""
        return self.list({"source": source})
    
    def mark_as_processed(self, document_id: str) -> bool:
        """Mark a document as processed."""
        db = self.connection_manager.get_db()
        collection = db.collection(self.collection_name)
        
        try:
            # Update document
            collection.update(
                document_id, 
                {"processed_at": datetime.now().isoformat()}
            )
            return True
        except Exception as e:
            print(f"Error marking document as processed: {str(e)}")
            return False
    
    def get_domains(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all domains within a document."""
        db = self.connection_manager.get_db()
        
        query = """
        FOR doc_domain IN DocumentContainsDomain
            FILTER doc_domain._from == CONCAT('Document/', @document_id)
            LET domain = DOCUMENT(doc_domain._to)
            RETURN domain
        """
        
        cursor = db.aql.execute(query, bind_vars={"document_id": document_id})
        return list(cursor)
    
    def link_to_domain(self, document_id: str, domain_id: str) -> str:
        """Link a document to a domain."""
        db = self.connection_manager.get_db()
        
        # Check if the edge collection exists
        if not db.has_collection("DocumentContainsDomain"):
            raise ValueError("Edge collection DocumentContainsDomain does not exist")
            
        edge_collection = db.collection("DocumentContainsDomain")
        
        # Create the edge
        edge_doc = {
            "_from": f"Document/{document_id}",
            "_to": f"Domain/{domain_id}"
        }
        
        result = edge_collection.insert(edge_doc)
        return result["_key"]
    
    def get_document_with_domains_and_predicates(self, document_id: str) -> Dict[str, Any]:
        """
        Get a document with all its domains and predicates.
        This provides a complete view of the document's semantic structure.
        """
        db = self.connection_manager.get_db()
        
        query = """
        LET doc = DOCUMENT(CONCAT('Document/', @document_id))
        
        LET domains = (
            FOR doc_domain IN DocumentContainsDomain
                FILTER doc_domain._from == CONCAT('Document/', @document_id)
                LET domain = DOCUMENT(doc_domain._to)
                
                LET predicates = (
                    FOR domain_pred IN DomainContainsPredicate
                        FILTER domain_pred._from == domain._id
                        LET predicate = DOCUMENT(domain_pred._to)
                        
                        LET subjects = (
                            FOR pred_subj IN PredicateHasSubject
                                FILTER pred_subj._from == predicate._id
                                LET subject = DOCUMENT(pred_subj._to)
                                RETURN subject
                        )
                        
                        LET objects = (
                            FOR pred_obj IN PredicateHasObject
                                FILTER pred_obj._from == predicate._id
                                LET object = DOCUMENT(pred_obj._to)
                                RETURN object
                        )
                        
                        RETURN MERGE(predicate, {
                            subjects: subjects,
                            objects: objects
                        })
                )
                
                RETURN MERGE(domain, {
                    predicates: predicates
                })
        )
        
        RETURN MERGE(doc, {
            domains: domains
        })
        """
        
        cursor = db.aql.execute(query, bind_vars={"document_id": document_id})
        
        try:
            return next(cursor)
        except StopIteration:
            return None
