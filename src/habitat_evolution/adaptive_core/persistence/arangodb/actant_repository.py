"""
Actant repository implementation for ArangoDB.
Manages entities that appear as subjects or objects in predicates across domains.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import uuid
from datetime import datetime

from .base_repository import ArangoDBBaseRepository

@dataclass
class Actant:
    """
    Represents an entity that appears as a subject or object in predicates.
    Actants can travel across domains, carrying predicates with them.
    """
    id: str
    name: str
    aliases: List[str]
    vector: Optional[List[float]] = None
    created_at: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, name: str, aliases: Optional[List[str]] = None, 
               vector: Optional[List[float]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Factory method to create a new Actant instance."""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            aliases=aliases or [],
            vector=vector,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert actant to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "aliases": self.aliases,
            "vector": self.vector,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "metadata": self.metadata
        }


class ActantRepository(ArangoDBBaseRepository[Actant]):
    """Repository for managing Actant entities in ArangoDB."""
    
    def __init__(self):
        super().__init__()
        self.collection_name = "Actant"
    
    def _dict_to_entity(self, entity_dict: Dict[str, Any]) -> Actant:
        """Convert a dictionary to an Actant entity."""
        # Convert ISO format string back to datetime if needed
        created_at = entity_dict.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now()
        elif created_at is None:
            created_at = datetime.now()
            
        return Actant(
            id=entity_dict.get("id"),
            name=entity_dict.get("name", ""),
            aliases=entity_dict.get("aliases", []),
            vector=entity_dict.get("vector"),
            created_at=created_at,
            metadata=entity_dict.get("metadata", {})
        )
    
    def find_by_name(self, name: str) -> Optional[Actant]:
        """Find an actant by name or alias."""
        db = self.connection_manager.get_db()
        
        # AQL query to find actant by name or alias
        query = """
        FOR a IN Actant
            FILTER a.name == @name OR @name IN a.aliases
            RETURN a
        """
        
        cursor = db.aql.execute(query, bind_vars={"name": name})
        
        # Get the first result if any
        try:
            doc = next(cursor)
            entity_dict = self._from_document_properties(doc)
            return self._dict_to_entity(entity_dict)
        except StopIteration:
            return None
    
    def find_or_create(self, name: str, aliases: Optional[List[str]] = None) -> Actant:
        """Find an actant by name or create if it doesn't exist."""
        actant = self.find_by_name(name)
        if actant:
            # Update aliases if new ones are provided
            if aliases:
                existing_aliases = set(actant.aliases)
                new_aliases = set(aliases)
                combined_aliases = list(existing_aliases.union(new_aliases))
                
                if len(combined_aliases) > len(actant.aliases):
                    actant.aliases = combined_aliases
                    self.update(actant.id, actant)
            
            return actant
        
        # Create new actant
        actant = Actant.create(name=name, aliases=aliases)
        actant_id = self.create(actant)
        actant.id = actant_id
        return actant
    
    def get_domain_journey(self, actant_id: str) -> List[Dict[str, Any]]:
        """
        Track an actant's journey across domains.
        Returns a chronological list of domains where the actant appears.
        """
        db = self.connection_manager.get_db()
        
        # AQL query to track actant journey
        query = """
        LET actant = DOCUMENT(CONCAT('Actant/', @actant_id))
        
        // Find all predicates where this actant appears as subject or object
        LET predicates = (
            FOR p IN Predicate
                FILTER p.subject == actant.name OR p.object == actant.name
                    OR p.subject IN actant.aliases OR p.object IN actant.aliases
                RETURN p
        )
        
        // Get domains containing these predicates
        LET domains = (
            FOR p IN predicates
                LET domain = DOCUMENT(CONCAT('Domain/', p.domain_id))
                RETURN {
                    domain: domain,
                    predicate: p,
                    role: p.subject == actant.name || p.subject IN actant.aliases ? 'subject' : 'object'
                }
        )
        
        // Sort by document and position
        FOR journey IN domains
            SORT journey.domain.document_id, journey.domain.start_position
            RETURN journey
        """
        
        cursor = db.aql.execute(query, bind_vars={"actant_id": actant_id})
        return list(cursor)
    
    def link_to_predicate(self, actant_id: str, predicate_id: str, role: str) -> str:
        """
        Link an actant to a predicate with a specific role (subject or object).
        
        Args:
            actant_id: The actant ID
            predicate_id: The predicate ID
            role: Either 'subject' or 'object'
            
        Returns:
            The edge ID
        """
        db = self.connection_manager.get_db()
        
        # Determine the edge collection based on role
        if role.lower() == 'subject':
            edge_collection_name = "PredicateHasSubject"
        elif role.lower() == 'object':
            edge_collection_name = "PredicateHasObject"
        else:
            raise ValueError(f"Invalid role: {role}. Must be 'subject' or 'object'")
        
        # Check if the edge collection exists
        if not db.has_collection(edge_collection_name):
            raise ValueError(f"Edge collection {edge_collection_name} does not exist")
            
        edge_collection = db.collection(edge_collection_name)
        
        # Create the edge
        edge_doc = {
            "_from": f"Predicate/{predicate_id}",
            "_to": f"Actant/{actant_id}"
        }
        
        result = edge_collection.insert(edge_doc)
        return result["_key"]
    
    def find_co_occurring_actants(self, actant_id: str, min_occurrences: int = 2) -> List[Dict[str, Any]]:
        """
        Find actants that co-occur with this actant in the same domains.
        
        Args:
            actant_id: The actant ID
            min_occurrences: Minimum number of co-occurrences to include
            
        Returns:
            List of co-occurring actants with occurrence counts
        """
        db = self.connection_manager.get_db()
        
        query = """
        LET actant = DOCUMENT(CONCAT('Actant/', @actant_id))
        
        // Find all predicates where this actant appears
        LET predicates = (
            FOR p IN Predicate
                FILTER p.subject == actant.name OR p.object == actant.name
                    OR p.subject IN actant.aliases OR p.object IN actant.aliases
                RETURN p
        )
        
        // Get domains containing these predicates
        LET domains = (
            FOR p IN predicates
                RETURN p.domain_id
        )
        
        // Find other actants in these domains
        LET co_actants = (
            FOR domain_id IN domains
                FOR p IN Predicate
                    FILTER p.domain_id == domain_id
                    LET subjects = [p.subject]
                    LET objects = [p.object]
                    FOR term IN APPEND(subjects, objects)
                        FILTER term != actant.name AND term NOT IN actant.aliases
                        COLLECT name = term WITH COUNT INTO count
                        FILTER count >= @min_occurrences
                        RETURN {
                            name: name,
                            occurrences: count
                        }
        )
        
        RETURN co_actants
        """
        
        cursor = db.aql.execute(
            query, 
            bind_vars={
                "actant_id": actant_id,
                "min_occurrences": min_occurrences
            }
        )
        
        return list(cursor)[0]  # Unwrap the outer array
