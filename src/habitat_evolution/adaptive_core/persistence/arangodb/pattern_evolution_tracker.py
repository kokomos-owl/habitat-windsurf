"""
Pattern Evolution Tracker for ArangoDB.
Tracks how actants and predicates evolve across domains and documents.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

from .connection import ArangoDBConnectionManager
from .actant_repository import ActantRepository
from .domain_repository import DomainRepository
from .predicate_repository import PredicateRepository

logger = logging.getLogger(__name__)

class PatternEvolutionTracker:
    """
    Tracks how patterns evolve across domains and documents.
    Focuses on actant journeys and predicate transformations.
    """
    
    def __init__(self):
        self.connection_manager = ArangoDBConnectionManager()
        self.actant_repository = ActantRepository()
        self.domain_repository = DomainRepository()
        self.predicate_repository = PredicateRepository()
    
    def track_actant_journey(self, actant_name: str) -> List[Dict[str, Any]]:
        """
        Track an actant's journey across domains and documents.
        
        Args:
            actant_name: The name of the actant to track
            
        Returns:
            A chronological list of the actant's appearances with context
        """
        # Find or create the actant
        actant = self.actant_repository.find_by_name(actant_name)
        if not actant:
            logger.warning(f"Actant '{actant_name}' not found")
            return []
        
        db = self.connection_manager.get_db()
        
        # Complex AQL query to track the actant's journey
        query = """
        LET actant = DOCUMENT(CONCAT('Actant/', @actant_id))
        
        // Find all predicates where this actant appears
        LET predicates = (
            FOR p IN Predicate
                FILTER p.subject == actant.name OR p.object == actant.name
                    OR p.subject IN actant.aliases OR p.object IN actant.aliases
                RETURN p
        )
        
        // Get domains and documents containing these predicates
        LET journey = (
            FOR p IN predicates
                LET domain = DOCUMENT(CONCAT('Domain/', p.domain_id))
                
                // Find the document containing this domain
                LET doc_edge = (
                    FOR edge IN DocumentContainsDomain
                        FILTER edge._to == domain._id
                        RETURN edge
                )[0]
                
                LET document = doc_edge ? DOCUMENT(doc_edge._from) : null
                
                // Get other actants in the same predicate
                LET other_actant_name = p.subject == actant.name || p.subject IN actant.aliases ? 
                                      p.object : p.subject
                
                LET other_actant = (
                    FOR a IN Actant
                        FILTER a.name == other_actant_name OR other_actant_name IN a.aliases
                        LIMIT 1
                        RETURN a
                )[0]
                
                RETURN {
                    document: document,
                    domain: domain,
                    predicate: p,
                    role: p.subject == actant.name || p.subject IN actant.aliases ? 'subject' : 'object',
                    other_actant: other_actant,
                    timestamp: domain.created_at
                }
        )
        
        // Sort chronologically
        FOR step IN journey
            SORT step.timestamp
            RETURN step
        """
        
        cursor = db.aql.execute(query, bind_vars={"actant_id": actant.id})
        return list(cursor)
    
    def detect_predicate_transformations(self, actant_name: str) -> List[Dict[str, Any]]:
        """
        Detect how predicates transform as an actant moves across domains.
        
        Args:
            actant_name: The name of the actant to track
            
        Returns:
            A list of predicate transformations
        """
        # Find the actant
        actant = self.actant_repository.find_by_name(actant_name)
        if not actant:
            logger.warning(f"Actant '{actant_name}' not found")
            return []
        
        db = self.connection_manager.get_db()
        
        # Query to detect predicate transformations
        query = """
        LET actant = DOCUMENT(CONCAT('Actant/', @actant_id))
        
        // Get the actant's journey
        LET journey = (
            FOR p IN Predicate
                FILTER p.subject == actant.name OR p.object == actant.name
                    OR p.subject IN actant.aliases OR p.object IN actant.aliases
                LET domain = DOCUMENT(CONCAT('Domain/', p.domain_id))
                
                // Find the document containing this domain
                LET doc_edge = (
                    FOR edge IN DocumentContainsDomain
                        FILTER edge._to == domain._id
                        RETURN edge
                )[0]
                
                LET document = doc_edge ? DOCUMENT(doc_edge._from) : null
                
                RETURN {
                    document: document,
                    domain: domain,
                    predicate: p,
                    role: p.subject == actant.name || p.subject IN actant.aliases ? 'subject' : 'object',
                    timestamp: domain.created_at
                }
        )
        
        // Sort chronologically
        LET sorted_journey = (
            FOR step IN journey
                SORT step.timestamp
                RETURN step
        )
        
        // Detect transformations by comparing adjacent predicates
        LET transformations = (
            FOR i IN 0..LENGTH(sorted_journey)-2
                LET current = sorted_journey[i]
                LET next = sorted_journey[i+1]
                
                // Only include if there's a change in verb or role
                FILTER current.predicate.verb != next.predicate.verb OR current.role != next.role
                
                RETURN {
                    from: current,
                    to: next,
                    verb_change: current.predicate.verb != next.predicate.verb,
                    role_change: current.role != next.role
                }
        )
        
        RETURN transformations
        """
        
        cursor = db.aql.execute(query, bind_vars={"actant_id": actant.id})
        return list(cursor)[0]  # Unwrap the outer array
    
    def find_resonating_domains(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find domains that resonate with each other based on shared actants and predicates.
        
        Args:
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of resonating domain pairs
        """
        db = self.connection_manager.get_db()
        
        # Query to find resonating domains
        query = """
        // For each pair of domains
        FOR d1 IN Domain
            FOR d2 IN Domain
                FILTER d1._key < d2._key  // Avoid duplicates
                
                // Get actants in d1
                LET d1_predicates = (
                    FOR edge IN DomainContainsPredicate
                        FILTER edge._from == d1._id
                        LET pred = DOCUMENT(edge._to)
                        RETURN pred
                )
                
                LET d1_actants = UNIQUE(
                    FLATTEN(
                        FOR pred IN d1_predicates
                            RETURN [pred.subject, pred.object]
                    )
                )
                
                // Get actants in d2
                LET d2_predicates = (
                    FOR edge IN DomainContainsPredicate
                        FILTER edge._from == d2._id
                        LET pred = DOCUMENT(edge._to)
                        RETURN pred
                )
                
                LET d2_actants = UNIQUE(
                    FLATTEN(
                        FOR pred IN d2_predicates
                            RETURN [pred.subject, pred.object]
                    )
                )
                
                // Calculate similarity based on shared actants
                LET shared_actants = LENGTH(
                    INTERSECTION(d1_actants, d2_actants)
                )
                
                LET total_actants = LENGTH(
                    UNION_DISTINCT(d1_actants, d2_actants)
                )
                
                LET similarity = total_actants > 0 ? shared_actants / total_actants : 0
                
                // Only include pairs above threshold
                FILTER similarity >= @threshold
                
                // Get documents for context
                LET d1_doc_edge = (
                    FOR edge IN DocumentContainsDomain
                        FILTER edge._to == d1._id
                        RETURN edge
                )[0]
                
                LET d2_doc_edge = (
                    FOR edge IN DocumentContainsDomain
                        FILTER edge._to == d2._id
                        RETURN edge
                )[0]
                
                LET d1_doc = d1_doc_edge ? DOCUMENT(d1_doc_edge._from) : null
                LET d2_doc = d2_doc_edge ? DOCUMENT(d2_doc_edge._from) : null
                
                RETURN {
                    domain1: d1,
                    domain2: d2,
                    document1: d1_doc,
                    document2: d2_doc,
                    similarity: similarity,
                    shared_actants: INTERSECTION(d1_actants, d2_actants)
                }
        """
        
        cursor = db.aql.execute(query, bind_vars={"threshold": threshold})
        return list(cursor)
    
    def record_domain_resonance(self, domain1_id: str, domain2_id: str, 
                               similarity: float, shared_actants: List[str]) -> str:
        """
        Record resonance between two domains.
        
        Args:
            domain1_id: ID of the first domain
            domain2_id: ID of the second domain
            similarity: Similarity score (0-1)
            shared_actants: List of actant names shared between domains
            
        Returns:
            ID of the created resonance edge
        """
        db = self.connection_manager.get_db()
        
        # Check if the edge collection exists
        if not db.has_collection("DomainResonatesWith"):
            raise ValueError("Edge collection DomainResonatesWith does not exist")
            
        edge_collection = db.collection("DomainResonatesWith")
        
        # Create the edge
        edge_doc = {
            "_from": f"Domain/{domain1_id}",
            "_to": f"Domain/{domain2_id}",
            "similarity": similarity,
            "shared_actants": shared_actants,
            "created_at": datetime.now().isoformat()
        }
        
        result = edge_collection.insert(edge_doc)
        return result["_key"]
    
    def analyze_actant_evolution(self, actant_name: str) -> Dict[str, Any]:
        """
        Analyze how an actant evolves across domains and documents.
        
        Args:
            actant_name: The name of the actant to analyze
            
        Returns:
            Analysis of the actant's evolution
        """
        # Get the actant's journey
        journey = self.track_actant_journey(actant_name)
        if not journey:
            return {"error": f"Actant '{actant_name}' not found or has no journey"}
        
        # Get predicate transformations
        transformations = self.detect_predicate_transformations(actant_name)
        
        # Analyze verb patterns
        verbs = {}
        for step in journey:
            verb = step["predicate"]["verb"]
            role = step["role"]
            
            if verb not in verbs:
                verbs[verb] = {"count": 0, "as_subject": 0, "as_object": 0}
                
            verbs[verb]["count"] += 1
            if role == "subject":
                verbs[verb]["as_subject"] += 1
            else:
                verbs[verb]["as_object"] += 1
        
        # Analyze co-occurring actants
        co_actants = {}
        for step in journey:
            other_actant = step.get("other_actant")
            if other_actant and "name" in other_actant:
                name = other_actant["name"]
                if name not in co_actants:
                    co_actants[name] = 0
                co_actants[name] += 1
        
        # Analyze domain transitions
        domain_transitions = []
        for i in range(len(journey) - 1):
            current = journey[i]
            next_step = journey[i + 1]
            
            if current["domain"]["_id"] != next_step["domain"]["_id"]:
                domain_transitions.append({
                    "from_domain": current["domain"],
                    "to_domain": next_step["domain"],
                    "from_predicate": current["predicate"],
                    "to_predicate": next_step["predicate"]
                })
        
        return {
            "actant_name": actant_name,
            "journey_length": len(journey),
            "unique_domains": len(set(step["domain"]["_id"] for step in journey)),
            "unique_documents": len(set(step["document"]["_id"] for step in journey if step["document"])),
            "verb_patterns": verbs,
            "transformations": transformations,
            "co_occurring_actants": co_actants,
            "domain_transitions": domain_transitions
        }
