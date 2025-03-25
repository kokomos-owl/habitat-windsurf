"""
ActantJourney Repository for ArangoDB.

Handles persistence of ActantJourney objects and related entities to ArangoDB.
Implements the cross-domain topology for tracking actant journeys.
"""

from typing import Dict, List, Any, Optional, Union
import logging
import uuid
from datetime import datetime

from .base_repository import ArangoDBBaseRepository
from .connection import ArangoDBConnectionManager
from ...transformation.actant_journey_tracker import ActantJourney, ActantJourneyPoint, DomainTransition

logger = logging.getLogger(__name__)

class ActantJourneyRepository(ArangoDBBaseRepository):
    """
    Repository for persisting ActantJourney objects to ArangoDB.
    
    Handles the storage and retrieval of actant journeys, including their
    journey points, domain transitions, and role shifts.
    """
    
    def __init__(self):
        """Initialize the ActantJourney repository."""
        super().__init__()
        self.collection_name = "ActantJourney"
        self.connection_manager = ArangoDBConnectionManager()
        self.db = self.connection_manager.get_db()
        
        # Edge collections we'll be working with
        self.actant_has_journey_collection = "ActantHasJourney"
        self.journey_contains_transition_collection = "JourneyContainsTransition"
        self.journey_contains_role_shift_collection = "JourneyContainsRoleShift"
        
        # Document collections for related entities
        self.domain_transition_collection = "DomainTransition"
        self.role_shift_collection = "RoleShift"
    
    def _dict_to_entity(self, data: Dict[str, Any]) -> ActantJourney:
        """
        Convert a dictionary to an ActantJourney object.
        
        Args:
            data: Dictionary containing ActantJourney data
            
        Returns:
            ActantJourney object
        """
        journey = ActantJourney.create(data.get("actant_name", ""))
        journey.id = data.get("id", str(uuid.uuid4()))
        
        # We'll load journey points and transitions separately
        return journey
    
    def save(self, journey: ActantJourney) -> str:
        """
        Save an ActantJourney to ArangoDB.
        
        This method saves the journey and all its related entities (journey points,
        domain transitions, role shifts) to their respective collections.
        
        Args:
            journey: ActantJourney object to save
            
        Returns:
            ID of the saved journey
        """
        # First, save the journey itself
        journey_dict = journey.to_dict()
        
        # Check if journey already exists
        existing_journey = self.find_by_actant_name(journey.actant_name)
        
        if existing_journey:
            # Update existing journey
            journey_id = existing_journey.id
            self.update(journey_id, journey)
        else:
            # Create new journey
            journey_id = self.create(journey)
            
            # Create edge from Actant to ActantJourney
            self._create_actant_journey_edge(journey.actant_name, journey_id)
        
        # Save domain transitions
        for transition in journey.domain_transitions:
            self._save_domain_transition(transition, journey_id)
        
        # Save role shifts
        role_shifts = journey.get_role_shifts()
        for shift in role_shifts:
            self._save_role_shift(shift, journey_id)
        
        return journey_id
    
    def _save_domain_transition(self, transition: DomainTransition, journey_id: str) -> str:
        """
        Save a DomainTransition to ArangoDB.
        
        Args:
            transition: DomainTransition object to save
            journey_id: ID of the parent ActantJourney
            
        Returns:
            ID of the saved transition
        """
        # Convert transition to dict
        transition_dict = transition.to_dict()
        
        # Check if transition already exists
        existing_transition = self._find_transition_by_properties(
            transition.actant_name,
            transition.source_predicate_id,
            transition.target_predicate_id
        )
        
        transition_id = None
        
        if existing_transition:
            # Update existing transition
            transition_id = existing_transition.get("_key")
            self.db.collection(self.domain_transition_collection).update(
                transition_id, transition_dict)
        else:
            # Create new transition
            result = self.db.collection(self.domain_transition_collection).insert(
                transition_dict, return_new=True)
            transition_id = result["_key"]
            
            # Create edge from ActantJourney to DomainTransition
            self._create_journey_transition_edge(journey_id, transition_id)
        
        return transition_id
    
    def _save_role_shift(self, transition: DomainTransition, journey_id: str) -> str:
        """
        Save a role shift to ArangoDB.
        
        Args:
            transition: DomainTransition object representing a role shift
            journey_id: ID of the parent ActantJourney
            
        Returns:
            ID of the saved role shift
        """
        # Only save if this is actually a role shift
        if not transition.has_role_shift:
            return None
        
        # Convert transition to role shift dict
        role_shift_dict = {
            "_key": transition.id,
            "actant_name": transition.actant_name,
            "source_role": transition.source_role,
            "target_role": transition.target_role,
            "source_predicate_id": transition.source_predicate_id,
            "target_predicate_id": transition.target_predicate_id,
            "timestamp": transition.timestamp
        }
        
        # Check if role shift already exists
        existing_shift = self._find_role_shift_by_properties(
            transition.actant_name,
            transition.source_predicate_id,
            transition.target_predicate_id
        )
        
        shift_id = None
        
        if existing_shift:
            # Update existing role shift
            shift_id = existing_shift.get("_key")
            self.db.collection(self.role_shift_collection).update(
                shift_id, role_shift_dict)
        else:
            # Create new role shift
            result = self.db.collection(self.role_shift_collection).insert(
                role_shift_dict, return_new=True)
            shift_id = result["_key"]
            
            # Create edge from ActantJourney to RoleShift
            self._create_journey_role_shift_edge(journey_id, shift_id)
        
        return shift_id
    
    def _create_actant_journey_edge(self, actant_name: str, journey_id: str) -> str:
        """
        Create an edge from an Actant to an ActantJourney.
        
        Args:
            actant_name: Name of the actant
            journey_id: ID of the ActantJourney
            
        Returns:
            ID of the created edge
        """
        # Find the actant document
        actant_cursor = self.db.collection("Actant").find({"name": actant_name}, limit=1)
        actant_list = list(actant_cursor)
        
        if not actant_list:
            # Create the actant if it doesn't exist
            actant_result = self.db.collection("Actant").insert({
                "name": actant_name,
                "aliases": []
            }, return_new=True)
            actant_id = actant_result["_id"]
        else:
            actant_id = actant_list[0]["_id"]
        
        # Create the edge
        edge = {
            "_from": actant_id,
            "_to": f"{self.collection_name}/{journey_id}",
            "created_at": datetime.now().isoformat()
        }
        
        result = self.db.collection(self.actant_has_journey_collection).insert(edge, return_new=True)
        return result["_key"]
    
    def _create_journey_transition_edge(self, journey_id: str, transition_id: str) -> str:
        """
        Create an edge from an ActantJourney to a DomainTransition.
        
        Args:
            journey_id: ID of the ActantJourney
            transition_id: ID of the DomainTransition
            
        Returns:
            ID of the created edge
        """
        # Create the edge
        edge = {
            "_from": f"{self.collection_name}/{journey_id}",
            "_to": f"{self.domain_transition_collection}/{transition_id}",
            "created_at": datetime.now().isoformat()
        }
        
        result = self.db.collection(self.journey_contains_transition_collection).insert(edge, return_new=True)
        return result["_key"]
    
    def _create_journey_role_shift_edge(self, journey_id: str, shift_id: str) -> str:
        """
        Create an edge from an ActantJourney to a RoleShift.
        
        Args:
            journey_id: ID of the ActantJourney
            shift_id: ID of the RoleShift
            
        Returns:
            ID of the created edge
        """
        # Create the edge
        edge = {
            "_from": f"{self.collection_name}/{journey_id}",
            "_to": f"{self.role_shift_collection}/{shift_id}",
            "created_at": datetime.now().isoformat()
        }
        
        result = self.db.collection(self.journey_contains_role_shift_collection).insert(edge, return_new=True)
        return result["_key"]
    
    def _find_transition_by_properties(self, actant_name: str, source_predicate_id: str, 
                                      target_predicate_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a DomainTransition by its properties.
        
        Args:
            actant_name: Name of the actant
            source_predicate_id: ID of the source predicate
            target_predicate_id: ID of the target predicate
            
        Returns:
            Dictionary containing the transition, or None if not found
        """
        cursor = self.db.collection(self.domain_transition_collection).find({
            "actant_name": actant_name,
            "source_predicate_id": source_predicate_id,
            "target_predicate_id": target_predicate_id
        }, limit=1)
        
        transitions = list(cursor)
        return transitions[0] if transitions else None
    
    def _find_role_shift_by_properties(self, actant_name: str, source_predicate_id: str, 
                                      target_predicate_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a RoleShift by its properties.
        
        Args:
            actant_name: Name of the actant
            source_predicate_id: ID of the source predicate
            target_predicate_id: ID of the target predicate
            
        Returns:
            Dictionary containing the role shift, or None if not found
        """
        cursor = self.db.collection(self.role_shift_collection).find({
            "actant_name": actant_name,
            "source_predicate_id": source_predicate_id,
            "target_predicate_id": target_predicate_id
        }, limit=1)
        
        shifts = list(cursor)
        return shifts[0] if shifts else None
    
    def find_by_actant_name(self, actant_name: str) -> Optional[ActantJourney]:
        """
        Find an ActantJourney by actant name.
        
        Args:
            actant_name: Name of the actant
            
        Returns:
            ActantJourney object, or None if not found
        """
        cursor = self.db.collection(self.collection_name).find({
            "actant_name": actant_name
        }, limit=1)
        
        journeys = list(cursor)
        if not journeys:
            return None
        
        journey_dict = journeys[0]
        journey = self._dict_to_entity(journey_dict)
        
        # Load domain transitions
        journey.domain_transitions = self.get_domain_transitions(journey.id)
        
        return journey
    
    def get_domain_transitions(self, journey_id: str) -> List[DomainTransition]:
        """
        Get all domain transitions for an ActantJourney.
        
        Args:
            journey_id: ID of the ActantJourney
            
        Returns:
            List of DomainTransition objects
        """
        # Use graph traversal to get all transitions
        query = """
        FOR v, e IN 1..1 OUTBOUND @journey_id JourneyContainsTransition
        SORT v.timestamp
        RETURN v
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "journey_id": f"{self.collection_name}/{journey_id}"
        })
        
        transitions = []
        for doc in cursor:
            transition = DomainTransition.create(
                actant_name=doc.get("actant_name", ""),
                source_domain_id=doc.get("source_domain_id", ""),
                target_domain_id=doc.get("target_domain_id", ""),
                source_predicate_id=doc.get("source_predicate_id", ""),
                target_predicate_id=doc.get("target_predicate_id", ""),
                source_role=doc.get("source_role", ""),
                target_role=doc.get("target_role", ""),
                timestamp=doc.get("timestamp", ""),
                transformation_id=doc.get("transformation_id")
            )
            transition.id = doc.get("_key", str(uuid.uuid4()))
            transitions.append(transition)
        
        return transitions
    
    def get_role_shifts(self, journey_id: str) -> List[DomainTransition]:
        """
        Get all role shifts for an ActantJourney.
        
        Args:
            journey_id: ID of the ActantJourney
            
        Returns:
            List of DomainTransition objects representing role shifts
        """
        # Use graph traversal to get all role shifts
        query = """
        FOR v, e IN 1..1 OUTBOUND @journey_id JourneyContainsRoleShift
        SORT v.timestamp
        RETURN v
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "journey_id": f"{self.collection_name}/{journey_id}"
        })
        
        role_shifts = []
        for doc in cursor:
            transition = DomainTransition.create(
                actant_name=doc.get("actant_name", ""),
                source_domain_id="",  # Role shifts may not have domain info
                target_domain_id="",
                source_predicate_id=doc.get("source_predicate_id", ""),
                target_predicate_id=doc.get("target_predicate_id", ""),
                source_role=doc.get("source_role", ""),
                target_role=doc.get("target_role", ""),
                timestamp=doc.get("timestamp", ""),
                transformation_id=None
            )
            transition.id = doc.get("_key", str(uuid.uuid4()))
            role_shifts.append(transition)
        
        return role_shifts
    
    def get_cross_domain_journey(self, actant_name: str) -> Dict[str, Any]:
        """
        Get a comprehensive cross-domain journey for an actant.
        
        This method uses specialized graph traversals to retrieve a complete
        picture of an actant's journey across domains, including role shifts,
        transitions, and temporal information.
        
        Args:
            actant_name: Name of the actant
            
        Returns:
            Dictionary containing the actant's cross-domain journey
        """
        # Use the specialized ACTANT_JOURNEY traversal function
        query = """
        LET traversal_function = DOCUMENT("AQLTraversals/ACTANT_JOURNEY")
        LET journey = EVAL(traversal_function.code)(@actant_name)
        
        RETURN {
            actant_name: @actant_name,
            journey: journey,
            metrics: {
                domain_count: COUNT(UNIQUE(
                    FOR event IN journey
                    FILTER event.event_type == 'transition'
                    RETURN event.target_domain_id
                )),
                role_shift_count: COUNT(
                    FOR event IN journey
                    FILTER event.event_type == 'role_shift'
                    RETURN event
                ),
                transition_count: COUNT(
                    FOR event IN journey
                    FILTER event.event_type == 'transition'
                    RETURN event
                )
            }
        }
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "actant_name": actant_name
        })
        
        results = list(cursor)
        return results[0] if results else {"actant_name": actant_name, "journey": [], "metrics": {}}
