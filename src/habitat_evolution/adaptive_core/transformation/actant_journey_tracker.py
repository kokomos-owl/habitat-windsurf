"""
Actant Journey Tracker

Tracks how actants carry predicates across domain boundaries, creating a form of
narrative structure or "character building" as concepts transform.

This module extends the observer pattern in LearningWindow to detect when actants
appear in different semantic domains and how their relationships change over time.
"""

from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass, field
import uuid
import math
import json
from datetime import datetime
import logging
from collections import defaultdict

from ..id.adaptive_id import AdaptiveID


@dataclass
class ActantJourneyPoint:
    """
    Represents a point in an actant's journey across semantic domains.
    
    Each journey point captures the actant's role in a predicate within a specific
    domain, along with temporal information about when this observation occurred.
    """
    id: str
    actant_name: str
    domain_id: str
    predicate_id: str
    role: str  # "subject", "object", etc.
    timestamp: str
    confidence: float = 0.8
    
    @classmethod
    def create(cls, actant_name: str, domain_id: str, predicate_id: str, 
               role: str, timestamp: str, confidence: float = 0.8):
        """Create a new journey point."""
        return cls(
            id=str(uuid.uuid4()),
            actant_name=actant_name,
            domain_id=domain_id,
            predicate_id=predicate_id,
            role=role,
            timestamp=timestamp,
            confidence=confidence
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "actant_name": self.actant_name,
            "domain_id": self.domain_id,
            "predicate_id": self.predicate_id,
            "role": self.role,
            "timestamp": self.timestamp,
            "confidence": self.confidence
        }


@dataclass
class DomainTransition:
    """
    Represents a transition of an actant between domains.
    
    Captures how an actant moves from one domain to another, carrying predicates
    with it and potentially changing its role in the process.
    """
    id: str
    actant_name: str
    source_domain_id: str
    target_domain_id: str
    source_predicate_id: str
    target_predicate_id: str
    source_role: str
    target_role: str
    timestamp: str
    transformation_id: Optional[str] = None
    
    @classmethod
    def create(cls, actant_name: str, source_domain_id: str, target_domain_id: str,
               source_predicate_id: str, target_predicate_id: str, 
               source_role: str, target_role: str, timestamp: str,
               transformation_id: Optional[str] = None):
        """Create a new domain transition."""
        return cls(
            id=str(uuid.uuid4()),
            actant_name=actant_name,
            source_domain_id=source_domain_id,
            target_domain_id=target_domain_id,
            source_predicate_id=source_predicate_id,
            target_predicate_id=target_predicate_id,
            source_role=source_role,
            target_role=target_role,
            timestamp=timestamp,
            transformation_id=transformation_id
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "actant_name": self.actant_name,
            "source_domain_id": self.source_domain_id,
            "target_domain_id": self.target_domain_id,
            "source_predicate_id": self.source_predicate_id,
            "target_predicate_id": self.target_predicate_id,
            "source_role": self.source_role,
            "target_role": self.target_role,
            "timestamp": self.timestamp,
            "transformation_id": self.transformation_id
        }
    
    @property
    def has_role_shift(self) -> bool:
        """Check if this transition involves a role shift."""
        return self.source_role != self.target_role


@dataclass
class ActantJourney:
    """
    Represents the complete journey of an actant across semantic domains.
    
    Captures all journey points and transitions for a specific actant, allowing
    for analysis of how the actant's role and relationships evolve over time.
    """
    id: str
    actant_name: str
    journey_points: List[ActantJourneyPoint] = field(default_factory=list)
    domain_transitions: List[DomainTransition] = field(default_factory=list)
    adaptive_id: Optional[AdaptiveID] = None
    
    @classmethod
    def create(cls, actant_name: str):
        """Create a new actant journey."""
        return cls(
            id=str(uuid.uuid4()),
            actant_name=actant_name
        )
    
    def add_journey_point(self, journey_point: ActantJourneyPoint) -> None:
        """Add a journey point to this actant's journey."""
        self.journey_points.append(journey_point)
    
    def add_domain_transition(self, transition: DomainTransition) -> None:
        """Add a domain transition to this actant's journey."""
        self.domain_transitions.append(transition)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "actant_name": self.actant_name,
            "journey_points": [jp.to_dict() for jp in self.journey_points],
            "domain_transitions": [dt.to_dict() for dt in self.domain_transitions]
        }
    
    def get_role_shifts(self) -> List[DomainTransition]:
        """Get all transitions where the actant's role shifted."""
        return [t for t in self.domain_transitions if t.has_role_shift]


class ActantJourneyTracker:
    """
    Tracks actant journeys across semantic domains.
    
    This class implements the observer pattern to detect when actants appear in
    different semantic domains and how their relationships change over time. It
    integrates with the LearningWindow to observe pattern evolution events.
    """
    
    def __init__(self):
        """Initialize the actant journey tracker."""
        self.actant_journeys = {}  # actant_name -> ActantJourney
        self.predicate_transformations = []  # List of transformation dictionaries
        self.predicates = {}  # predicate_id -> predicate dictionary
        self.domains = {}  # domain_id -> domain dictionary
        self.logger = logging.getLogger(__name__)
    
    def observe_pattern_evolution(self, context: Dict[str, Any]) -> None:
        """
        Observe pattern evolution events from a LearningWindow.
        
        This method is called by the LearningWindow when a pattern evolution event
        occurs. It extracts information about predicates, domains, and transformations
        from the context and updates the actant journeys accordingly.
        
        Args:
            context: Context information about the pattern evolution
        """
        try:
            # Check if this is a predicate transformation event
            if context.get("change_type") == "predicate_transformation" and "transformation" in context:
                transformation = context["transformation"]
                self.predicate_transformations.append(transformation)
                
                # Extract information about the predicates involved
                source_id = transformation.get("source_id")
                target_id = transformation.get("target_id")
                carrying_actants = transformation.get("carrying_actants", [])
                
                # If we have the predicates in our cache, process them
                if source_id in self.predicates and target_id in self.predicates:
                    source_predicate = self.predicates[source_id]
                    target_predicate = self.predicates[target_id]
                    
                    # Process each actant that carries across the transformation
                    for actant_name in carrying_actants:
                        self._process_actant_transition(
                            actant_name=actant_name,
                            source_predicate=source_predicate,
                            target_predicate=target_predicate,
                            transformation_id=transformation.get("id"),
                            timestamp=context.get("timestamp", datetime.now().isoformat())
                        )
            
            # Check if this is a predicate event (adding a new predicate)
            elif context.get("change_type") == "predicate_added" and "predicate" in context:
                predicate = context["predicate"]
                self.predicates[predicate["id"]] = predicate
                
                # Process the actants in this predicate
                self._process_predicate_actants(
                    predicate=predicate,
                    timestamp=context.get("timestamp", datetime.now().isoformat())
                )
            
            # Check if this is a domain event (adding a new domain)
            elif context.get("change_type") == "domain_added" and "domain" in context:
                domain = context["domain"]
                self.domains[domain["id"]] = domain
        
        except Exception as e:
            self.logger.error(f"Error observing pattern evolution: {e}")
    
    def _process_predicate_actants(self, predicate: Dict[str, Any], timestamp: str) -> None:
        """
        Process the actants in a predicate.
        
        Creates journey points for each actant in the predicate.
        
        Args:
            predicate: Dictionary containing predicate information
            timestamp: Timestamp of the observation
        """
        domain_id = predicate.get("domain_id")
        predicate_id = predicate.get("id")
        
        # Process subject
        if "subject" in predicate:
            self._add_journey_point(
                actant_name=predicate["subject"],
                domain_id=domain_id,
                predicate_id=predicate_id,
                role="subject",
                timestamp=timestamp
            )
        
        # Process object
        if "object" in predicate:
            self._add_journey_point(
                actant_name=predicate["object"],
                domain_id=domain_id,
                predicate_id=predicate_id,
                role="object",
                timestamp=timestamp
            )
    
    def _add_journey_point(self, actant_name: str, domain_id: str, 
                          predicate_id: str, role: str, timestamp: str) -> None:
        """
        Add a journey point for an actant.
        
        Creates a new journey point and adds it to the actant's journey.
        
        Args:
            actant_name: Name of the actant
            domain_id: ID of the domain
            predicate_id: ID of the predicate
            role: Role of the actant in the predicate
            timestamp: Timestamp of the observation
        """
        # Create journey point
        journey_point = ActantJourneyPoint.create(
            actant_name=actant_name,
            domain_id=domain_id,
            predicate_id=predicate_id,
            role=role,
            timestamp=timestamp
        )
        
        # Get or create actant journey
        if actant_name not in self.actant_journeys:
            self.actant_journeys[actant_name] = ActantJourney.create(actant_name)
        
        # Add journey point to actant journey
        self.actant_journeys[actant_name].add_journey_point(journey_point)
    
    def _process_actant_transition(self, actant_name: str, source_predicate: Dict[str, Any],
                                  target_predicate: Dict[str, Any], transformation_id: str,
                                  timestamp: str) -> None:
        """
        Process an actant's transition between predicates.
        
        Creates a domain transition for the actant and adds it to the actant's journey.
        
        Args:
            actant_name: Name of the actant
            source_predicate: Dictionary containing source predicate information
            target_predicate: Dictionary containing target predicate information
            transformation_id: ID of the transformation
            timestamp: Timestamp of the observation
        """
        # Determine the actant's role in each predicate
        source_role = self._determine_actant_role(actant_name, source_predicate)
        target_role = self._determine_actant_role(actant_name, target_predicate)
        
        # If we couldn't determine the role in either predicate, skip
        if not source_role or not target_role:
            return
        
        # Create domain transition
        transition = DomainTransition.create(
            actant_name=actant_name,
            source_domain_id=source_predicate.get("domain_id"),
            target_domain_id=target_predicate.get("domain_id"),
            source_predicate_id=source_predicate.get("id"),
            target_predicate_id=target_predicate.get("id"),
            source_role=source_role,
            target_role=target_role,
            timestamp=timestamp,
            transformation_id=transformation_id
        )
        
        # Get or create actant journey
        if actant_name not in self.actant_journeys:
            self.actant_journeys[actant_name] = ActantJourney.create(actant_name)
        
        # Add domain transition to actant journey
        self.actant_journeys[actant_name].add_domain_transition(transition)
    
    def _determine_actant_role(self, actant_name: str, predicate: Dict[str, Any]) -> Optional[str]:
        """
        Determine an actant's role in a predicate.
        
        Args:
            actant_name: Name of the actant
            predicate: Dictionary containing predicate information
            
        Returns:
            Role of the actant in the predicate, or None if not found
        """
        if predicate.get("subject") == actant_name:
            return "subject"
        elif predicate.get("object") == actant_name:
            return "object"
        return None
    
    def get_actant_journey(self, actant_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the journey for a specific actant.
        
        Args:
            actant_name: Name of the actant
            
        Returns:
            Dictionary containing the actant's journey, or None if not found
        """
        if actant_name in self.actant_journeys:
            return self.actant_journeys[actant_name].to_dict()
        return None
    
    def get_role_shifts(self, actant_name: str) -> List[Dict[str, Any]]:
        """
        Get all role shifts for a specific actant.
        
        Args:
            actant_name: Name of the actant
            
        Returns:
            List of domain transitions where the actant's role shifted
        """
        if actant_name in self.actant_journeys:
            return [t.to_dict() for t in self.actant_journeys[actant_name].get_role_shifts()]
        return []
    
    def get_predicate_transformations(self, actant_name: str) -> List[Dict[str, Any]]:
        """
        Get all predicate transformations involving a specific actant.
        
        Args:
            actant_name: Name of the actant
            
        Returns:
            List of transformations involving the actant
        """
        transformations = []
        
        for transformation in self.predicate_transformations:
            carrying_actants = transformation.get("carrying_actants", [])
            if actant_name in carrying_actants:
                transformations.append(transformation)
        
        return transformations
    
    def get_all_actant_journeys(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all actant journeys.
        
        Returns:
            Dictionary mapping actant names to their journeys
        """
        return {name: journey.to_dict() for name, journey in self.actant_journeys.items()}
    
    def get_journey_metrics(self, actant_name: str) -> Dict[str, Any]:
        """
        Get metrics for a specific actant's journey.
        
        Args:
            actant_name: Name of the actant
            
        Returns:
            Dictionary containing metrics about the actant's journey
        """
        if actant_name not in self.actant_journeys:
            return {}
        
        journey = self.actant_journeys[actant_name]
        
        # Count domains visited
        domains_visited = set(jp.domain_id for jp in journey.journey_points)
        
        # Count role shifts
        role_shifts = len(journey.get_role_shifts())
        
        # Calculate domain transition frequency
        if len(journey.journey_points) > 1:
            transition_frequency = len(journey.domain_transitions) / (len(journey.journey_points) - 1)
        else:
            transition_frequency = 0
        
        return {
            "actant_name": actant_name,
            "domains_visited": len(domains_visited),
            "domain_transitions": len(journey.domain_transitions),
            "role_shifts": role_shifts,
            "transition_frequency": transition_frequency
        }


# Example usage:
if __name__ == "__main__":
    # Create tracker
    tracker = ActantJourneyTracker()
    
    # This would be replaced with actual integration with a LearningWindow
    print("ActantJourneyTracker initialized and ready to observe pattern evolution events.")
