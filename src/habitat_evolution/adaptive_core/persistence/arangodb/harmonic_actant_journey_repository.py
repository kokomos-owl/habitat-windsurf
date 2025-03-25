"""
Harmonic ActantJourney Repository for ArangoDB.

This repository extends the standard ActantJourneyRepository with harmonic I/O
capabilities, ensuring that database operations don't disrupt the natural
evolution of eigenspaces and pattern detection.

It integrates with the AdaptiveID system to provide versioning, relationship
tracking, and state change notifications for actant journeys.
"""

from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

from .actant_journey_repository import ActantJourneyRepository
from ...io.harmonic_repository_mixin import HarmonicRepositoryMixin
from ...io.harmonic_io_service import HarmonicIOService
from ...transformation.actant_journey_tracker import ActantJourney, ActantJourneyPoint, DomainTransition
from ...id.adaptive_id import AdaptiveID

logger = logging.getLogger(__name__)


class HarmonicActantJourneyRepository(ActantJourneyRepository, HarmonicRepositoryMixin):
    """
    ActantJourneyRepository with harmonic I/O capabilities.
    
    This repository ensures that actant journey persistence operations
    don't disrupt the natural evolution of patterns and eigenspaces.
    It also enhances integration with the AdaptiveID system.
    """
    
    def __init__(self, io_service: HarmonicIOService):
        """
        Initialize the harmonic actant journey repository.
        
        Args:
            io_service: Harmonic I/O service to use for scheduling
        """
        ActantJourneyRepository.__init__(self)
        HarmonicRepositoryMixin.__init__(self, io_service)
        
        # Register direct methods that will be called by the harmonic service
        self._register_direct_methods()
        
        # Track registered learning windows and field observers
        self.learning_windows = []
        self.field_observers = []
        
    def _register_direct_methods(self):
        """Register direct methods that will be called by the harmonic service."""
        # For each public method, create a corresponding _direct_ method
        # that will be called by the harmonic service
        
        # Save journey
        self._direct_save_journey = lambda journey: super(HarmonicActantJourneyRepository, self).save_journey(journey)
        
        # Save journey point
        self._direct_save_journey_point = lambda journey_id, point: super(HarmonicActantJourneyRepository, self).save_journey_point(journey_id, point)
        
        # Save domain transition
        self._direct_save_domain_transition = lambda journey_id, transition: super(HarmonicActantJourneyRepository, self).save_domain_transition(journey_id, transition)
        
        # Save role shift
        self._direct_save_role_shift = lambda journey_id, role_shift: super(HarmonicActantJourneyRepository, self).save_role_shift(journey_id, role_shift)
        
        # Get journey by actant name
        self._direct_get_journey_by_actant_name = lambda actant_name: super(HarmonicActantJourneyRepository, self).get_journey_by_actant_name(actant_name)
        
        # Get journey points
        self._direct_get_journey_points = lambda journey_id: super(HarmonicActantJourneyRepository, self).get_journey_points(journey_id)
        
        # Get domain transitions
        self._direct_get_domain_transitions = lambda journey_id: super(HarmonicActantJourneyRepository, self).get_domain_transitions(journey_id)
        
        # Get role shifts
        self._direct_get_role_shifts = lambda journey_id: super(HarmonicActantJourneyRepository, self).get_role_shifts(journey_id)
        
        # Get journey metrics
        self._direct_get_journey_metrics = lambda journey_id: super(HarmonicActantJourneyRepository, self).get_journey_metrics(journey_id)
        
    def save_journey(self, journey: Union[Dict[str, Any], ActantJourney]) -> str:
        """
        Save an actant journey with harmonic timing.
        
        This method ensures that the journey has an associated AdaptiveID
        and that the AdaptiveID is properly updated with journey information.
        
        Args:
            journey: ActantJourney object or dictionary representation
            
        Returns:
            ID of the saved journey
        """
        # Convert to dictionary if necessary
        if isinstance(journey, ActantJourney):
            journey_dict = journey.to_dict()
            
            # Ensure journey has an AdaptiveID
            if journey.adaptive_id is None:
                journey.initialize_adaptive_id()
                
            # Update the dictionary with the AdaptiveID
            journey_dict["adaptive_id"] = journey.adaptive_id.to_dict() if journey.adaptive_id else None
        else:
            journey_dict = journey
            
            # Check if journey has an AdaptiveID
            if "adaptive_id" not in journey_dict or not journey_dict["adaptive_id"]:
                # Create a new AdaptiveID
                adaptive_id = AdaptiveID(
                    base_concept=journey_dict.get("actant_name", "unknown"),
                    creator_id="harmonic_actant_journey_repository",
                    weight=1.0,
                    confidence=0.8,
                    uncertainty=0.2
                )
                
                # Add initial temporal context
                adaptive_id.update_temporal_context(
                    "creation_time",
                    datetime.now().isoformat(),
                    "initialization"
                )
                
                # Add initial journey state
                adaptive_id.update_temporal_context(
                    "journey_state",
                    {
                        "journey_points": len(journey_dict.get("journey_points", [])),
                        "domain_transitions": len(journey_dict.get("domain_transitions", [])),
                        "role_shifts": 0,
                        "domains_visited": set()
                    },
                    "initialization"
                )
                
                # Add to journey dictionary
                journey_dict["adaptive_id"] = adaptive_id.to_dict()
        
        # Create data context for harmonic scheduling
        data_context = self._create_data_context(journey_dict, "save_journey")
        data_context["data_type"] = "actant_journey"
        
        # Schedule through harmonic service
        return self._harmonic_write("save_journey", journey_dict, _data_context=data_context)
        
    def save_journey_point(self, journey_id: str, point: Union[Dict[str, Any], ActantJourneyPoint]) -> str:
        """
        Save a journey point with harmonic timing.
        
        This method also updates the associated AdaptiveID with information
        about the new journey point.
        
        Args:
            journey_id: ID of the journey to add the point to
            point: ActantJourneyPoint object or dictionary representation
            
        Returns:
            ID of the saved journey point
        """
        # Convert to dictionary if necessary
        if isinstance(point, ActantJourneyPoint):
            point_dict = point.to_dict()
        else:
            point_dict = point
            
        # Create data context for harmonic scheduling
        data_context = self._create_data_context(point_dict, "save_journey_point")
        data_context["data_type"] = "journey_point"
        data_context["journey_id"] = journey_id
        
        # Get the journey to update its AdaptiveID
        journey = self._direct_get_journey_by_actant_name(point_dict.get("actant_name"))
        
        if journey and "adaptive_id" in journey and journey["adaptive_id"]:
            # Convert AdaptiveID dictionary to object if necessary
            if isinstance(journey["adaptive_id"], dict):
                adaptive_id = AdaptiveID.from_dict(journey["adaptive_id"])
            else:
                adaptive_id = journey["adaptive_id"]
                
            # Update AdaptiveID with new journey point information
            adaptive_id.update_temporal_context(
                "journey_point_added",
                {
                    "domain_id": point_dict.get("domain_id"),
                    "role": point_dict.get("role"),
                    "timestamp": point_dict.get("timestamp")
                },
                "journey_point_addition"
            )
            
            # Update journey state
            journey_state = adaptive_id.get_temporal_context("journey_state") or {
                "journey_points": 0,
                "domain_transitions": 0,
                "role_shifts": 0,
                "domains_visited": set()
            }
            
            journey_state["journey_points"] += 1
            if "domains_visited" not in journey_state:
                journey_state["domains_visited"] = set()
                
            if isinstance(journey_state["domains_visited"], set):
                journey_state["domains_visited"].add(point_dict.get("domain_id"))
            else:
                journey_state["domains_visited"] = set([point_dict.get("domain_id")])
                
            adaptive_id.update_temporal_context(
                "journey_state",
                journey_state,
                "journey_point_addition"
            )
            
            # Notify learning windows and field observers
            self._notify_observers(
                journey["actant_name"],
                "journey_point_added",
                None,
                point_dict,
                adaptive_id
            )
            
            # Update the journey with the modified AdaptiveID
            journey["adaptive_id"] = adaptive_id.to_dict()
            self._harmonic_update("save_journey", journey)
        
        # Schedule through harmonic service
        return self._harmonic_write("save_journey_point", journey_id, point_dict, _data_context=data_context)
        
    def save_domain_transition(self, journey_id: str, transition: Union[Dict[str, Any], DomainTransition]) -> str:
        """
        Save a domain transition with harmonic timing.
        
        This method also updates the associated AdaptiveID with information
        about the new domain transition.
        
        Args:
            journey_id: ID of the journey to add the transition to
            transition: DomainTransition object or dictionary representation
            
        Returns:
            ID of the saved domain transition
        """
        # Convert to dictionary if necessary
        if isinstance(transition, DomainTransition):
            transition_dict = transition.to_dict()
        else:
            transition_dict = transition
            
        # Create data context for harmonic scheduling
        data_context = self._create_data_context(transition_dict, "save_domain_transition")
        data_context["data_type"] = "domain_transition"
        data_context["journey_id"] = journey_id
        
        # Get the journey to update its AdaptiveID
        journey = self._direct_get_journey_by_actant_name(transition_dict.get("actant_name"))
        
        if journey and "adaptive_id" in journey and journey["adaptive_id"]:
            # Convert AdaptiveID dictionary to object if necessary
            if isinstance(journey["adaptive_id"], dict):
                adaptive_id = AdaptiveID.from_dict(journey["adaptive_id"])
            else:
                adaptive_id = journey["adaptive_id"]
                
            # Update AdaptiveID with new domain transition information
            adaptive_id.update_temporal_context(
                "domain_transition_added",
                {
                    "source_domain_id": transition_dict.get("source_domain_id"),
                    "target_domain_id": transition_dict.get("target_domain_id"),
                    "source_role": transition_dict.get("source_role"),
                    "target_role": transition_dict.get("target_role"),
                    "timestamp": transition_dict.get("timestamp")
                },
                "domain_transition_addition"
            )
            
            # Update journey state
            journey_state = adaptive_id.get_temporal_context("journey_state") or {
                "journey_points": 0,
                "domain_transitions": 0,
                "role_shifts": 0,
                "domains_visited": set()
            }
            
            journey_state["domain_transitions"] += 1
            
            # Check for role shift
            if transition_dict.get("source_role") != transition_dict.get("target_role"):
                journey_state["role_shifts"] += 1
                
            adaptive_id.update_temporal_context(
                "journey_state",
                journey_state,
                "domain_transition_addition"
            )
            
            # Notify learning windows and field observers
            self._notify_observers(
                journey["actant_name"],
                "domain_transition_added",
                None,
                transition_dict,
                adaptive_id
            )
            
            # Update the journey with the modified AdaptiveID
            journey["adaptive_id"] = adaptive_id.to_dict()
            self._harmonic_update("save_journey", journey)
        
        # Schedule through harmonic service
        return self._harmonic_write("save_domain_transition", journey_id, transition_dict, _data_context=data_context)
        
    def get_journey_by_actant_name(self, actant_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a journey by actant name with harmonic timing.
        
        Args:
            actant_name: Name of the actant
            
        Returns:
            Journey dictionary, or None if not found
        """
        # Create data context for harmonic scheduling
        data_context = {
            "data_type": "actant_journey",
            "entity_id": actant_name
        }
        
        # Schedule through harmonic service
        return self._harmonic_read("get_journey_by_actant_name", actant_name, _data_context=data_context)
        
    def get_journey_points(self, journey_id: str) -> List[Dict[str, Any]]:
        """
        Get journey points with harmonic timing.
        
        Args:
            journey_id: ID of the journey
            
        Returns:
            List of journey point dictionaries
        """
        # Create data context for harmonic scheduling
        data_context = {
            "data_type": "journey_points",
            "entity_id": journey_id
        }
        
        # Schedule through harmonic service
        return self._harmonic_read("get_journey_points", journey_id, _data_context=data_context)
        
    def get_domain_transitions(self, journey_id: str) -> List[Dict[str, Any]]:
        """
        Get domain transitions with harmonic timing.
        
        Args:
            journey_id: ID of the journey
            
        Returns:
            List of domain transition dictionaries
        """
        # Create data context for harmonic scheduling
        data_context = {
            "data_type": "domain_transitions",
            "entity_id": journey_id
        }
        
        # Schedule through harmonic service
        return self._harmonic_read("get_domain_transitions", journey_id, _data_context=data_context)
        
    def get_role_shifts(self, journey_id: str) -> List[Dict[str, Any]]:
        """
        Get role shifts with harmonic timing.
        
        Args:
            journey_id: ID of the journey
            
        Returns:
            List of role shift dictionaries
        """
        # Create data context for harmonic scheduling
        data_context = {
            "data_type": "role_shifts",
            "entity_id": journey_id
        }
        
        # Schedule through harmonic service
        return self._harmonic_read("get_role_shifts", journey_id, _data_context=data_context)
        
    def get_journey_metrics(self, journey_id: str) -> Dict[str, Any]:
        """
        Get journey metrics with harmonic timing.
        
        Args:
            journey_id: ID of the journey
            
        Returns:
            Dictionary of journey metrics
        """
        # Create data context for harmonic scheduling
        data_context = {
            "data_type": "journey_metrics",
            "entity_id": journey_id
        }
        
        # Schedule through harmonic service
        return self._harmonic_read("get_journey_metrics", journey_id, _data_context=data_context)
        
    def register_learning_window(self, learning_window):
        """
        Register a learning window for notifications about actant journeys.
        
        Args:
            learning_window: The learning window to register
        """
        if learning_window not in self.learning_windows:
            self.learning_windows.append(learning_window)
            
    def register_field_observer(self, field_observer):
        """
        Register a field observer for notifications about actant journeys.
        
        Args:
            field_observer: The field observer to register
        """
        if field_observer not in self.field_observers:
            self.field_observers.append(field_observer)
            
    def _notify_observers(self, 
                         actant_name: str, 
                         change_type: str, 
                         old_value: Any, 
                         new_value: Any,
                         adaptive_id: AdaptiveID):
        """
        Notify observers about changes to actant journeys.
        
        Args:
            actant_name: Name of the actant
            change_type: Type of change
            old_value: Previous value
            new_value: New value
            adaptive_id: Associated AdaptiveID
        """
        # Notify learning windows
        for window in self.learning_windows:
            if hasattr(window, "record_state_change"):
                window.record_state_change(
                    entity_id=actant_name,
                    change_type=change_type,
                    old_value=old_value,
                    new_value=new_value,
                    origin="harmonic_actant_journey_repository",
                    adaptive_id=adaptive_id
                )
                
        # Notify field observers
        for observer in self.field_observers:
            if hasattr(observer, "observe_field_state"):
                observer.observe_field_state({
                    "entity_id": actant_name,
                    "change_type": change_type,
                    "old_value": old_value,
                    "new_value": new_value,
                    "adaptive_id": adaptive_id,
                    "timestamp": datetime.now().isoformat()
                })
