"""
Query Actant

Implements queries as first-class actants in the Habitat Evolution system, allowing
them to participate in semantic relationships and transformations across modalities
and AI systems.

This module enables a modality-agnostic and AI-agnostic approach to queries, where
queries can form meaning bridges with other actants and evolve through interactions.
"""

from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
import uuid
from datetime import datetime
import logging
import json

from ..id.adaptive_id import AdaptiveID
from ..transformation.actant_journey_tracker import ActantJourney, ActantJourneyPoint

@dataclass
class QueryActant:
    """
    Represents a query as a first-class actant in the system.
    
    A query actant can participate in semantic relationships and transformations,
    forming meaning bridges with other actants and evolving through interactions.
    It maintains its semantic identity across different modalities and AI systems.
    """
    id: str
    query_text: str
    modality: str = "text"  # Default modality is text, but can be image, audio, video, etc.
    context: Dict[str, Any] = field(default_factory=dict)
    adaptive_id: Optional[AdaptiveID] = None
    actant_journey: Optional[ActantJourney] = None
    
    @classmethod
    def create(cls, query_text: str, modality: str = "text", context: Dict[str, Any] = None):
        """Create a new query actant."""
        query_id = f"query_{str(uuid.uuid4())[:8]}"
        query = cls(
            id=query_id,
            query_text=query_text,
            modality=modality,
            context=context or {}
        )
        
        # Initialize the AdaptiveID for this query
        query.initialize_adaptive_id()
        
        # Initialize the actant journey for this query
        query.initialize_actant_journey()
        
        return query
    
    def initialize_adaptive_id(self) -> None:
        """Initialize the AdaptiveID for this query actant.
        
        This sets up the AdaptiveID instance with appropriate initial state and context,
        allowing it to function as a first-class entity that can track the query's
        evolution across semantic domains.
        """
        if self.adaptive_id is None:
            self.adaptive_id = AdaptiveID(
                base_concept=self.query_text,
                creator_id="query_actant",
                weight=1.0,
                confidence=0.8,
                uncertainty=0.2
            )
            
            # Add initial temporal context
            self.adaptive_id.update_temporal_context(
                "creation_time",
                datetime.now().isoformat(),
                "initialization"
            )
            
            # Add initial query state
            self.adaptive_id.update_temporal_context(
                "query_state",
                {
                    "modality": self.modality,
                    "context": self.context,
                    "version": 1  # Add versioning to track query evolution
                },
                "initialization"
            )
    
    def initialize_actant_journey(self) -> None:
        """Initialize the actant journey for this query.
        
        This creates an ActantJourney instance for the query, allowing it to
        participate in semantic relationships and transformations like any other actant.
        """
        if self.actant_journey is None:
            self.actant_journey = ActantJourney.create(self.id)
            
            # Add initial journey point in the query domain
            initial_point = ActantJourneyPoint(
                id=str(uuid.uuid4()),
                actant_name=self.id,
                domain_id="query_domain",
                predicate_id="initial_query",
                role="subject",
                timestamp=datetime.now().isoformat()
            )
            
            self.actant_journey.add_journey_point(initial_point)
    
    def transform_modality(self, new_modality: str, transformation_context: Dict[str, Any] = None) -> 'QueryActant':
        """Transform the query to a new modality.
        
        This method creates a new query actant with the same semantic content but in a
        different modality, while maintaining the semantic identity through the AdaptiveID.
        
        Args:
            new_modality: The new modality for the query (e.g., "image", "audio", "video")
            transformation_context: Additional context for the transformation
            
        Returns:
            A new QueryActant instance in the new modality
        """
        # Store the previous state for change notification
        old_state = self.to_dict()
        
        # Create a new query actant in the new modality
        new_query = QueryActant.create(
            query_text=self.query_text,
            modality=new_modality,
            context={**self.context, **(transformation_context or {})}
        )
        
        # Link the new query's AdaptiveID to the original
        if self.adaptive_id and new_query.adaptive_id:
            self.adaptive_id.add_relationship(
                relationship_type="modality_transformation",
                target_id=new_query.adaptive_id.id,
                context={
                    "source_modality": self.modality,
                    "target_modality": new_modality,
                    "transformation_time": datetime.now().isoformat()
                }
            )
            
            # Notify about the state change
            self.adaptive_id.notify_state_change(
                "modality_transformed",
                old_state,
                new_query.to_dict(),
                "query_actant"
            )
        
        # Add a journey point to the actant journey for this transformation
        if self.actant_journey:
            transformation_point = ActantJourneyPoint(
                id=str(uuid.uuid4()),
                actant_name=self.id,
                domain_id=f"{new_modality}_domain",
                predicate_id="modality_transformation",
                role="subject",
                timestamp=datetime.now().isoformat()
            )
            
            self.actant_journey.add_journey_point(transformation_point)
        
        return new_query
    
    def evolve(self, new_query_text: str, evolution_context: Dict[str, Any] = None) -> 'QueryActant':
        """Evolve the query based on interactions with the system.
        
        This method creates a new query actant that represents an evolution of the
        original query, while maintaining the semantic identity through the AdaptiveID.
        
        Args:
            new_query_text: The evolved query text
            evolution_context: Additional context for the evolution
            
        Returns:
            A new QueryActant instance representing the evolved query
        """
        # Store the previous state for change notification
        old_state = self.to_dict()
        
        # Create a new query actant with the evolved text
        new_query = QueryActant.create(
            query_text=new_query_text,
            modality=self.modality,
            context={**self.context, **(evolution_context or {})}
        )
        
        # Link the new query's AdaptiveID to the original
        if self.adaptive_id and new_query.adaptive_id:
            self.adaptive_id.add_relationship(
                relationship_type="query_evolution",
                target_id=new_query.adaptive_id.id,
                context={
                    "original_query": self.query_text,
                    "evolved_query": new_query_text,
                    "evolution_time": datetime.now().isoformat()
                }
            )
            
            # Notify about the state change
            self.adaptive_id.notify_state_change(
                "query_evolved",
                old_state,
                new_query.to_dict(),
                "query_actant"
            )
        
        # Add a journey point to the actant journey for this evolution
        if self.actant_journey:
            evolution_point = ActantJourneyPoint(
                id=str(uuid.uuid4()),
                actant_name=self.id,
                domain_id="evolved_query_domain",
                predicate_id="query_evolution",
                role="subject",
                timestamp=datetime.now().isoformat()
            )
            
            self.actant_journey.add_journey_point(evolution_point)
        
        return new_query
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "query_text": self.query_text,
            "modality": self.modality,
            "context": self.context
        }
        
        # Add AdaptiveID information if available
        if self.adaptive_id:
            result["adaptive_id"] = self.adaptive_id.to_dict()
        
        # Add actant journey information if available
        if self.actant_journey:
            result["actant_journey"] = self.actant_journey.to_dict()
        
        return result
    
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryActant':
        """Create a QueryActant from a dictionary representation."""
        query = cls(
            id=data["id"],
            query_text=data["query_text"],
            modality=data["modality"],
            context=data["context"]
        )
        
        # Restore AdaptiveID if available
        if "adaptive_id" in data and data["adaptive_id"]:
            query.adaptive_id = AdaptiveID.from_dict(data["adaptive_id"])
        
        # Restore actant journey if available
        if "actant_journey" in data and data["actant_journey"]:
            query.actant_journey = ActantJourney.from_dict(data["actant_journey"])
        
        return query
