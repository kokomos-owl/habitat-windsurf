"""
Semantic Current Observer

This module implements the SemanticCurrentObserver class, which observes the flow
of concepts through semantic space without imposing predefined structures.
"""

from typing import Dict, List, Any, Tuple, Set, Optional
from datetime import datetime
import uuid
import logging

from ..id.adaptive_id import AdaptiveID
from ...field.field_navigator import FieldNavigator
from ..transformation.actant_journey_tracker import ActantJourneyTracker


class SemanticCurrentObserver:
    """
    Observes semantic currents without imposing predefined structures.
    
    This class tracks how concepts flow through semantic space, recording
    relationships between actants and predicates without categorizing them
    in advance. It allows patterns to emerge naturally from observations.
    """
    
    def __init__(self, field_navigator: FieldNavigator, journey_tracker: ActantJourneyTracker):
        """
        Initialize a semantic current observer.
        
        Args:
            field_navigator: Navigator for the semantic field
            journey_tracker: Tracker for actant journeys
        """
        self.field_navigator = field_navigator
        self.journey_tracker = journey_tracker
        self.observed_relationships = {}
        self.relationship_frequency = {}
        self.observation_history = []
        
        # Create an AdaptiveID for this observer
        self.adaptive_id = AdaptiveID(
            base_concept="semantic_current_observer",
            creator_id="system"
        )
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
    def observe_semantic_currents(self, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observe new data flowing through the system without imposing structure.
        
        Args:
            new_data: New data to observe
            
        Returns:
            Observation results
        """
        # Extract actants and their relationships from the data
        actants, relationships = self._extract_actants_and_relationships(new_data)
        
        # Track these relationships without categorizing them
        observation_results = {
            "observation_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "actants_observed": len(actants),
            "relationships_observed": len(relationships),
            "relationships": []
        }
        
        for rel in relationships:
            rel_key = f"{rel['source']}_{rel['predicate']}_{rel['target']}"
            
            # Track frequency
            if rel_key not in self.relationship_frequency:
                self.relationship_frequency[rel_key] = 0
            self.relationship_frequency[rel_key] += 1
            
            # Store relationship details
            if rel_key not in self.observed_relationships:
                self.observed_relationships[rel_key] = {
                    "source": rel["source"],
                    "predicate": rel["predicate"],
                    "target": rel["target"],
                    "first_observed": datetime.now().isoformat(),
                    "last_observed": datetime.now().isoformat(),
                    "frequency": 1,
                    "contexts": []
                }
            else:
                self.observed_relationships[rel_key]["frequency"] += 1
                self.observed_relationships[rel_key]["last_observed"] = datetime.now().isoformat()
            
            # Add context from this observation
            self.observed_relationships[rel_key]["contexts"].append({
                "observation_id": observation_results["observation_id"],
                "timestamp": observation_results["timestamp"],
                "context": rel.get("context", {})
            })
            
            # Add to results
            observation_results["relationships"].append({
                "source": rel["source"],
                "predicate": rel["predicate"],
                "target": rel["target"],
                "frequency": self.relationship_frequency[rel_key]
            })
            
            # Update the AdaptiveID with this observation
            self.adaptive_id.update_context({
                "relationship_observed": rel_key,
                "timestamp": datetime.now().isoformat(),
                "source": rel["source"],
                "predicate": rel["predicate"],
                "target": rel["target"],
                "frequency": self.relationship_frequency[rel_key]
            })
        
        # Add to observation history
        self.observation_history.append(observation_results)
        
        # Register with field observers to participate in field analysis
        if hasattr(self.field_navigator, 'register_observer'):
            self.adaptive_id.register_with_field_observer(self.field_navigator)
            
        return observation_results
    
    def _extract_actants_and_relationships(self, data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract actants and their relationships from data.
        
        Args:
            data: Data to extract from
            
        Returns:
            Tuple of (actants, relationships)
        """
        actants = []
        relationships = []
        
        # Extract from predicates if available
        if "predicates" in data:
            for pred in data["predicates"]:
                # Extract actants
                if "subject" in pred and pred["subject"] not in [a.get("name") for a in actants]:
                    actants.append({"name": pred["subject"], "role": "subject"})
                    
                if "object" in pred and pred["object"] not in [a.get("name") for a in actants]:
                    actants.append({"name": pred["object"], "role": "object"})
                
                # Extract relationship
                if "subject" in pred and "verb" in pred and "object" in pred:
                    relationships.append({
                        "source": pred["subject"],
                        "predicate": pred["verb"],
                        "target": pred["object"],
                        "context": pred.get("context", {})
                    })
        
        # Extract from actant journeys if available
        if "actant_journeys" in data:
            for journey in data["actant_journeys"]:
                if journey["actant_name"] not in [a.get("name") for a in actants]:
                    actants.append({"name": journey["actant_name"], "role": "traveler"})
                
                # Extract relationships from transitions
                if "domain_transitions" in journey:
                    for transition in journey["domain_transitions"]:
                        relationships.append({
                            "source": journey["actant_name"],
                            "predicate": "transitions_from",
                            "target": transition["source_domain_id"],
                            "context": {"transition_id": transition["id"]}
                        })
                        
                        relationships.append({
                            "source": journey["actant_name"],
                            "predicate": "transitions_to",
                            "target": transition["target_domain_id"],
                            "context": {"transition_id": transition["id"]}
                        })
        
        # Extract from field analysis if available
        if "field_analysis" in data:
            field = data["field_analysis"]
            
            # Extract patterns as actants
            if "patterns" in field:
                for pattern_id, pattern_data in field["patterns"].items():
                    actants.append({"name": pattern_id, "role": "pattern"})
            
            # Extract resonance relationships
            if "resonance_relationships" in field:
                for rel_id, rel_data in field["resonance_relationships"].items():
                    if "patterns" in rel_data and "type" in rel_data:
                        for pattern in rel_data["patterns"]:
                            relationships.append({
                                "source": pattern,
                                "predicate": f"participates_in_{rel_data['type']}",
                                "target": rel_id,
                                "context": {"strength": rel_data.get("strength", 0.0)}
                            })
        
        return actants, relationships
    
    def get_frequent_relationships(self, threshold: int = 3) -> List[Dict[str, Any]]:
        """
        Get relationships that occur frequently.
        
        Args:
            threshold: Minimum frequency to include
            
        Returns:
            List of frequent relationships
        """
        frequent = []
        
        for rel_key, frequency in self.relationship_frequency.items():
            if frequency >= threshold:
                rel_data = self.observed_relationships[rel_key]
                frequent.append({
                    "source": rel_data["source"],
                    "predicate": rel_data["predicate"],
                    "target": rel_data["target"],
                    "frequency": frequency,
                    "first_observed": rel_data["first_observed"],
                    "last_observed": rel_data["last_observed"]
                })
        
        return frequent
    
    def register_with_learning_window(self, learning_window) -> None:
        """
        Register this observer with a learning window.
        
        Args:
            learning_window: The learning window to register with
        """
        self.adaptive_id.register_with_learning_window(learning_window)
