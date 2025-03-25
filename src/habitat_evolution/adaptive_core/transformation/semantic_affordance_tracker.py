#!/usr/bin/env python3
"""
Semantic Affordance Tracker

This module provides functionality for tracking and nurturing semantic affordances -
potential suppositions that function not just as indexes but as referents themselves.
This creates a supple IO space where meaning emerges from relationships rather than
being statically defined.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict

from habitat_evolution.adaptive_core.transformation.actant_journey_tracker import ActantJourney
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

logger = logging.getLogger(__name__)


class SemanticAffordance:
    """
    Represents a semantic affordance - a potential relationship or transformation
    that emerges from the interaction of actants across domains.
    
    Semantic affordances are not just static relationships but dynamic potentials
    that can influence future transformations and pattern emergence.
    """
    
    def __init__(self, 
                 source_actant_id: str,
                 target_actant_id: str,
                 affordance_type: str,
                 propensity: float,
                 context: Dict[str, Any] = None):
        """
        Initialize a semantic affordance.
        
        Args:
            source_actant_id: ID of the source actant
            target_actant_id: ID of the target actant
            affordance_type: Type of affordance (e.g., "transformation", "resonance", "influence")
            propensity: Propensity value (0.0 to 1.0) indicating likelihood of this affordance
            context: Additional context for this affordance
        """
        self.source_actant_id = source_actant_id
        self.target_actant_id = target_actant_id
        self.affordance_type = affordance_type
        self.propensity = propensity
        self.context = context or {}
        self.creation_time = datetime.now()
        self.last_update_time = self.creation_time
        self.activation_count = 0
        self.affordance_id = f"{source_actant_id}:{target_actant_id}:{affordance_type}:{self.creation_time.isoformat()}"
    
    def activate(self, activation_context: Dict[str, Any] = None) -> None:
        """
        Activate this affordance, increasing its propensity.
        
        Args:
            activation_context: Context for this activation
        """
        self.activation_count += 1
        self.last_update_time = datetime.now()
        
        # Increase propensity with diminishing returns
        self.propensity = min(1.0, self.propensity + (1.0 - self.propensity) * 0.1)
        
        if activation_context:
            # Update context with new information
            self.context.update(activation_context)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation of this affordance
        """
        return {
            "affordance_id": self.affordance_id,
            "source_actant_id": self.source_actant_id,
            "target_actant_id": self.target_actant_id,
            "affordance_type": self.affordance_type,
            "propensity": self.propensity,
            "creation_time": self.creation_time.isoformat(),
            "last_update_time": self.last_update_time.isoformat(),
            "activation_count": self.activation_count,
            "context": self.context
        }


class SemanticAffordanceTracker:
    """
    Tracks and nurtures semantic affordances across the system.
    
    This tracker identifies potential semantic relationships that aren't explicitly
    modeled yet, allowing for the emergence of new patterns and transformations.
    """
    
    def __init__(self, propensity_threshold: float = 0.3):
        """
        Initialize the semantic affordance tracker.
        
        Args:
            propensity_threshold: Threshold for considering an affordance significant
        """
        self.affordances = {}  # affordance_id -> SemanticAffordance
        self.actant_affordances = defaultdict(list)  # actant_id -> [affordance_ids]
        self.domain_affordances = defaultdict(list)  # domain_id -> [affordance_ids]
        self.propensity_threshold = propensity_threshold
        self.logger = logging.getLogger(__name__)
    
    def detect_affordances(self, 
                          actant_journeys: List[ActantJourney],
                          transformation_log: List[Dict[str, Any]]) -> List[SemanticAffordance]:
        """
        Detect potential semantic affordances from actant journeys and transformation logs.
        
        Args:
            actant_journeys: List of actant journeys
            transformation_log: Transformation log from HarmonicIO
            
        Returns:
            List of newly detected semantic affordances
        """
        self.logger.info("Detecting semantic affordances")
        
        new_affordances = []
        
        # Group transformations by actant
        actant_transformations = defaultdict(list)
        for transform in transformation_log:
            actant_transformations[transform["actant_id"]].append(transform)
        
        # Detect co-occurrence patterns
        self._detect_co_occurrence_affordances(actant_journeys, actant_transformations, new_affordances)
        
        # Detect transformation sequence patterns
        self._detect_sequence_affordances(actant_transformations, new_affordances)
        
        # Detect domain crossing patterns
        self._detect_domain_crossing_affordances(actant_journeys, new_affordances)
        
        self.logger.info(f"Detected {len(new_affordances)} new semantic affordances")
        return new_affordances
    
    def _detect_co_occurrence_affordances(self,
                                         actant_journeys: List[ActantJourney],
                                         actant_transformations: Dict[str, List[Dict[str, Any]]],
                                         new_affordances: List[SemanticAffordance]) -> None:
        """
        Detect affordances based on co-occurrence of actants in the same domains.
        
        Args:
            actant_journeys: List of actant journeys
            actant_transformations: Transformations grouped by actant
            new_affordances: List to append new affordances to
        """
        # Map domains to actants
        domain_actants = defaultdict(set)
        for journey in actant_journeys:
            for point in journey.journey_points:
                domain_actants[point.domain_id].add(journey.actant_name)
        
        # Find co-occurring actants
        for domain, actants in domain_actants.items():
            if len(actants) < 2:
                continue
                
            actant_list = list(actants)
            for i in range(len(actant_list)):
                for j in range(i + 1, len(actant_list)):
                    source_actant = actant_list[i]
                    target_actant = actant_list[j]
                    
                    # Calculate propensity based on co-occurrence frequency
                    propensity = 0.3  # Base propensity
                    
                    # Create bidirectional affordances
                    affordance_id = f"{source_actant}:{target_actant}:co-occurrence:{domain}"
                    if affordance_id not in self.affordances:
                        affordance = SemanticAffordance(
                            source_actant_id=source_actant,
                            target_actant_id=target_actant,
                            affordance_type="co-occurrence",
                            propensity=propensity,
                            context={"domain": domain}
                        )
                        self.affordances[affordance.affordance_id] = affordance
                        self.actant_affordances[source_actant].append(affordance.affordance_id)
                        self.actant_affordances[target_actant].append(affordance.affordance_id)
                        self.domain_affordances[domain].append(affordance.affordance_id)
                        new_affordances.append(affordance)
                    else:
                        self.affordances[affordance_id].activate({"domain": domain})
    
    def _detect_sequence_affordances(self,
                                    actant_transformations: Dict[str, List[Dict[str, Any]]],
                                    new_affordances: List[SemanticAffordance]) -> None:
        """
        Detect affordances based on transformation sequences.
        
        Args:
            actant_transformations: Transformations grouped by actant
            new_affordances: List to append new affordances to
        """
        for actant_id, transformations in actant_transformations.items():
            # Sort transformations by timestamp
            sorted_transforms = sorted(transformations, key=lambda t: t["timestamp"])
            
            if len(sorted_transforms) < 2:
                continue
                
            # Look for sequential patterns
            for i in range(len(sorted_transforms) - 1):
                current = sorted_transforms[i]
                next_transform = sorted_transforms[i + 1]
                
                source_domain = current["source_domain"]
                intermediate_domain = current["target_domain"]
                target_domain = next_transform["target_domain"]
                
                # Only consider true sequences (where intermediate is the source of next)
                if intermediate_domain != next_transform["source_domain"]:
                    continue
                
                # Create sequence affordance
                affordance_type = "sequence"
                affordance_id = f"{actant_id}:{source_domain}:{intermediate_domain}:{target_domain}:{affordance_type}"
                
                if affordance_id not in self.affordances:
                    affordance = SemanticAffordance(
                        source_actant_id=actant_id,
                        target_actant_id=actant_id,  # Self-referential for sequences
                        affordance_type=affordance_type,
                        propensity=0.4,  # Higher base propensity for sequences
                        context={
                            "source_domain": source_domain,
                            "intermediate_domain": intermediate_domain,
                            "target_domain": target_domain,
                            "transformation_types": [current["transformation_type"], next_transform["transformation_type"]]
                        }
                    )
                    self.affordances[affordance.affordance_id] = affordance
                    self.actant_affordances[actant_id].append(affordance.affordance_id)
                    new_affordances.append(affordance)
                else:
                    self.affordances[affordance_id].activate()
    
    def _detect_domain_crossing_affordances(self,
                                           actant_journeys: List[ActantJourney],
                                           new_affordances: List[SemanticAffordance]) -> None:
        """
        Detect affordances based on domain crossings.
        
        Args:
            actant_journeys: List of actant journeys
            new_affordances: List to append new affordances to
        """
        # Track domain crossings by actant
        domain_crossings = defaultdict(lambda: defaultdict(int))
        
        for journey in actant_journeys:
            for transition in journey.domain_transitions:
                source = transition.source_domain_id
                target = transition.target_domain_id
                domain_crossings[journey.actant_name][(source, target)] += 1
        
        # Create affordances for frequent domain crossings
        for actant_id, crossings in domain_crossings.items():
            for (source_domain, target_domain), count in crossings.items():
                # Calculate propensity based on crossing frequency
                propensity = min(0.8, 0.2 + (count * 0.1))  # Increases with count but caps at 0.8
                
                affordance_type = "domain_crossing"
                affordance_id = f"{actant_id}:{source_domain}:{target_domain}:{affordance_type}"
                
                if affordance_id not in self.affordances:
                    affordance = SemanticAffordance(
                        source_actant_id=actant_id,
                        target_actant_id=actant_id,  # Self-referential for domain crossings
                        affordance_type=affordance_type,
                        propensity=propensity,
                        context={
                            "source_domain": source_domain,
                            "target_domain": target_domain,
                            "crossing_count": count
                        }
                    )
                    self.affordances[affordance.affordance_id] = affordance
                    self.actant_affordances[actant_id].append(affordance.affordance_id)
                    self.domain_affordances[source_domain].append(affordance.affordance_id)
                    self.domain_affordances[target_domain].append(affordance.affordance_id)
                    new_affordances.append(affordance)
                else:
                    self.affordances[affordance_id].activate({"crossing_count": count})
    
    def get_actant_affordances(self, actant_id: str) -> List[SemanticAffordance]:
        """
        Get all affordances for a specific actant.
        
        Args:
            actant_id: ID of the actant
            
        Returns:
            List of semantic affordances for the actant
        """
        return [self.affordances[affordance_id] 
                for affordance_id in self.actant_affordances.get(actant_id, [])
                if self.affordances[affordance_id].propensity >= self.propensity_threshold]
    
    def get_domain_affordances(self, domain_id: str) -> List[SemanticAffordance]:
        """
        Get all affordances for a specific domain.
        
        Args:
            domain_id: ID of the domain
            
        Returns:
            List of semantic affordances for the domain
        """
        return [self.affordances[affordance_id] 
                for affordance_id in self.domain_affordances.get(domain_id, [])
                if self.affordances[affordance_id].propensity >= self.propensity_threshold]
    
    def get_all_affordances(self, threshold: Optional[float] = None) -> List[SemanticAffordance]:
        """
        Get all semantic affordances above the specified threshold.
        
        Args:
            threshold: Propensity threshold (defaults to instance threshold)
            
        Returns:
            List of semantic affordances
        """
        threshold = threshold if threshold is not None else self.propensity_threshold
        return [affordance for affordance in self.affordances.values() 
                if affordance.propensity >= threshold]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation of this tracker
        """
        return {
            "affordances": {
                affordance_id: affordance.to_dict()
                for affordance_id, affordance in self.affordances.items()
                if affordance.propensity >= self.propensity_threshold
            },
            "actant_affordance_counts": {
                actant_id: len(affordance_ids)
                for actant_id, affordance_ids in self.actant_affordances.items()
            },
            "domain_affordance_counts": {
                domain_id: len(affordance_ids)
                for domain_id, affordance_ids in self.domain_affordances.items()
            },
            "propensity_threshold": self.propensity_threshold,
            "total_affordances": len(self.affordances),
            "significant_affordances": len([a for a in self.affordances.values() 
                                          if a.propensity >= self.propensity_threshold])
        }
