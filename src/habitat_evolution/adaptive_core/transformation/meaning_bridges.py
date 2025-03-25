#!/usr/bin/env python3
"""
Meaning Bridges

This module provides functionality for tracking and nurturing meaning bridges -
potential connections that function not just as indexes but as referents themselves.
This creates a supple IO space where meaning emerges from relationships rather than
being statically defined.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict

from .actant_journey_tracker import ActantJourney
from ..id.adaptive_id import AdaptiveID

logger = logging.getLogger(__name__)


class MeaningBridge:
    """
    Represents a meaning bridge - a potential relationship or transformation
    that emerges from the interaction of actants across domains.
    
    Meaning bridges are not just static relationships but dynamic potentials
    that can influence future transformations and pattern emergence.
    """
    
    def __init__(self, 
                 source_actant_id: str,
                 target_actant_id: str,
                 bridge_type: str,
                 propensity: float,
                 context: Dict[str, Any] = None):
        """
        Initialize a meaning bridge.
        
        Args:
            source_actant_id: ID of the source actant
            target_actant_id: ID of the target actant
            bridge_type: Type of bridge (e.g., "co-occurrence", "sequence", "domain_crossing")
            propensity: Propensity value (0.0 to 1.0) indicating likelihood of this bridge
            context: Additional context for this bridge
        """
        self.source_actant_id = source_actant_id
        self.target_actant_id = target_actant_id
        self.bridge_type = bridge_type
        self.propensity = propensity
        self.context = context or {}
        self.created_at = datetime.now().isoformat()
        self.id = f"bridge_{self.source_actant_id}_{self.target_actant_id}_{self.bridge_type}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation of this meaning bridge
        """
        return {
            "id": self.id,
            "source_actant_id": self.source_actant_id,
            "target_actant_id": self.target_actant_id,
            "bridge_type": self.bridge_type,
            "propensity": self.propensity,
            "context": self.context,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeaningBridge':
        """
        Create a meaning bridge from a dictionary.
        
        Args:
            data: Dictionary representation of a meaning bridge
            
        Returns:
            A new MeaningBridge instance
        """
        bridge = cls(
            source_actant_id=data["source_actant_id"],
            target_actant_id=data["target_actant_id"],
            bridge_type=data["bridge_type"],
            propensity=data["propensity"],
            context=data.get("context", {})
        )
        bridge.created_at = data.get("created_at", bridge.created_at)
        bridge.id = data.get("id", bridge.id)
        return bridge


class MeaningBridgeTracker:
    """
    Tracks and nurtures meaning bridges across the system.
    
    This tracker identifies potential semantic relationships that aren't explicitly
    modeled yet, allowing for the emergence of new patterns and transformations.
    """
    
    def __init__(self, propensity_threshold: float = 0.3):
        """
        Initialize the meaning bridge tracker.
        
        Args:
            propensity_threshold: Threshold for considering a bridge significant
        """
        self.bridges = {}  # bridge_id -> MeaningBridge
        self.actant_bridges = defaultdict(list)  # actant_id -> [bridge_ids]
        self.domain_bridges = defaultdict(list)  # domain_id -> [bridge_ids]
        self.propensity_threshold = propensity_threshold
        self.logger = logging.getLogger(__name__)
    
    def detect_bridges(self, 
                      actant_journeys: List[ActantJourney],
                      transformation_log: List[Dict[str, Any]]) -> List[MeaningBridge]:
        """
        Detect potential meaning bridges from actant journeys and transformation logs.
        
        Args:
            actant_journeys: List of actant journeys
            transformation_log: Transformation log from HarmonicIO
            
        Returns:
            List of newly detected meaning bridges
        """
        self.logger.info("Detecting meaning bridges")
        
        # Group transformations by actant
        actant_transformations = defaultdict(list)
        for transform in transformation_log:
            actant_id = transform.get("actant_id")
            if actant_id:
                actant_transformations[actant_id].append(transform)
        
        # Create new bridges
        new_bridges = []
        
        # Detect different types of bridges
        self._detect_co_occurrence_bridges(actant_journeys, actant_transformations, new_bridges)
        self._detect_sequence_bridges(actant_transformations, new_bridges)
        self._detect_domain_crossing_bridges(actant_journeys, new_bridges)
        
        # Store new bridges
        for bridge in new_bridges:
            self.bridges[bridge.id] = bridge
            self.actant_bridges[bridge.source_actant_id].append(bridge.id)
            self.actant_bridges[bridge.target_actant_id].append(bridge.id)
            
            # Store by domain if available in context
            source_domain = bridge.context.get("source_domain")
            if source_domain:
                self.domain_bridges[source_domain].append(bridge.id)
            
            target_domain = bridge.context.get("target_domain")
            if target_domain:
                self.domain_bridges[target_domain].append(bridge.id)
        
        self.logger.info(f"Detected {len(new_bridges)} new meaning bridges")
        return new_bridges
    
    def _detect_co_occurrence_bridges(self,
                                     actant_journeys: List[ActantJourney],
                                     actant_transformations: Dict[str, List[Dict[str, Any]]],
                                     new_bridges: List[MeaningBridge]):
        """
        Detect bridges based on co-occurrence of actants in the same domains.
        
        Args:
            actant_journeys: List of actant journeys
            actant_transformations: Transformations grouped by actant
            new_bridges: List to append new bridges to
        """
        # Track actants by domain
        domain_actants = defaultdict(set)
        
        # Collect domains for each actant
        for journey in actant_journeys:
            actant_name = journey.actant_name
            # Extract domains from journey points
            visited_domains = set(jp.domain_id for jp in journey.journey_points)
            for domain in visited_domains:
                domain_actants[domain].add(actant_name)
        
        # Find co-occurrences
        for domain, actants in domain_actants.items():
            if len(actants) < 2:
                continue
                
            actant_list = list(actants)
            for i in range(len(actant_list)):
                for j in range(i + 1, len(actant_list)):
                    source_id = actant_list[i]
                    target_id = actant_list[j]
                    
                    # Calculate propensity based on domain significance
                    # For now, use a simple heuristic
                    propensity = 0.5  # Base propensity
                    
                    # Create bidirectional bridges
                    bridge1 = MeaningBridge(
                        source_actant_id=source_id,
                        target_actant_id=target_id,
                        bridge_type="co-occurrence",
                        propensity=propensity,
                        context={"domain": domain}
                    )
                    
                    bridge2 = MeaningBridge(
                        source_actant_id=target_id,
                        target_actant_id=source_id,
                        bridge_type="co-occurrence",
                        propensity=propensity,
                        context={"domain": domain}
                    )
                    
                    new_bridges.append(bridge1)
                    new_bridges.append(bridge2)
    
    def _detect_sequence_bridges(self,
                                actant_transformations: Dict[str, List[Dict[str, Any]]],
                                new_bridges: List[MeaningBridge]):
        """
        Detect bridges based on transformation sequences.
        
        Args:
            actant_transformations: Transformations grouped by actant
            new_bridges: List to append new bridges to
        """
        # Track transformation sequences
        for actant_id, transforms in actant_transformations.items():
            if len(transforms) < 2:
                continue
                
            # Sort by timestamp
            sorted_transforms = sorted(transforms, key=lambda t: t.get("timestamp", ""))
            
            # Look for sequences
            for i in range(len(sorted_transforms) - 1):
                curr_transform = sorted_transforms[i]
                next_transform = sorted_transforms[i + 1]
                
                source_domain = curr_transform.get("source_domain")
                intermediate_domain = curr_transform.get("target_domain")
                target_domain = next_transform.get("target_domain")
                
                if not all([source_domain, intermediate_domain, target_domain]):
                    continue
                
                # Create sequence bridge
                bridge = MeaningBridge(
                    source_actant_id=actant_id,
                    target_actant_id=actant_id,  # Self-reference for sequence
                    bridge_type="sequence",
                    propensity=0.7,  # Higher propensity for sequences
                    context={
                        "source_domain": source_domain,
                        "intermediate_domain": intermediate_domain,
                        "target_domain": target_domain
                    }
                )
                
                new_bridges.append(bridge)
    
    def _detect_domain_crossing_bridges(self,
                                       actant_journeys: List[ActantJourney],
                                       new_bridges: List[MeaningBridge]):
        """
        Detect bridges based on domain crossings.
        
        Args:
            actant_journeys: List of actant journeys
            new_bridges: List to append new bridges to
        """
        # Track domain crossings
        domain_crossings = defaultdict(int)
        
        for journey in actant_journeys:
            # Extract domains from journey points
            domains = set(jp.domain_id for jp in journey.journey_points)
            if len(domains) < 2:
                continue
                
            # Count domain crossings
            domains_list = list(domains)
            for i in range(len(domains_list)):
                for j in range(len(domains_list)):
                    if i != j:
                        crossing_key = (domains_list[i], domains_list[j])
                        domain_crossings[crossing_key] += 1
        
        # Create bridges for significant crossings
        for (source_domain, target_domain), count in domain_crossings.items():
            # Find actants that made this crossing
            crossing_actants = []
            for journey in actant_journeys:
                # Extract domains from journey points
                visited_domains = set(jp.domain_id for jp in journey.journey_points)
                if source_domain in visited_domains and target_domain in visited_domains:
                    crossing_actants.append(journey.actant_name)
            
            if len(crossing_actants) < 2:
                continue
                
            # Create bridges between actants that share this crossing
            for i in range(len(crossing_actants)):
                for j in range(i + 1, len(crossing_actants)):
                    source_id = crossing_actants[i]
                    target_id = crossing_actants[j]
                    
                    # Calculate propensity based on crossing frequency
                    propensity = min(0.9, 0.3 + (count * 0.1))  # Cap at 0.9
                    
                    # Create bidirectional bridges
                    bridge1 = MeaningBridge(
                        source_actant_id=source_id,
                        target_actant_id=target_id,
                        bridge_type="domain_crossing",
                        propensity=propensity,
                        context={
                            "source_domain": source_domain,
                            "target_domain": target_domain,
                            "crossing_count": count
                        }
                    )
                    
                    bridge2 = MeaningBridge(
                        source_actant_id=target_id,
                        target_actant_id=source_id,
                        bridge_type="domain_crossing",
                        propensity=propensity,
                        context={
                            "source_domain": source_domain,
                            "target_domain": target_domain,
                            "crossing_count": count
                        }
                    )
                    
                    new_bridges.append(bridge1)
                    new_bridges.append(bridge2)
    
    def get_actant_bridges(self, actant_id: str) -> List[MeaningBridge]:
        """
        Get all bridges for a specific actant.
        
        Args:
            actant_id: ID of the actant
            
        Returns:
            List of meaning bridges for the actant
        """
        bridge_ids = self.actant_bridges.get(actant_id, [])
        return [self.bridges[bid] for bid in bridge_ids if bid in self.bridges]
    
    def get_domain_bridges(self, domain_id: str) -> List[MeaningBridge]:
        """
        Get all bridges for a specific domain.
        
        Args:
            domain_id: ID of the domain
            
        Returns:
            List of meaning bridges for the domain
        """
        bridge_ids = self.domain_bridges.get(domain_id, [])
        return [self.bridges[bid] for bid in bridge_ids if bid in self.bridges]
    
    def get_all_bridges(self, threshold: Optional[float] = None) -> List[MeaningBridge]:
        """
        Get all meaning bridges above the specified threshold.
        
        Args:
            threshold: Propensity threshold (defaults to instance threshold)
            
        Returns:
            List of meaning bridges
        """
        threshold = threshold if threshold is not None else self.propensity_threshold
        return [bridge for bridge in self.bridges.values() if bridge.propensity >= threshold]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation of this tracker
        """
        return {
            "bridges": {bid: bridge.to_dict() for bid, bridge in self.bridges.items()},
            "propensity_threshold": self.propensity_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeaningBridgeTracker':
        """
        Create a meaning bridge tracker from a dictionary.
        
        Args:
            data: Dictionary representation of a meaning bridge tracker
            
        Returns:
            A new MeaningBridgeTracker instance
        """
        tracker = cls(propensity_threshold=data.get("propensity_threshold", 0.3))
        
        # Restore bridges
        for bridge_id, bridge_data in data.get("bridges", {}).items():
            bridge = MeaningBridge.from_dict(bridge_data)
            tracker.bridges[bridge_id] = bridge
            tracker.actant_bridges[bridge.source_actant_id].append(bridge_id)
            tracker.actant_bridges[bridge.target_actant_id].append(bridge_id)
            
            # Restore domain bridges
            source_domain = bridge.context.get("source_domain")
            if source_domain:
                tracker.domain_bridges[source_domain].append(bridge_id)
            
            target_domain = bridge.context.get("target_domain")
            if target_domain:
                tracker.domain_bridges[target_domain].append(bridge_id)
        
        return tracker
