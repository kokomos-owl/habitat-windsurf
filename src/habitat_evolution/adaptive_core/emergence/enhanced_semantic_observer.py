"""
Enhanced Semantic Current Observer

This module extends the SemanticCurrentObserver class to add direct relationship
observation capabilities and better integration with the event bus.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import logging

from .semantic_current_observer import SemanticCurrentObserver
from ..id.adaptive_id import AdaptiveID

class EnhancedSemanticObserver(SemanticCurrentObserver):
    """
    Enhanced semantic observer with direct relationship observation capabilities.
    
    This class extends the SemanticCurrentObserver to add methods for directly
    observing relationships between actants, making it easier to integrate
    with the event bus and test the pattern detection system.
    """
    
    def observe_relationship(self, source: str, predicate: str, target: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Directly observe a relationship between actants.
        
        Args:
            source: Source actant
            predicate: Relationship predicate
            target: Target actant
            context: Optional context information
            
        Returns:
            Observation result
        """
        if context is None:
            context = {}
            
        # Create a relationship structure
        relationship = {
            "source": source,
            "predicate": predicate,
            "target": target,
            "context": context
        }
        
        # Create a data structure that matches what observe_semantic_currents expects
        data = {
            "predicates": [{
                "subject": source,
                "verb": predicate,
                "object": target,
                "context": context
            }]
        }
        
        # Use the parent method to process this
        result = self.observe_semantic_currents(data)
        
        # Return a simplified result
        return {
            "observation_id": result["observation_id"],
            "timestamp": result["timestamp"],
            "relationship": f"{source}_{predicate}_{target}",
            "frequency": self.relationship_frequency.get(f"{source}_{predicate}_{target}", 0)
        }
    
    def get_relationship_frequency(self, source: str, predicate: str, target: str) -> int:
        """
        Get the frequency of a specific relationship.
        
        Args:
            source: Source actant
            predicate: Relationship predicate
            target: Target actant
            
        Returns:
            Frequency count
        """
        rel_key = f"{source}_{predicate}_{target}"
        return self.relationship_frequency.get(rel_key, 0)
    
    def get_relationship_data(self, source: str, predicate: str, target: str) -> Optional[Dict[str, Any]]:
        """
        Get data for a specific relationship.
        
        Args:
            source: Source actant
            predicate: Relationship predicate
            target: Target actant
            
        Returns:
            Relationship data or None if not found
        """
        rel_key = f"{source}_{predicate}_{target}"
        return self.observed_relationships.get(rel_key)
