"""
Quality transition tracking for pattern evolution.

This module provides the QualityTransitionTracker class which tracks quality
state transitions for entities as they evolve through the system.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from collections import defaultdict

from .quality_transitions_import_adapter import PatternState

logger = logging.getLogger(__name__)

class QualityTransitionTracker:
    """Track quality transitions for entities.
    
    This class tracks how entities move between quality states over time,
    providing insights into pattern evolution and the effectiveness of
    the self-reinforcing feedback mechanism.
    """
    
    def __init__(self):
        """Initialize the quality transition tracker."""
        self.transitions = defaultdict(list)  # entity -> list of transitions
        self.transition_counts = {
            "uncertain_to_good": 0,
            "uncertain_to_poor": 0,
            "good_to_poor": 0,
            "poor_to_uncertain": 0,
            "poor_to_good": 0,
            "good_to_uncertain": 0
        }
        self.pattern_state_transitions = defaultdict(int)  # (from_state, to_state) -> count
    
    def record_transition(self, entity: str, from_state: str, to_state: str, 
                         metrics: Dict[str, float], pattern_state: PatternState) -> None:
        """Record a quality state transition for an entity.
        
        Args:
            entity: The entity transitioning
            from_state: Starting quality state
            to_state: Ending quality state
            metrics: Quality metrics at transition time
            pattern_state: Current pattern state
        """
        transition = {
            "from": from_state,
            "to": to_state,
            "metrics": metrics,
            "pattern_state": pattern_state.name,
            "timestamp": datetime.now().isoformat()
        }
        
        self.transitions[entity].append(transition)
        
        # Update transition counts
        transition_key = f"{from_state}_to_{to_state}"
        if transition_key in self.transition_counts:
            self.transition_counts[transition_key] += 1
        
        # Update pattern state transitions
        if "previous_pattern_state" in self.transitions[entity][-2] if len(self.transitions[entity]) > 1 else False:
            prev_pattern_state = self.transitions[entity][-2]["pattern_state"]
            self.pattern_state_transitions[(prev_pattern_state, pattern_state.name)] += 1
        
        logger.info(f"Recorded transition for '{entity}': {from_state} -> {to_state}")
    
    def get_entity_transitions(self, entity: str) -> List[Dict[str, Any]]:
        """Get all transitions for an entity.
        
        Args:
            entity: The entity to get transitions for
            
        Returns:
            List of transitions for the entity
        """
        return self.transitions.get(entity, [])
    
    def get_all_transitions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all transitions for all entities.
        
        Returns:
            Dictionary of entity -> transitions
        """
        return dict(self.transitions)
    
    def get_transition_summary(self) -> Dict[str, Any]:
        """Get a summary of transitions.
        
        Returns:
            Dictionary with transition summary
        """
        return {
            "transition_counts": self.transition_counts,
            "pattern_state_transitions": dict(self.pattern_state_transitions),
            "total_entities": len(self.transitions),
            "total_transitions": sum(len(transitions) for transitions in self.transitions.values()),
            "entities_with_multiple_transitions": sum(1 for transitions in self.transitions.values() if len(transitions) > 1)
        }
    
    def get_quality_improvement_metrics(self) -> Dict[str, float]:
        """Get metrics on quality improvement over time.
        
        Returns:
            Dictionary with quality improvement metrics
        """
        # Calculate improvement ratio
        positive_transitions = (
            self.transition_counts["uncertain_to_good"] +
            self.transition_counts["poor_to_uncertain"] +
            self.transition_counts["poor_to_good"]
        )
        
        negative_transitions = (
            self.transition_counts["good_to_uncertain"] +
            self.transition_counts["good_to_poor"] +
            self.transition_counts["uncertain_to_poor"]
        )
        
        total_transitions = positive_transitions + negative_transitions
        
        if total_transitions == 0:
            improvement_ratio = 0.0
        else:
            improvement_ratio = positive_transitions / total_transitions
        
        # Calculate average transitions per entity
        total_entities = len(self.transitions)
        avg_transitions = (sum(len(transitions) for transitions in self.transitions.values()) / 
                          max(1, total_entities))
        
        return {
            "improvement_ratio": improvement_ratio,
            "positive_transitions": positive_transitions,
            "negative_transitions": negative_transitions,
            "avg_transitions_per_entity": avg_transitions,
            "entities_with_improvements": sum(1 for transitions in self.transitions.values() 
                                           if any(t["from"] in ["uncertain", "poor"] and t["to"] == "good" 
                                                for t in transitions))
        }
    
    def get_evolution_trajectory(self, entity: str) -> Dict[str, Any]:
        """Get the evolution trajectory for an entity.
        
        Args:
            entity: The entity to get trajectory for
            
        Returns:
            Dictionary with evolution trajectory
        """
        transitions = self.get_entity_transitions(entity)
        
        if not transitions:
            return {"entity": entity, "has_trajectory": False}
        
        # Extract states and metrics over time
        states = [t["from"] for t in transitions]
        if transitions:
            states.append(transitions[-1]["to"])
            
        metrics_over_time = []
        for t in transitions:
            if "metrics" in t:
                metrics_over_time.append({
                    "timestamp": t["timestamp"],
                    "metrics": t["metrics"]
                })
        
        # Determine if entity has improved
        has_improved = any(t["from"] in ["uncertain", "poor"] and t["to"] == "good" for t in transitions)
        
        # Determine stability
        state_changes = len(set(states))
        is_stable = state_changes <= 2  # At most 2 different states
        
        return {
            "entity": entity,
            "has_trajectory": True,
            "states": states,
            "current_state": states[-1] if states else None,
            "metrics_over_time": metrics_over_time,
            "has_improved": has_improved,
            "is_stable": is_stable,
            "transition_count": len(transitions)
        }
