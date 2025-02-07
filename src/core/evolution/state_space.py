"""State-space evolution tracking for pattern dynamics.

This module tracks the evolution of patterns through state-space,
not just temporal sequences. It maintains the relationship between
adaptive_ids and their state transitions, allowing us to understand
how patterns emerge from interface interactions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
import numpy as np
from collections import defaultdict

@dataclass
class StateTransition:
    """Represents a transition in state-space."""
    from_state: Dict[str, Any]
    to_state: Dict[str, Any]
    adaptive_id: str
    interface_context: Dict[str, Any]
    energy_delta: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_vector(self) -> np.ndarray:
        """Convert transition to vector form for analysis."""
        # TODO: Implement proper state vectorization
        return np.array([
            self.energy_delta,
            len(self.interface_context),
            # Add more dimensions as we understand the state space
        ])

@dataclass
class StateSpace:
    """Tracks pattern evolution through state-space."""
    transitions: Dict[str, List[StateTransition]] = field(default_factory=lambda: defaultdict(list))
    interface_patterns: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    def record_transition(self, 
                        adaptive_id: str,
                        from_state: Dict[str, Any],
                        to_state: Dict[str, Any],
                        interface_context: Dict[str, Any],
                        energy_delta: float) -> None:
        """Record a state transition with interface context."""
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            adaptive_id=adaptive_id,
            interface_context=interface_context,
            energy_delta=energy_delta
        )
        self.transitions[adaptive_id].append(transition)
        
        # Track patterns emerging from interfaces
        for interface_id in interface_context.get('interface_ids', []):
            self.interface_patterns[interface_id].add(adaptive_id)
    
    def get_evolution_trajectory(self, adaptive_id: str) -> List[Dict[str, Any]]:
        """Get the evolution trajectory for a pattern."""
        transitions = self.transitions.get(adaptive_id, [])
        return [
            {
                'from_state': t.from_state,
                'to_state': t.to_state,
                'energy_delta': t.energy_delta,
                'timestamp': t.timestamp.isoformat(),
                'interface_context': t.interface_context
            }
            for t in sorted(transitions, key=lambda x: x.timestamp)
        ]
    
    def get_interface_patterns(self, interface_id: str) -> Set[str]:
        """Get patterns that emerged from an interface."""
        return self.interface_patterns.get(interface_id, set())
    
    def analyze_trajectory(self, adaptive_id: str) -> Dict[str, Any]:
        """Analyze the evolution trajectory of a pattern."""
        transitions = self.transitions.get(adaptive_id, [])
        if not transitions:
            return {}
            
        vectors = np.array([t.to_vector() for t in transitions])
        
        return {
            'total_energy_delta': sum(t.energy_delta for t in transitions),
            'transition_count': len(transitions),
            'trajectory_variance': np.var(vectors, axis=0).tolist(),
            'interface_diversity': len(set(
                iface for t in transitions 
                for iface in t.interface_context.get('interface_ids', [])
            ))
        }

    def get_state_space_metrics(self) -> Dict[str, Any]:
        """Calculate metrics about the state space evolution."""
        all_transitions = [
            t for transitions in self.transitions.values()
            for t in transitions
        ]
        
        if not all_transitions:
            return {}
            
        return {
            'total_patterns': len(self.transitions),
            'total_transitions': len(all_transitions),
            'total_interfaces': len(self.interface_patterns),
            'avg_transitions_per_pattern': len(all_transitions) / len(self.transitions),
            'avg_patterns_per_interface': sum(len(patterns) for patterns in self.interface_patterns.values()) / len(self.interface_patterns) if self.interface_patterns else 0
        }
