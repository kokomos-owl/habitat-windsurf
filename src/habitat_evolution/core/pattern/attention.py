"""Attention filters for pattern observation."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
@dataclass
class NeighborContext:
    """Context from neighboring field positions."""
    position: Dict[str, float]
    neighbors: Dict[str, Dict]  # Observations at neighboring positions
    distances: Dict[str, float]  # Distances to neighbors
    gradients: Dict[str, Dict]   # Gradients to neighbors

@dataclass
class AttentionFilter:
    """Filter for focusing pattern observation."""
    name: str
    conditions: Dict[str, Callable]
    weight: float = 1.0
    neighbor_conditions: Dict[str, Callable] = None  # Conditions that check neighbor relationships
    
    def evaluate(self, observations: Dict, neighbor_context: Optional[NeighborContext] = None) -> float:
        """Evaluate how well observations match this filter's conditions."""
        scores = []
        
        # Evaluate local conditions
        for metric, condition in self.conditions.items():
            path = metric.split('.')
            value = observations
            for key in path:
                value = value[key]
            scores.append(condition(value))
            
        # Evaluate neighbor conditions if present
        if self.neighbor_conditions and neighbor_context:
            for check_name, check_fn in self.neighbor_conditions.items():
                score = check_fn(neighbor_context)
                scores.append(score)
                
                # Log significant neighbor relationships
                if score >= 0.8:
                    logger.info(f"[NEIGHBOR FACT] Strong {check_name} detected")
                    
        return np.mean(scores) * self.weight

class AttentionSet:
    """Set of attention filters that define what patterns to seek."""
    
    def __init__(self):
        self.filters: List[AttentionFilter] = []
        
    def add_filter(self, attention_filter: AttentionFilter):
        """Add an attention filter to the set."""
        self.filters.append(attention_filter)
        
    def evaluate(self, observations: Dict, neighbor_context: Optional[NeighborContext] = None) -> Dict[str, float]:
        """Evaluate observations against all filters.
        
        Args:
            observations: Current position observations
            neighbor_context: Optional context from neighboring positions
        """
        results = {}
        for f in self.filters:
            results[f.name] = f.evaluate(observations, neighbor_context)
        return results

# Example attention filters for climate domain
def check_gradient_alignment(context: NeighborContext) -> float:
    """Check if gradients between neighbors align in meaningful ways."""
    alignments = []
    for n1, g1 in context.gradients.items():
        for n2, g2 in context.gradients.items():
            if n1 != n2:
                # Calculate cosine similarity between gradient directions
                dir1, dir2 = g1.direction, g2.direction
                dot_product = sum(dir1[k] * dir2[k] for k in ['x', 'y'])
                mag1 = np.sqrt(sum(v*v for v in dir1.values()))
                mag2 = np.sqrt(sum(v*v for v in dir2.values()))
                
                if mag1 > 0 and mag2 > 0:
                    # Weight alignment by gradient magnitudes
                    alignment = abs(dot_product / (mag1 * mag2))
                    weight = (g1.magnitude * g2.magnitude) / 4.0  # Normalize to [0,1]
                    alignments.append(alignment * weight)
    return np.mean(alignments) if alignments else 0.0

def check_phase_coherence(context: NeighborContext) -> float:
    """Check if phase relationships between neighbors are coherent."""
    phases = [n['wave']['phase'] for n in context.neighbors.values()]
    if not phases:
        return 0.0
    # Calculate circular variance of phases
    phases = np.array(phases)
    mean_vector = np.mean(np.exp(1j * phases))
    return abs(mean_vector)  # 1.0 = perfectly coherent, 0.0 = random

def check_field_stability(context: NeighborContext) -> float:
    """Check if the local field neighborhood is stable."""
    stabilities = [n['field']['stability'] for n in context.neighbors.values()]
    if not stabilities:
        return 0.0
    return np.mean(stabilities)

def create_climate_attention_set() -> AttentionSet:
    attention_set = AttentionSet()
    
    # Filter for stable concept formation
    attention_set.add_filter(AttentionFilter(
        name="concept_stability",
        conditions={
            'wave.coherence': lambda x: x >= 0.8,
            'field.stability': lambda x: x >= 0.7,
            'flow.turbulence': lambda x: x <= 0.3
        }
    ))
    
    # Filter for relationship formation
    attention_set.add_filter(AttentionFilter(
        name="relationship_flow",
        conditions={
            'wave.phase': lambda x: 0.2 <= x <= 0.8,  # Looking for moderate phase relationships
            'field.gradient_magnitude': lambda x: x >= 0.5,  # Strong gradients
            'flow.viscosity': lambda x: 0.3 <= x <= 0.7  # Goldilocks zone for flow
        }
    ))
    
    # Filter for hazard intensification with neighbor awareness
    attention_set.add_filter(AttentionFilter(
        name="hazard_intensification",
        conditions={
            'wave.potential': lambda x: x >= 0.7,
            'field.gradient_magnitude': lambda x: x >= 0.8,
            'flow.turbulence': lambda x: x >= 0.6
        },
        weight=1.5,  # Higher weight for hazard detection
        neighbor_conditions={
            'gradient_alignment': lambda ctx: check_gradient_alignment(ctx) >= 0.7,  # Strong alignment indicates organized hazard
            'phase_coherence': lambda ctx: check_phase_coherence(ctx) >= 0.6,     # Some phase coherence but allowing for dynamics
            'field_stability': lambda ctx: check_field_stability(ctx) <= 0.4      # Low stability indicates active hazard
        }
    ))
    
    # Filter for adaptation opportunity
    attention_set.add_filter(AttentionFilter(
        name="adaptation_potential",
        conditions={
            'wave.coherence': lambda x: x >= 0.6,
            'field.stability': lambda x: x >= 0.6,
            'flow.viscosity': lambda x: x >= 0.5,
            'flow.turbulence': lambda x: x <= 0.4
        }
    ))
    
    return attention_set
