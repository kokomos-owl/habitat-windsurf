"""
Pattern transformation and adaptation.
"""
from typing import Dict, Any, List
import numpy as np
from dataclasses import dataclass

from ...core.pattern.types import Pattern, FieldState
from ...core.pattern.metrics import PatternMetrics

@dataclass
class TransformConfig:
    """Configuration for pattern transformation."""
    coherence_boost: float = 0.2
    energy_decay: float = 0.1
    merge_threshold: float = 0.8

class PatternTransformer:
    """Transforms patterns between representations."""
    
    def __init__(self, config: TransformConfig = None):
        self.config = config or TransformConfig()
    
    def merge_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Merge similar patterns.
        
        Args:
            patterns: Patterns to merge
            
        Returns:
            Merged patterns
        """
        if not patterns:
            return []
            
        merged = []
        used = set()
        
        for i, p1 in enumerate(patterns):
            if i in used:
                continue
                
            # Find similar patterns
            similar = []
            for j, p2 in enumerate(patterns):
                if j not in used and self._similarity(p1, p2) >= self.config.merge_threshold:
                    similar.append(p2)
                    used.add(j)
            
            # Merge similar patterns
            if similar:
                merged_pattern = self._combine_patterns(similar)
                merged.append(merged_pattern)
            else:
                merged.append(p1)
                
        return merged
    
    def adapt_to_field(self, pattern: Pattern,
                      field_state: FieldState) -> Pattern:
        """Adapt pattern to field conditions.
        
        Args:
            pattern: Pattern to adapt
            field_state: Current field state
            
        Returns:
            Adapted pattern
        """
        # Calculate field influence
        coherence_pull = (field_state.gradients.coherence - pattern['coherence'])
        energy_pull = (field_state.gradients.energy - pattern['energy'])
        
        # Update pattern properties
        new_coherence = pattern['coherence'] + (coherence_pull * self.config.coherence_boost)
        new_energy = pattern['energy'] + (energy_pull * (1.0 - self.config.energy_decay))
        
        # Ensure valid ranges
        pattern['coherence'] = max(0.0, min(1.0, new_coherence))
        pattern['energy'] = max(0.0, min(1.0, new_energy))
        
        # Update metrics
        pattern['metrics'] = PatternMetrics(
            coherence=pattern['coherence'],
            emergence_rate=0.5,  # Default for now
            cross_pattern_flow=0.0,  # Calculated elsewhere
            energy_state=pattern['energy'],
            adaptation_rate=self.config.coherence_boost,
            stability=0.5  # Default for now
        ).to_dict()
        
        return pattern
    
    def _similarity(self, p1: Pattern, p2: Pattern) -> float:
        """Calculate similarity between patterns."""
        # Compare core metrics
        coherence_diff = abs(p1['coherence'] - p2['coherence'])
        energy_diff = abs(p1['energy'] - p2['energy'])
        
        # Weighted similarity
        return 1.0 - ((coherence_diff * 0.7) + (energy_diff * 0.3))
    
    def _combine_patterns(self, patterns: List[Pattern]) -> Pattern:
        """Combine multiple patterns into one."""
        if not patterns:
            return None
            
        # Average core metrics
        coherence = np.mean([p['coherence'] for p in patterns])
        energy = np.mean([p['energy'] for p in patterns])
        
        # Combine relationships
        relationships = set()
        for p in patterns:
            relationships.update(p.get('relationships', []))
        
        # Create merged pattern
        return {
            'id': f"merged_{patterns[0]['id']}",
            'coherence': coherence,
            'energy': energy,
            'state': 'ACTIVE',
            'metrics': PatternMetrics(
                coherence=coherence,
                emergence_rate=0.5,
                cross_pattern_flow=0.0,
                energy_state=energy,
                adaptation_rate=0.5,
                stability=0.5
            ).to_dict(),
            'relationships': list(relationships)
        }
