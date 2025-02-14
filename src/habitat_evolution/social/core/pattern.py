"""Social pattern detection and evolution tracking."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

from .field import FieldState, SocialField

@dataclass
class PatternMetrics:
    """Metrics for social pattern analysis."""
    # Field dynamics
    field_energy: float
    field_coherence: float
    field_flow: float
    
    # Social dynamics
    adoption_rate: float
    influence_reach: float
    stability_index: float
    
    # Practice formation
    practice_maturity: float
    institutionalization: float

@dataclass
class PatternConfig:
    """Configuration for social pattern detection."""
    # Detection thresholds
    min_coherence: float = 0.3
    min_energy: float = 0.2
    min_flow: float = 0.1
    
    # Evolution parameters
    maturity_rate: float = 0.1
    influence_decay: float = 0.2
    adoption_momentum: float = 0.3

class SocialPattern:
    """Represents and manages a social pattern."""
    
    def __init__(self, 
                 pattern_id: str,
                 config: PatternConfig,
                 initial_field: SocialField):
        self.id = pattern_id
        self.config = config
        self.creation_time = datetime.now()
        self.field = initial_field
        self.metrics = PatternMetrics(
            field_energy=initial_field.state.energy,
            field_coherence=initial_field.state.coherence,
            field_flow=initial_field.state.flow,
            adoption_rate=0.0,
            influence_reach=0.0,
            stability_index=0.0,
            practice_maturity=0.0,
            institutionalization=0.0
        )
        self.relationships: Dict[str, float] = {}
    
    def update(self, dt: float, related_patterns: Dict[str, 'SocialPattern']):
        """Update pattern state and metrics."""
        # Update field state
        self.field.update(dt, {p.id: p.field for p in related_patterns.values()})
        
        # Update metrics
        self._update_field_metrics()
        self._update_social_metrics(related_patterns)
        self._update_practice_metrics(dt)
        
        # Update relationships
        self._update_relationships(related_patterns)
    
    def _update_field_metrics(self):
        """Update field-based metrics."""
        self.metrics.field_energy = self.field.state.energy
        self.metrics.field_coherence = self.field.state.coherence
        self.metrics.field_flow = self.field.state.flow
    
    def _update_social_metrics(self, related_patterns: Dict[str, 'SocialPattern']):
        """Update social interaction metrics."""
        if not related_patterns:
            return
            
        # Calculate adoption rate from field flow and relationships
        adoption_influences = [
            p.metrics.field_flow * self.relationships.get(pid, 0.0)
            for pid, p in related_patterns.items()
        ]
        self.metrics.adoption_rate = np.mean(adoption_influences) if adoption_influences else 0.0
        
        # Calculate influence reach
        self.metrics.influence_reach = sum(
            strength for strength in self.relationships.values()
        ) / len(related_patterns)
        
        # Update stability based on field and relationships
        self.metrics.stability_index = (
            self.field.state.stability * 0.6 +
            self.metrics.influence_reach * 0.4
        )
    
    def _update_practice_metrics(self, dt: float):
        """Update practice formation metrics."""
        # Practice maturity grows with stability and adoption
        maturity_delta = (
            self.metrics.stability_index * 0.7 +
            self.metrics.adoption_rate * 0.3
        ) * self.config.maturity_rate * dt
        
        self.metrics.practice_maturity = min(
            1.0, self.metrics.practice_maturity + maturity_delta
        )
        
        # Institutionalization follows maturity with momentum
        if self.metrics.practice_maturity > 0.5:
            inst_delta = (
                self.metrics.practice_maturity * 0.8 +
                self.metrics.influence_reach * 0.2
            ) * self.config.adoption_momentum * dt
            
            self.metrics.institutionalization = min(
                1.0, self.metrics.institutionalization + inst_delta
            )
    
    def _update_relationships(self, related_patterns: Dict[str, 'SocialPattern']):
        """Update relationship strengths with other patterns."""
        for pid, pattern in related_patterns.items():
            # Calculate relationship strength based on:
            # - Field coherence similarity
            # - Practice maturity alignment
            # - Spatial/temporal proximity (TODO)
            coherence_sim = 1 - abs(
                self.metrics.field_coherence - pattern.metrics.field_coherence
            )
            practice_sim = 1 - abs(
                self.metrics.practice_maturity - pattern.metrics.practice_maturity
            )
            
            strength = (coherence_sim * 0.6 + practice_sim * 0.4)
            
            # Apply influence decay
            current_strength = self.relationships.get(pid, 0.0)
            self.relationships[pid] = current_strength * (1 - self.config.influence_decay) + strength * self.config.influence_decay
