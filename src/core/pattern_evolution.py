"""Pattern evolution and emergence tracking system.

This module facilitates the natural emergence and evolution of patterns
within the system. Rather than enforcing strict pattern definitions,
it allows patterns to form, evolve, and stabilize naturally while
maintaining coherence through dynamic relationships.

Key Concepts:
    - Natural Emergence: Patterns form organically from system dynamics
    - Evolution: Patterns adapt and change based on interactions
    - Dynamic Relationships: Connections between patterns shift naturally
    - Adaptive Confidence: Trust in patterns grows with stability
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class PatternState:
    """Dynamic state of an evolving pattern.
    
    Represents a pattern's current evolutionary state, including its
    relationships, energy, and stability characteristics. The state
    adapts naturally as the pattern interacts with the system.
    """
    pattern: str
    confidence: float
    temporal_context: Dict[str, any]
    related_patterns: Set[str] = field(default_factory=set)
    flow_velocity: float = 0.0
    flow_direction: float = 0.0
    energy: float = 0.0  # Pattern formation energy
    stability: float = 0.0  # Pattern stability measure
    emergence_phase: float = 0.0  # 0.0 (forming) to 1.0 (stable)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def is_emerging(self) -> bool:
        """Check if pattern is in emergence phase."""
        return self.emergence_phase < 0.3
    
    def is_stable(self) -> bool:
        """Check if pattern has stabilized."""
        return self.emergence_phase > 0.7 and self.stability > 0.7

@dataclass
class EvolutionMetrics:
    """Metrics tracking pattern evolution and emergence.
    
    These metrics capture both the current state and the evolutionary
    trajectory of patterns, helping understand their natural development.
    """
    coherence: float = 0.0       # Pattern relationship strength
    stability: float = 0.0       # Pattern stability level
    emergence_rate: float = 0.0   # Rate of new pattern formation
    cross_pattern_flow: float = 0.0  # Inter-pattern energy flow
    energy_state: float = 0.0    # Current pattern energy level
    adaptation_rate: float = 0.0  # How quickly patterns adapt

class PatternEvolutionTracker:
    """Tracks and analyzes pattern evolution in climate risk documents."""
    
    def __init__(self, state_file: Optional[str] = None):
        self.patterns: Dict[str, PatternState] = {}
        self.state_file = state_file or "pattern_evolution.json"
        self.load_state()
        
    def load_state(self):
        """Load pattern evolution state from file."""
        try:
            path = Path(self.state_file)
            if path.exists():
                with path.open() as f:
                    data = json.load(f)
                    for p_data in data.get('patterns', []):
                        pattern = PatternState(
                            pattern=p_data['pattern'],
                            confidence=p_data['confidence'],
                            temporal_context=p_data['temporal_context'],
                            related_patterns=set(p_data['related_patterns']),
                            flow_velocity=p_data['flow_velocity'],
                            flow_direction=p_data['flow_direction'],
                            last_updated=datetime.fromisoformat(p_data['last_updated'])
                        )
                        self.patterns[pattern.pattern] = pattern
        except Exception as e:
            logger.warning(f"Could not load pattern state: {e}")
    
    def save_state(self):
        """Save pattern evolution state to file."""
        try:
            data = {
                'patterns': [{
                    'pattern': p.pattern,
                    'confidence': p.confidence,
                    'temporal_context': p.temporal_context,
                    'related_patterns': list(p.related_patterns),
                    'flow_velocity': p.flow_velocity,
                    'flow_direction': p.flow_direction,
                    'last_updated': p.last_updated.isoformat()
                } for p in self.patterns.values()],
                'last_updated': datetime.utcnow().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save pattern state: {e}")
    
    def calculate_similarity(self, pattern1: str, pattern2: str) -> Tuple[float, float]:
        """Calculate semantic similarity and energy between patterns.
        
        Returns both similarity score and energy transfer potential.
        Energy transfer indicates how likely patterns are to influence
        each other's evolution.
        """
        # Calculate base similarity
        words1 = set(pattern1.lower().split())
        words2 = set(pattern2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0.0
        
        # Calculate energy transfer potential
        # Higher when patterns share some but not all characteristics
        energy_potential = similarity * (1 - similarity) * 4  # Peaks at 0.5 similarity
        
        return similarity, energy_potential
    
    def update_flow_dynamics(self, pattern: str,
                           related_patterns: List[str],
                           similarities: List[Tuple[float, float]]):
        """Update flow dynamics based on natural pattern interactions.
        
        Allows patterns to naturally influence each other's evolution
        through energy exchange and dynamic relationships.
        """
        if not similarities:
            return
            
        state = self.patterns.get(pattern)
        if not state:
            return
            
        # Unpack similarity scores and energy potentials
        scores, energies = zip(*similarities)
        
        # Update velocity based on relationship strengths
        state.flow_velocity = sum(scores) / len(scores)
        
        # Calculate energy exchange
        total_energy = sum(energies)
        state.energy = 0.7 * state.energy + 0.3 * total_energy
        
        # Update stability based on relationship consistency
        state.stability = min(1.0, state.stability + 0.1) if state.flow_velocity > 0.5 else max(0.0, state.stability - 0.1)
        
        # Update emergence phase based on stability and energy
        if state.stability > 0.7 and state.energy < 0.3:
            state.emergence_phase = min(1.0, state.emergence_phase + 0.1)  # Pattern stabilizing
        elif state.energy > 0.7:
            state.emergence_phase = max(0.0, state.emergence_phase - 0.1)  # Pattern evolving
        
        # Update relationships based on strongest interactions
        strongest_related = sorted(zip(related_patterns, scores),
                                 key=lambda x: x[1],
                                 reverse=True)[:3]
        if strongest_related:
            related_patterns = [r[0] for r in strongest_related]
            state.flow_direction = sum(r[1] for r in strongest_related) / len(strongest_related)
            state.related_patterns.update(related_patterns)
    
    def observe_pattern(self, pattern: str,
                       confidence: float,
                       temporal_context: Dict[str, any],
                       related_patterns: Optional[List[str]] = None) -> EvolutionMetrics:
        """Observe and facilitate natural pattern evolution.
        
        This method observes pattern behavior and facilitates natural evolution
        without enforcing specific outcomes. It allows patterns to emerge,
        adapt, and stabilize based on their interactions and energy states.
        
        Args:
            pattern: The pattern being observed
            confidence: Initial confidence in the pattern
            temporal_context: Time-based context for evolution
            related_patterns: Other patterns that may influence evolution
            
        Returns:
            EvolutionMetrics capturing the current evolutionary state
        """
        # Get or create pattern state with emergence tracking
        state = self.patterns.get(pattern)
        if not state:
            state = PatternState(
                pattern=pattern,
                confidence=confidence,
                temporal_context=temporal_context,
                energy=0.5,  # Initial energy for emergence
                emergence_phase=0.0  # Start in emergence phase
            )
            self.patterns[pattern] = state
        
        # Calculate relationship dynamics
        relationship_dynamics = []
        if related_patterns:
            relationship_dynamics = [
                self.calculate_similarity(pattern, rel)
                for rel in related_patterns
            ]
            
            # Allow natural evolution through interactions
            self.update_flow_dynamics(pattern, related_patterns, relationship_dynamics)
        
        # Update pattern state naturally
        state.confidence = 0.7 * state.confidence + 0.3 * confidence  # Smooth transitions
        state.temporal_context.update(temporal_context)  # Accumulate context
        state.last_updated = datetime.utcnow()
        
        # Calculate evolution metrics
        metrics = EvolutionMetrics()
        
        if relationship_dynamics:
            # Unpack similarity scores and energy potentials
            scores, energies = zip(*relationship_dynamics)
            
            # Coherence emerges from relationship strength
            metrics.coherence = sum(scores) / len(scores)
            
            # Energy state reflects pattern formation activity
            metrics.energy_state = state.energy
            
            # Adaptation rate based on energy exchange
            metrics.adaptation_rate = sum(energies) / len(energies)
        
        # Stability emerges from consistent behavior
        metrics.stability = state.stability
        
        # Emergence rate reflects new pattern formation
        metrics.emergence_rate = (1.0 - state.emergence_phase) * state.energy
        
        # Cross-pattern flow emerges from mutual interactions
        if state.related_patterns:
            cross_flows = []
            for rel in state.related_patterns:
                if rel in self.patterns:
                    rel_state = self.patterns[rel]
                    if pattern in rel_state.related_patterns:
                        # Calculate bidirectional energy exchange
                        energy_exchange = (
                            state.energy * rel_state.energy *
                            state.flow_velocity * rel_state.flow_velocity
                        ) ** 0.5  # Geometric mean for stability
                        cross_flows.append(energy_exchange)
            
            if cross_flows:
                metrics.cross_pattern_flow = sum(cross_flows) / len(cross_flows)
        
        # Allow patterns to naturally stabilize or evolve
        self._adjust_pattern_phase(state, metrics)
        
        self.save_state()
        return metrics
        
    def _adjust_pattern_phase(self, state: PatternState, metrics: EvolutionMetrics):
        """Allow patterns to naturally transition between phases.
        
        Patterns can naturally stabilize or return to evolution based on
        their energy states and interactions.
        """
        if metrics.stability > 0.8 and metrics.energy_state < 0.3:
            # Pattern is stabilizing
            state.emergence_phase = min(1.0, state.emergence_phase + 0.1)
        elif metrics.energy_state > 0.7 or metrics.adaptation_rate > 0.6:
            # Pattern is actively evolving
            state.emergence_phase = max(0.0, state.emergence_phase - 0.1)
    
    def get_evolved_patterns(self, min_confidence: float = 0.5,
                           min_energy: float = 0.3) -> List[Dict[str, Any]]:
        """Get patterns based on their evolutionary state.
        
        Returns patterns in various stages of evolution, from emerging
        to stable, based on their energy states and confidence levels.
        
        Args:
            min_confidence: Minimum confidence threshold
            min_energy: Minimum energy threshold for evolution
            
        Returns:
            List of pattern states with evolutionary metrics
        """
        evolved_patterns = []
        for pattern, state in self.patterns.items():
            if state.confidence >= min_confidence:
                pattern_info = {
                    'pattern': pattern,
                    'confidence': state.confidence,
                    'energy': state.energy,
                    'stability': state.stability,
                    'emergence_phase': state.emergence_phase,
                    'is_emerging': state.is_emerging(),
                    'is_stable': state.is_stable(),
                    'flow_velocity': state.flow_velocity,
                    'last_updated': state.last_updated.isoformat()
                }
                
                # Include patterns that are either stable or actively evolving
                if (state.is_stable() or 
                    (state.energy >= min_energy and state.flow_velocity > 0)):
                    evolved_patterns.append(pattern_info)
                    
        # Sort by evolution phase - emerging patterns first, then by confidence
        return sorted(evolved_patterns,
                     key=lambda x: (x['is_emerging'], x['confidence']),
                     reverse=True)
    
    def get_pattern_relationships(self, pattern: str,
                                min_strength: float = 0.3) -> List[Dict[str, Any]]:
        """Get dynamic pattern relationships.
        
        Identifies relationships between patterns based on their
        evolutionary states and energy exchange potential.
        
        Args:
            pattern: Pattern to find relationships for
            min_strength: Minimum relationship strength threshold
            
        Returns:
            List of relationship information including energy dynamics
        """
        state = self.patterns.get(pattern)
        if not state:
            return []
            
        relationships = []
        for rel in state.related_patterns:
            if rel in self.patterns:
                rel_state = self.patterns[rel]
                similarity, energy = self.calculate_similarity(pattern, rel)
                
                if similarity >= min_strength:
                    relationship = {
                        'pattern': rel,
                        'similarity': similarity,
                        'energy_exchange': energy,
                        'mutual_flow': state.flow_velocity * rel_state.flow_velocity,
                        'combined_stability': (state.stability + rel_state.stability) / 2,
                        'emergence_alignment': abs(state.emergence_phase - rel_state.emergence_phase)
                    }
                    relationships.append(relationship)
                    
        # Sort by overall relationship strength (combination of similarity and energy)
        return sorted(relationships,
                     key=lambda x: x['similarity'] * (1 + x['energy_exchange']),
                     reverse=True)
