"""Domain-aware pattern evolution tracking."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.core.interfaces.gradient_interface import (
    GradientInterface,
    GradientState
)
from src.core.analysis.pattern_emergence import (
    PatternEmergenceTracker,
    EmergentPattern
)
from src.core.types import (
    DensityMetrics,
    PatternEvolutionMetrics,
    PatternEvidence,
    TemporalContext
)

class DomainPatternTracker:
    """
    Tracks pattern evolution with domain-specific gradient awareness.
    
    This class bridges the PatternEmergenceTracker with the GradientInterface,
    allowing patterns to evolve naturally while respecting domain-specific
    constraints and relationships.
    """
    
    def __init__(
        self,
        domain_name: str,
        dimensions: int = 4,
        similarity_threshold: float = 0.85,
        evolution_rate: float = 0.05
    ):
        self.domain_name = domain_name
        self.pattern_tracker = PatternEmergenceTracker()
        self.gradient_interface = GradientInterface(
            dimensions=dimensions,
            similarity_threshold=similarity_threshold,
            evolution_rate=evolution_rate
        )
        self.domain_states: Dict[str, List[GradientState]] = {}
        self.pattern_to_state: Dict[str, GradientState] = {}
    
    async def process_evidence(
        self,
        evidence: PatternEvidence,
        context: TemporalContext
    ) -> Tuple[List[EmergentPattern], List[GradientState]]:
        """
        Process new pattern evidence with domain awareness.
        
        Args:
            evidence: New pattern evidence to process
            context: Temporal context for evidence
            
        Returns:
            Tuple of (emergent patterns, gradient states)
        """
        # First, let pattern tracker process evidence
        patterns = await self.pattern_tracker.observe_elements(
            [evidence],
            context
        )
        
        # Calculate gradient state for evidence
        gradient_state = self._evidence_to_gradient_state(evidence)
        
        # Track domain state
        pattern_type = evidence.pattern_type
        if pattern_type not in self.domain_states:
            self.domain_states[pattern_type] = []
        self.domain_states[pattern_type].append(gradient_state)
        
        # Evolve existing patterns in gradient space
        evolved_states = []
        for pattern in patterns["evolved_patterns"]:
            if pattern.pattern_id in self.pattern_to_state:
                current_state = self.pattern_to_state[pattern.pattern_id]
                evolved = self.gradient_interface.evolve_state(
                    current_state,
                    gradient_state
                )
                self.pattern_to_state[pattern.pattern_id] = evolved
                evolved_states.append(evolved)
        
        # Record new patterns
        for pattern in patterns["new_patterns"]:
            self.pattern_to_state[pattern.pattern_id] = gradient_state
        
        return patterns["all_patterns"], evolved_states
    
    def _evidence_to_gradient_state(
        self,
        evidence: PatternEvidence
    ) -> GradientState:
        """Convert pattern evidence to gradient state."""
        if not evidence.density_metrics or not evidence.evolution_metrics:
            raise ValueError("Evidence must have density and evolution metrics")
            
        dims = self._calculate_dimensions(
            evidence.density_metrics,
            evidence.evolution_metrics
        )
        
        return GradientState(
            dimensions=dims,
            confidence=evidence.stability_score,
            timestamp=evidence.timestamp,
            metadata={
                "pattern_type": evidence.pattern_type,
                "source": evidence.source_data
            }
        )
    
    def _calculate_dimensions(
        self,
        density: DensityMetrics,
        evolution: PatternEvolutionMetrics
    ) -> List[float]:
        """Calculate gradient dimensions from metrics."""
        return [
            density.local_density,
            density.cross_domain_strength,
            evolution.coherence_level,
            evolution.stability
        ]
    
    def get_domain_patterns(
        self,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.7
    ) -> List[Tuple[EmergentPattern, GradientState]]:
        """Get patterns with their gradient states."""
        patterns = []
        
        for pattern_id, state in self.pattern_to_state.items():
            if state.confidence < min_confidence:
                continue
                
            pattern = self.pattern_tracker.emergent_patterns.get(pattern_id)
            if not pattern:
                continue
                
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
                
            patterns.append((pattern, state))
        
        return patterns
    
    def analyze_pattern_relationships(
        self,
        pattern_type: str
    ) -> Dict[str, Any]:
        """Analyze pattern relationships in gradient space."""
        if pattern_type not in self.domain_states:
            return {}
            
        states = self.domain_states[pattern_type]
        if len(states) < 2:
            return {}
            
        # Calculate stability metrics
        transitions = [
            states[i].distance_to(states[i-1])
            for i in range(1, len(states))
        ]
        
        # Analyze pattern evolution
        confidences = [s.confidence for s in states]
        
        return {
            "mean_transition": float(np.mean(transitions)),
            "std_transition": float(np.std(transitions)),
            "mean_confidence": float(np.mean(confidences)),
            "stability_score": float(1.0 - np.std(transitions)),
            "evolution_rate": float(np.mean(transitions) / np.mean(confidences)),
            "state_count": len(states)
        }
    
    def find_related_patterns(
        self,
        pattern_id: str,
        min_similarity: float = 0.8
    ) -> List[Tuple[EmergentPattern, float]]:
        """Find patterns related in gradient space."""
        if pattern_id not in self.pattern_to_state:
            return []
            
        source_state = self.pattern_to_state[pattern_id]
        related = []
        
        for other_id, other_state in self.pattern_to_state.items():
            if other_id == pattern_id:
                continue
                
            _, similarity = self.gradient_interface.is_near_io(
                source_state,
                other_state
            )
            
            if similarity >= min_similarity:
                pattern = self.pattern_tracker.emergent_patterns.get(other_id)
                if pattern:
                    related.append((pattern, similarity))
        
        return sorted(
            related,
            key=lambda x: x[1],
            reverse=True
        )
