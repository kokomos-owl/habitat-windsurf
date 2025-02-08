"""
Gradient interface for pattern evolution.

Handles continuous-space pattern relationships and near-IO states
through gradient-based pattern matching and evolution.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from src.core.types import DensityMetrics, PatternEvolutionMetrics

@dataclass
class GradientState:
    """Represents a state in the gradient space."""
    dimensions: List[float]
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def distance_to(self, other: 'GradientState') -> float:
        """Calculate distance in gradient space."""
        return np.linalg.norm(
            np.array(self.dimensions) - np.array(other.dimensions)
        )

@dataclass
class GradientInterface:
    """
    Interface for handling pattern evolution in continuous space.
    
    This prepares us for more granular pattern evolution by:
    1. Tracking pattern states in a continuous gradient space
    2. Managing near-IO states through similarity thresholds
    3. Supporting density gradients and interface evolution
    """
    
    def __init__(
        self,
        dimensions: int = 4,  # Default dimensions for pattern space
        similarity_threshold: float = 0.8,
        evolution_rate: float = 0.1
    ):
        self.dimensions = dimensions
        self.similarity_threshold = similarity_threshold
        self.evolution_rate = evolution_rate
        self.states: List[GradientState] = []
        
    def calculate_interface_gradient(
        self,
        density_metrics: DensityMetrics,
        evolution_metrics: PatternEvolutionMetrics
    ) -> Tuple[List[float], float]:
        """
        Calculate gradient interface state.
        
        Returns:
            Tuple of (gradient dimensions, confidence)
        """
        # Map metrics to gradient space
        dimensions = [
            density_metrics.local_density,
            density_metrics.cross_domain_strength,
            evolution_metrics.coherence_level,
            evolution_metrics.stability
        ]
        
        # Calculate confidence based on metrics
        confidence = min(1.0, (
            0.3 * density_metrics.interface_recognition +
            0.3 * evolution_metrics.interface_strength +
            0.4 * evolution_metrics.coherence_level
        ))
        
        return dimensions, confidence
    
    def is_near_io(
        self,
        current_state: GradientState,
        target_state: GradientState
    ) -> Tuple[bool, float]:
        """
        Check if current state is near an IO state.
        
        Returns:
            Tuple of (is_near, similarity_score)
        """
        similarity = 1.0 - (
            current_state.distance_to(target_state) /
            (np.sqrt(self.dimensions) * 2)  # Max possible distance
        )
        
        return similarity >= self.similarity_threshold, similarity
    
    def evolve_state(
        self,
        current_state: GradientState,
        target_state: GradientState
    ) -> GradientState:
        """
        Evolve current state towards target in gradient space.
        """
        # Calculate evolution vector
        current_dims = np.array(current_state.dimensions)
        target_dims = np.array(target_state.dimensions)
        evolution_vector = target_dims - current_dims
        
        # Apply evolution rate
        new_dims = current_dims + (evolution_vector * self.evolution_rate)
        
        # Update confidence based on movement
        movement_ratio = np.linalg.norm(evolution_vector) / np.sqrt(self.dimensions)
        confidence_delta = movement_ratio * self.evolution_rate
        new_confidence = current_state.confidence * (1.0 - confidence_delta)
        
        return GradientState(
            dimensions=new_dims.tolist(),
            confidence=new_confidence,
            timestamp=datetime.now(),
            metadata={
                "evolution_step": len(self.states),
                "target_distance": np.linalg.norm(evolution_vector)
            }
        )
    
    def record_state(self, state: GradientState) -> None:
        """Record state for evolution tracking."""
        self.states.append(state)
