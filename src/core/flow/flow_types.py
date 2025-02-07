"""Core flow types and states for habitat-windsurf.

This module defines the fundamental types and states used in flow management,
adapted from the original habitat_poc implementation but streamlined for the POC.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

@dataclass
class FlowDynamics:
    """Essential dynamics of pattern flow."""
    velocity: float = 0.0    # Rate of pattern change
    direction: float = 0.0   # -1.0 (diverging) to 1.0 (converging)
    energy: float = 0.0      # Pattern formation energy
    propensity: float = 0.0  # Tendency toward certain states
    
    @property
    def is_active(self) -> bool:
        """Check if flow has significant movement and propensity."""
        return (abs(self.velocity) > 0.1 and 
                self.energy > 0.2 and 
                self.propensity > 0.3)
    
    @property
    def emergence_readiness(self) -> float:
        """Measure readiness for emergence based on current conditions."""
        base_readiness = (self.energy * 0.4 + 
                         abs(self.velocity) * 0.3 +
                         self.propensity * 0.3)
        return base_readiness * (1.0 + (0.5 * self.direction if self.direction > 0 else 0))

@dataclass
class FlowState:
    """Current state of flow with dynamics awareness."""
    dynamics: FlowDynamics = field(default_factory=FlowDynamics)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def is_valid(self) -> bool:
        """Validate flow state meets minimum thresholds."""
        return self.dynamics.is_active

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'dynamics': {
                'velocity': self.dynamics.velocity,
                'direction': self.dynamics.direction,
                'energy': self.dynamics.energy,
                'propensity': self.dynamics.propensity
            },
            'temporal_context': self.temporal_context,
            'last_updated': self.last_updated.isoformat()
        }

class FlowType(Enum):
    """Types of flows in the system."""
    STRUCTURAL = "structural"  # Structure-based flow
    SEMANTIC = "semantic"      # Meaning-based flow
    TEMPORAL = "temporal"      # Time-based flow
    EMERGENT = "emergent"      # Newly emerging flow

@dataclass
class FlowMetrics:
    """Metrics tracking flow characteristics."""
    coherence: float = 0.0       # Overall coherence
    stability: float = 0.0       # Flow stability
    emergence_rate: float = 0.0  # Rate of pattern emergence
    cross_flow: float = 0.0      # Cross-pattern flow strength

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'coherence': self.coherence,
            'stability': self.stability,
            'emergence_rate': self.emergence_rate,
            'cross_flow': self.cross_flow
        }

@dataclass
class ProcessingContext:
    """Context for flow processing."""
    flow_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    flow_type: FlowType = FlowType.STRUCTURAL
    metrics: FlowMetrics = field(default_factory=FlowMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
