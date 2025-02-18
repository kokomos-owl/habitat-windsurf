"""Flow dynamics and state management for pattern coherence."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class FlowState(Enum):
    """States for coherence flow."""
    STABLE = "STABLE"
    ADAPTING = "ADAPTING"
    EMERGING = "EMERGING"
    TRANSITIONING = "TRANSITIONING"

@dataclass
class FlowDynamics:
    """Dynamics for coherence flow."""
    stability: float = 0.8
    adaptation_rate: float = 0.6
    emergence_potential: float = 0.7
    transition_threshold: float = 0.5
    
    def calculate_flow_metrics(self) -> Dict[str, float]:
        """Calculate flow metrics."""
        return {
            "stability": self.stability,
            "adaptation_rate": self.adaptation_rate,
            "emergence_potential": self.emergence_potential,
            "transition_threshold": self.transition_threshold,
            "flow_score": (
                self.stability * 0.4 +
                self.adaptation_rate * 0.3 +
                self.emergence_potential * 0.3
            )
        }
    
    def determine_state(self) -> FlowState:
        """Determine current flow state."""
        metrics = self.calculate_flow_metrics()
        flow_score = metrics["flow_score"]
        
        if flow_score >= 0.8:
            return FlowState.STABLE
        elif flow_score >= 0.6:
            return FlowState.ADAPTING
        elif flow_score >= 0.4:
            return FlowState.EMERGING
        else:
            return FlowState.TRANSITIONING
