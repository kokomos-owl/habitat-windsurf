"""
Pattern metrics and measurement types.
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PatternMetrics:
    """Metrics for pattern evaluation."""
    coherence: float
    emergence_rate: float
    cross_pattern_flow: float
    energy_state: float
    adaptation_rate: float
    stability: float

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "coherence": self.coherence,
            "emergence_rate": self.emergence_rate,
            "cross_pattern_flow": self.cross_pattern_flow,
            "energy_state": self.energy_state,
            "adaptation_rate": self.adaptation_rate,
            "stability": self.stability
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PatternMetrics':
        """Create metrics from dictionary."""
        return cls(
            coherence=data.get("coherence", 0.0),
            emergence_rate=data.get("emergence_rate", 0.0),
            cross_pattern_flow=data.get("cross_pattern_flow", 0.0),
            energy_state=data.get("energy_state", 0.0),
            adaptation_rate=data.get("adaptation_rate", 0.0),
            stability=data.get("stability", 0.0)
        )
