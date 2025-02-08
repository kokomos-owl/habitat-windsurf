from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.core.interfaces.base_states import BaseProjectState

@dataclass
class DensityMetrics:
    """Metrics for pattern density and interface strength."""
    global_density: float = 0.0
    local_density: float = 0.0
    cross_domain_strength: float = 0.0
    interface_recognition: float = 0.0
    viscosity: float = 0.0
    
    def calculate_interface_strength(self) -> float:
        """Calculate interface recognition strength."""
        return (
            0.4 * self.cross_domain_strength +
            0.3 * self.viscosity +
            0.3 * self.local_density
        )

@dataclass
class UncertaintyMetrics:
    """Metrics for pattern uncertainty."""
    confidence: float = 0.0
    interface_confidence: float = 0.0
    viscosity_stability: float = 0.0
    temporal_stability: float = 1.0
    
    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence score."""
        return (
            0.4 * self.confidence +
            0.3 * self.interface_confidence +
            0.3 * self.viscosity_stability
        )

@dataclass
class PatternEvolutionMetrics:
    """Metrics for tracking pattern evolution."""
    gradient: float = 0.0
    interface_strength: float = 0.0
    stability: float = 0.0
    emergence_rate: float = 0.0
    coherence_level: float = 0.0
    
    def calculate_recognition_threshold(self) -> float:
        """Calculate pattern recognition threshold."""
        return min(1.0, (
            0.3 * self.stability +
            0.3 * self.interface_strength +
            0.4 * self.coherence_level
        ))

@dataclass
class TemporalContext:
    """Temporal context for pattern evidence."""
    start_time: datetime
    learning_window: 'LearningWindow'

@dataclass
class PatternEvidence:
    """Evidence for pattern recognition."""
    evidence_id: str
    timestamp: datetime
    pattern_type: str
    source_data: Dict[str, Any]
    temporal_context: TemporalContext
    uncertainty_metrics: Optional[UncertaintyMetrics] = None
    density_metrics: Optional[DensityMetrics] = None
    evolution_metrics: Optional[PatternEvolutionMetrics] = None
    stability_score: float = 0.0
    emergence_rate: float = 0.0
    
    def calculate_density_score(self) -> float:
        """Calculate overall density score."""
        if not self.density_metrics:
            return 0.0
        return (
            0.4 * self.density_metrics.global_density +
            0.4 * self.density_metrics.local_density +
            0.2 * self.density_metrics.cross_domain_strength
        )

@dataclass
class LearningWindow:
    """Container for managing pattern evolution."""
    window_id: str
    start_time: datetime
    patterns: List[str]
    density_metrics: DensityMetrics
    coherence_level: float
    viscosity_gradient: float
