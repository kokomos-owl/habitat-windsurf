"""
Core metrics models for the Adaptive Core system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class WaveMetrics:
    """Wave mechanics metrics"""
    phase_coherence: float
    interference_pattern: Dict[str, float]
    wave_propagation: Dict[str, float]
    amplitude: float
    frequency: float
    wavelength: float

@dataclass
class FieldMetrics:
    """Field theory metrics"""
    gradient_strength: float
    field_decay: Dict[str, float]
    interaction_potential: float
    field_density: float
    field_uniformity: float
    field_stability: float

@dataclass
class InformationMetrics:
    """Information theory metrics"""
    signal_to_noise: float
    entropy: float
    information_flow: Dict[str, float]
    complexity: float
    predictability: float
    mutual_information: float

@dataclass
class FlowMetrics:
    """Flow dynamics metrics"""
    viscosity: float
    vorticity: Dict[str, float]
    flow_stability: float
    turbulence: float
    pressure_gradient: float
    flow_coherence: float

@dataclass
class PatternMetrics:
    """Comprehensive pattern metrics"""
    wave_metrics: WaveMetrics
    field_metrics: FieldMetrics
    information_metrics: InformationMetrics
    flow_metrics: FlowMetrics
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation"""
        return {
            "wave_metrics": {
                "phase_coherence": self.wave_metrics.phase_coherence,
                "interference_pattern": self.wave_metrics.interference_pattern,
                "wave_propagation": self.wave_metrics.wave_propagation,
                "amplitude": self.wave_metrics.amplitude,
                "frequency": self.wave_metrics.frequency,
                "wavelength": self.wave_metrics.wavelength
            },
            "field_metrics": {
                "gradient_strength": self.field_metrics.gradient_strength,
                "field_decay": self.field_metrics.field_decay,
                "interaction_potential": self.field_metrics.interaction_potential,
                "field_density": self.field_metrics.field_density,
                "field_uniformity": self.field_metrics.field_uniformity,
                "field_stability": self.field_metrics.field_stability
            },
            "information_metrics": {
                "signal_to_noise": self.information_metrics.signal_to_noise,
                "entropy": self.information_metrics.entropy,
                "information_flow": self.information_metrics.information_flow,
                "complexity": self.information_metrics.complexity,
                "predictability": self.information_metrics.predictability,
                "mutual_information": self.information_metrics.mutual_information
            },
            "flow_metrics": {
                "viscosity": self.flow_metrics.viscosity,
                "vorticity": self.flow_metrics.vorticity,
                "flow_stability": self.flow_metrics.flow_stability,
                "turbulence": self.flow_metrics.turbulence,
                "pressure_gradient": self.flow_metrics.pressure_gradient,
                "flow_coherence": self.flow_metrics.flow_coherence
            },
            "timestamp": self.timestamp
        }
