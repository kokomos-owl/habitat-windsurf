"""Field configuration for pattern evolution."""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

class AnalysisMode(Enum):
    """Analysis modes for field-driven pattern evolution.
    
    Modes:
        COHERENCE: Pattern coherence and relationships
        WAVE: Phase relationships and wave mechanics
        INFORMATION: Signal processing and entropy
        FLOW: Field dynamics and viscosity
        QUANTUM: Quantum-inspired correlation effects
    """
    COHERENCE = auto()
    WAVE = auto()
    INFORMATION = auto()
    FLOW = auto()
    QUANTUM = auto()

@dataclass
class FieldConfig:
    """Configuration for field-driven pattern evolution.
    
    This class defines parameters for different analysis modes:
    1. COHERENCE mode: Pattern relationships and signal quality
    2. WAVE mode: Wave mechanics and phase evolution
    3. INFORMATION mode: Signal processing and conservation laws
    4. FLOW mode: Field dynamics and viscosity effects
    5. QUANTUM mode: Correlation functions and entanglement
    """
    # Field properties
    field_size: int = 10
    active_modes: List[AnalysisMode] = None
    
    # Wave parameters
    propagation_speed: float = 1.0
    wavelength: float = 2.0
    group_velocity: float = 0.5
    phase_velocity: float = 1.0
    phase_resolution: float = 0.1
    
    # Coherence parameters
    coherence_length: float = 2.0
    correlation_time: float = 1.0
    noise_threshold: float = 0.3
    
    # Flow parameters
    viscosity: float = 0.1
    reynolds_number: float = 100.0
    turbulence_threshold: float = 0.7
    
    # Conservation parameters
    energy_tolerance: float = 0.1
    information_tolerance: float = 0.1
    
    # Boundary conditions
    boundary_condition: str = 'periodic'  # 'periodic', 'reflective', or 'absorbing'
    
    def __post_init__(self):
        """Initialize default active modes if none provided."""
        if self.active_modes is None:
            self.active_modes = [AnalysisMode.COHERENCE]
    
    def is_mode_active(self, mode: AnalysisMode) -> bool:
        """Check if a specific analysis mode is active."""
        return mode in self.active_modes
    
    def get_active_parameters(self) -> Dict[str, Any]:
        """Get parameters relevant to active analysis modes."""
        params = {
            'field_size': self.field_size,
            'boundary_condition': self.boundary_condition
        }
        
        if self.is_mode_active(AnalysisMode.WAVE):
            params.update({
                'propagation_speed': self.propagation_speed,
                'wavelength': self.wavelength,
                'group_velocity': self.group_velocity,
                'phase_velocity': self.phase_velocity,
                'phase_resolution': self.phase_resolution
            })
        
        if self.is_mode_active(AnalysisMode.COHERENCE):
            params.update({
                'coherence_length': self.coherence_length,
                'correlation_time': self.correlation_time,
                'noise_threshold': self.noise_threshold
            })
        
        if self.is_mode_active(AnalysisMode.FLOW):
            params.update({
                'viscosity': self.viscosity,
                'reynolds_number': self.reynolds_number,
                'turbulence_threshold': self.turbulence_threshold
            })
        
        if self.is_mode_active(AnalysisMode.INFORMATION):
            params.update({
                'energy_tolerance': self.energy_tolerance,
                'information_tolerance': self.information_tolerance
            })
        
        return params
