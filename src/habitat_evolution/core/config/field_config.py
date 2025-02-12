"""Field configuration and analysis modes.

This module defines configuration options for field behavior and analysis modes
that determine which aspects of pattern behavior are active during testing
and execution.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Any

class AnalysisMode(Enum):
    """Defines different modes of pattern analysis"""
    COHERENCE = auto()      # Basic coherence analysis
    WAVE = auto()          # Wave mechanics analysis
    FLOW = auto()          # Flow dynamics analysis
    QUANTUM = auto()       # Quantum analog analysis
    INFORMATION = auto()   # Information theory analysis
    ALL = auto()           # All parameters active

@dataclass
class FieldConfig:
    """Configuration for field behavior.
    
    Parameters are organized by their analytical domain and can be
    selectively activated based on the type of analysis being performed.
    
    Usage:
        config = FieldConfig(active_modes=[AnalysisMode.COHERENCE])
        if config.is_mode_active(AnalysisMode.COHERENCE):
            # Use coherence parameters
    """
    field_size: int = 10
    active_modes: List[AnalysisMode] = None
    boundary_condition: str = 'periodic'
    
    # Core thresholds - minimal set to avoid bias
    coherence_threshold: float = 0.6  # Minimum for stable patterns
    noise_threshold: float = 0.3    # Maximum for incoherent patterns
    
    # Wave parameters
    propagation_speed: float = 1.0
    wavelength: float = 2.0
    group_velocity: float = 0.5
    phase_velocity: float = 1.0
    phase_resolution: float = 0.1
    
    # Field parameters
    coherence_length: float = 2.0
    correlation_time: float = 1.0
    noise_threshold: float = 0.3
    energy_tolerance: float = 0.1
    information_tolerance: float = 0.1
    
    def __post_init__(self):
        if self.active_modes is None:
            self.active_modes = [AnalysisMode.COHERENCE]
    
    def is_mode_active(self, mode: AnalysisMode) -> bool:
        """Check if a specific analysis mode is active."""
        return mode in self.active_modes or AnalysisMode.ALL in self.active_modes
    
    def get_active_parameters(self) -> Dict[str, Any]:
        """Get parameters relevant to active analysis modes."""
        params = {}
        
        # Always include core parameters
        params.update({
            'field_size': self.field_size,
            'boundary_condition': self.boundary_condition,
            'noise_threshold': self.noise_threshold
        })
        
        # Add mode-specific parameters
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
                'correlation_time': self.correlation_time
            })

        if self.is_mode_active(AnalysisMode.INFORMATION):
            params.update({
                'energy_tolerance': self.energy_tolerance,
                'information_tolerance': self.information_tolerance
            })
            
        return params
