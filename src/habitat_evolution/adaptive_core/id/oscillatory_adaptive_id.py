"""
Oscillatory Adaptive ID

This module extends the AdaptiveID class with oscillatory properties, enabling
pattern recognition, coherence maintenance, and predictive capabilities through
wave-like signatures.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import uuid
from datetime import datetime
import numpy as np
from copy import deepcopy

from .adaptive_id import AdaptiveID
from ..oscillation.oscillatory_signature import HarmonicComponent

class OscillatoryAdaptiveID(AdaptiveID):
    """
    Extends AdaptiveID with oscillatory properties.
    
    This class adds wave-like properties to adaptive identities, enabling:
    1. Pattern recognition across time scales
    2. Coherence maintenance during inactivity
    3. Emergent grammar of relationships
    4. Predictive capability
    
    The oscillatory properties form a signature that characterizes the entity's
    behavior in the semantic field, allowing for recognition, resonance detection,
    and future state prediction.
    """
    
    def __init__(self, 
                base_concept: str,
                creator_id: str = None,
                weight: float = 1.0,
                confidence: float = 1.0,
                uncertainty: float = 0.0,
                fundamental_frequency: float = 0.1,
                fundamental_amplitude: float = 0.5,
                fundamental_phase: float = 0.0,
                harmonics: Optional[List[Dict[str, float]]] = None):
        """
        Initialize an oscillatory adaptive ID.
        
        Args:
            base_concept: Base concept string
            creator_id: ID of the creator
            weight: Weight of the concept
            confidence: Confidence in the concept
            uncertainty: Uncertainty about the concept
            fundamental_frequency: Base frequency of oscillation (Hz)
            fundamental_amplitude: Base amplitude of oscillation (0-1)
            fundamental_phase: Initial phase of oscillation (radians)
            harmonics: List of harmonic components as dictionaries
        """
        # Initialize base AdaptiveID
        super().__init__(
            base_concept=base_concept,
            creator_id=creator_id,
            weight=weight,
            confidence=confidence,
            uncertainty=uncertainty
        )
        
        # Initialize oscillatory properties
        self.oscillatory_properties = {
            "fundamental_frequency": fundamental_frequency,
            "fundamental_amplitude": fundamental_amplitude,
            "fundamental_phase": fundamental_phase,
            "harmonics": harmonics or [],
            "energy_level": 1.0,
            "energy_capacity": 1.0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Initialize oscillation history
        self.oscillation_history = []
        self.max_history_length = 20
        
        # Record initial state in temporal context
        self.update_temporal_context(
            "oscillatory_state_initial",
            deepcopy(self.oscillatory_properties),
            "initialization"
        )
    
    def update_oscillatory_state(self, 
                               time_delta: float = 1.0, 
                               amplitude_change: float = 0.0,
                               energy_change: float = 0.0,
                               metadata_updates: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the oscillatory state based on time passage and changes.
        
        Args:
            time_delta: Time elapsed since last update (in arbitrary time units)
            amplitude_change: Change in amplitude (-1 to 1)
            energy_change: Change in energy level (-1 to 1)
            metadata_updates: Updates to metadata
        """
        # Record current state in history before updating
        current_state = {
            "timestamp": datetime.now().isoformat(),
            "fundamental_frequency": self.oscillatory_properties["fundamental_frequency"],
            "fundamental_amplitude": self.oscillatory_properties["fundamental_amplitude"],
            "fundamental_phase": self.oscillatory_properties["fundamental_phase"],
            "energy_level": self.oscillatory_properties["energy_level"]
        }
        
        self.oscillation_history.append(current_state)
        if len(self.oscillation_history) > self.max_history_length:
            self.oscillation_history = self.oscillation_history[-self.max_history_length:]
        
        # Update phase based on frequency and time delta
        self.oscillatory_properties["fundamental_phase"] = (
            self.oscillatory_properties["fundamental_phase"] + 
            (self.oscillatory_properties["fundamental_frequency"] * time_delta)
        ) % (2 * np.pi)
        
        # Update harmonics
        for harmonic in self.oscillatory_properties["harmonics"]:
            harmonic["phase"] = (
                harmonic["phase"] + (harmonic["frequency"] * time_delta)
            ) % (2 * np.pi)
        
        # Update amplitude with constraints
        self.oscillatory_properties["fundamental_amplitude"] = max(
            0.0, 
            min(1.0, self.oscillatory_properties["fundamental_amplitude"] + amplitude_change)
        )
        
        # Update energy with constraints
        self.oscillatory_properties["energy_level"] = max(
            0.0, 
            min(self.oscillatory_properties["energy_capacity"], 
                self.oscillatory_properties["energy_level"] + energy_change)
        )
        
        # Update metadata
        if metadata_updates:
            self.oscillatory_properties.update(metadata_updates)
        
        # Update timestamp
        self.oscillatory_properties["last_updated"] = datetime.now().isoformat()
        
        # Record update in temporal context
        self.update_temporal_context(
            f"oscillatory_state_{datetime.now().isoformat()}",
            {
                "fundamental_frequency": self.oscillatory_properties["fundamental_frequency"],
                "fundamental_amplitude": self.oscillatory_properties["fundamental_amplitude"],
                "fundamental_phase": self.oscillatory_properties["fundamental_phase"],
                "energy_level": self.oscillatory_properties["energy_level"],
                "time_delta": time_delta,
                "amplitude_change": amplitude_change,
                "energy_change": energy_change
            },
            "oscillatory_update"
        )
    
    def add_harmonic(self, frequency: float, amplitude: float, phase: float = 0.0) -> None:
        """
        Add a new harmonic component to the oscillatory properties.
        
        Args:
            frequency: Frequency of the harmonic (Hz)
            amplitude: Amplitude of the harmonic (0-1)
            phase: Phase of the harmonic (radians)
        """
        harmonic = {
            "frequency": frequency,
            "amplitude": amplitude,
            "phase": phase
        }
        
        self.oscillatory_properties["harmonics"].append(harmonic)
        
        # Record in temporal context
        self.update_temporal_context(
            f"harmonic_added_{datetime.now().isoformat()}",
            harmonic,
            "harmonic_addition"
        )
    
    def get_current_value(self) -> float:
        """
        Get the current instantaneous value of the oscillation.
        
        Returns:
            The current value as a sum of the fundamental and all harmonics
        """
        # Calculate fundamental component
        fundamental = self.oscillatory_properties["fundamental_amplitude"] * np.sin(
            self.oscillatory_properties["fundamental_phase"]
        )
        
        # Add all harmonic components
        harmonic_sum = sum(
            h["amplitude"] * np.sin(h["phase"]) 
            for h in self.oscillatory_properties["harmonics"]
        )
        
        # Combine and normalize to -1 to 1 range
        total = fundamental + harmonic_sum
        max_possible = self.oscillatory_properties["fundamental_amplitude"] + sum(
            h["amplitude"] for h in self.oscillatory_properties["harmonics"]
        )
        
        # Avoid division by zero
        if max_possible > 0:
            return total / max_possible
        return 0.0
    
    def project_future_state(self, time_delta: float) -> Dict[str, Any]:
        """
        Project the oscillatory state into the future.
        
        Args:
            time_delta: Time to project forward
            
        Returns:
            Dictionary with projected oscillatory properties
        """
        # Copy current properties
        projected = deepcopy(self.oscillatory_properties)
        
        # Project fundamental phase
        projected["fundamental_phase"] = (
            projected["fundamental_phase"] + 
            (projected["fundamental_frequency"] * time_delta)
        ) % (2 * np.pi)
        
        # Project harmonics
        for harmonic in projected["harmonics"]:
            harmonic["phase"] = (
                harmonic["phase"] + (harmonic["frequency"] * time_delta)
            ) % (2 * np.pi)
        
        # Add projection metadata
        projected["projected"] = True
        projected["projection_time_delta"] = time_delta
        projected["projection_timestamp"] = datetime.now().isoformat()
        
        return projected
    
    def calculate_resonance(self, other: 'OscillatoryAdaptiveID') -> float:
        """
        Calculate resonance between this entity and another.
        
        Resonance is highest when:
        1. Frequencies are in harmonic ratios (1:1, 1:2, 2:3, etc.)
        2. Phases are aligned
        3. Amplitudes are similar
        
        Args:
            other: Another oscillatory adaptive ID
            
        Returns:
            Resonance value between 0 (no resonance) and 1 (perfect resonance)
        """
        # Calculate frequency resonance using harmonic ratios
        freq_ratio = min(
            self.oscillatory_properties["fundamental_frequency"], 
            other.oscillatory_properties["fundamental_frequency"]
        ) / max(
            self.oscillatory_properties["fundamental_frequency"], 
            other.oscillatory_properties["fundamental_frequency"]
        )
        
        if freq_ratio < 0.01:  # Avoid division by zero or very small numbers
            freq_ratio = 0.01
            
        # Check if ratio is close to a simple fraction (1/1, 1/2, 2/3, etc.)
        harmonic_ratios = [1.0, 0.5, 0.667, 0.75, 0.8, 0.833]
        freq_resonance = max(1.0 - min(abs(freq_ratio - ratio) for ratio in harmonic_ratios), 0.0)
        
        # Calculate phase alignment (normalized to 0-1)
        phase_diff = abs(
            ((self.oscillatory_properties["fundamental_phase"] - 
              other.oscillatory_properties["fundamental_phase"]) + np.pi) % (2 * np.pi) - np.pi
        ) / np.pi
        
        phase_resonance = 1.0 - phase_diff
        
        # Calculate amplitude similarity (normalized to 0-1)
        amp_diff = abs(
            self.oscillatory_properties["fundamental_amplitude"] - 
            other.oscillatory_properties["fundamental_amplitude"]
        )
        
        amp_resonance = 1.0 - amp_diff
        
        # Combine the three factors (with frequency being most important)
        resonance = (0.5 * freq_resonance) + (0.3 * phase_resonance) + (0.2 * amp_resonance)
        
        # Record resonance in temporal context
        self.update_temporal_context(
            f"resonance_with_{other.id}_{datetime.now().isoformat()}",
            {
                "target_id": other.id,
                "resonance_value": resonance,
                "freq_resonance": freq_resonance,
                "phase_resonance": phase_resonance,
                "amp_resonance": amp_resonance
            },
            "resonance_calculation"
        )
        
        return resonance
    
    def calculate_interference(self, other: 'OscillatoryAdaptiveID') -> Tuple[float, str]:
        """
        Calculate interference pattern between this entity and another.
        
        Args:
            other: Another oscillatory adaptive ID
            
        Returns:
            Tuple of (interference_value, interference_type)
            where interference_value is between -1 (destructive) and 1 (constructive)
            and interference_type is "constructive", "destructive", or "neutral"
        """
        # Calculate phase difference
        phase_diff = abs(
            self.oscillatory_properties["fundamental_phase"] - 
            other.oscillatory_properties["fundamental_phase"]
        ) % (2 * np.pi)
        
        # Determine if constructive or destructive
        # Constructive when phases are aligned (diff near 0 or 2π)
        # Destructive when phases are opposite (diff near π)
        if phase_diff < np.pi/4 or phase_diff > 7*np.pi/4:
            interference_type = "constructive"
            interference_value = 1.0 - (phase_diff / (np.pi/4))
        elif phase_diff > 3*np.pi/4 and phase_diff < 5*np.pi/4:
            interference_type = "destructive"
            interference_value = -1.0 + (abs(phase_diff - np.pi) / (np.pi/4))
        else:
            interference_type = "neutral"
            # Smoothly transition between constructive and destructive
            if phase_diff <= np.pi:
                interference_value = 1.0 - (2.0 * phase_diff / np.pi)
            else:
                interference_value = -1.0 + (2.0 * (phase_diff - np.pi) / np.pi)
        
        # Record interference in temporal context
        self.update_temporal_context(
            f"interference_with_{other.id}_{datetime.now().isoformat()}",
            {
                "target_id": other.id,
                "interference_value": interference_value,
                "interference_type": interference_type,
                "phase_difference": phase_diff
            },
            "interference_calculation"
        )
        
        return interference_value, interference_type
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary with all properties
        """
        # Get base AdaptiveID properties
        base_dict = super().to_dict()
        
        # Add oscillatory properties
        base_dict["oscillatory_properties"] = deepcopy(self.oscillatory_properties)
        base_dict["oscillation_history"] = deepcopy(self.oscillation_history)
        
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OscillatoryAdaptiveID':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary with properties
            
        Returns:
            New OscillatoryAdaptiveID instance
        """
        # Create instance with base properties
        instance = cls(
            base_concept=data["base_concept"],
            creator_id=data.get("creator_id"),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            uncertainty=data.get("uncertainty", 0.0)
        )
        
        # Set ID and versions
        instance.id = data["id"]
        instance.versions = data.get("versions", [])
        
        # Set relationships
        instance.relationships = data.get("relationships", {})
        
        # Set temporal context
        instance.temporal_context = data.get("temporal_context", {})
        
        # Set oscillatory properties
        if "oscillatory_properties" in data:
            instance.oscillatory_properties = data["oscillatory_properties"]
        
        # Set oscillation history
        if "oscillation_history" in data:
            instance.oscillation_history = data["oscillation_history"]
        
        return instance
    
    @classmethod
    def from_adaptive_id(cls, adaptive_id: AdaptiveID, 
                        fundamental_frequency: float = 0.1,
                        fundamental_amplitude: float = 0.5,
                        fundamental_phase: float = 0.0,
                        harmonics: Optional[List[Dict[str, float]]] = None) -> 'OscillatoryAdaptiveID':
        """
        Create from an existing AdaptiveID.
        
        Args:
            adaptive_id: Existing AdaptiveID instance
            fundamental_frequency: Base frequency of oscillation (Hz)
            fundamental_amplitude: Base amplitude of oscillation (0-1)
            fundamental_phase: Initial phase of oscillation (radians)
            harmonics: List of harmonic components as dictionaries
            
        Returns:
            New OscillatoryAdaptiveID instance with properties from existing AdaptiveID
        """
        # Create new instance
        instance = cls(
            base_concept=adaptive_id.base_concept,
            creator_id=adaptive_id.creator_id,
            weight=adaptive_id.weight,
            confidence=adaptive_id.confidence,
            uncertainty=adaptive_id.uncertainty,
            fundamental_frequency=fundamental_frequency,
            fundamental_amplitude=fundamental_amplitude,
            fundamental_phase=fundamental_phase,
            harmonics=harmonics
        )
        
        # Copy ID and versions
        instance.id = adaptive_id.id
        instance.versions = deepcopy(adaptive_id.versions)
        
        # Copy relationships
        instance.relationships = deepcopy(adaptive_id.relationships)
        
        # Copy temporal context
        instance.temporal_context = deepcopy(adaptive_id.temporal_context)
        
        return instance
