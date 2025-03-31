"""
Oscillatory Signature

This module implements oscillatory signatures for entities in the Habitat Evolution system.
Signatures capture the wave-like properties of entities and relationships, enabling
pattern recognition, coherence maintenance, and predictive capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import uuid
from datetime import datetime
import numpy as np
from collections import deque

@dataclass
class HarmonicComponent:
    """A single harmonic component of an oscillatory signature."""
    frequency: float
    amplitude: float
    phase: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "frequency": self.frequency,
            "amplitude": self.amplitude,
            "phase": self.phase
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'HarmonicComponent':
        """Create from dictionary representation."""
        return cls(
            frequency=data["frequency"],
            amplitude=data["amplitude"],
            phase=data["phase"]
        )
    
    def project(self, time_delta: float) -> 'HarmonicComponent':
        """Project this harmonic component forward in time."""
        new_phase = (self.phase + (self.frequency * time_delta)) % (2 * np.pi)
        return HarmonicComponent(
            frequency=self.frequency,
            amplitude=self.amplitude,
            phase=new_phase
        )

@dataclass
class OscillatorySignature:
    """
    Represents the oscillatory properties of an entity or relationship.
    
    The signature captures wave-like properties including:
    - Fundamental frequency and phase
    - Harmonic components (overtones)
    - Amplitude envelope
    - Energy level and capacity
    - Recent oscillation history
    
    This enables:
    1. Pattern recognition across time scales
    2. Coherence maintenance during inactivity
    3. Emergent grammar of relationships
    4. Predictive capability
    """
    id: str
    entity_id: str
    entity_type: str  # "actant", "predicate", "domain", "relationship"
    
    # Primary oscillatory properties
    fundamental_frequency: float
    fundamental_amplitude: float
    fundamental_phase: float
    
    # Harmonic components (overtones)
    harmonics: List[HarmonicComponent] = field(default_factory=list)
    
    # Energy properties
    energy_level: float = 1.0
    energy_capacity: float = 1.0
    
    # Temporal properties
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Recent oscillation history (limited to prevent excessive storage)
    oscillation_history: List[Dict[str, Any]] = field(default_factory=list)
    max_history_length: int = 20
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, entity_id: str, entity_type: str, 
               fundamental_frequency: float = 0.1,
               fundamental_amplitude: float = 0.5,
               fundamental_phase: float = 0.0,
               harmonics: Optional[List[Dict[str, float]]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> 'OscillatorySignature':
        """
        Create a new oscillatory signature.
        
        Args:
            entity_id: ID of the entity this signature belongs to
            entity_type: Type of entity ("actant", "predicate", "domain", "relationship")
            fundamental_frequency: Base frequency of oscillation (Hz)
            fundamental_amplitude: Base amplitude of oscillation (0-1)
            fundamental_phase: Initial phase of oscillation (radians)
            harmonics: List of harmonic components as dictionaries
            metadata: Additional metadata
            
        Returns:
            A new OscillatorySignature instance
        """
        signature_id = f"sig_{str(uuid.uuid4())[:8]}"
        
        # Convert harmonic dictionaries to HarmonicComponent objects
        harmonic_components = []
        if harmonics:
            for h in harmonics:
                harmonic_components.append(HarmonicComponent(
                    frequency=h["frequency"],
                    amplitude=h["amplitude"],
                    phase=h["phase"]
                ))
        
        return cls(
            id=signature_id,
            entity_id=entity_id,
            entity_type=entity_type,
            fundamental_frequency=fundamental_frequency,
            fundamental_amplitude=fundamental_amplitude,
            fundamental_phase=fundamental_phase,
            harmonics=harmonic_components,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signature to dictionary representation."""
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "fundamental_frequency": self.fundamental_frequency,
            "fundamental_amplitude": self.fundamental_amplitude,
            "fundamental_phase": self.fundamental_phase,
            "harmonics": [h.to_dict() for h in self.harmonics],
            "energy_level": self.energy_level,
            "energy_capacity": self.energy_capacity,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "oscillation_history": self.oscillation_history[-self.max_history_length:] if self.oscillation_history else [],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OscillatorySignature':
        """Create signature from dictionary representation."""
        # Convert string dates to datetime objects
        created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        last_updated = datetime.fromisoformat(data["last_updated"]) if isinstance(data["last_updated"], str) else data["last_updated"]
        
        # Convert harmonic dictionaries to HarmonicComponent objects
        harmonics = [HarmonicComponent.from_dict(h) for h in data["harmonics"]]
        
        return cls(
            id=data["id"],
            entity_id=data["entity_id"],
            entity_type=data["entity_type"],
            fundamental_frequency=data["fundamental_frequency"],
            fundamental_amplitude=data["fundamental_amplitude"],
            fundamental_phase=data["fundamental_phase"],
            harmonics=harmonics,
            energy_level=data["energy_level"],
            energy_capacity=data["energy_capacity"],
            created_at=created_at,
            last_updated=last_updated,
            oscillation_history=data["oscillation_history"],
            metadata=data["metadata"]
        )
    
    def update_state(self, time_delta: float = 1.0, 
                     amplitude_change: float = 0.0,
                     energy_change: float = 0.0,
                     metadata_updates: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the signature state based on time passage and changes.
        
        Args:
            time_delta: Time elapsed since last update (in arbitrary time units)
            amplitude_change: Change in amplitude (-1 to 1)
            energy_change: Change in energy level (-1 to 1)
            metadata_updates: Updates to metadata
        """
        # Record current state in history before updating
        current_state = {
            "timestamp": datetime.now().isoformat(),
            "fundamental_frequency": self.fundamental_frequency,
            "fundamental_amplitude": self.fundamental_amplitude,
            "fundamental_phase": self.fundamental_phase,
            "energy_level": self.energy_level
        }
        
        self.oscillation_history.append(current_state)
        if len(self.oscillation_history) > self.max_history_length:
            self.oscillation_history = self.oscillation_history[-self.max_history_length:]
        
        # Update phase based on frequency and time delta
        self.fundamental_phase = (self.fundamental_phase + (self.fundamental_frequency * time_delta)) % (2 * np.pi)
        
        # Update harmonics
        for harmonic in self.harmonics:
            harmonic.phase = (harmonic.phase + (harmonic.frequency * time_delta)) % (2 * np.pi)
        
        # Update amplitude with constraints
        self.fundamental_amplitude = max(0.0, min(1.0, self.fundamental_amplitude + amplitude_change))
        
        # Update energy with constraints
        self.energy_level = max(0.0, min(self.energy_capacity, self.energy_level + energy_change))
        
        # Update metadata
        if metadata_updates:
            self.metadata.update(metadata_updates)
        
        # Update timestamp
        self.last_updated = datetime.now()
    
    def add_harmonic(self, frequency: float, amplitude: float, phase: float = 0.0) -> None:
        """
        Add a new harmonic component to the signature.
        
        Args:
            frequency: Frequency of the harmonic (Hz)
            amplitude: Amplitude of the harmonic (0-1)
            phase: Phase of the harmonic (radians)
        """
        self.harmonics.append(HarmonicComponent(
            frequency=frequency,
            amplitude=amplitude,
            phase=phase
        ))
    
    def get_current_value(self) -> float:
        """
        Get the current instantaneous value of the oscillation.
        
        Returns:
            The current value as a sum of the fundamental and all harmonics
        """
        # Calculate fundamental component
        fundamental = self.fundamental_amplitude * np.sin(self.fundamental_phase)
        
        # Add all harmonic components
        harmonic_sum = sum(h.amplitude * np.sin(h.phase) for h in self.harmonics)
        
        # Combine and normalize to -1 to 1 range
        total = fundamental + harmonic_sum
        max_possible = self.fundamental_amplitude + sum(h.amplitude for h in self.harmonics)
        
        # Avoid division by zero
        if max_possible > 0:
            return total / max_possible
        return 0.0
    
    def project_future_state(self, time_delta: float) -> 'OscillatorySignature':
        """
        Project the signature's state into the future.
        
        Args:
            time_delta: Time to project forward
            
        Returns:
            A new OscillatorySignature representing the projected future state
        """
        # Project fundamental phase
        new_phase = (self.fundamental_phase + (self.fundamental_frequency * time_delta)) % (2 * np.pi)
        
        # Project harmonics
        projected_harmonics = [h.project(time_delta) for h in self.harmonics]
        
        # Create projected signature
        projected = OscillatorySignature(
            id=f"{self.id}_projected",
            entity_id=self.entity_id,
            entity_type=self.entity_type,
            fundamental_frequency=self.fundamental_frequency,
            fundamental_amplitude=self.fundamental_amplitude,
            fundamental_phase=new_phase,
            harmonics=projected_harmonics,
            energy_level=self.energy_level,
            energy_capacity=self.energy_capacity,
            created_at=self.created_at,
            last_updated=self.last_updated,
            oscillation_history=self.oscillation_history.copy(),
            metadata={**self.metadata, "projected": True, "projection_time_delta": time_delta}
        )
        
        return projected
    
    def calculate_resonance(self, other: 'OscillatorySignature') -> float:
        """
        Calculate resonance between this signature and another.
        
        Resonance is highest when:
        1. Frequencies are in harmonic ratios (1:1, 1:2, 2:3, etc.)
        2. Phases are aligned
        3. Amplitudes are similar
        
        Args:
            other: Another oscillatory signature
            
        Returns:
            Resonance value between 0 (no resonance) and 1 (perfect resonance)
        """
        # Calculate frequency resonance using harmonic ratios
        freq_ratio = min(self.fundamental_frequency, other.fundamental_frequency) / max(self.fundamental_frequency, other.fundamental_frequency)
        if freq_ratio < 0.01:  # Avoid division by zero or very small numbers
            freq_ratio = 0.01
            
        # Check if ratio is close to a simple fraction (1/1, 1/2, 2/3, etc.)
        harmonic_ratios = [1.0, 0.5, 0.667, 0.75, 0.8, 0.833]
        freq_resonance = max(1.0 - min(abs(freq_ratio - ratio) for ratio in harmonic_ratios), 0.0)
        
        # Calculate phase alignment (normalized to 0-1)
        phase_diff = abs(((self.fundamental_phase - other.fundamental_phase) + np.pi) % (2 * np.pi) - np.pi) / np.pi
        phase_resonance = 1.0 - phase_diff
        
        # Calculate amplitude similarity (normalized to 0-1)
        amp_diff = abs(self.fundamental_amplitude - other.fundamental_amplitude)
        amp_resonance = 1.0 - amp_diff
        
        # Combine the three factors (with frequency being most important)
        resonance = (0.5 * freq_resonance) + (0.3 * phase_resonance) + (0.2 * amp_resonance)
        
        return resonance
    
    def calculate_interference(self, other: 'OscillatorySignature') -> Tuple[float, str]:
        """
        Calculate interference pattern between this signature and another.
        
        Args:
            other: Another oscillatory signature
            
        Returns:
            Tuple of (interference_value, interference_type)
            where interference_value is between -1 (destructive) and 1 (constructive)
            and interference_type is "constructive", "destructive", or "neutral"
        """
        # Calculate phase difference
        phase_diff = abs(self.fundamental_phase - other.fundamental_phase) % (2 * np.pi)
        
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
        
        return interference_value, interference_type
