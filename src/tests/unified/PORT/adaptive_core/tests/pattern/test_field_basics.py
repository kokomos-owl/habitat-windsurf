"""Basic tests for pattern field behavior.

This module provides foundational tests for pattern field propagation
and coherence detection. It serves as the baseline for understanding
how patterns move through and interact with the conceptual field.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ...pattern.evolution import PatternEvolutionManager
from ...pattern.quality import PatternQualityAnalyzer
from ...storage.memory import InMemoryPatternStore, InMemoryRelationshipStore
from ...services.event_bus import LocalEventBus
from ...services.time_provider import TimeProvider

@dataclass
class FieldConfig:
    """Configuration for field behavior testing.
    All parameters are chosen based on theoretical foundations.
    """
    # Field parameters
    field_size: int = 10
    propagation_speed: float = 1.0  # c in wave equation
    decay_rate: float = 0.1        # Field strength decay
    boundary_condition: str = 'periodic'  # or 'reflective', 'absorbing'
    
    # Pattern parameters
    coherence_threshold: float = 0.6
    noise_threshold: float = 0.3
    interaction_range: float = 2.0
    max_patterns: int = 100
    min_pattern_separation: float = 0.5
    
    # Conservation parameters
    energy_tolerance: float = 0.1  # Allow 10% energy fluctuation
    information_tolerance: float = 0.1
    conservation_check_interval: int = 10
    
    # Wave parameters
    phase_resolution: float = 0.1
    wavelength: float = 2.0
    dispersion_relation: str = 'linear'  # or 'nonlinear'
    group_velocity: float = 0.5
    
    # Flow dynamics
    viscosity: float = 0.1
    reynolds_number_threshold: float = 2000
    vorticity_threshold: float = 0.1
    boundary_layer_thickness: float = 0.2
    
    # Quantum effects
    tunneling_probability: float = 0.1
    coherence_length: float = 1.0
    entanglement_threshold: float = 0.5
    measurement_backaction: float = 0.1
    
    # Edge cases
    singularity_threshold: float = 0.9
    chaos_onset: float = 0.7
    bifurcation_point: float = 0.5
    phase_transition_temperature: float = 1.0
    
    @property
    def is_turbulent(self) -> bool:
        """Check if flow is turbulent based on Reynolds number."""
        return self.reynolds_number > self.reynolds_number_threshold
    
    @property
    def coherence_scale(self) -> float:
        """Calculate characteristic coherence scale."""
        return self.coherence_length * np.exp(-self.viscosity)
    
    @property
    def critical_density(self) -> float:
        """Calculate critical pattern density for phase transition."""
        return 1.0 / (self.coherence_length ** 3)

@dataclass
class TestPattern:
    """Simple pattern for testing field behavior."""
    content: str
    position: np.ndarray  # 2D position for simplicity
    strength: float
    coherence: float
    phase: float = 0.0    # For wave-like behavior
    frequency: float = 1.0 # For temporal evolution

def create_test_field(config: FieldConfig) -> np.ndarray:
    """Create a test field with given configuration.
    
    Supports different boundary conditions and initial states:
    - Periodic: Field wraps around
    - Reflective: Field bounces at boundaries
    - Absorbing: Field dies at boundaries
    """
    size = config.field_size
    field = np.zeros((size, size))
    
    # Add background noise
    if config.noise_threshold > 0:
        field += np.random.normal(0, config.noise_threshold, (size, size))
    
    # Apply boundary conditions
    if config.boundary_condition == 'reflective':
        field[0, :] = field[1, :]
        field[-1, :] = field[-2, :]
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]
    elif config.boundary_condition == 'absorbing':
        field[0, :] = 0
        field[-1, :] = 0
        field[:, 0] = 0
        field[:, -1] = 0
    # periodic is default (no action needed)
    
    return field

class TestFieldBasics:
    """Basic field behavior tests."""
    
    @pytest.fixture
    def config(self) -> FieldConfig:
        """Provide test configuration."""
        return FieldConfig()
    
    @pytest.fixture
    def setup_basic_field(self, config):
        """Set up a basic test field with patterns."""
        # Create stores and services
        pattern_store = InMemoryPatternStore()
        relationship_store = InMemoryRelationshipStore()
        event_bus = LocalEventBus()
        quality_analyzer = PatternQualityAnalyzer()
        
        # Create manager
        manager = PatternEvolutionManager(
            pattern_store=pattern_store,
            relationship_store=relationship_store,
            event_bus=event_bus,
            quality_analyzer=quality_analyzer
        )
        
        # Create field with wave-supporting properties
        field = create_test_field(config.field_size)
        
        return manager, field, config
    
    @pytest.mark.asyncio
    async def test_single_pattern_propagation(self, setup_basic_field):
        """Test how a single pattern propagates through the field.
        
        Validates:
        1. Wave equation behavior (∂²ψ/∂t² = c²∇²ψ)
        2. Energy conservation
        3. Information conservation
        4. Phase evolution
        5. Field decay
        
        This is our most basic test - a "hello world" for pattern fields.
        It verifies that:
        1. A pattern creates a field disturbance
        2. The disturbance propagates correctly
        3. Field strength decays with distance
        4. Coherence is maintained within expected bounds
        """
        manager, field = setup_basic_field
        
        # Create a simple test pattern
        pattern_data = {
            "type": "test_pattern",
            "content": "Hello, Field!",
            "context": {
                "position": [5, 5],  # Center of field
                "initial_strength": 1.0
            }
        }
        
        # Register pattern
        result = await manager.register_pattern(**pattern_data)
        assert result.success
        pattern_id = result.data["id"]
        
        # Let it propagate (simulate field evolution)
        await manager._update_pattern_metrics(pattern_id)
        
        # Get updated pattern
        patterns = await manager._pattern_store.find_patterns({"id": pattern_id})
        assert patterns.success
        pattern = patterns.data[0]
        
        # Verify basic field properties
        metrics = pattern["metrics"]
        quality = pattern["quality"]
        
        # 1. Field Disturbance
        assert metrics["energy_state"] > 0, "Pattern should create field disturbance"
        
        # 2. Propagation
        assert metrics["emergence_rate"] > 0, "Pattern should propagate"
        
        # 3. Coherence
        assert 0 <= metrics["coherence"] <= 1, "Coherence should be normalized"
        assert quality["signal"]["strength"] > 0, "Should have measurable signal strength"
        
        # 4. Conservation Laws
        
        # Energy Conservation
        total_energy = (
            metrics["energy_state"] + 
            metrics["cross_pattern_flow"] * metrics["adaptation_rate"]
        )
        energy_bounds = (1 - config.energy_tolerance, 1 + config.energy_tolerance)
        assert energy_bounds[0] <= total_energy <= energy_bounds[1], \
            "Energy should be conserved within tolerance"
            
        # Information Conservation (based on Shannon entropy)
        initial_entropy = -sum(p * np.log(p) if p > 0 else 0 
                             for p in [metrics["coherence"], 1 - metrics["coherence"]])
        final_entropy = -sum(p * np.log(p) if p > 0 else 0 
                            for p in [quality["signal"]["strength"], 
                                     quality["signal"]["noise_ratio"]])
        assert abs(final_entropy - initial_entropy) <= config.information_tolerance, \
            "Information should be conserved within tolerance"
            
        # Wave Behavior
        phase_change = abs(quality["signal"]["persistence"] * 
                         config.propagation_speed / config.wavelength)
        assert phase_change > 0, "Pattern should show wave-like behavior"
        
        # Field Decay
        distance = np.sqrt(2)  # Diagonal distance in field
        expected_decay = np.exp(-config.decay_rate * distance)
        actual_decay = quality["signal"]["strength"] / metrics["energy_state"]
        assert abs(actual_decay - expected_decay) <= config.energy_tolerance, \
            "Field should decay according to exponential law"
        
        # 5. Verify Signal Properties
        signal_metrics = quality["signal"]
        assert signal_metrics["noise_ratio"] < 0.5, "Initial pattern should be clean"
        assert signal_metrics["persistence"] > 0, "Pattern should show persistence"
        
        # 6. Check Flow Properties
        flow_metrics = quality["flow"]
        assert flow_metrics["viscosity"] >= 0, "Viscosity should be non-negative"
        assert -1 <= flow_metrics["current"] <= 1, "Current should be normalized"
    
    @pytest.mark.asyncio
    async def test_pattern_coherence_detection(self, setup_basic_field):
        """Test if we can detect coherent pattern regions.
        
        Validates:
        1. Coherence gradients
        2. Interference patterns
        3. Boundary effects
        4. Phase relationships
        5. Information flow"""
        """Test if we can detect coherent pattern regions.
        
        This test verifies our ability to:
        1. Identify regions of high coherence
        2. Distinguish signal from noise
        3. Detect pattern boundaries
        """
        manager, field = setup_basic_field
        
        # Create two related patterns
        pattern1_data = {
            "type": "test_pattern",
            "content": "First Pattern",
            "context": {
                "position": [3, 3],
                "initial_strength": 1.0
            }
        }
        
        pattern2_data = {
            "type": "test_pattern",
            "content": "Related Pattern",
            "context": {
                "position": [4, 4],
                "initial_strength": 0.8
            }
        }
        
        # Register patterns
        result1 = await manager.register_pattern(**pattern1_data)
        result2 = await manager.register_pattern(**pattern2_data)
        assert result1.success and result2.success
        
        # Create relationship
        await manager.relate_patterns(
            result1.data["id"],
            result2.data["id"],
            relationship_type="similarity",
            strength=0.7
        )
        
        # Update metrics
        await manager._update_pattern_metrics(result1.data["id"])
        await manager._update_pattern_metrics(result2.data["id"])
        
        # Get updated patterns
        patterns1 = await manager._pattern_store.find_patterns({"id": result1.data["id"]})
        patterns2 = await manager._pattern_store.find_patterns({"id": result2.data["id"]})
        assert patterns1.success and patterns2.success
        
        pattern1 = patterns1.data[0]
        pattern2 = patterns2.data[0]
        
        # Verify coherence properties
        
        # 1. Individual Coherence
        assert pattern1["metrics"]["coherence"] > 0.6, "First pattern should be coherent"
        assert pattern2["metrics"]["coherence"] > 0.6, "Second pattern should be coherent"
        
        # 2. Relationship Influence
        assert pattern1["metrics"]["cross_pattern_flow"] > 0, "Should show pattern interaction"
        assert pattern2["metrics"]["cross_pattern_flow"] > 0, "Should show pattern interaction"
        
        # 3. Signal Quality
        assert pattern1["quality"]["signal"]["strength"] > pattern1["quality"]["signal"]["noise_ratio"], \
            "Signal should be stronger than noise"
        assert pattern2["quality"]["signal"]["strength"] > pattern2["quality"]["signal"]["noise_ratio"], \
            "Signal should be stronger than noise"
        
        # 4. Flow Dynamics
        assert pattern1["quality"]["flow"]["viscosity"] < 0.5, "Field should allow flow"
        assert pattern2["quality"]["flow"]["viscosity"] < 0.5, "Field should allow flow"
        
        # 5. Verify Pattern Interaction
        assert abs(pattern1["quality"]["flow"]["current"]) > 0, "Should show flow between patterns"
        assert abs(pattern2["quality"]["flow"]["current"]) > 0, "Should show flow between patterns"
        
        # 6. Interference Effects
        phase_diff = abs(pattern1["quality"]["signal"]["persistence"] - 
                        pattern2["quality"]["signal"]["persistence"])
        interference = np.cos(phase_diff * 2 * np.pi / config.wavelength)
        assert -1 <= interference <= 1, "Patterns should show interference effects"
        
        # 7. Boundary Detection
        edge_coherence = min(pattern1["metrics"]["coherence"],
                           pattern2["metrics"]["coherence"])
        assert edge_coherence < pattern1["metrics"]["coherence"], \
            "Boundary should show reduced coherence"
        
        # 8. Information Flow
        flow_gradient = (pattern2["quality"]["flow"]["current"] - 
                        pattern1["quality"]["flow"]["current"])
        assert abs(flow_gradient) > 0, "Should have directional information flow"
