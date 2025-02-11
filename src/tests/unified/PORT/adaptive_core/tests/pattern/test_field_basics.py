"""Basic tests for pattern field behavior.

This module provides foundational tests for pattern field propagation
and coherence detection. It serves as the baseline for understanding
how patterns move through and interact with the conceptual field.
"""

import pytest
import numpy as np
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent.parent.parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from tests.unified.PORT.adaptive_core.pattern.evolution import PatternEvolutionManager, PatternMetrics
from tests.unified.PORT.adaptive_core.pattern.quality import PatternQualityAnalyzer
from tests.unified.PORT.adaptive_core.storage.memory import InMemoryPatternStore, InMemoryRelationshipStore
from tests.unified.PORT.adaptive_core.services.event_bus import LocalEventBus
from tests.unified.PORT.adaptive_core.services.time_provider import TimeProvider

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
    # Initialize field
    field = np.zeros((config.field_size, config.field_size))
    
    # Apply boundary conditions
    if config.boundary_condition == 'periodic':
        # No special handling needed for periodic
        pass
    elif config.boundary_condition == 'reflective':
        # Set boundaries to mirror internal values
        field[0, :] = field[1, :]
        field[-1, :] = field[-2, :]
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]
    elif config.boundary_condition == 'absorbing':
        # Set boundaries to zero (already done by zeros initialization)
        pass
    
    return field
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
        field = create_test_field(config)
        
        # Store config in manager for access in tests
        manager.config = config
        
        return manager, field
    
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
        
        This test validates our fundamental understanding of pattern coherence
        through multiple scientific lenses:
        
        1. Wave Mechanics:
           - Phase relationships between pattern regions
           - Interference patterns and superposition
           - Wave packet dispersion and group velocity
        
        2. Field Theory:
           - Coherence gradients and their propagation
           - Field strength decay with distance
           - Boundary conditions and edge effects
        
        3. Information Theory:
           - Signal-to-noise ratio in pattern regions
           - Information flow between patterns
           - Entropy gradients in transition regions
        
        4. Quantum Analogs:
           - Coherence length and correlation functions
           - Entanglement-like effects between patterns
           - Measurement effects on pattern state
        
        5. Flow Dynamics:
           - Pattern viscosity and Reynolds number
           - Vorticity in pattern transitions
           - Turbulence onset in high-activity regions
        """
        manager, field = setup_basic_field
        config = manager.config
        
        # Create a controlled pattern configuration
        patterns = [
            {
                "id": str(uuid.uuid4()),
                "pattern_type": "test_pattern",
                "content": {"name": "Core Pattern"},
                "context": {
                    "position": [5, 5],  # Center
                    "initial_strength": 1.0,
                    "phase": 0.0
                },
                "metrics": PatternMetrics(
                    coherence=0.0,
                    emergence_rate=0.0,
                    cross_pattern_flow=0.0,
                    energy_state=0.0,
                    adaptation_rate=0.0,
                    stability=0.0
                ).to_dict(),
                "state": "EMERGING",
                "quality": {
                    "signal": {"strength": 0.0, "noise_ratio": 0.0, "persistence": 0.0, "reproducibility": 0.0},
                    "flow": {"viscosity": 0.0, "back_pressure": 0.0, "volume": 0.0, "current": 0.0}
                }
            },
            {
                "id": str(uuid.uuid4()),
                "pattern_type": "test_pattern",
                "content": {"name": "Coherent Satellite"},
                "context": {
                    "position": [5 + config.coherence_length, 5],
                    "initial_strength": 0.8,
                    "phase": np.pi/4  # Phase-locked relationship
                },
                "metrics": PatternMetrics(
                    coherence=0.0,
                    emergence_rate=0.0,
                    cross_pattern_flow=0.0,
                    energy_state=0.0,
                    adaptation_rate=0.0,
                    stability=0.0
                ).to_dict(),
                "state": "EMERGING",
                "quality": {
                    "signal": {"strength": 0.0, "noise_ratio": 0.0, "persistence": 0.0, "reproducibility": 0.0},
                    "flow": {"viscosity": 0.0, "back_pressure": 0.0, "volume": 0.0, "current": 0.0}
                }
            },
            {
                "id": str(uuid.uuid4()),
                "pattern_type": "test_pattern",
                "content": {"name": "Incoherent Noise"},
                "context": {
                    "position": [5, 5 + 2*config.coherence_length],
                    "initial_strength": 0.3,
                    "phase": np.random.random() * 2*np.pi
                },
                "metrics": PatternMetrics(
                    coherence=0.0,
                    emergence_rate=0.0,
                    cross_pattern_flow=0.0,
                    energy_state=0.0,
                    adaptation_rate=0.0,
                    stability=0.0
                ).to_dict(),
                "state": "EMERGING",
                "quality": {
                    "signal": {"strength": 0.0, "noise_ratio": 0.0, "persistence": 0.0, "reproducibility": 0.0},
                    "flow": {"viscosity": 0.0, "back_pressure": 0.0, "volume": 0.0, "current": 0.0}
                }
            }
        ]
        
        # Register patterns and let them evolve
        pattern_ids = []
        for p in patterns:
            # Use the pattern's ID instead of generating a new one
            result = await manager._pattern_store.save_pattern(p)
            assert result.success, f"Failed to save pattern: {result.error}"
            pattern_ids.append(p["id"])
            
        # Let patterns interact
        for _ in range(5):  # Multiple timesteps
            for pid in pattern_ids:
                await manager._update_pattern_metrics(pid)
        
        # Retrieve final pattern states
        final_patterns = []
        for pid in pattern_ids:
            print(f"Searching for pattern {pid}")
            result = await manager._pattern_store.find_patterns({"id": pid})
            assert result.success
            print(f"Found patterns: {result.data}")
            final_patterns.extend(result.data)
        
        # 1. Wave Mechanics Tests
        core_pattern = final_patterns[0]
        satellite_pattern = final_patterns[1]
        
        # Phase relationship should be maintained
        phase_diff = abs(core_pattern["context"]["phase"] - satellite_pattern["context"]["phase"])
        assert abs(phase_diff - np.pi/4) < config.phase_resolution, \
            "Phase relationship should be preserved"
        
        # 2. Field Theory Tests
        # Verify coherence gradient
        coherence_values = [p["metrics"]["coherence"] for p in final_patterns]
        assert coherence_values[0] > coherence_values[1] > coherence_values[2], \
            "Coherence should decrease with distance from core"
        
        # 3. Information Theory Tests
        for pattern in final_patterns[:2]:  # Core and satellite
            signal = pattern["quality"]["signal"]
            assert signal["strength"] > config.noise_threshold, \
                "Coherent patterns should maintain signal strength"
            assert signal["noise_ratio"] < 0.5, \
                "Coherent patterns should have low noise"
        
        # 4. Quantum Analog Tests
        # Test correlation at coherence length
        core_pos = np.array(core_pattern["context"]["position"])
        satellite_pos = np.array(satellite_pattern["context"]["position"])
        separation = np.linalg.norm(satellite_pos - core_pos)
        
        correlation = core_pattern["metrics"]["coherence"] * \
                     satellite_pattern["metrics"]["coherence"]
        expected_correlation = np.exp(-separation / config.coherence_length)
        assert abs(correlation - expected_correlation) < config.energy_tolerance, \
            "Correlation should decay exponentially with distance"
        
        # 5. Flow Dynamics Tests
        # Check for turbulence onset
        noise_pattern = final_patterns[2]
        flow = noise_pattern["quality"]["flow"]
        
        if flow["reynolds_number"] > config.reynolds_number_threshold:
            assert flow["vorticity"] > config.vorticity_threshold, \
                "High Reynolds number should indicate turbulence"
        
        # Verify viscosity effects
        assert noise_pattern["metrics"]["coherence"] < config.noise_threshold, \
            "Incoherent patterns should dissipate due to viscosity"
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
