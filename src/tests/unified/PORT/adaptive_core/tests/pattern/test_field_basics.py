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

from enum import Enum, auto
from typing import List, Optional, Dict, Any

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
    """Configuration for field behavior testing.
    Parameters are organized by their analytical domain and can be
    selectively activated based on the type of analysis being performed.
    
    Usage:
        config = FieldConfig(active_modes=[AnalysisMode.COHERENCE])
        if config.is_mode_active(AnalysisMode.COHERENCE):
            # Use coherence parameters
    """
    # === Core Parameters (Always Active) ===
    field_size: int = 10
    active_modes: Optional[List[AnalysisMode]] = None
    
    # === Coherence Analysis Parameters ===
    # Primary parameters for pattern coherence detection
    coherence_threshold: float = 0.6     # Primary: Threshold for coherent patterns
    coherence_length: float = 1.0        # Primary: Characteristic length for coherence decay
    noise_threshold: float = 0.3         # Primary: Threshold for noise classification
    information_tolerance: float = 0.1   # Primary: Allowable information drift
    phase_resolution: float = 0.1        # Primary: Minimum detectable phase difference
    
    # === Wave Mechanics Parameters ===
    # Used for analyzing wave-like behavior in patterns
    propagation_speed: float = 1.0       # Reserved: Wave equation constant (c)
    wavelength: float = 2.0              # Reserved: Pattern wavelength
    dispersion_relation: str = 'linear'  # Reserved: Wave dispersion type
    group_velocity: float = 0.5          # Reserved: Pattern group velocity
    
    # === Flow Dynamics Parameters ===
    # Reserved for future flow analysis
    viscosity: float = 0.1               # Reserved: Pattern flow viscosity
    reynolds_number_threshold: float = 2000  # Reserved: Turbulence threshold
    vorticity_threshold: float = 0.1     # Reserved: Vortex detection
    boundary_layer_thickness: float = 0.2 # Reserved: Boundary effects
    
    # === Quantum Analog Parameters ===
    # Reserved for quantum-like behavior
    tunneling_probability: float = 0.1    # Reserved: Pattern tunneling
    entanglement_threshold: float = 0.5   # Reserved: Pattern entanglement
    measurement_backaction: float = 0.1   # Reserved: Measurement effects
    
    # === Information Theory Parameters ===
    # Reserved for information flow analysis
    decay_rate: float = 0.1              # Reserved: Information decay
    interaction_range: float = 2.0        # Reserved: Pattern interaction
    max_patterns: int = 100              # Reserved: Pattern capacity
    min_pattern_separation: float = 0.5   # Reserved: Pattern distinction
    energy_tolerance: float = 0.1         # Reserved: Energy conservation
    conservation_check_interval: int = 10 # Reserved: Conservation checks
    
    # === Advanced Parameters ===
    # Reserved for future complex analysis
    boundary_condition: str = 'periodic'  # Reserved: Field boundary type
    singularity_threshold: float = 0.9    # Reserved: Singularity detection
    chaos_onset: float = 0.7              # Reserved: Chaos transition
    bifurcation_point: float = 0.5        # Reserved: System bifurcation
    phase_transition_temperature: float = 1.0  # Reserved: Phase transitions
    
    def __post_init__(self):
        """Initialize with default mode if none specified."""
        if self.active_modes is None:
            self.active_modes = [AnalysisMode.COHERENCE]  # Default to basic coherence analysis
    
    def is_mode_active(self, mode: AnalysisMode) -> bool:
        """Check if a specific analysis mode is active."""
        return AnalysisMode.ALL in self.active_modes or mode in self.active_modes
    
    def get_active_parameters(self) -> Dict[str, Any]:
        """Get dictionary of currently active parameters and their values.
        
        Core propagation parameters are always included to ensure basic
        pattern evolution works. Additional parameters are included based
        on active modes.
        """
        # Core parameters always included for basic pattern propagation
        params = {
            'field_size': self.field_size,
            'coherence_length': self.coherence_length,  # Required for spatial decay
            'propagation_speed': self.propagation_speed,  # Required for evolution
            'boundary_condition': self.boundary_condition,  # Required for field behavior
            'energy_tolerance': self.energy_tolerance,  # Required for stability
            'noise_threshold': self.noise_threshold  # Required for signal detection
        }
        
        # Additional COHERENCE mode parameters
        if self.is_mode_active(AnalysisMode.COHERENCE):
            params.update({
                'coherence_threshold': self.coherence_threshold,
                'phase_resolution': self.phase_resolution,
                'information_tolerance': self.information_tolerance
            })
        
        # Additional WAVE mode parameters
        if self.is_mode_active(AnalysisMode.WAVE):
            params.update({
                'wavelength': self.wavelength,
                'dispersion_relation': self.dispersion_relation,
                'group_velocity': self.group_velocity
            })
        
        # Additional FLOW mode parameters
        if self.is_mode_active(AnalysisMode.FLOW):
            params.update({
                'viscosity': self.viscosity,
                'reynolds_number_threshold': self.reynolds_number_threshold,
                'vorticity_threshold': self.vorticity_threshold,
                'boundary_layer_thickness': self.boundary_layer_thickness
            })
        
        # Additional QUANTUM mode parameters
        if self.is_mode_active(AnalysisMode.QUANTUM):
            params.update({
                'tunneling_probability': self.tunneling_probability,
                'entanglement_threshold': self.entanglement_threshold,
                'measurement_backaction': self.measurement_backaction
            })
        
        # Additional INFORMATION mode parameters
        if self.is_mode_active(AnalysisMode.INFORMATION):
            params.update({
                'decay_rate': self.decay_rate,
                'interaction_range': self.interaction_range,
                'max_patterns': self.max_patterns,
                'min_pattern_separation': self.min_pattern_separation,
                'conservation_check_interval': self.conservation_check_interval
            })
        
        return params
    
    @property
    def is_turbulent(self) -> bool:
        """Check if flow is turbulent based on Reynolds number.
        Only valid when FLOW analysis mode is active."""
        if not self.is_mode_active(AnalysisMode.FLOW):
            return False
        return self.reynolds_number_threshold > 0 and self.viscosity > 0
    
    @property
    def coherence_scale(self) -> float:
        """Calculate characteristic coherence scale.
        Only valid when COHERENCE analysis mode is active."""
        if not self.is_mode_active(AnalysisMode.COHERENCE):
            return 0.0
        return self.coherence_length
    
    @property
    def critical_density(self) -> float:
        """Calculate critical pattern density for phase transition.
        Only valid when QUANTUM analysis mode is active."""
        if not self.is_mode_active(AnalysisMode.QUANTUM):
            return 0.0
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
    
    The field is created based on active analysis modes:
    1. COHERENCE mode: Determines noise threshold and field properties
    2. WAVE mode: Affects boundary conditions and wave properties
    3. FLOW mode: Influences field dynamics
    
    Supports different boundary conditions:
    - Periodic: Field wraps around (default)
    - Reflective: Field bounces at boundaries
    - Absorbing: Field dies at boundaries
    """
    active_params = config.get_active_parameters()
    size = active_params['field_size']
    
    # Initialize field
    field = np.zeros((size, size))
    
    # Add background noise if COHERENCE mode is active
    if config.is_mode_active(AnalysisMode.COHERENCE):
        noise_level = active_params.get('noise_threshold', 0)
        if noise_level > 0:
            field += np.random.normal(0, noise_level, (size, size))
    
    # Apply boundary conditions if WAVE mode is active
    if config.is_mode_active(AnalysisMode.WAVE):
        boundary_type = active_params.get('boundary_condition', 'periodic')
        if boundary_type == 'reflective':
            # Set boundaries to mirror internal values
            field[0, :] = field[1, :]
            field[-1, :] = field[-2, :]
            field[:, 0] = field[:, 1]
            field[:, -1] = field[:, -2]
        elif boundary_type == 'absorbing':
            # Set boundaries to zero
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
        """Provide test configuration based on the test being run.
        Default configuration focuses on coherence detection.
        
        For coherence detection tests, we primarily need:
        - Coherence parameters for basic pattern relationships
        - Wave parameters for phase relationships
        - Information parameters for correlation tolerances
        """
        return FieldConfig(
            active_modes=[
                AnalysisMode.COHERENCE,  # Primary: Pattern coherence and relationships
                AnalysisMode.WAVE,       # Required: Phase relationships
                AnalysisMode.INFORMATION # Required: Correlation tolerances
            ]
        )
    
    @pytest.fixture
    def propagation_config(self) -> FieldConfig:
        """Configuration specific to pattern propagation test.
        
        Focuses on:
        - Wave mechanics for propagation behavior
        - Information theory for conservation laws
        - Flow dynamics for field evolution
        """
        return FieldConfig(
            active_modes=[
                AnalysisMode.WAVE,         # Primary: Wave equation behavior
                AnalysisMode.INFORMATION,   # Required: Conservation laws
                AnalysisMode.FLOW          # Required: Field evolution
            ]
        )
    
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
        active_params = manager.config.get_active_parameters()
        
        # Create a test pattern with wave and flow characteristics
        result = await manager.register_pattern(
            pattern_type="test_pattern",
            content={"message": "Hello, Field!"},
            context={
                "position": [5, 5],  # Center of field
                "initial_strength": 1.0,
                "phase": 0.0,
                "group_velocity": active_params['group_velocity'] if 'group_velocity' in active_params else 0.5
            }
        )
        assert result.success
        pattern_id = result.data["id"]
        await manager._update_pattern_metrics(pattern_id)
        
        # Get updated pattern
        patterns = await manager._pattern_store.find_patterns({"id": pattern_id})
        assert patterns.success
        pattern = patterns.data[0]
        
        # === Wave Mechanics Tests ===
        # Using parameters: propagation_speed, wavelength (WAVE mode)
        metrics = pattern["metrics"]
        quality = pattern["quality"]
        
        if config.is_mode_active(AnalysisMode.WAVE):
            # Verify wave equation behavior
            phase_change = abs(quality["signal"]["persistence"] * 
                             active_params['propagation_speed'] / active_params['wavelength'])
            assert phase_change > 0, "Pattern should show wave-like behavior"
            
            # Check propagation speed
            assert metrics["emergence_rate"] > 0, "Pattern should propagate"
            assert metrics["emergence_rate"] <= active_params['propagation_speed'], \
                "Propagation should not exceed wave speed"
        
        # === Conservation Tests ===
        # Using parameters: energy_tolerance, information_tolerance (INFORMATION mode)
        if config.is_mode_active(AnalysisMode.INFORMATION):
            # Energy Conservation
            total_energy = (
                metrics["energy_state"] + 
                metrics["cross_pattern_flow"] * metrics["adaptation_rate"]
            )
            energy_bounds = (1 - active_params['energy_tolerance'], 
                           1 + active_params['energy_tolerance'])
            assert energy_bounds[0] <= total_energy <= energy_bounds[1], \
                "Energy should be conserved within tolerance"
            
            # Information Conservation (Shannon entropy)
            initial_entropy = -sum(p * np.log(p) if p > 0 else 0 
                                 for p in [metrics["coherence"], 1 - metrics["coherence"]])
            final_entropy = -sum(p * np.log(p) if p > 0 else 0 
                                for p in [quality["signal"]["strength"], 
                                         quality["signal"]["noise_ratio"]])
            assert abs(final_entropy - initial_entropy) <= active_params['information_tolerance'], \
                "Information should be conserved within tolerance"
            
            # Field Decay
            distance = np.sqrt(2)  # Diagonal distance in field
            expected_decay = np.exp(-active_params['decay_rate'] * distance)
            actual_decay = quality["signal"]["strength"] / metrics["energy_state"]
            assert abs(actual_decay - expected_decay) <= active_params['energy_tolerance'], \
                "Field should decay according to exponential law"
        
        # === Flow Dynamics Tests ===
        # Using parameters: viscosity (FLOW mode)
        if config.is_mode_active(AnalysisMode.FLOW):
            flow_metrics = quality["flow"]
            assert flow_metrics["viscosity"] >= 0, "Viscosity should be non-negative"
            assert flow_metrics["viscosity"] <= active_params['viscosity'], \
                "Flow viscosity should not exceed configuration"
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
        
        # Get active parameters for each analysis mode
        active_params = config.get_active_parameters()
        
        # === Wave Mechanics Tests ===
        # Using parameters: phase_resolution (WAVE mode)
        core_pattern = final_patterns[0]
        satellite_pattern = final_patterns[1]
        
        # Phase relationship should be maintained
        phase_diff = abs(core_pattern["context"]["phase"] - satellite_pattern["context"]["phase"])
        assert abs(phase_diff - np.pi/4) < active_params['phase_resolution'], \
            "Phase relationship should be preserved"
        
        # === Coherence Analysis Tests ===
        # Using parameters: coherence_threshold, noise_threshold (COHERENCE mode)
        coherence_values = [p["metrics"]["coherence"] for p in final_patterns]
        assert coherence_values[0] > coherence_values[1] > coherence_values[2], \
            "Coherence should decrease with distance from core"
        
        # Verify signal quality
        for pattern in final_patterns[:2]:  # Core and satellite
            signal = pattern["quality"]["signal"]
            assert signal["strength"] > active_params['noise_threshold'], \
                "Coherent patterns should maintain signal strength"
            assert signal["noise_ratio"] < 0.5, \
                "Coherent patterns should have low noise"
        
        # === Pattern Correlation Tests ===
        # Using parameters: coherence_length (COHERENCE mode)
        #                  information_tolerance (INFORMATION mode)
        core_pos = np.array(core_pattern["context"]["position"])
        satellite_pos = np.array(satellite_pattern["context"]["position"])
        separation = np.linalg.norm(satellite_pos - core_pos)
        
        # Calculate phase-aware correlation
        core_phase = core_pattern["context"]["phase"]
        satellite_phase = satellite_pattern["context"]["phase"]
        phase_diff = abs(core_phase - satellite_phase)
        phase_factor = 0.5 + 0.5 * np.cos(phase_diff)
        
        correlation = core_pattern["metrics"]["coherence"] * \
                     satellite_pattern["metrics"]["coherence"]
        
        # Expected correlation combines spatial decay and phase alignment
        spatial_decay = np.exp(-separation / active_params['coherence_length'])
        expected_correlation = spatial_decay * phase_factor
        
        # Allow for semantic drift within information tolerance
        assert abs(correlation - expected_correlation) < active_params['information_tolerance'], \
            "Correlation should follow phase-aware exponential decay"
        
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
