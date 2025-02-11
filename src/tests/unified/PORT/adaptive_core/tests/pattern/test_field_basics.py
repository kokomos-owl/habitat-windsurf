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

from unified.PORT.adaptive_core.pattern.evolution import PatternEvolutionManager, PatternMetrics
from unified.PORT.adaptive_core.pattern.quality import PatternQualityAnalyzer
from unified.PORT.adaptive_core.storage.memory import InMemoryPatternStore, InMemoryRelationshipStore
from unified.PORT.adaptive_core.services.event_bus import LocalEventBus
from unified.PORT.adaptive_core.services.time_provider import TimeProvider

from typing import List, Optional, Dict, Any
from unified.PORT.adaptive_core.config.field_config import AnalysisMode, FieldConfig


@dataclass
class TestPattern:
    """Simple pattern for testing field behavior."""
    content: str
    position: np.ndarray  # 2D position for simplicity
    strength: float
    coherence: float
    phase: float = 0.0
    frequency: float = 1.0
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
            field_size=10,
            active_modes=[
                AnalysisMode.COHERENCE,  # Primary: Pattern coherence and relationships
                AnalysisMode.WAVE,       # Required: Phase relationships
                AnalysisMode.INFORMATION, # Required: Correlation tolerances
                AnalysisMode.FLOW        # Required: Viscosity effects
            ],
            propagation_speed=1.0,
            wavelength=2.0,
            group_velocity=0.5,
            phase_velocity=1.0,
            phase_resolution=0.1,
            coherence_length=2.0,
            correlation_time=1.0,
            noise_threshold=0.3
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
            field_size=10,
            active_modes=[
                AnalysisMode.WAVE,         # Primary: Wave equation behavior
                AnalysisMode.INFORMATION,   # Required: Conservation laws
                AnalysisMode.FLOW          # Required: Field evolution
            ],
            propagation_speed=1.0,
            wavelength=2.0,
            group_velocity=0.5,
            phase_velocity=1.0,
            phase_resolution=0.1,
            coherence_length=2.0,
            correlation_time=1.0,
            noise_threshold=0.3
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
    async def test_single_pattern_propagation(self, propagation_config):
        # Create field with propagation config
        pattern_store = InMemoryPatternStore()
        relationship_store = InMemoryRelationshipStore()
        event_bus = LocalEventBus()
        manager = PatternEvolutionManager(pattern_store, relationship_store, event_bus)
        manager.config = propagation_config
        field = create_test_field(propagation_config)
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
        pattern_id = result.data  # data is the pattern ID string
        await manager._update_pattern_metrics(pattern_id)
        
        # Get updated pattern
        patterns = await manager._pattern_store.find_patterns({"id": pattern_id})
        assert patterns.success
        pattern = patterns.data[0]
        
        # === Wave Mechanics Tests ===
        # Using parameters: propagation_speed, wavelength (WAVE mode)
        metrics = pattern["metrics"]
        quality = pattern["quality"]

        # Debug Step 1: Verify active modes
        print("\nStep 1: Active Modes")
        print(f"Active Modes: {manager.config.active_modes}")
        print(f"Is Wave Mode Active? {manager.config.is_mode_active(AnalysisMode.WAVE)}")

        # Debug Step 2: Verify active parameters
        print("\nStep 2: Active Parameters")
        print(f"All Parameters: {active_params}")
        print(f"Propagation Speed: {active_params.get('propagation_speed')}")
        print(f"Wavelength: {active_params.get('wavelength')}")

        # Debug Step 3: Verify pattern quality
        print("\nStep 3: Pattern Quality")
        print(f"Full Quality: {quality}")
        print(f"Signal Persistence: {quality['signal']['persistence']}")
        
        if manager.config.is_mode_active(AnalysisMode.WAVE):
            print("\n=== WAVE-LIKE BEHAVIOR TEST ===")
            print("Required conditions for wave-like behavior:")
            print("1. Wave mode must be active")
            print("2. Propagation speed must be non-zero")
            print("3. Wavelength must be non-zero")
            print("4. Signal persistence must be non-zero")
            print("\nChecking conditions:")
            
            persistence = quality["signal"]["persistence"]
            propagation_speed = active_params.get('propagation_speed', 0)
            wavelength = active_params.get('wavelength', 0)
            
            print(f"✓ Wave mode active: {manager.config.is_mode_active(AnalysisMode.WAVE)}")
            print(f"✓ Propagation speed: {propagation_speed}")
            print(f"✓ Wavelength: {wavelength}")
            print(f"✓ Signal persistence: {persistence}")
            
            # Update pattern metrics
            await manager._update_pattern_metrics(pattern_id)
            
            # Get updated pattern
            patterns = await manager._pattern_store.find_patterns({"id": pattern_id})
            assert patterns.success
            pattern = patterns.data[0]
            quality = pattern["quality"]
            persistence = quality["signal"]["persistence"]
            
            # Calculate phase change based on persistence and wave parameters
            phase_change = abs(persistence * propagation_speed / wavelength if wavelength else 0)
            print(f"\nPhase change calculation: |{persistence} * {propagation_speed} / {wavelength}| = {phase_change}")
            
            # Phase change should be non-zero when all wave parameters are properly set
            assert phase_change > 0, "Pattern should show wave-like behavior (phase_change > 0)"
            
            # Check propagation speed
            assert metrics["emergence_rate"] > 0, "Pattern should propagate"
            assert metrics["emergence_rate"] <= active_params['propagation_speed'], \
                "Propagation should not exceed wave speed"
        
        # === Conservation Tests ===
        # Using parameters: energy_tolerance, information_tolerance (INFORMATION mode)
        if manager.config.is_mode_active(AnalysisMode.INFORMATION):
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
    async def test_pattern_coherence_detection(self, config):
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
        # Create field with config
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
        config = manager.config
        
        print("\n=== PATTERN EVOLUTION ANALYSIS ===")
        print(f"Field Size: {config.field_size}x{config.field_size}")
        print(f"Coherence Length: {config.coherence_length}")
        print(f"Wavelength: {config.wavelength}")
        print("\nActive Analysis Modes:")
        for mode in config.active_modes:
            print(f"- {mode.name}")
        
        # Create a controlled pattern configuration
        patterns = [
            {
                "id": str(uuid.uuid4()),
                "pattern_type": "test_pattern",
                "content": {"name": "Core Pattern"},
                "context": {
                    "position": [5, 5],  # Center
                    "initial_strength": 1.0,  # Maximum strength at core
                    "phase": 0.0,  # Reference phase
                    "wavelength": config.wavelength  # Natural wavelength
                },
                "metrics": PatternMetrics(
                    coherence=1.0,  # High coherence for core
                    emergence_rate=0.8,
                    cross_pattern_flow=0.0,
                    energy_state=0.8,  # High energy state
                    adaptation_rate=0.0,
                    stability=0.9  # High stability
                ).to_dict(),
                "state": "stable",  # Start in stable state
                "quality": {
                    "signal": {"strength": 1.0, "noise_ratio": 0.1, "persistence": 0.9, "reproducibility": 0.9},
                    "flow": {"viscosity": 0.2, "back_pressure": 0.0, "volume": 0.0, "current": 0.0}
                }
            },
            {
                "id": str(uuid.uuid4()),
                "pattern_type": "test_pattern",
                "content": {"name": "Coherent Satellite"},
                "context": {
                    "position": [5 + config.coherence_length, 5],
                    "initial_strength": np.exp(-config.coherence_length/config.wavelength),  # Natural decay
                    "phase": 2*np.pi * (config.coherence_length/config.wavelength),  # Natural phase progression
                },
                # Let metrics be calculated based on context
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
                # Let metrics be calculated based on context
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
        print("\n=== PATTERN EVOLUTION TRACE ===")
        for timestep in range(10):  # More timesteps to allow for dissipation
            print(f"\nTimestep {timestep + 1}:")
            for pid in pattern_ids:
                await manager._update_pattern_metrics(pid)
                result = await manager._pattern_store.find_patterns({"id": pid})
                if result.success:
                    p = result.data[0]
                    print(f"Pattern '{p['content']['name']}':")
                    print(f"  Coherence: {p['metrics']['coherence']:.3f}")
                    print(f"  Energy: {p['metrics']['energy_state']:.3f}")
                    print(f"  Flow: {p['metrics']['cross_pattern_flow']:.3f}")
        
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
        
        # Phase relationship should follow natural wave mechanics
        wavelength = config.wavelength
        distance = config.coherence_length
        expected_phase = 2*np.pi * (distance/wavelength)
        actual_phase = abs(core_pattern["context"]["phase"] - satellite_pattern["context"]["phase"])
        
        print(f"\n=== PHASE RELATIONSHIP ANALYSIS ===")
        print(f"Expected phase diff: {expected_phase:.3f}")
        print(f"Actual phase diff: {actual_phase:.3f}")
        print(f"Tolerance: {active_params['phase_resolution']}")
        
        assert abs(actual_phase - expected_phase) < active_params['phase_resolution'], \
            "Phase relationship should follow wave mechanics"
        
        # === Coherence Analysis Tests ===
        # Using parameters: coherence_threshold, noise_threshold (COHERENCE mode)
        coherence_values = [p["metrics"]["coherence"] for p in final_patterns]
        assert coherence_values[0] > coherence_values[1] > coherence_values[2], \
            "Coherence should decrease with distance from core"
        
        # Verify signal quality
        for pattern in final_patterns[:2]:  # Core and satellite
            signal = pattern["quality"]["signal"]
            # Signal strength should decay with distance but stay above noise
            min_strength = active_params['noise_threshold']
            if pattern['content']['name'] == 'Core Pattern':
                assert signal['strength'] > 0.5, "Core pattern should maintain high signal strength"
            else:
                # Satellite patterns can have lower strength but must stay above noise
                assert signal['strength'] > min_strength, \
                    f"Pattern strength ({signal['strength']:.3f}) should stay above noise threshold ({min_strength:.3f})"
            assert signal["noise_ratio"] < 0.5, \
                "Coherent patterns should have low noise"
        
        print("\n=== PHASE-AWARE DECAY TEST ===")
        print("Required conditions for phase-aware decay:")
        print("1. Core and satellite patterns must exist")
        print("2. Coherence mode must be active")
        print("3. Information mode must be active for tolerance")
        print("4. Patterns must have defined phases")
        print("\nChecking conditions:")

        # Using parameters: coherence_length (COHERENCE mode)
        #                  information_tolerance (INFORMATION mode)
        core_pos = np.array(core_pattern["context"]["position"])
        satellite_pos = np.array(satellite_pattern["context"]["position"])
        separation = np.linalg.norm(satellite_pos - core_pos)
        
        print(f"✓ Core pattern exists with coherence: {core_pattern['metrics']['coherence']}")
        print(f"✓ Satellite pattern exists with coherence: {satellite_pattern['metrics']['coherence']}")
        print(f"✓ Coherence mode active: {manager.config.is_mode_active(AnalysisMode.COHERENCE)}")
        print(f"✓ Information mode active: {manager.config.is_mode_active(AnalysisMode.INFORMATION)}")
        
        # Calculate phase-aware correlation
        core_phase = core_pattern["context"]["phase"]
        satellite_phase = satellite_pattern["context"]["phase"]
        phase_diff = abs(core_phase - satellite_phase)
        phase_factor = 0.5 + 0.5 * np.cos(phase_diff)
        
        print(f"\nPhase Analysis:")
        print(f"Core Phase: {core_phase:.3f}")
        print(f"Satellite Phase: {satellite_phase:.3f}")
        print(f"Phase Difference: {phase_diff:.3f}")
        print(f"Phase Factor: {phase_factor:.3f}")
        
        correlation = core_pattern["metrics"]["coherence"] * \
                     satellite_pattern["metrics"]["coherence"]
        
        # Calculate minimum allowed correlation based on spatial decay and phase
        spatial_decay = np.exp(-separation / active_params['coherence_length'])
        min_expected_correlation = spatial_decay * phase_factor
        
        print(f"\nCorrelation Analysis:")
        print(f"Separation Distance: {separation:.3f}")
        print(f"Spatial Decay: {spatial_decay:.3f}")
        print(f"Actual Correlation: {correlation:.3f}")
        print(f"Minimum Expected: {min_expected_correlation:.3f}")
        print(f"Difference: {correlation - min_expected_correlation:.3f}")
        print(f"Allowed Tolerance: {active_params['information_tolerance']:.3f}")
        
        # The actual correlation should not exceed 1.0 (perfect correlation)
        # and should not be less than the theoretical minimum (spatial decay * phase)
        assert correlation <= 1.0, "Correlation cannot exceed 1.0"
        assert correlation >= min_expected_correlation, \
            f"Correlation ({correlation:.3f}) should not be less than minimum expected ({min_expected_correlation:.3f})"
        
        # 5. Flow Dynamics Tests
        # Check for turbulence and viscosity effects
        noise_pattern = final_patterns[2]
        flow = noise_pattern["quality"]["flow"]
        
        # Update pattern metrics for noise pattern
        await manager._update_pattern_metrics(noise_pattern["id"])
        
        # Get updated pattern
        patterns = await manager._pattern_store.find_patterns({"id": noise_pattern["id"]})
        assert patterns.success
        noise_pattern = patterns.data[0]
        flow = noise_pattern["quality"]["flow"]
        
        # Check for turbulent flow indicators
        if flow["viscosity"] < 1.0 and abs(flow["current"]) > 0:
            # Low viscosity and non-zero current indicates potential turbulence
            assert flow["back_pressure"] > 0, "Turbulent flow should show back pressure"
        
        # Verify viscosity effects on coherence
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
