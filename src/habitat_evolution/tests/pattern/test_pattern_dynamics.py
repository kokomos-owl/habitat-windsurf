"""Test pattern dynamics with simultaneous propagation and coherence analysis."""

import pytest
import numpy as np
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from .test_field_visualization import FieldVisualizer, VisualizationConfig
from pathlib import Path

# Local test configuration
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@dataclass
class DynamicsConfig:
    """Configuration for pattern dynamics testing."""
    # Field properties
    field_size: int = 20
    time_steps: int = 30
    dt: float = 0.05  # Smaller timestep for stability
    
    # Pattern properties
    initial_separation: float = 5.0
    initial_strength: float = 1.0
    
    # Wave properties
    propagation_speed: float = 0.5  # Slower propagation
    decay_rate: float = 0.1  # Faster decay
    
    # Coherence properties
    coherence_threshold: float = 0.6
    interaction_range: float = 2.0
    
    # Analysis
    save_plots: bool = True
    plot_interval: int = 5

class TestPatternDynamics:
    """Test pattern propagation and coherence together."""
    
    @pytest.fixture
    def config(self) -> DynamicsConfig:
        return DynamicsConfig()
    
    @pytest.fixture
    def visualizer(self) -> FieldVisualizer:
        return FieldVisualizer(VisualizationConfig())
    
    def create_initial_patterns(self, config: DynamicsConfig) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Create initial field with two patterns."""
        field = np.zeros((config.field_size, config.field_size))
        center = config.field_size // 2
        
        # Create two patterns with offset
        patterns = [
            {
                "id": "pattern1",
                "position": np.array([center - config.initial_separation//2, center]),
                "strength": config.initial_strength,
                "metrics": {
                    "coherence": 1.0,
                    "energy_state": config.initial_strength,
                    "phase": 0.0
                }
            },
            {
                "id": "pattern2",
                "position": np.array([center + config.initial_separation//2, center]),
                "strength": config.initial_strength,
                "metrics": {
                    "coherence": 1.0,
                    "energy_state": config.initial_strength,
                    "phase": np.pi  # Start with opposite phase
                }
            }
        ]
        
        # Set initial field values
        for pattern in patterns:
            pos = pattern["position"]
            field[int(pos[0]), int(pos[1])] = pattern["strength"]
        
        return field, patterns
    
    def evolve_field(self, field: np.ndarray, config: DynamicsConfig) -> np.ndarray:
        """Evolve field one time step."""
        # Wave equation discretization
        laplacian = (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) + 
                    np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) - 4*field)
        
        # Calculate damping factor
        damping = np.exp(-config.decay_rate * config.dt)
        
        # Evolve field with damped diffusion
        new_field = field * damping + config.propagation_speed * config.dt * laplacian
        
        return new_field
    
    def calculate_pattern_coherence(self, field: np.ndarray, pos: np.ndarray, config: DynamicsConfig) -> float:
        """Calculate pattern coherence using multiple metrics."""
        # Get local field
        x_start = int(max(0, pos[0] - config.interaction_range))
        x_end = int(min(field.shape[0], pos[0] + config.interaction_range + 1))
        y_start = int(max(0, pos[1] - config.interaction_range))
        y_end = int(min(field.shape[1], pos[1] + config.interaction_range + 1))
        local_field = field[x_start:x_end, y_start:y_end]
        
        if local_field.size == 0:
            return 0.0
        
        # 1. Density gradient coherence
        gradients = np.gradient(local_field)
        gradient_magnitude = np.sqrt(gradients[0]**2 + gradients[1]**2)
        density_coherence = np.mean(gradient_magnitude)
        
        # 2. Signal-to-noise ratio
        signal = np.mean(local_field)
        noise = np.std(local_field)
        snr = signal / (noise + 1e-10)  # Avoid division by zero
        
        # 3. Local structure coherence
        structure_coherence = np.mean(local_field > config.coherence_threshold)
        
        # Combine metrics with weights
        weights = {
            'density': 0.4,
            'snr': 0.3,
            'structure': 0.3
        }
        
        coherence = (
            weights['density'] * density_coherence +
            weights['snr'] * snr +
            weights['structure'] * structure_coherence
        )
        
        return np.clip(coherence, 0, 1)
    
    def update_pattern_metrics(self, 
                             patterns: List[Dict[str, Any]], 
                             field: np.ndarray,
                             config: DynamicsConfig) -> List[Dict[str, Any]]:
        """Update pattern metrics based on field state."""
        for pattern in patterns:
            pos = pattern["position"]
            
            # Calculate coherence using enhanced metric
            pattern["metrics"]["coherence"] = self.calculate_pattern_coherence(field, pos, config)
            
            # Calculate energy state in local region
            x_start = int(max(0, pos[0] - config.interaction_range))
            x_end = int(min(field.shape[0], pos[0] + config.interaction_range + 1))
            y_start = int(max(0, pos[1] - config.interaction_range))
            y_end = int(min(field.shape[1], pos[1] + config.interaction_range + 1))
            local_field = field[x_start:x_end, y_start:y_end]
            pattern["metrics"]["energy_state"] = np.sum(local_field**2)
            
            # Track phase with damping
            pattern["metrics"]["phase"] += config.dt * config.propagation_speed
            pattern["metrics"]["phase"] = pattern["metrics"]["phase"] % (2*np.pi)
        
        return patterns
    
    def analyze_cross_talk(self, 
                          patterns: List[Dict[str, Any]], 
                          field: np.ndarray,
                          config: DynamicsConfig) -> Dict[str, float]:
        """Analyze pattern interactions and cross-talk."""
        # Calculate midpoint between patterns
        pos1 = patterns[0]["position"]
        pos2 = patterns[1]["position"]
        mid_x = int((pos1[0] + pos2[0]) // 2)
        mid_y = int((pos1[1] + pos2[1]) // 2)
        
        # Analyze interference region
        interference_strength = field[mid_x, mid_y]
        expected_strength = (patterns[0]["metrics"]["energy_state"] + 
                           patterns[1]["metrics"]["energy_state"]) / 2
        
        return {
            "interference_ratio": interference_strength / expected_strength,
            "phase_difference": abs(patterns[0]["metrics"]["phase"] - 
                                  patterns[1]["metrics"]["phase"]) % (2*np.pi),
            "coherence_product": (patterns[0]["metrics"]["coherence"] * 
                                patterns[1]["metrics"]["coherence"])
        }
    
    @pytest.mark.asyncio
    async def test_pattern_propagation_and_coherence(self, 
                                                   config: DynamicsConfig,
                                                   visualizer: FieldVisualizer):
        """Test pattern propagation and coherence simultaneously."""
        # Initialize
        field, patterns = self.create_initial_patterns(config)
        field_states = [field.copy()]
        pattern_states = [patterns.copy()]
        cross_talk_data = []
        
        # Evolution loop
        for t in range(config.time_steps):
            # Evolve field
            field = self.evolve_field(field, config)
            
            # Update patterns
            patterns = self.update_pattern_metrics(patterns, field, config)
            
            # Analyze cross-talk
            cross_talk = self.analyze_cross_talk(patterns, field, config)
            cross_talk_data.append(cross_talk)
            
            # Store states
            field_states.append(field.copy())
            pattern_states.append(patterns.copy())
            
            # Create visualization at intervals
            if config.save_plots and t % config.plot_interval == 0:
                fig = visualizer.plot_pattern_evolution(field_states, pattern_states)
                
                # Add cross-talk analysis
                ax = fig.add_subplot(2, 2, 4)
                times = range(len(cross_talk_data))
                ax.plot(times, [d["interference_ratio"] for d in cross_talk_data], 
                       label="Interference")
                ax.plot(times, [d["coherence_product"] for d in cross_talk_data],
                       label="Coherence")
                ax.set_title("Cross-Talk Analysis")
                ax.legend()
                
                # Save plot
                plot_dir = Path(__file__).parent / "plots"
                plot_dir.mkdir(exist_ok=True)
                fig.savefig(plot_dir / f"dynamics_t{t:03d}.png")
                plt.close(fig)
        
        # Verify fundamental properties
        final_energy = np.sum(field**2)
        initial_energy = np.sum(field_states[0]**2)
        
        # Energy should decay over time but allow for some numerical diffusion
        assert final_energy < initial_energy, "Energy should decrease over time"
        assert final_energy > 0, "Energy should remain positive"
        
        # Patterns should maintain some coherence
        for pattern in patterns:
            assert pattern["metrics"]["coherence"] > 0.3, \
                "Patterns should maintain minimal coherence"
        
        # Cross-talk should show interference effects
        interference_ratios = [d["interference_ratio"] for d in cross_talk_data]
        assert max(interference_ratios) != min(interference_ratios), \
            "Should see interference effects"
        
        return field_states, pattern_states, cross_talk_data
