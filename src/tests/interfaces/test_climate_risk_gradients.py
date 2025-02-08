"""Tests for gradient interface using climate risk domain patterns."""

import pytest
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any, Tuple

from src.core.interfaces.gradient_interface import (
    GradientInterface,
    GradientState
)
from src.core.types import (
    DensityMetrics,
    PatternEvolutionMetrics
)

class ClimateRiskPatterns:
    """Domain-specific climate risk patterns."""
    
    def __init__(self):
        # Historical baselines
        self.historical_drought_probability = 0.085  # 8.5% from doc
        self.historical_extreme_rain_baseline = 0.01  # 1% heaviest events
        self.historical_high_danger_fire_days = 100  # Baseline for comparison
        
        # Projected changes
        self.mid_century_drought_probability = 0.13   # 13% by 2041-2060
        self.late_century_drought_probability = 0.26  # 26% by 2071-2090
        self.precipitation_intensity_increase = 0.55  # 55% increase in Northeast
        
    def generate_drought_evolution(
        self,
        num_steps: int,
        time_horizon: str
    ) -> List[float]:
        """Generate drought pattern evolution."""
        if time_horizon == "mid":
            target = self.mid_century_drought_probability
        else:  # late century
            target = self.late_century_drought_probability
            
        return np.linspace(
            self.historical_drought_probability,
            target,
            num_steps
        ).tolist()
    
    def calculate_risk_dimensions(
        self,
        drought_prob: float,
        precip_intensity: float,
        fire_danger: float,
        storm_intensity: float
    ) -> List[float]:
        """Calculate risk dimensions for gradient space."""
        return [
            min(1.0, drought_prob / 0.3),  # Normalize to [0,1]
            min(1.0, precip_intensity / 2.0),
            min(1.0, fire_danger / 200),
            min(1.0, storm_intensity / 2.0)
        ]

@pytest.fixture
def climate_patterns():
    """Create climate risk pattern generator."""
    return ClimateRiskPatterns()

@pytest.fixture
def domain_interface():
    """Create domain-aware gradient interface."""
    return GradientInterface(
        dimensions=4,  # drought, precipitation, fire, storms
        similarity_threshold=0.85,  # Higher threshold for climate patterns
        evolution_rate=0.05  # Slower evolution for climate systems
    )

@pytest.mark.asyncio
async def test_drought_pattern_evolution(climate_patterns, domain_interface):
    """Test drought pattern evolution matches projections."""
    # Generate mid-century evolution
    drought_steps = climate_patterns.generate_drought_evolution(20, "mid")
    
    states = []
    base_time = datetime.now()
    
    for i, drought_prob in enumerate(drought_steps):
        # Calculate related metrics based on drought
        precip_intensity = 1.0 + (drought_prob * 0.5)  # More intense rain during droughts
        fire_danger = 100 * (1 + drought_prob)  # Fire danger increases with drought
        storm_intensity = 1.0 + (i/len(drought_steps)) * 0.3  # Gradual storm intensification
        
        dims = climate_patterns.calculate_risk_dimensions(
            drought_prob, precip_intensity, fire_danger, storm_intensity
        )
        
        state = GradientState(
            dimensions=dims,
            confidence=0.9 - (i/len(drought_steps)) * 0.2,  # Decreasing confidence over time
            timestamp=base_time + timedelta(days=i*30)  # Monthly steps
        )
        states.append(state)
        domain_interface.record_state(state)
    
    # Verify evolution matches projections
    final_state = states[-1]
    assert abs(final_state.dimensions[0] - 
              climate_patterns.mid_century_drought_probability/0.3) < 0.01, \
        "Drought evolution didn't match projections"
    
    # Check for smooth transitions
    for i in range(1, len(states)):
        transition = states[i].distance_to(states[i-1])
        assert transition < 0.1, "Too rapid state transition"

@pytest.mark.asyncio
async def test_compound_risk_patterns(climate_patterns, domain_interface):
    """Test detection of compound risk patterns (drought + fire)."""
    num_steps = 30
    base_time = datetime.now()
    
    # Generate compound evolution
    drought_steps = climate_patterns.generate_drought_evolution(num_steps, "late")
    states = []
    
    for i, drought_prob in enumerate(drought_steps):
        # Intensify fire danger more rapidly during drought
        fire_base = climate_patterns.historical_high_danger_fire_days
        fire_increase = 1.0 + (i/num_steps) * 0.94  # 94% increase by late century
        fire_drought_multiplier = 1.0 + drought_prob  # Additional increase during drought
        
        fire_danger = fire_base * fire_increase * fire_drought_multiplier
        
        dims = climate_patterns.calculate_risk_dimensions(
            drought_prob=drought_prob,
            precip_intensity=1.0 + (i/num_steps) * 0.55,  # 55% increase
            fire_danger=fire_danger,
            storm_intensity=1.0 + (i/num_steps) * 0.4
        )
        
        state = GradientState(
            dimensions=dims,
            confidence=0.85,  # High confidence in compound effects
            timestamp=base_time + timedelta(days=i*30)
        )
        states.append(state)
        domain_interface.record_state(state)
    
    # Verify compound risk detection
    high_risk_states = [
        s for s in states
        if s.dimensions[0] > 0.5 and s.dimensions[2] > 0.5  # High drought and fire risk
    ]
    
    assert len(high_risk_states) > 0, "No compound risk states detected"
    
    # Check compound risk emergence timing
    first_compound = high_risk_states[0]
    compound_time = (first_compound.timestamp - base_time).days / 30
    assert compound_time > 6, "Compound risks emerged too quickly"

@pytest.mark.asyncio
async def test_pattern_stability_during_extremes(climate_patterns, domain_interface):
    """Test pattern stability during extreme events."""
    base_time = datetime.now()
    
    # Simulate rapid onset drought
    drought_baseline = climate_patterns.historical_drought_probability
    drought_extreme = 0.30  # 30% probability (beyond late century)
    
    states = []
    
    # Generate rapid onset
    onset_steps = 10
    for i in range(onset_steps):
        drought_prob = drought_baseline + (
            (drought_extreme - drought_baseline) * (i/onset_steps)
        )
        
        dims = climate_patterns.calculate_risk_dimensions(
            drought_prob=drought_prob,
            precip_intensity=1.5,  # Intense precipitation during transition
            fire_danger=150,  # Elevated fire risk
            storm_intensity=1.2
        )
        
        state = GradientState(
            dimensions=dims,
            confidence=0.7,  # Lower confidence during extreme transition
            timestamp=base_time + timedelta(days=i)
        )
        states.append(state)
        domain_interface.record_state(state)
    
    # Verify stability properties
    for i in range(1, len(states)):
        current = states[i]
        previous = states[i-1]
        
        # Check transition bounds
        transition = current.distance_to(previous)
        assert transition < 0.15, "State transition too extreme"
        
        # Verify dimension bounds
        assert all(0 <= d <= 1 for d in current.dimensions), \
            "Dimensions outside valid range"

@pytest.mark.asyncio
async def test_multi_hazard_pattern_recognition(climate_patterns, domain_interface):
    """Test recognition of multi-hazard patterns in climate risk."""
    num_steps = 40
    base_time = datetime.now()
    
    # Generate three concurrent hazard evolutions
    drought_steps = climate_patterns.generate_drought_evolution(num_steps, "late")
    precip_steps = np.linspace(1.0, 1.55, num_steps)  # 55% increase
    fire_steps = np.linspace(100, 194, num_steps)  # 94% increase
    
    states = []
    hazard_patterns = []
    
    for i in range(num_steps):
        # Create primary state
        dims = climate_patterns.calculate_risk_dimensions(
            drought_prob=drought_steps[i],
            precip_intensity=precip_steps[i],
            fire_danger=fire_steps[i],
            storm_intensity=1.0 + (i/num_steps) * 0.3
        )
        
        state = GradientState(
            dimensions=dims,
            confidence=0.8,
            timestamp=base_time + timedelta(days=i*30)
        )
        states.append(state)
        
        # Track hazard patterns
        if i > 0:
            for prev_state in states[-3:]:  # Look at recent states
                is_near, similarity = domain_interface.is_near_io(state, prev_state)
                if is_near:
                    hazard_patterns.append((state, prev_state, similarity))
        
        domain_interface.record_state(state)
    
    # Analyze hazard pattern relationships
    assert len(hazard_patterns) > 0, "No hazard patterns detected"
    
    # Verify pattern relationships
    pattern_similarities = [p[2] for p in hazard_patterns]
    assert np.mean(pattern_similarities) > 0.7, \
        "Hazard patterns not sufficiently related"
    
    # Check for pattern clusters
    pattern_distances = [
        p[0].distance_to(p[1])
        for p in hazard_patterns
    ]
    assert np.std(pattern_distances) < 0.2, \
        "Hazard pattern relationships too volatile"
