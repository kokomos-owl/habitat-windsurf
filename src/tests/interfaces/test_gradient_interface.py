"""Tests for gradient interface state evolution under high load."""

import pytest
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Tuple

from src.core.interfaces.gradient_interface import (
    GradientInterface,
    GradientState
)
from src.core.types import (
    DensityMetrics,
    PatternEvolutionMetrics
)

def generate_state_sequence(
    start_dims: List[float],
    end_dims: List[float],
    steps: int
) -> List[List[float]]:
    """Generate a sequence of states between start and end."""
    return [
        [
            start_dims[i] + (end_dims[i] - start_dims[i]) * (step / steps)
            for i in range(len(start_dims))
        ]
        for step in range(steps + 1)
    ]

@pytest.fixture
def mock_high_volume_states() -> List[Tuple[DensityMetrics, PatternEvolutionMetrics]]:
    """Generate mock states simulating high-volume document ingestion."""
    base_time = datetime.now()
    states = []
    
    # Simulate rapid state changes over time
    for i in range(100):  # Simulate 100 rapid state changes
        density = DensityMetrics(
            global_density=0.5 + np.sin(i/10) * 0.3,  # Oscillating density
            local_density=0.6 + np.cos(i/8) * 0.2,
            cross_domain_strength=0.7 + np.sin(i/12) * 0.2,
            interface_recognition=0.8 + np.cos(i/15) * 0.1,
            viscosity=0.4 + np.sin(i/20) * 0.3
        )
        
        evolution = PatternEvolutionMetrics(
            gradient=0.6 + np.cos(i/10) * 0.2,
            interface_strength=0.7 + np.sin(i/15) * 0.2,
            stability=0.8 - (i/100) * 0.3,  # Decreasing stability
            emergence_rate=0.5 + np.cos(i/12) * 0.3,
            coherence_level=0.9 - (i/100) * 0.4  # Decreasing coherence
        )
        
        states.append((density, evolution))
    
    return states

@pytest.mark.asyncio
async def test_high_volume_state_evolution(mock_high_volume_states):
    """Test gradient interface under high-volume state changes."""
    interface = GradientInterface(dimensions=4)
    previous_state = None
    state_transitions = []
    
    for density, evolution in mock_high_volume_states:
        # Calculate new state
        dims, conf = interface.calculate_interface_gradient(density, evolution)
        current_state = GradientState(
            dimensions=dims,
            confidence=conf,
            timestamp=datetime.now()
        )
        
        if previous_state:
            # Track state transition
            transition = current_state.distance_to(previous_state)
            state_transitions.append(transition)
            
            # Verify state evolution is bounded
            assert transition < np.sqrt(interface.dimensions), \
                "State transition too large"
            
            # Check confidence reflects stability
            assert current_state.confidence <= previous_state.confidence + 0.2, \
                "Confidence increased too rapidly"
            
        previous_state = current_state
        interface.record_state(current_state)
    
    # Verify overall system stability
    assert len(interface.states) == len(mock_high_volume_states), \
        "Missing state recordings"
    
    # Check for reasonable transition distribution
    transitions = np.array(state_transitions)
    assert np.mean(transitions) < 0.5, \
        "Average state transition too large"
    assert np.std(transitions) < 0.3, \
        "State transitions too volatile"

@pytest.mark.asyncio
async def test_near_io_state_detection():
    """Test near-IO state detection during rapid evolution."""
    interface = GradientInterface(dimensions=4)
    
    # Generate sequence of states approaching an IO state
    target_dims = [0.9, 0.9, 0.9, 0.9]  # Ideal IO state
    current_dims = [0.2, 0.3, 0.4, 0.3]  # Starting state
    
    states = generate_state_sequence(current_dims, target_dims, 20)
    near_io_detected = False
    transition_point = None
    
    for i, dims in enumerate(states):
        current_state = GradientState(
            dimensions=dims,
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        target_state = GradientState(
            dimensions=target_dims,
            confidence=1.0,
            timestamp=datetime.now()
        )
        
        is_near, similarity = interface.is_near_io(current_state, target_state)
        
        if is_near and not near_io_detected:
            near_io_detected = True
            transition_point = i
            
        interface.record_state(current_state)
    
    assert near_io_detected, "Never reached near-IO state"
    assert transition_point > 0, "Near-IO detected too early"
    assert transition_point < len(states), "Near-IO detected too late"

@pytest.mark.asyncio
async def test_pattern_aware_rag_simulation():
    """Simulate pattern-aware RAG interface evolution."""
    interface = GradientInterface(dimensions=4)
    
    # Simulate RAG query refinement over time
    query_evolution_steps = 50
    base_dims = [0.5, 0.5, 0.5, 0.5]
    
    # Simulate multiple competing patterns
    pattern_sequences = [
        generate_state_sequence(base_dims, [0.9, 0.8, 0.9, 0.8], query_evolution_steps),
        generate_state_sequence(base_dims, [0.8, 0.9, 0.8, 0.9], query_evolution_steps),
        generate_state_sequence(base_dims, [0.7, 0.7, 0.9, 0.9], query_evolution_steps)
    ]
    
    pattern_states = []
    for step in range(query_evolution_steps):
        # Track competing patterns
        current_patterns = []
        for sequence in pattern_sequences:
            state = GradientState(
                dimensions=sequence[step],
                confidence=0.7 + (step/query_evolution_steps) * 0.2,
                timestamp=datetime.now()
            )
            current_patterns.append(state)
            interface.record_state(state)
        pattern_states.append(current_patterns)
        
        # Verify pattern separation
        for i, pattern1 in enumerate(current_patterns):
            for j, pattern2 in enumerate(current_patterns):
                if i != j:
                    distance = pattern1.distance_to(pattern2)
                    assert distance > 0.1, \
                        f"Patterns {i} and {j} too similar at step {step}"
    
    # Verify evolution stability
    for pattern_idx in range(len(pattern_sequences)):
        pattern_evolution = [
            states[pattern_idx].dimensions 
            for states in pattern_states
        ]
        
        # Check for smooth evolution
        for step in range(1, len(pattern_evolution)):
            change = np.linalg.norm(
                np.array(pattern_evolution[step]) - 
                np.array(pattern_evolution[step-1])
            )
            assert change < 0.1, f"Pattern {pattern_idx} evolved too rapidly"

@pytest.mark.asyncio
async def test_state_space_stability():
    """Test stability of state space under rapid changes."""
    interface = GradientInterface(dimensions=4)
    
    # Simulate rapid state changes
    num_changes = 200
    change_interval = timedelta(milliseconds=50)
    base_time = datetime.now()
    
    states = []
    for i in range(num_changes):
        # Create rapidly changing state
        dims = [
            0.5 + np.sin(i/10 + d) * 0.3
            for d in range(interface.dimensions)
        ]
        
        state = GradientState(
            dimensions=dims,
            confidence=0.7 + np.sin(i/20) * 0.2,
            timestamp=base_time + (change_interval * i)
        )
        states.append(state)
        interface.record_state(state)
    
    # Analyze state space stability
    time_deltas = [
        (states[i+1].timestamp - states[i].timestamp).total_seconds()
        for i in range(len(states)-1)
    ]
    
    state_changes = [
        states[i+1].distance_to(states[i])
        for i in range(len(states)-1)
    ]
    
    # Verify temporal stability
    assert np.mean(time_deltas) > 0, "Time not advancing"
    assert np.std(time_deltas) < 0.1, "Irregular time progression"
    
    # Verify state space stability
    assert np.mean(state_changes) < 0.3, "Average state change too large"
    assert np.std(state_changes) < 0.2, "State changes too volatile"
    
    # Verify system bounds
    all_dims = np.array([s.dimensions for s in states])
    assert np.all(all_dims >= 0) and np.all(all_dims <= 1), \
        "State dimensions outside bounds"
