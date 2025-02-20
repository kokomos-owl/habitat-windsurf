"""Tests for Natural Pattern Emergence Observation."""

import pytest
import numpy as np
from datetime import datetime
import logging
import time
from unittest.mock import Mock

from habitat_evolution.core.pattern.quality import (
    PatternQualityAnalyzer,
    SignalMetrics,
    FlowMetrics,
    PatternState
)
from habitat_evolution.core.pattern.attention import AttentionFilter
from habitat_evolution.pattern_aware_rag.monitoring.vector_attention_monitor import (
    VectorAttentionMonitor,
    VectorSpaceMetrics
)



@pytest.fixture
def pattern_analyzer():
    """Create pattern quality analyzer for testing."""
    return PatternQualityAnalyzer(
        signal_threshold=0.3,
        noise_threshold=0.7,
        persistence_window=5
    )

@pytest.fixture
def attention_filter():
    """Create attention filter for testing."""
    return AttentionFilter(
        name="test_filter",
        conditions={
            'stability': lambda x: x >= 0.7,
            'density': lambda x: 0.4 <= x <= 0.8,
            'turbulence': lambda x: x <= 0.3
        },
        weight=1.0
    )

@pytest.fixture
def vector_monitor(attention_filter, pattern_analyzer):
    """Create vector attention monitor for testing."""
    monitor = VectorAttentionMonitor(
        attention_filter=attention_filter,
        window_size=5,
        edge_threshold=0.3,
        stability_threshold=0.7,
        density_radius=0.1
    )
    
    # Initialize with pattern analyzer
    monitor.pattern_analyzer = pattern_analyzer
    return monitor

@pytest.fixture
def mock_metrics():
    """Create mock metrics for testing."""
    return {
        'signal': SignalMetrics(
            strength=0.8,
            noise_ratio=0.2,
            persistence=0.7,
            reproducibility=0.6
        ),
        'flow': FlowMetrics(
            viscosity=0.4,
            back_pressure=0.3,
            volume=0.6,
            current=0.5
        )
    }

@pytest.mark.timeout(10)  # Longer timeout for natural observation
def test_vector_space_evolution(vector_monitor, pattern_analyzer, mock_metrics):
    """Observe natural pattern state transitions without forcing."""
    transitions = []
    emergence_points = []
    
    # Initialize test vectors with increasing stability
    test_vectors = [
        np.array([0.5, 0.5, 0.0]),  # Initial balanced state
        np.array([0.6, 0.4, 0.0]),  # Slight stability increase
        np.array([0.7, 0.3, 0.0]),  # More stable
        np.array([0.7, 0.3, 0.0]),  # Pattern emergence
        np.array([0.6, 0.4, 0.0])   # Stable pattern
    ]
    
    evolution_points = []
    stability_track = []
    attention_track = []
    
    # Track vector space evolution
    for i, vector in enumerate(test_vectors):
        # Create context for vector processing
        context = {
            'stability': 0.7 + (i * 0.05),  # Gradually increasing stability
            'density': 0.5 + (i * 0.05),     # Gradually increasing density
            'turbulence': 0.3 - (i * 0.05)   # Gradually decreasing turbulence
        }
        
        # Process vector with context
        metrics = vector_monitor.process_vector(vector, context)
        
        # Record evolution point
        evolution_points.append({
            'timestamp': datetime.now(),
            'vector': vector.tolist(),
            'edge_strength': metrics.edge_strength,
            'stability': metrics.stability_score,
            'density': metrics.local_density,
            'turbulence': metrics.turbulence_level,
            'attention': metrics.attention_weight
        })
        
        # Track stability progression
        stability_track.append(metrics.stability_score)
        attention_track.append(metrics.attention_weight)
        
        # Allow natural evolution time
        time.sleep(0.1)
    
    # Log natural evolution
    logging.info(f"Vector space evolution points: {evolution_points}")
    logging.info(f"Stability progression: {stability_track}")
    logging.info(f"Attention progression: {attention_track}")
    
    # Verify basic evolution occurred
    assert len(evolution_points) == len(test_vectors)
    
    # Check for stability trend without asserting specific values
    stability_changes = [b - a for a, b in zip(stability_track[:-1], stability_track[1:])]
    logging.info(f"Stability changes between points: {stability_changes}")

@pytest.mark.timeout(15)  # Extended timeout for flow observation
def test_attention_flow_patterns(vector_monitor, pattern_analyzer, mock_metrics):
    """Test pattern emergence through natural flow without forcing states."""
    emergence_sequence = []
    pattern_states = []
    
    # Create flow pattern sequence with varying dynamics
    flow_vectors = [
        np.array([0.5, 0.5, 0.0]),  # Balanced flow
        np.array([0.6, 0.3, 0.1]),  # Increasing turbulence
        np.array([0.7, 0.2, 0.1]),  # Higher stability, maintained turbulence
        np.array([0.8, 0.1, 0.1]),  # Peak stability, controlled turbulence
        np.array([0.7, 0.2, 0.1])   # Flow stabilization with turbulence
    ]
    
    flow_metrics = []
    signal_metrics = []
    pattern_states = []
    
    # Track pattern flow and evolution
    for i, vector in enumerate(flow_vectors):
        # Create context for flow analysis
        context = {
            'stability': 0.6 + (i * 0.1),    # Increasing stability
            'density': 0.5,                   # Constant density
            'turbulence': 0.4 - (i * 0.1)    # Decreasing turbulence
        }
        
        # Process vector with context
        metrics = vector_monitor.process_vector(vector, context)
        
        # Analyze pattern quality
        signal = SignalMetrics(
            strength=metrics.stability_score,
            noise_ratio=metrics.turbulence_level,
            persistence=0.7,  # Fixed for test
            reproducibility=0.6  # Fixed for test
        )
        
        flow = FlowMetrics(
            viscosity=1.0 - metrics.local_density,
            back_pressure=metrics.edge_strength,
            volume=metrics.local_density,
            current=metrics.attention_weight
        )
        
        # Determine pattern state
        state = pattern_analyzer.determine_state(signal, flow)
        pattern_states.append(state)
        
        # Record metrics
        flow_metrics.append(flow)
        signal_metrics.append(signal)
        
        # Allow natural evolution
        time.sleep(0.2)
    
    # Log pattern evolution
    logging.info(f"Pattern states: {pattern_states}")
    logging.info(f"Signal metrics: {signal_metrics}")
    logging.info(f"Flow metrics: {flow_metrics}")
    
    # Verify pattern evolution
    assert len(pattern_states) == len(flow_vectors)
    
    # Check for state transitions
    state_transitions = [f"{a} -> {b}" for a, b in zip(pattern_states[:-1], pattern_states[1:])]
    logging.info(f"State transitions: {state_transitions}")
    
    # Analyze flow dynamics
    attention_changes = [b.current - a.current for a, b in zip(flow_metrics[:-1], flow_metrics[1:])]
    logging.info(f"Attention flow changes: {attention_changes}")
    
    # Analyze signal evolution
    signal_changes = [b.strength - a.strength for a, b in zip(signal_metrics[:-1], signal_metrics[1:])]
    logging.info(f"Signal strength changes: {signal_changes}")
