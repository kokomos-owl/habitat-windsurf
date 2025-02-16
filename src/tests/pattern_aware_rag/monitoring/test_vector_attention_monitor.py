"""
Tests for vector-based attention monitoring system.

Validates:
1. Edge detection in semantic space
2. Stability analysis with attention weights
3. Density-based pattern detection
4. Turbulence monitoring
5. Drift tracking
6. Metric logging for navigation
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
from habitat_evolution.pattern_aware_rag.monitoring.vector_attention_monitor import (
    VectorAttentionMonitor,
    VectorSpaceMetrics
)
from habitat_evolution.core.pattern.attention import AttentionFilter

@pytest.fixture
def attention_filter():
    """Create test attention filter."""
    return AttentionFilter(
        name="test_filter",
        conditions={"test": lambda x: 0.8},
        weight=1.0
    )

@pytest.fixture
def vector_monitor(attention_filter):
    """Create vector attention monitor for testing."""
    return VectorAttentionMonitor(
        attention_filter=attention_filter,
        window_size=5,
        edge_threshold=0.3,
        stability_threshold=0.7,
        density_radius=0.1
    )

class TestVectorAttentionMonitor:
    """Test suite for vector attention monitoring."""
    
    def test_edge_detection(self, vector_monitor):
        """Test semantic edge detection with attention weights."""
        # Create vectors with clear edge
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        
        # Process first vector
        metrics1 = vector_monitor.process_vector(v1, {"test": True})
        assert metrics1.edge_strength == 0.0  # First vector has no edge
        
        # Process second vector (should detect edge)
        metrics2 = vector_monitor.process_vector(v2, {"test": True})
        assert metrics2.edge_strength > vector_monitor.edge_threshold
        assert metrics2.attention_weight == 0.8  # From attention filter
    
    def test_stability_analysis(self, vector_monitor):
        """Test stability analysis with vector field characteristics."""
        # Generate stable sequence
        stable_vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.98, 0.02, 0.0]),
            np.array([0.97, 0.03, 0.0]),
            np.array([0.99, 0.01, 0.0])
        ]
        
        # Process stable sequence
        metrics = []
        for v in stable_vectors:
            m = vector_monitor.process_vector(v, {"test": True})
            metrics.append(m)
        
        # Verify stability scores
        assert all(m.stability_score > vector_monitor.stability_threshold for m in metrics[1:])
        
        # Generate unstable sequence
        unstable_vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0])
        ]
        
        # Process unstable sequence
        metrics = []
        for v in unstable_vectors:
            m = vector_monitor.process_vector(v, {"test": True})
            metrics.append(m)
        
        # Verify instability detection
        assert any(m.stability_score < vector_monitor.stability_threshold for m in metrics[1:])
    
    def test_density_patterns(self, vector_monitor):
        """Test density-based pattern detection."""
        # Create cluster of vectors
        cluster = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.95, 0.05, 0.0]),
            np.array([0.98, 0.02, 0.0]),
            np.array([0.97, 0.03, 0.0])
        ]
        
        # Process cluster
        metrics = []
        for v in cluster:
            m = vector_monitor.process_vector(v, {"test": True})
            metrics.append(m)
        
        # Verify density increases with more vectors in cluster
        densities = [m.local_density for m in metrics]
        assert all(densities[i] <= densities[i+1] for i in range(len(densities)-1))
    
    def test_turbulence_detection(self, vector_monitor):
        """Test turbulence detection in vector dynamics."""
        # Generate laminar flow
        laminar = [
            np.array([float(i), 0.0, 0.0]) for i in range(5)
        ]
        
        # Process laminar flow
        laminar_metrics = []
        for v in laminar:
            m = vector_monitor.process_vector(v, {"test": True})
            laminar_metrics.append(m)
        
        # Generate turbulent flow
        turbulent = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0])
        ]
        
        # Process turbulent flow
        turbulent_metrics = []
        for v in turbulent:
            m = vector_monitor.process_vector(v, {"test": True})
            turbulent_metrics.append(m)
        
        # Verify turbulence detection
        max_laminar = max(m.turbulence_level for m in laminar_metrics)
        min_turbulent = min(m.turbulence_level for m in turbulent_metrics)
        assert max_laminar < min_turbulent
    
    def test_drift_tracking(self, vector_monitor):
        """Test semantic drift tracking."""
        # Create sequence with consistent drift
        drift_sequence = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.1, 0.0]),
            np.array([0.2, 0.2, 0.0]),
            np.array([0.3, 0.3, 0.0])
        ]
        
        # Process drift sequence
        metrics = []
        for v in drift_sequence:
            m = vector_monitor.process_vector(v, {"test": True})
            metrics.append(m)
        
        # Verify drift detection
        final_drift = metrics[-1].drift_velocity
        assert np.all(final_drift > 0)  # Positive drift in x and y
    
    @patch('habitat_evolution.pattern_aware_rag.monitoring.vector_attention_monitor.MetricsLogger')
    def test_metric_logging(self, mock_logger, vector_monitor):
        """Test metric logging for monitoring and navigation."""
        # Process vector
        vector = np.array([1.0, 0.0, 0.0])
        metrics = vector_monitor.process_vector(vector, {"test": True})
        
        # Verify metrics were logged
        mock_logger.return_value.log_metrics.assert_called_once()
        logged_data = mock_logger.return_value.log_metrics.call_args[0][1]
        
        # Verify logged data structure
        assert "timestamp" in logged_data
        assert "edge_strength" in logged_data
        assert "stability_score" in logged_data
        assert "local_density" in logged_data
        assert "turbulence_level" in logged_data
        assert "drift_magnitude" in logged_data
        assert "attention_weight" in logged_data
