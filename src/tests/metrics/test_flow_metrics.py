"""Tests for flow-based metric extraction."""

import pytest
from datetime import datetime
from src.core.metrics.flow_metrics import MetricFlowManager, MetricFlow

@pytest.fixture
def flow_manager():
    """Create fresh flow manager for each test."""
    return MetricFlowManager()

def test_metric_extraction():
    """Test basic metric extraction."""
    manager = MetricFlowManager()
    
    text = """
    Temperature increased by 2.5 degrees.
    Precipitation changed by 55%.
    Drought conditions occurred 8.5% of the time.
    Fire danger days increased from 10 to 94.
    """
    
    metrics = manager.extract_metrics(text)
    
    # Check metric count
    assert len(metrics) >= 6
    
    # Verify metric types
    types = {m['type'] for m in metrics}
    assert 'number' in types
    assert 'percentage' in types
    assert 'range' in types
    
    # Check confidence scores
    for metric in metrics:
        assert 0.0 <= metric['confidence'] <= 1.0
        assert 'flow_id' in metric

def test_flow_confidence_calculation():
    """Test flow confidence calculation."""
    manager = MetricFlowManager()
    flow = manager.create_flow("test_pattern")
    
    # Test base confidence
    assert flow.calculate_flow_confidence() > 0.8
    
    # Test with modified metrics
    flow.confidence = 0.8
    flow.viscosity = 0.5
    flow.density = 0.7
    flow.temporal_stability = 0.9
    flow.cross_validation_score = 0.85
    
    confidence = flow.calculate_flow_confidence()
    assert 0.7 <= confidence <= 0.9

def test_context_adjusted_confidence():
    """Test confidence adjustment based on context."""
    manager = MetricFlowManager()
    
    text = "Temperature will increase by 3.5 degrees"
    
    # Test with near-term context
    near_context = {
        'temporal_distance': 5,
        'source_reliability': 0.9,
        'cross_validation_score': 0.85
    }
    near_metrics = manager.extract_metrics(text, near_context)
    
    # Test with far-future context
    far_context = {
        'temporal_distance': 80,
        'source_reliability': 0.7,
        'cross_validation_score': 0.6
    }
    far_metrics = manager.extract_metrics(text, far_context)
    
    # Far future should have lower confidence
    assert near_metrics[0]['confidence'] > far_metrics[0]['confidence']

def test_pattern_confidence_aggregation():
    """Test pattern confidence aggregation."""
    manager = MetricFlowManager()
    
    # Create multiple flows for same pattern
    flow1 = manager.create_flow("test_pattern")
    flow2 = manager.create_flow("test_pattern")
    
    # Update flow metrics
    manager.update_flow_metrics(flow1.flow_id, {
        'confidence': 0.8,
        'viscosity': 0.4,
        'density': 0.9
    })
    
    manager.update_flow_metrics(flow2.flow_id, {
        'confidence': 0.9,
        'viscosity': 0.3,
        'density': 0.95
    })
    
    # Get aggregated confidence
    confidence = manager.get_pattern_confidence("test_pattern")
    assert 0.8 <= confidence <= 0.9

def test_flow_history_tracking():
    """Test flow metric history tracking."""
    manager = MetricFlowManager()
    flow = manager.create_flow("test_pattern")
    
    # Update metrics multiple times
    updates = [
        {'confidence': 0.8, 'viscosity': 0.4},
        {'confidence': 0.85, 'viscosity': 0.35},
        {'confidence': 0.9, 'viscosity': 0.3}
    ]
    
    for update in updates:
        manager.update_flow_metrics(flow.flow_id, update)
        
    # Check history
    assert len(flow.history) == len(updates)
    
    # Verify increasing confidence trend
    confidences = [h['confidence'] for h in flow.history]
    assert all(confidences[i] <= confidences[i+1] for i in range(len(confidences)-1))

def test_unusual_value_handling():
    """Test handling of unusual metric values."""
    manager = MetricFlowManager()
    
    # Test very large number
    large_text = "Temperature increased by 1000000 degrees"
    large_metrics = manager.extract_metrics(large_text)
    
    # Test very small number
    small_text = "Changed by 0.0000001 percent"
    small_metrics = manager.extract_metrics(small_text)
    
    # Unusual values should have lower confidence
    assert large_metrics[0]['confidence'] < 0.8
    assert small_metrics[0]['confidence'] < 0.8

def test_cross_validation_impact():
    """Test impact of cross-validation on confidence."""
    manager = MetricFlowManager()
    
    # Create flow and update with different cross-validation scores
    flow = manager.create_flow("test_pattern")
    
    # Test with high cross-validation
    manager.update_flow_metrics(flow.flow_id, {
        'confidence': 0.8,
        'cross_validation_score': 0.9
    })
    high_confidence = manager.get_flow_confidence(flow.flow_id)
    
    # Test with low cross-validation
    manager.update_flow_metrics(flow.flow_id, {
        'confidence': 0.8,
        'cross_validation_score': 0.5
    })
    low_confidence = manager.get_flow_confidence(flow.flow_id)
    
    # Higher cross-validation should result in higher confidence
    assert high_confidence > low_confidence
