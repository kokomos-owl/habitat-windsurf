"""Integration tests for climate metric extraction system."""

import pytest
from datetime import datetime, timedelta
import json
from pathlib import Path

from src.core.processor import ClimateRiskProcessor
from src.core.metrics.pattern_recognition import ClimatePatternRecognizer
from src.core.metrics.temporal_stability import TemporalStabilityTracker
from src.core.metrics.flow_metrics import MetricFlowManager

@pytest.fixture
def test_document(tmp_path):
    """Create a test document with climate metrics."""
    content = """
    Climate Risk Assessment - Martha's Vineyard 2025-2100
    
    Temperature Changes:
    - Average temperature will increase by 2.5°C by 2050
    - Summer temperatures may rise between 1.8°C to 3.2°C
    - Winter temperature changes are projected to be +1.5°C
    
    Precipitation:
    - Annual rainfall will increase by 15% by mid-century
    - Heavy precipitation events increased by 25% since 2000
    - Winter precipitation may change by -5% to +20%
    
    Sea Level:
    - Sea level rise is projected to be 0.5m by 2050
    - Storm surge heights increased by 0.3m
    - Extreme water levels may rise by 0.8m to 1.2m
    
    Extreme Events:
    - Drought frequency increased to 3 times per decade
    - Heat waves occur 5.5 times more frequently
    - Flood events increased by 45% in frequency
    """
    
    doc_path = tmp_path / "test_climate_risk.txt"
    doc_path.write_text(content)
    return doc_path

@pytest.fixture
def processor():
    """Create climate risk processor."""
    return ClimateRiskProcessor()

@pytest.mark.asyncio
async def test_document_processing(processor, test_document):
    """Test end-to-end document processing."""
    result = await processor.process_document(str(test_document))
    
    # Check basic processing
    assert result.metrics
    assert not result.validation_errors
    assert result.evolution_metrics
    
    # Verify metric types
    metric_types = {m.risk_type for m in result.metrics}
    expected_types = {'temperature', 'precipitation', 'sea_level'}
    assert metric_types.intersection(expected_types)
    
    # Check confidence levels
    assert all(m.confidence > 0.6 for m in result.metrics)
    
    # Verify evolution metrics
    assert result.evolution_metrics.stability > 0
    assert result.evolution_metrics.trend in {'stable', 'improving', 'degrading', 'unknown'}

def test_pattern_recognition(test_document):
    """Test pattern recognition capabilities."""
    recognizer = ClimatePatternRecognizer()
    content = Path(test_document).read_text()
    
    matches = recognizer.find_patterns(content)
    
    # Check pattern types
    pattern_types = {m.pattern_type for m in matches}
    assert 'temperature_change' in pattern_types
    assert 'precipitation_change' in pattern_types
    assert 'sea_level_rise' in pattern_types
    
    # Verify context extraction
    for match in matches:
        assert match.context
        if 'temporal_indicators' in match.context:
            assert match.context['temporal_indicators']
        assert 'surrounding_text' in match.context

def test_temporal_stability():
    """Test temporal stability tracking."""
    tracker = TemporalStabilityTracker()
    
    # Add observations over time
    base_time = datetime.now()
    for i in range(5):
        time = base_time + timedelta(days=i*30)
        tracker.add_observation(time, 'temperature', 25.0 + i*0.5)
    
    # Check stability scores
    score = tracker.get_stability_score('temperature')
    assert 0 <= score <= 1
    
    # Verify trend detection
    report = tracker.get_stability_report('temperature')
    assert report['trend'] in {'stable', 'improving', 'degrading', 'unknown'}
    assert 'trend_confidence' in report

def test_flow_metrics():
    """Test flow metric management."""
    manager = MetricFlowManager()
    
    # Create flows
    flow1 = manager.create_flow('temperature')
    flow2 = manager.create_flow('precipitation')
    
    # Update metrics
    manager.update_flow_metrics(flow1.flow_id, {
        'confidence': 0.8,
        'viscosity': 0.4,
        'density': 0.9
    })
    
    manager.update_flow_metrics(flow2.flow_id, {
        'confidence': 0.75,
        'viscosity': 0.5,
        'density': 0.85
    })
    
    # Check pattern confidence
    temp_conf = manager.get_pattern_confidence('temperature')
    precip_conf = manager.get_pattern_confidence('precipitation')
    
    assert 0.7 <= temp_conf <= 0.9
    assert 0.7 <= precip_conf <= 0.9

@pytest.mark.asyncio
async def test_confidence_evolution(processor, test_document):
    """Test confidence evolution over multiple processes."""
    results = []
    
    # Process document multiple times
    for _ in range(3):
        result = await processor.process_document(str(test_document))
        results.append(result)
    
    # Check confidence evolution
    confidences = [
        [m.confidence for m in result.metrics]
        for result in results
        if result.metrics  # Only include results with metrics
    ]
    
    # Verify we have results to compare
    assert len(confidences) > 1, "Not enough results with metrics to compare confidence evolution"
    
    # Later runs should maintain or improve confidence
    for conf, prev in zip(confidences[1:], confidences[:-1]):
        if not conf or not prev:  # Skip if either list is empty
            continue
        assert sum(conf)/len(conf) >= sum(prev)/len(prev), "Confidence decreased over time"

def test_visualization_data(processor, test_document):
    """Test generation of visualization data."""
    manager = MetricFlowManager()
    
    # Create and update flows
    flow_data = []
    base_time = datetime.now()
    
    for i in range(5):
        flow = manager.create_flow(f'metric_{i}')
        time = base_time + timedelta(days=i)
        
        metrics = {
            'confidence': 0.7 + i*0.05,
            'viscosity': 0.5 - i*0.02,
            'density': 0.8 + i*0.03
        }
        
        manager.update_flow_metrics(flow.flow_id, metrics)
        
        flow_data.append({
            'timestamp': time.isoformat(),
            'metricType': f'metric_{i}',
            **metrics
        })
    
    # Verify visualization data structure
    assert all('timestamp' in d for d in flow_data)
    assert all('confidence' in d for d in flow_data)
    assert all('viscosity' in d for d in flow_data)
    assert all('density' in d for d in flow_data)
    
    # Check data ranges
    confidences = [d['confidence'] for d in flow_data]
    assert all(0 <= c <= 1 for c in confidences)
    assert len(set(confidences)) > 1  # Ensure variation
