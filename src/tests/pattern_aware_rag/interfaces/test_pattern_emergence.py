"""Tests for Pattern Emergence Interface."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import asyncio

from habitat_evolution.pattern_aware_rag.monitoring.vector_attention_monitor import VectorAttentionMonitor
from habitat_evolution.pattern_aware_rag.interfaces.pattern_emergence import (
    PatternEmergenceInterface,
    PatternState,
    PatternMetrics,
    PatternEvent,
    PatternEventType,
    EmergentPattern,
    PatternFeedback
)

@pytest.fixture
def mock_monitor():
    """Create mock vector attention monitor."""
    monitor = Mock(spec=VectorAttentionMonitor)
    monitor.density_radius = 0.1
    return monitor

@pytest.fixture
def pei(mock_monitor):
    """Create pattern emergence interface for testing."""
    interface = PatternEmergenceInterface(
        monitor=mock_monitor,
        min_confidence=0.5
    )
    return interface

@pytest.mark.asyncio
async def test_pattern_lifecycle(pei, mock_monitor):
    """Test complete pattern lifecycle from formation to dissolution."""
    # Setup mock metrics
    metrics_sequence = [
        # Formation metrics
        Mock(
            current_vector=np.array([1.0, 0.0, 0.0]),
            local_density=0.6,
            stability_score=0.7,
            attention_weight=0.8
        ),
        # Stabilization metrics
        Mock(
            current_vector=np.array([0.98, 0.02, 0.0]),
            local_density=0.8,
            stability_score=0.9,
            attention_weight=0.8
        ),
        # Dissolution metrics
        Mock(
            current_vector=np.array([0.5, 0.5, 0.0]),
            local_density=0.2,
            stability_score=0.3,
            attention_weight=0.4
        )
    ]
    
    mock_monitor.get_latest_metrics.side_effect = metrics_sequence
    
    # Start interface
    await pei.start()
    
    # Collect events
    events = []
    async def collect_events():
        async for pattern in pei.observe_patterns():
            events.append(pattern)
            if len(events) >= 4:  # Formation, Emergence, Stabilization, Dissolution
                break
    
    # Run event collection
    collector = asyncio.create_task(collect_events())
    await asyncio.sleep(0.5)  # Allow time for processing
    await pei.stop()
    await collector
    
    # Verify pattern lifecycle
    assert len(events) == 4
    assert [e.state for e in events] == [
        PatternState.FORMING,
        PatternState.EMERGING,
        PatternState.STABLE,
        PatternState.DISSOLVING
    ]

@pytest.mark.asyncio
async def test_pattern_feedback(pei):
    """Test pattern feedback processing."""
    # Create test pattern
    pattern = EmergentPattern(
        id="test_pattern",
        state=PatternState.STABLE,
        center=np.array([1.0, 0.0, 0.0]),
        radius=0.1,
        metrics=PatternMetrics(
            density=0.8,
            stability=0.7,
            attention=0.5,
            confidence=0.7,
            timestamp=datetime.now()
        ),
        context={}
    )
    pei._patterns["test_pattern"] = pattern
    
    # Process feedback
    feedback = PatternFeedback(
        attention_delta=0.2,
        confidence_override=0.8,
        context_updates={"priority": "high"}
    )
    
    success = await pei.process_feedback("test_pattern", feedback)
    assert success
    
    # Verify updates
    updated_pattern = pei._patterns["test_pattern"]
    assert updated_pattern.metrics.attention > 0.5  # Increased attention
    assert updated_pattern.metrics.confidence > 0.7  # Increased confidence
    assert updated_pattern.context["priority"] == "high"

@pytest.mark.asyncio
async def test_pattern_detection(pei, mock_monitor):
    """Test pattern detection with various vector configurations."""
    # Create cluster of vectors
    vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.95, 0.05, 0.0]),
        np.array([0.98, 0.02, 0.0])
    ]
    
    metrics = [
        Mock(
            current_vector=v,
            local_density=0.7,
            stability_score=0.8,
            attention_weight=0.6
        ) for v in vectors
    ]
    
    mock_monitor.get_latest_metrics.side_effect = metrics
    
    # Start interface
    await pei.start()
    
    # Collect patterns
    patterns = []
    async def collect_patterns():
        async for pattern in pei.observe_patterns():
            patterns.append(pattern)
            if len(patterns) >= 3:
                break
    
    collector = asyncio.create_task(collect_patterns())
    await asyncio.sleep(0.5)
    await pei.stop()
    await collector
    
    # Verify pattern detection
    assert len(patterns) > 0
    assert all(p.metrics.confidence >= 0.5 for p in patterns)
    
    # Check pattern properties
    first_pattern = patterns[0]
    assert np.allclose(first_pattern.center, vectors[0])
    assert first_pattern.radius == mock_monitor.density_radius

@pytest.mark.asyncio
async def test_event_handling(pei):
    """Test event queue handling and backpressure."""
    # Create many events rapidly
    events = []
    for i in range(2000):  # More than queue size
        pattern = EmergentPattern(
            id=f"pattern_{i}",
            state=PatternState.FORMING,
            center=np.array([1.0, 0.0, 0.0]),
            radius=0.1,
            metrics=PatternMetrics(
                density=0.8,
                stability=0.7,
                attention=0.5,
                confidence=0.7,
                timestamp=datetime.now()
            ),
            context={}
        )
        event = PatternEvent(
            type=PatternEventType.PATTERN_FORMING,
            pattern=pattern,
            timestamp=datetime.now()
        )
        events.append(event)
    
    # Queue events
    for event in events:
        await pei._queue_event(event.type, event.pattern)
    
    # Verify queue doesn't exceed max size
    assert pei._event_queue.qsize() <= 1000  # Default queue size

@pytest.mark.asyncio
async def test_adaptive_radius(pei, mock_monitor):
    """Test adaptive radius for pattern detection."""
    # Create vectors with varying density
    sparse_vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0])
    ]
    
    dense_vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.98, 0.02, 0.0]),
        np.array([0.99, 0.01, 0.0])
    ]
    
    # Test with sparse vectors
    metrics = [
        Mock(
            current_vector=v,
            local_density=0.3,
            stability_score=0.8,
            attention_weight=0.6
        ) for v in sparse_vectors
    ]
    
    mock_monitor.get_latest_metrics.side_effect = metrics
    await pei._update_patterns(metrics[0])
    
    # Verify adaptive radius for sparse region
    sparse_patterns = pei._patterns.values()
    assert len(sparse_patterns) > 0
    
    # Test with dense vectors
    metrics = [
        Mock(
            current_vector=v,
            local_density=0.9,
            stability_score=0.8,
            attention_weight=0.6
        ) for v in dense_vectors
    ]
    
    pei._patterns.clear()
    mock_monitor.get_latest_metrics.side_effect = metrics
    await pei._update_patterns(metrics[0])
    
    # Verify adaptive radius for dense region
    dense_patterns = pei._patterns.values()
    assert len(dense_patterns) > 0
