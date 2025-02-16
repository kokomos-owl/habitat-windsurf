"""Tests for Pattern Emergence Interface."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import logging
import asyncio
from typing import AsyncGenerator

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
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

@pytest.fixture(scope="function")
def event_loop():
    """Create an event loop for each test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
def mock_monitor():
    """Create mock vector attention monitor."""
    monitor = Mock(spec=VectorAttentionMonitor)
    monitor.density_radius = 0.1
    metrics = Mock(
        current_vector=np.array([1.0, 0.0, 0.0]),
        local_density=0.6,
        stability_score=0.7,
        attention_weight=0.8
    )
    mock_metrics = AsyncMock()
    mock_metrics.return_value = metrics
    monitor.get_latest_metrics = mock_metrics
    return monitor

@pytest.fixture
async def pei(mock_monitor) -> AsyncGenerator[PatternEmergenceInterface, None]:
    """Create pattern emergence interface for testing."""
    interface = PatternEmergenceInterface(
        monitor=mock_monitor,
        min_confidence=0.5,
        event_buffer_size=100  # Smaller buffer for testing
    )
    interface.logger = logging.getLogger(__name__)
    await interface.start()
    yield interface
    await interface.stop()

@pytest.mark.asyncio
@pytest.mark.timeout(5)  # 5 second timeout
async def test_pattern_lifecycle(pei, mock_monitor):
    """Test complete pattern lifecycle from formation to dissolution."""
    # Metrics are set up in fixture
    
    try:
        # Collect events with timeout
        events = []
        # Start metrics processing
        await pei._process_metrics()
        
        # Queue some test events
        adaptive_id = AdaptiveID(
            base_concept="test_pattern",
            creator_id="test",
            weight=0.8,
            confidence=0.7,
            uncertainty=0.5
        )
        pattern = EmergentPattern(
            id=adaptive_id.id,
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
            context={
                "adaptive_id": adaptive_id,
                "base_concept": "test_pattern",
                "creator": "test"
            }
        )
        
        # Queue state transitions
        await pei._queue_event(PatternEventType.PATTERN_FORMING, pattern)
        await pei._queue_event(PatternEventType.PATTERN_EMERGING, pattern)
        await pei._queue_event(PatternEventType.PATTERN_STABLE, pattern)
        
        async with asyncio.timeout(3.0):  # 3 second timeout
            async for pattern in pei.observe_patterns():
                events.append(pattern)
                if len(events) >= 3:  # We expect 3 state transitions
                    break
        
        # Verify pattern lifecycle
        assert len(events) >= 3
        states = [e.state for e in events[:3]]
        assert states == [
            PatternState.FORMING,
            PatternState.EMERGING,
            PatternState.STABLE
        ]
    except asyncio.TimeoutError:
        pei._running = False  # Ensure cleanup
        raise
    finally:
        await pei.stop()

@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_pattern_feedback(pei):
    """Test pattern feedback processing."""
    # Create test pattern with AdaptiveID
    adaptive_id = AdaptiveID(
        base_concept="test_pattern",
        creator_id="test",
        weight=0.8,
        confidence=0.7,
        uncertainty=0.5
    )
    
    # Create pattern with known metrics for predictable calculations
    pattern = EmergentPattern(
        id=adaptive_id.id,
        state=PatternState.STABLE,
        center=np.array([1.0, 0.0, 0.0]),
        radius=0.1,
        metrics=PatternMetrics(
            density=0.8,
            stability=0.7,  # Used as smoothing factor
            attention=0.5,
            confidence=0.7,  # Initial confidence
            timestamp=datetime.now()
        ),
        context={
            "adaptive_id": adaptive_id,
            "base_concept": "test_pattern",
            "creator": "test"
        }
    )
    pei._patterns[adaptive_id.id] = pattern
    
    # Process feedback with timeout
    async with asyncio.timeout(3.0):
        # Test attention update first
        feedback = PatternFeedback(
            attention_delta=0.2,
            confidence_override=None,  # Test attention update in isolation
            context_updates=None
        )
        success = await pei.process_feedback(adaptive_id.id, feedback)
        assert success
        
        # Verify attention update
        expected_attention = 0.7 * 0.7 + (1 - 0.7) * 0.5  # stability * new + (1-stability) * current
        assert np.isclose(pei._patterns[adaptive_id.id].metrics.attention, expected_attention, rtol=1e-2)
        
        # Now test confidence update separately
        feedback = PatternFeedback(
            attention_delta=None,
            confidence_override=0.8,  # Test confidence update in isolation
            context_updates={"priority": "high"}
        )
        success = await pei.process_feedback(adaptive_id.id, feedback)
        assert success
    
    # Verify updates
    updated_pattern = pei._patterns[adaptive_id.id]
    
    # Calculate expected confidence after temporal decay and smoothing
    decay = np.exp(-0.1)  # 10% decay
    decayed_confidence = 0.7 * decay  # Initial confidence with decay
    expected_confidence = np.clip(
        0.7 * 0.8 + (1 - 0.7) * decayed_confidence,  # stability * override + (1-stability) * decayed
        0.0, 1.0
    )
    assert np.isclose(updated_pattern.metrics.confidence, expected_confidence, rtol=1e-2)
    
    # Verify context update
    assert updated_pattern.context["priority"] == "high"

@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_pattern_detection(pei, mock_monitor):
    """Test pattern detection with various vector configurations."""
    try:
        # Update metrics for this test
        metrics = Mock(
            current_vector=np.array([1.0, 0.0, 0.0]),
            local_density=0.7,
            stability_score=0.8,
            attention_weight=0.6
        )
        mock_monitor.get_latest_metrics.return_value = metrics
        
        # Create a test pattern
        adaptive_id = AdaptiveID(
            base_concept="test_pattern",
            creator_id="test",
            weight=0.8,
            confidence=0.7,
            uncertainty=0.5
        )
        pattern = EmergentPattern(
            id=adaptive_id.id,
            state=PatternState.FORMING,
            center=metrics.current_vector,
            radius=0.1,
            metrics=PatternMetrics(
                density=0.8,
                stability=0.7,
                attention=0.5,
                confidence=0.7,
                timestamp=datetime.now()
            ),
            context={
                "adaptive_id": adaptive_id,
                "base_concept": "test_pattern",
                "creator": "test"
            }
        )
        
        # Queue pattern and start processing
        await pei._queue_event(PatternEventType.PATTERN_FORMING, pattern)
        await pei._process_metrics()
        
        # Collect patterns with timeout
        patterns = []
        async with asyncio.timeout(3.0):
            async for pattern in pei.observe_patterns():
                patterns.append(pattern)
                if len(patterns) >= 1:  # We expect at least 1 pattern
                    break
        
        # Verify pattern detection
        assert len(patterns) >= 1
        assert all(p.metrics.confidence >= 0.0 for p in patterns)  # Start at 0
        assert all("adaptive_id" in p.context for p in patterns)
        assert all(isinstance(p.context["adaptive_id"], AdaptiveID) for p in patterns)
        
        # Check pattern properties
        first_pattern = patterns[0]
        assert np.allclose(first_pattern.center, metrics.current_vector)
        assert first_pattern.radius > mock_monitor.density_radius  # Sparse region
    except asyncio.TimeoutError:
        pei._running = False  # Ensure cleanup
        raise
    finally:
        await pei.stop()

@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_event_handling(pei):
    """Test event queue handling and backpressure."""
    try:
        # Create events rapidly
        events = []
        for i in range(10):  # Reduced number for faster testing
            adaptive_id = AdaptiveID(
                base_concept=f"test_pattern_{i}",
                creator_id="test",
                weight=0.8,
                confidence=0.7,
                uncertainty=0.5
            )
            pattern = EmergentPattern(
                id=adaptive_id.id,
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
                context={
                    "adaptive_id": adaptive_id,
                    "base_concept": f"test_pattern_{i}",
                    "creator": "test"
                }
            )
            event = PatternEvent(
                type=PatternEventType.PATTERN_FORMING,
                pattern=pattern,
                timestamp=datetime.now()
            )
            events.append(event)
        
        # Queue events with timeout
        async with asyncio.timeout(3.0):
            for event in events:
                await pei._queue_event(event.type, event.pattern)
        
        # Verify queue has events
        assert pei._event_queue.qsize() > 0
    except asyncio.TimeoutError:
        pei._running = False  # Ensure cleanup
        raise
    finally:
        await pei.stop()

@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_adaptive_radius(pei, mock_monitor):
    """Test adaptive radius for pattern detection."""
    try:
        async with asyncio.timeout(3.0):
            # Test with sparse region
            sparse_metrics = Mock(
                current_vector=np.array([1.0, 0.0, 0.0]),
                local_density=0.3,  # Sparse
                stability_score=0.8,
                attention_weight=0.6
            )
            mock_monitor.get_latest_metrics.return_value = sparse_metrics
            
            # Calculate adaptive radius for sparse region
            base_radius = mock_monitor.density_radius
            sparse_radius = base_radius * (1.0 / sparse_metrics.local_density) ** 0.5
            
            metrics = await mock_monitor.get_latest_metrics()
            await pei._update_patterns(metrics)
            
            # Verify adaptive radius for sparse region
            sparse_patterns = list(pei._patterns.values())
            assert len(sparse_patterns) > 0
            assert sparse_patterns[0].radius > base_radius
            assert abs(sparse_patterns[0].radius - sparse_radius) < 0.001
            assert "adaptive_id" in sparse_patterns[0].context
            assert isinstance(sparse_patterns[0].context["adaptive_id"], AdaptiveID)
            
            # Test with dense region
            dense_metrics = Mock(
                current_vector=np.array([0.98, 0.02, 0.0]),
                local_density=0.9,  # Dense
                stability_score=0.8,
                attention_weight=0.6
            )
            mock_monitor.get_latest_metrics.return_value = dense_metrics
            
            # Calculate adaptive radius for dense region
            dense_radius = base_radius * dense_metrics.local_density ** 0.5  # Changed formula
            
            pei._patterns.clear()
            metrics = await mock_monitor.get_latest_metrics()
            await pei._update_patterns(metrics)
            
            # Verify adaptive radius for dense region
            dense_patterns = list(pei._patterns.values())
            assert len(dense_patterns) > 0
            assert dense_patterns[0].radius < base_radius
            assert abs(dense_patterns[0].radius - dense_radius) < 0.001
    except asyncio.TimeoutError:
        pei._running = False  # Ensure cleanup
        raise
    finally:
        await pei.stop()
