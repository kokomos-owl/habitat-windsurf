"""
Integration tests for Learning Window Control in Pattern-Aware RAG.

These tests validate:
1. Learning window lifecycle and thresholds
2. Back pressure mechanisms
3. Event coordination under learning windows
4. Stability control
"""

import pytest
from datetime import datetime, timedelta
from habitat_evolution.pattern_aware_rag.learning.learning_control import (
    LearningWindow,
    BackPressureController,
    EventCoordinator
)

@pytest.fixture
def event_coordinator():
    """Create a fresh event coordinator for each test."""
    return EventCoordinator(max_queue_size=1000)

@pytest.fixture
def back_pressure():
    """Create a back pressure controller with test settings."""
    return BackPressureController(
        base_delay=0.1,
        max_delay=1.0,
        stability_threshold=0.7,
        window_size=5
    )

class TestLearningWindowControl:
    """Test suite for learning window control and stability."""
    
    async def test_window_lifecycle(self, event_coordinator):
        """Test learning window creation and lifecycle."""
        # Create a 5-minute learning window
        window = event_coordinator.create_learning_window(
            duration_minutes=5,
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes=10
        )
        
        assert window.is_active
        assert not window.is_saturated
        assert window.stability_threshold == 0.7
        assert window.coherence_threshold == 0.6
        assert window.max_changes_per_window == 10
        
        # Simulate changes up to saturation
        for _ in range(10):
            event = await event_coordinator.queue_event(
                event_type="test_change",
                entity_id="test_1",
                data={"change": "test"},
                stability_score=0.8
            )
            assert event is not None
        
        # Verify window is now saturated
        assert window.is_saturated
        
        # Verify window stats
        stats = event_coordinator.get_window_stats()
        assert stats["total_changes"] == 10
        assert stats["stability_avg"] > 0.7
    
    async def test_back_pressure_control(self, back_pressure):
        """Test back pressure mechanism and stability control."""
        # Test initial state
        assert back_pressure.get_current_delay() == back_pressure.base_delay
        
        # Simulate declining stability
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        delays = []
        
        for score in scores:
            back_pressure.record_stability(score)
            delays.append(back_pressure.get_current_delay())
        
        # Verify increasing delays with decreasing stability
        assert all(delays[i] <= delays[i+1] for i in range(len(delays)-1))
        assert delays[-1] > back_pressure.base_delay
    
    async def test_event_coordination(self, event_coordinator):
        """Test event coordination under learning window constraints."""
        # Create active learning window
        window = event_coordinator.create_learning_window(
            duration_minutes=5,
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes=5
        )
        
        # Queue events with varying stability
        events = []
        for i in range(3):
            event = await event_coordinator.queue_event(
                event_type="test_event",
                entity_id=f"entity_{i}",
                data={"test": i},
                stability_score=0.8 - (i * 0.1)  # Decreasing stability
            )
            events.append(event)
        
        # Get pending events
        pending = event_coordinator.get_pending_events(max_events=5)
        assert len(pending) == 3
        
        # Process events
        for event in events:
            event_coordinator.mark_processed(event["id"])
        
        # Verify window stats reflect processed events
        stats = event_coordinator.get_window_stats()
        assert stats["processed_events"] == 3
        assert 0.6 < stats["stability_avg"] < 0.8
    
    async def test_stability_thresholds(self, event_coordinator):
        """Test system behavior around stability thresholds."""
        window = event_coordinator.create_learning_window(
            duration_minutes=5,
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes=10
        )
        
        # Test below threshold
        low_stability_event = await event_coordinator.queue_event(
            event_type="test_event",
            entity_id="test_1",
            data={"test": "data"},
            stability_score=0.6  # Below threshold
        )
        
        # Test above threshold
        high_stability_event = await event_coordinator.queue_event(
            event_type="test_event",
            entity_id="test_2",
            data={"test": "data"},
            stability_score=0.8  # Above threshold
        )
        
        # Get window stats
        stats = event_coordinator.get_window_stats()
        assert stats["below_threshold_count"] == 1
        assert stats["above_threshold_count"] == 1
    
    async def test_concurrent_window_operations(self, event_coordinator):
        """Test concurrent operations within learning window."""
        window = event_coordinator.create_learning_window(
            duration_minutes=5,
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes=10
        )
        
        # Simulate concurrent events
        import asyncio
        
        async def queue_event(id: int, stability: float):
            return await event_coordinator.queue_event(
                event_type="concurrent_test",
                entity_id=f"entity_{id}",
                data={"test": id},
                stability_score=stability
            )
        
        # Queue 5 events concurrently
        events = await asyncio.gather(*[
            queue_event(i, 0.7 + (i * 0.02))
            for i in range(5)
        ])
        
        assert len(events) == 5
        assert all(event is not None for event in events)
        
        # Verify window integrity
        stats = event_coordinator.get_window_stats()
        assert stats["total_changes"] == 5
        assert stats["stability_avg"] > 0.7
