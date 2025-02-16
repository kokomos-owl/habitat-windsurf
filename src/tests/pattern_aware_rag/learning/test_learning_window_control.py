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
    
    def test_window_lifecycle(self, event_coordinator):
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
        delays = []
        for _ in range(10):
            delay = event_coordinator.queue_event(
                event_type="test_change",
                entity_id="test_1",
                data={"change": "test"},
                stability_score=0.8
            )
            delays.append(delay)
            assert isinstance(delay, float)
        
        # Verify window is now saturated
        assert window.is_saturated
        
        # Verify window stats
        stats = event_coordinator.get_window_stats()
        assert stats["change_count"] == 10
        assert stats["is_saturated"] == True
        assert "current_pressure" in stats
    
    async def test_back_pressure_control(self, back_pressure):
        """Test back pressure mechanism and stability control."""
        # Simulate declining stability
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        delays = []
        
        for score in scores:
            delay = back_pressure.calculate_delay(score)
            delays.append(delay)
        
        # Verify increasing delays with decreasing stability
        assert all(delays[i] <= delays[i+1] for i in range(len(delays)-1))
        assert delays[-1] > back_pressure.base_delay
        assert delays[-1] <= back_pressure.max_delay
    
    def test_event_coordination(self, event_coordinator):
        """Test event coordination under learning window constraints."""
        # Create active learning window
        window = event_coordinator.create_learning_window(
            duration_minutes=5,
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes=5
        )
        
        # Queue events with varying stability
        delays = []
        event_ids = []
        for i in range(3):
            delay = event_coordinator.queue_event(
                event_type="test_event",
                entity_id=f"entity_{i}",
                data={"test": i},
                stability_score=0.8 - (i * 0.1)  # Decreasing stability
            )
            delays.append(delay)
            event_ids.append(f"test_event_entity_{i}_{datetime.now().isoformat()}")
        
        # Verify delays increase with decreasing stability
        assert all(delays[i] <= delays[i+1] for i in range(len(delays)-1))
        
        # Get pending events
        pending = event_coordinator.get_pending_events(max_events=5)
        assert len(pending) == 3
        
        # Process events
        for event_id in event_ids:
            event_coordinator.mark_processed(event_id)
        
        # Verify window stats
        stats = event_coordinator.get_window_stats()
        assert stats["change_count"] == 3
        assert stats["pending_events"] == 0
    
    def test_stability_thresholds(self, event_coordinator):
        """Test system behavior around stability thresholds."""
        window = event_coordinator.create_learning_window(
            duration_minutes=5,
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes=10
        )
        
        # Test below threshold
        # Test below threshold
        low_delay = event_coordinator.queue_event(
            event_type="test_event",
            entity_id="test_1",
            data={"test": "data"},
            stability_score=0.6  # Below threshold
        )
        
        # Test above threshold
        high_delay = event_coordinator.queue_event(
            event_type="test_event",
            entity_id="test_2",
            data={"test": "data"},
            stability_score=0.8  # Above threshold
        )
        
        # Verify both events were queued
        stats = event_coordinator.get_window_stats()
        assert stats["change_count"] == 2
        
        # Get window stats
        stats = event_coordinator.get_window_stats()
        assert stats["change_count"] == 2
        assert stats["pending_events"] == 2
        assert "current_pressure" in stats
    
    def test_stability_gradients(self, event_coordinator):
        """Test granular stability responses with fine-grained gradients."""
        window = event_coordinator.create_learning_window(
            duration_minutes=5,
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes=20
        )
        
        # Test fine-grained stability scores
        gradient_scores = [
            0.95,  # Highly stable
            0.85,  # Very stable
            0.75,  # Above threshold
            0.70,  # At threshold
            0.65,  # Below threshold
            0.55,  # Unstable
            0.45   # Very unstable
        ]
        
        delays = []
        for score in gradient_scores:
            delay = event_coordinator.queue_event(
                event_type="gradient_test",
                entity_id=f"test_{score}",
                data={"stability": score},
                stability_score=score
            )
            delays.append(delay)
        
        # Verify delay gradients
        # 1. Delays should increase as stability decreases
        assert all(delays[i] <= delays[i+1] for i in range(len(delays)-1))
        
        # 2. Verify delay differences are proportional
        delay_diffs = [delays[i+1] - delays[i] for i in range(len(delays)-1)]
        assert all(diff >= 0 for diff in delay_diffs), "Delays should never decrease"
        
        # 3. Check gradient sensitivity around threshold
        threshold_idx = gradient_scores.index(0.70)
        below_threshold = delays[threshold_idx + 1] - delays[threshold_idx]
        above_threshold = delays[threshold_idx] - delays[threshold_idx - 1]
        assert below_threshold > above_threshold, "Larger delay increase below threshold"
        
        # Get final stats
        stats = event_coordinator.get_window_stats()
        assert stats["change_count"] == len(gradient_scores)
        assert not stats["is_saturated"]
    
    def test_concurrent_window_operations(self, event_coordinator):
        """Test concurrent operations within learning window."""
        window = event_coordinator.create_learning_window(
            duration_minutes=5,
            stability_threshold=0.7,
            coherence_threshold=0.6,
            max_changes=10
        )
        
        # Queue multiple events with increasing stability
        delays = []
        for i in range(5):
            delay = event_coordinator.queue_event(
                event_type="concurrent_test",
                entity_id=f"entity_{i}",
                data={"test": i},
                stability_score=0.7 + (i * 0.02)
            )
            delays.append(delay)
        
        # Verify delays decrease with increasing stability
        assert all(delays[i] >= delays[i+1] for i in range(len(delays)-1))
        # Verify window integrity
        stats = event_coordinator.get_window_stats()
        assert stats["change_count"] == 5
        assert stats["pending_events"] == 5
        assert not stats["is_saturated"]
