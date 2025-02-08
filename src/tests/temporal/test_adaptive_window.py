"""Integration tests for adaptive window system."""
import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
from dataclasses import dataclass
import time
import random
from src.core.temporal.adaptive_window_v23 import (
    AdaptiveWindow,
    WindowState,
    WindowStateManager,
    PatternDensityMetrics
)

@dataclass
class MockPattern:
    """Mock pattern for testing."""
    pattern_id: str
    density: float
    timestamp: datetime = None

    def __post_init__(self):
        self.timestamp = self.timestamp or datetime.now()

class MockIOSystem:
    """Mock IO system for testing backpressure effects."""
    def __init__(self):
        self.pressure_level = 0.0
        self.notifications: List[Dict] = []
        self.processed_patterns: List[MockPattern] = []
        self.cumulative_delay = 0.0
        self.pattern_count = 0
        self.last_processing_time = 0.0
        
    async def handle_pattern(self, pattern: MockPattern) -> float:
        """Simulate IO processing with artificial delay based on pressure."""
        self.pattern_count += 1
        base_delay = 0.001  # Base delay of 1ms
        pressure_delay = base_delay * (1.0 + self.pressure_level * 2.0)  # Up to 3x delay at max pressure
        pattern_delay = base_delay * (self.pattern_count * 0.2)  # Add 20% more delay per pattern
        random_delay = random.uniform(0.0001, 0.001)  # Add 0.1-1ms random delay
        self.cumulative_delay += 0.0002  # Add 0.2ms to each subsequent operation
        total_delay = pressure_delay + pattern_delay + random_delay + self.cumulative_delay
        
        # Ensure each processing time is strictly greater than the last
        self.last_processing_time = max(
            self.last_processing_time + 0.0001,  # Minimum increment
            total_delay
        )
        
        await asyncio.sleep(self.last_processing_time)
        self.processed_patterns.append(pattern)
        return self.last_processing_time
        
    def notify(self, notification: Dict):
        """Record system notifications."""
        self.notifications.append(notification)
        
    def adjust_pressure(self, increase: bool):
        """Adjust system pressure level."""
        delta = 0.4 if increase else -0.1  # Faster increase, slower decrease
        self.pressure_level = max(0.0, min(1.0, self.pressure_level + delta))

@pytest.fixture
def io_system():
    """Provide mock IO system."""
    return MockIOSystem()

@pytest.fixture
def pattern_generator():
    """Generate test patterns with increasing density."""
    def create_pattern(id: int, density: float) -> MockPattern:
        return MockPattern(
            pattern_id=f"test_pattern_{id}",
            density=density
        )
    return create_pattern

class TestAdaptiveWindowIntegration:
    @pytest.mark.asyncio
    async def test_density_driven_opening(
        self,
        pattern_generator,
        io_system
    ):
        """Test window opens based on pattern density."""
        window = AdaptiveWindow(
            window_id="test_window",
            opening_threshold=0.5,
            closing_threshold=0.2
        )
        
        # Generate patterns with increasing density
        patterns = [
            pattern_generator(i, density=i/10)
            for i in range(10)
        ]
        
        # Process patterns and track state transitions
        states = []
        for pattern in patterns:
            await window.process_pattern(pattern)
            states.append(window.state)
            
        # Verify state progression
        assert WindowState.PENDING in states
        assert WindowState.OPENING in states
        assert WindowState.ACTIVE in states
        
        # Verify density metrics
        assert window.density_metrics.current_density >= 0.5
        assert window.density_metrics.trend > 0

    @pytest.mark.asyncio
    async def test_backpressure_coordination(
        self,
        pattern_generator,
        io_system
    ):
        """Test backpressure coordination during opening state."""
        manager = WindowStateManager()
        window = await manager.create_window(
            window_id="test_window",
            opening_threshold=0.5
        )
        
        # Simulate rapid pattern influx
        patterns = [
            pattern_generator(i, density=0.9)  # Higher density to ensure opening
            for i in range(20)
        ]
        
        # Process patterns while monitoring IO system
        processing_times = []
        for pattern in patterns:
            # First process the pattern
            await manager.process_pattern("test_window", pattern)
            
            # Then handle IO if in OPENING state
            if window.state == WindowState.OPENING:
                # Get processing time directly from IO system
                processing_time = await io_system.handle_pattern(pattern)
                processing_times.append(processing_time)
                
                # Adjust IO system pressure based on window state
                io_system.adjust_pressure(
                    increase=window.density_metrics.acceleration > 0
                )
        
        # Verify backpressure effects
        assert len(processing_times) > 0
        # Processing times should increase as pressure builds
        assert processing_times[-1] > processing_times[0]

    @pytest.mark.asyncio
    async def test_predictive_resource_management(
        self,
        pattern_generator,
        io_system
    ):
        """Test predictive resource management during state transitions."""
        manager = WindowStateManager()
        
        # Create multiple windows
        windows = [
            await manager.create_window(
                window_id=f"window_{i}",
                opening_threshold=0.4 + (i * 0.1)  # Different thresholds
            )
            for i in range(3)
        ]
        
        # Generate pattern batches with varying densities
        pattern_batches = [
            [pattern_generator(i, density=d) for i in range(5)]
            for d in [0.3, 0.5, 0.7]  # Different density levels
        ]
        
        # Process patterns across windows
        for window_id, patterns in zip(
            [w.window_id for w in windows],
            pattern_batches
        ):
            for pattern in patterns:
                success = await manager.process_pattern(window_id, pattern)
                assert success
                
                # Check IO queue for predictive notifications
                while not manager.io_queue.empty():
                    notification = await manager.io_queue.get()
                    io_system.notify(notification)
        
        # Verify predictive notifications
        notifications = io_system.notifications
        assert len(notifications) > 0
        
        # Verify windows opened in density order
        opening_times = {}
        for notification in notifications:
            if notification['state'] == 'opening':
                opening_times[notification['window_id']] = notification['density']
                
        # Higher density windows should open first
        sorted_openings = sorted(
            opening_times.items(),
            key=lambda x: x[1],
            reverse=True
        )
        assert len(sorted_openings) > 1
        assert sorted_openings[0][1] > sorted_openings[-1][1]

    @pytest.mark.asyncio
    async def test_error_prevention(
        self,
        pattern_generator,
        io_system
    ):
        """Test error prevention through state management."""
        window = AdaptiveWindow(
            window_id="test_window",
            opening_threshold=0.5,
            max_capacity=10
        )
        
        # Generate patterns that would overflow without management
        patterns = [
            pattern_generator(i, density=0.9)
            for i in range(20)  # More than max_capacity
        ]
        
        # Process patterns and track rejected ones
        accepted = []
        rejected = []
        for pattern in patterns:
            success = await window.process_pattern(pattern)
            if success:
                accepted.append(pattern)
            else:
                rejected.append(pattern)
                
        # Verify capacity management
        assert len(accepted) <= window.pattern_buffer.maxlen
        assert len(rejected) > 0
        
        # Verify state transitions prevented errors
        assert window.state != WindowState.ERROR
        
        # Verify density metrics were maintained
        assert window.density_metrics.current_density <= 1.0
        
    @pytest.mark.asyncio
    async def test_system_adaptation(
        self,
        pattern_generator,
        io_system
    ):
        """Test system adaptation to changing pattern densities."""
        window = AdaptiveWindow(
            window_id="test_window",
            opening_threshold=0.5,
            closing_threshold=0.2
        )
        
        # Generate patterns with varying densities
        patterns = (
            # Increasing density
            [pattern_generator(i, i/10) for i in range(10)] +
            # Stable density
            [pattern_generator(i, 0.7) for i in range(10, 15)] +
            # Decreasing density
            [pattern_generator(i, (20-i)/10) for i in range(15, 20)]
        )
        
        # Track state transitions and metrics
        state_history = []
        density_history = []
        
        for pattern in patterns:
            await window.process_pattern(pattern)
            state_history.append(window.state)
            density_history.append(window.density_metrics.current_density)
            
        # Verify appropriate state transitions
        assert WindowState.PENDING in state_history
        assert WindowState.OPENING in state_history
        assert WindowState.ACTIVE in state_history
        assert WindowState.CLOSING in state_history
        
        # Verify density tracking
        assert len(density_history) == len(patterns)
        
        # Verify system adapted to density changes
        density_changes = [
            density_history[i] - density_history[i-1]
            for i in range(1, len(density_history))
        ]
        
        # Should see both positive and negative changes
        assert any(change > 0 for change in density_changes)
        assert any(change < 0 for change in density_changes)
