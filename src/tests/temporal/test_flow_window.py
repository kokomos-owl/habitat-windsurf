"""Integration tests for flow-based adaptive window system.

This module contains integration tests that validate the flow-based pattern
processing system. The tests focus on three key aspects:

1. Flow Rate Adaptation:
   - Verifies the system adapts flow rates based on pressure differentials
   - Tests response to varying pattern densities and intensities
   - Validates flow metric calculations

2. Backpressure Regulation:
   - Tests the system's ability to handle and regulate backpressure
   - Verifies pressure differential calculations
   - Validates output pressure adjustments

3. Capacity Management:
   - Tests buffer capacity handling
   - Verifies flow rate adjustments based on buffer fullness
   - Validates system stability under high load

The tests use mock patterns and a mock IO system to simulate real-world
conditions and verify system behavior.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
from dataclasses import dataclass
import time
import random

from src.core.temporal.adaptive_window import (
    AdaptiveWindow,
    WindowStateManager,
    FlowMetrics
)

@dataclass
class MockPattern:
    """Mock pattern for testing flow-based window behavior.
    
    Simulates patterns with configurable density and intensity to test
    different flow conditions and pressure scenarios.
    
    Attributes:
        pattern_id (str): Unique identifier for the pattern
        density (float): Pattern density, affects input pressure
        intensity (float): Pattern intensity, affects processing complexity
        timestamp (datetime): Creation time of the pattern
    """
    """Mock pattern for testing."""
    pattern_id: str
    density: float
    intensity: float = 0.5
    timestamp: datetime = None

    def __post_init__(self):
        self.timestamp = self.timestamp or datetime.now()

class MockIOSystem:
    """Mock IO system for testing flow effects and backpressure.
    
    Simulates an IO system that processes patterns and provides feedback
    through flow metrics. Includes artificial delays and pressure effects
    to test system adaptation.
    
    Attributes:
        flow_rate (float): Current flow rate of the system
        pressure_differential (float): Current pressure differential
        notifications (List[Dict]): Record of system notifications
        processed_patterns (List[MockPattern]): Patterns processed
        cumulative_delay (float): Accumulated processing delay
    """
    """Mock IO system for testing flow effects."""
    def __init__(self):
        self.flow_rate = 0.0
        self.pressure_differential = 0.0
        self.notifications: List[Dict] = []
        self.processed_patterns: List[MockPattern] = []
        self.cumulative_delay = 0.0
        
    async def handle_pattern(self, pattern: MockPattern) -> float:
        """Simulate IO processing with artificial delay based on flow metrics."""
        # Calculate delay based on flow rate and pressure
        base_delay = 0.001
        flow_delay = base_delay * (1.0 - self.flow_rate)  # More delay when flow is low
        pressure_delay = base_delay * abs(self.pressure_differential)  # More delay with pressure differential
        self.cumulative_delay += 0.0002  # Progressive delay
        
        total_delay = flow_delay + pressure_delay + self.cumulative_delay
        await asyncio.sleep(total_delay)
        
        self.processed_patterns.append(pattern)
        return total_delay
        
    def notify(self, notification: Dict):
        """Record system notifications."""
        self.notifications.append(notification)
        self.flow_rate = notification.get('flow_rate', 0.0)
        self.pressure_differential = notification.get('pressure_differential', 0.0)

@pytest.fixture
def io_system():
    """Provide mock IO system."""
    return MockIOSystem()

@pytest.fixture
def pattern_generator():
    """Generate test patterns with specified characteristics."""
    def _generate(index: int, density: float = 0.5, intensity: float = 0.5):
        return MockPattern(
            pattern_id=f"test_pattern_{index}",
            density=density,
            intensity=intensity
        )
    return _generate

class TestFlowWindowIntegration:
    """Integration tests for flow-based window system.
    
    Tests the complete flow-based pattern processing system, including:
    - Flow rate adaptation
    - Backpressure regulation
    - Capacity management
    
    Each test validates a specific aspect of the system while ensuring
    overall stability and correct behavior under various conditions.
    """
    """Integration tests for flow-based window system."""
    
    @pytest.mark.asyncio
    async def test_flow_rate_adaptation(
        self,
        pattern_generator,
        io_system
    ):
        """Test window adapts flow rate based on pressure differential."""
        manager = WindowStateManager()
        window = await manager.create_window(
            window_id="test_window",
            flow_threshold=0.5
        )
        
        # Generate patterns with increasing density and intensity
        patterns = [
            pattern_generator(
                i,
                density=min(1.0, 0.4 + (i * 0.1)),
                intensity=min(1.0, 0.3 + (i * 0.1))
            )
            for i in range(10)
        ]
        
        # Process patterns while monitoring flow metrics
        flow_rates = []
        pressure_diffs = []
        
        for pattern in patterns:
            await manager.process_pattern("test_window", pattern)
            flow_rates.append(window.flow_metrics.flow_rate)
            pressure_diffs.append(window.flow_metrics.pressure_differential)
            
            # Simulate IO system handling
            await io_system.handle_pattern(pattern)
        
        # Verify flow adaptation
        assert len(flow_rates) > 0
        # Flow rate should respond to pressure changes
        assert any(rate > 0.0 for rate in flow_rates)
        # Pressure differential should vary
        assert max(pressure_diffs) > min(pressure_diffs)
        
    @pytest.mark.asyncio
    async def test_backpressure_regulation(
        self,
        pattern_generator,
        io_system
    ):
        """Test backpressure regulation through flow control."""
        manager = WindowStateManager()
        window = await manager.create_window(
            window_id="test_window",
            flow_threshold=0.6
        )
        
        # Generate high-pressure patterns
        patterns = [
            pattern_generator(i, density=0.8, intensity=0.9)
            for i in range(15)
        ]
        
        # Process patterns while monitoring output pressure
        output_pressures = []
        
        for pattern in patterns:
            await manager.process_pattern("test_window", pattern)
            output_pressures.append(window.flow_metrics.output_pressure)
            
            # Simulate IO system handling
            await io_system.handle_pattern(pattern)
        
        # Verify pressure regulation
        assert len(output_pressures) > 0
        # Output pressure should increase to regulate flow
        assert output_pressures[-1] > output_pressures[0]
        
    @pytest.mark.asyncio
    async def test_capacity_management(
        self,
        pattern_generator,
        io_system
    ):
        """Test flow adjustment based on buffer capacity."""
        manager = WindowStateManager()
        window = await manager.create_window(
            window_id="test_window",
            flow_threshold=0.5
        )
        
        # Generate enough patterns to approach capacity
        patterns = [
            pattern_generator(i, density=0.7)
            for i in range(80)  # 80% of max_capacity=100
        ]
        
        # Process patterns while monitoring buffer size
        buffer_sizes = []
        flow_rates = []
        
        for pattern in patterns:
            success = await manager.process_pattern("test_window", pattern)
            if success:
                buffer_sizes.append(len(window.pattern_buffer))
                flow_rates.append(window.flow_metrics.flow_rate)
            
            # Simulate IO system handling
            await io_system.handle_pattern(pattern)
        
        # Verify capacity management
        assert len(buffer_sizes) > 0
        # Flow rate should decrease as buffer fills
        assert flow_rates[-1] < flow_rates[0]
        # Buffer should not exceed capacity
        assert max(buffer_sizes) <= window.pattern_buffer.maxlen
