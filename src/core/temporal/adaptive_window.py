"""Flow-based adaptive window with pressure differentials.

This module implements a fluid dynamics inspired approach to pattern processing,
using concepts from flow mechanics to manage system load and backpressure.

Key Components:
    - FlowMetrics: Tracks and calculates flow-related measurements
    - AdaptiveWindow: Main window implementation using flow-based control
    - WindowStateManager: Coordinates multiple windows and IO systems

The system uses continuous metrics rather than discrete states, allowing for:
    - Natural adaptation to changing conditions
    - Smooth transitions without state boundaries
    - Effective backpressure handling through pressure differentials
    - Built-in capacity management affecting flow rates

Flow Mechanics:
    - Input Pressure: Derived from pattern density and intensity
    - Output Pressure: Represents downstream system backpressure
    - Pressure Differential: Drives the flow rate
    - Flow Rate: Determines pattern processing speed
    - Buffer Capacity: Influences pressure and flow calculations

Example:
    ```python
    # Create a window manager
    manager = WindowStateManager()
    
    # Create a new adaptive window
    window = await manager.create_window(
        window_id="my_window",
        flow_threshold=0.5
    )
    
    # Process patterns through the window
    success = await manager.process_pattern("my_window", pattern)
    ```
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Callable, Any
from enum import Enum
import asyncio
from collections import deque
import logging
import statistics
import math
import random
import time

@dataclass
class FlowMetrics:
    """Tracks and calculates flow-related measurements for adaptive windows.
    
    This class maintains the core metrics that drive the flow-based pattern
    processing system. It uses concepts from fluid dynamics to model system
    behavior and manage processing rates.
    
    Attributes:
        input_pressure (float): Pressure from incoming patterns, based on their
            density and intensity. Range [0.0, 1.0]
        output_pressure (float): Backpressure from downstream systems.
            Range [0.0, 1.0]
        flow_rate (float): Current rate of pattern processing. Range [0.0, 1.0]
        pressure_differential (float): Difference between input and output
            pressure. Range [-1.0, 1.0]
        last_update (datetime): Timestamp of last metrics update
        history (List[float]): Recent flow rate history for trend analysis
        window_size (int): Size of the history window for trend analysis
    """
    input_pressure: float = 0.0  # Incoming pattern pressure
    output_pressure: float = 0.0  # Backpressure from downstream systems
    flow_rate: float = 0.0  # Current flow rate through the window
    pressure_differential: float = 0.0  # Difference between input and output pressure
    last_update: datetime = field(default_factory=datetime.now)
    history: List[float] = field(default_factory=list)
    window_size: int = 5
    
    def calculate_flow_rate(self, buffer_size: int, max_capacity: int) -> float:
        """Calculate flow rate based on buffer size and pressure differential."""
        capacity_factor = buffer_size / max_capacity
        # Flow rate increases with positive pressure differential and decreases with capacity
        return max(0.0, self.pressure_differential * (1.0 - capacity_factor))
    
    def update(self, pattern: Any, buffer_size: int, max_capacity: int):
        """Update flow metrics."""
        now = datetime.now()
        
        # Calculate input pressure from pattern characteristics
        pattern_density = getattr(pattern, 'density', 0.5)
        pattern_intensity = getattr(pattern, 'intensity', 0.5)
        self.input_pressure = pattern_density * (1.0 + pattern_intensity)
        
        # Calculate pressure differential
        self.pressure_differential = self.input_pressure - self.output_pressure
        
        # Calculate flow rate
        self.flow_rate = self.calculate_flow_rate(buffer_size, max_capacity)
        
        # Store history
        self.history.append(self.flow_rate)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        self.last_update = now
        
        logging.debug(
            f"Flow metrics updated - input: {self.input_pressure:.3f}, "
            f"output: {self.output_pressure:.3f}, "
            f"differential: {self.pressure_differential:.3f}, "
            f"flow_rate: {self.flow_rate:.3f}"
        )

class AdaptiveWindow:
    """Flow-based adaptive window for pattern processing.
    
    This class implements a fluid dynamics inspired approach to pattern
    processing, using flow metrics to naturally adapt to changing system
    conditions and manage backpressure.
    
    The window automatically:
        - Increases flow when pressure differential is positive
        - Reduces flow when buffer capacity is high
        - Adjusts processing delays based on flow metrics
        - Maintains stability through gradual pressure changes
    
    Args:
        window_id (str): Unique identifier for the window
        flow_threshold (float): Target flow rate threshold. Range [0.0, 1.0]
        max_capacity (int): Maximum number of patterns in buffer
    
    Attributes:
        flow_metrics (FlowMetrics): Current flow measurements and calculations
        pattern_buffer (deque): Buffer for pattern storage and processing
        flow_handlers (List[Callable]): Registered flow management handlers
    """
    def __init__(
        self,
        window_id: str,
        flow_threshold: float = 0.5,
        max_capacity: int = 1000
    ):
        self.window_id = window_id
        self.flow_metrics = FlowMetrics()
        self.pattern_buffer = deque(maxlen=max_capacity)
        self.flow_threshold = flow_threshold
        self.flow_handlers: List[Callable] = []
        self.base_processing_delay = 0.0001
        self.pattern_count = 0
        self._setup_flow_handlers()
        
        logging.debug(
            f"Window {window_id} created with flow threshold: {flow_threshold}"
        )

    def _setup_flow_handlers(self):
        """Setup handlers for flow management."""
        self.flow_handlers = [
            self._manage_flow_rate,
            self._adjust_backpressure,
            self._check_capacity
        ]

    async def process_pattern(self, pattern: Any) -> bool:
        """Process incoming pattern and update flow metrics."""
        if len(self.pattern_buffer) >= self.pattern_buffer.maxlen:
            logging.warning(f"Window {self.window_id} at capacity, rejecting pattern")
            return False

        # Update flow metrics
        self.flow_metrics.update(
            pattern,
            len(self.pattern_buffer),
            self.pattern_buffer.maxlen
        )
        
        # Run flow handlers
        for handler in self.flow_handlers:
            await handler(pattern)
            
        # Buffer pattern if flow rate is positive
        if self.flow_metrics.flow_rate > 0:
            self.pattern_buffer.append(pattern)
            self.pattern_count += 1
            
            # Add processing delay based on flow metrics
            delay = self._calculate_processing_delay()
            await asyncio.sleep(delay)
            
            return True
        return False

    def _calculate_processing_delay(self) -> float:
        """Calculate processing delay based on flow metrics."""
        # Base delay increases as flow rate decreases
        flow_factor = 1.0 + (1.0 - self.flow_metrics.flow_rate) * 2.0
        # Additional delay based on output pressure
        pressure_factor = 1.0 + self.flow_metrics.output_pressure * 3.0
        # Progressive delay based on pattern count
        pattern_factor = 1.0 + (self.pattern_count * 0.01)
        
        return self.base_processing_delay * flow_factor * pressure_factor * pattern_factor

    async def _manage_flow_rate(self, pattern: Any):
        """Manage flow rate based on pressure differential."""
        if self.flow_metrics.flow_rate < self.flow_threshold:
            # Reduce output pressure to increase flow
            self.flow_metrics.output_pressure = max(
                0.0,
                self.flow_metrics.output_pressure - 0.1
            )
        else:
            # Increase output pressure to regulate flow
            self.flow_metrics.output_pressure = min(
                1.0,
                self.flow_metrics.output_pressure + 0.1
            )

    async def _adjust_backpressure(self, pattern: Any):
        """Adjust backpressure based on flow metrics."""
        if self.flow_metrics.pressure_differential > 0:
            # Add delay proportional to pressure differential
            delay = self.base_processing_delay * self.flow_metrics.pressure_differential
            await asyncio.sleep(delay)

    async def _check_capacity(self, pattern: Any):
        """Check capacity and adjust flow metrics."""
        capacity_used = len(self.pattern_buffer) / self.pattern_buffer.maxlen
        if capacity_used > 0.8:  # 80% capacity threshold
            # Increase output pressure to slow down flow
            self.flow_metrics.output_pressure = min(
                1.0,
                self.flow_metrics.output_pressure + 0.2
            )

class WindowStateManager:
    """Manages multiple adaptive windows and coordinates with IO systems.
    
    This class serves as the primary interface for creating and managing
    adaptive windows. It coordinates pattern processing across windows and
    handles communication with IO systems through an async queue.
    
    The manager:
        - Creates and tracks multiple adaptive windows
        - Routes patterns to appropriate windows
        - Coordinates flow metrics with IO systems
        - Maintains system-wide stability
    
    Attributes:
        windows (Dict[str, AdaptiveWindow]): Active windows by ID
        io_queue (asyncio.Queue): Queue for IO system notifications
    """
    """Manages window flow states and coordinates with IO systems."""
    
    def __init__(self):
        self.windows: Dict[str, AdaptiveWindow] = {}
        self.io_queue = asyncio.Queue()
        
    async def create_window(
        self,
        window_id: str,
        flow_threshold: float = 0.5
    ) -> AdaptiveWindow:
        """Create new adaptive window."""
        window = AdaptiveWindow(
            window_id,
            flow_threshold=flow_threshold,
            max_capacity=100  # Smaller capacity for testing
        )
        self.windows[window_id] = window
        return window
        
    async def process_pattern(
        self,
        window_id: str,
        pattern: Any
    ) -> bool:
        """Process pattern and manage flow."""
        if window_id not in self.windows:
            return False
            
        window = self.windows[window_id]
        success = await window.process_pattern(pattern)
        
        # Notify IO systems of flow metrics
        await self.io_queue.put({
            'window_id': window_id,
            'flow_rate': window.flow_metrics.flow_rate,
            'pressure_differential': window.flow_metrics.pressure_differential,
            'timestamp': datetime.now()
        })
            
        return success
