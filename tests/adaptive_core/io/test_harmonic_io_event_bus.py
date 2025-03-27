"""
Test for HarmonicIOService event bus integration.

This test verifies the current behavior of the HarmonicIOService
with respect to event bus integration, serving as a baseline
before making any modifications to address technical debt.
"""

import unittest
import logging
from datetime import datetime

from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService, OperationType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestHarmonicIOEventBus(unittest.TestCase):
    """Test suite for HarmonicIOService event bus integration."""

    def setUp(self):
        """Set up test environment."""
        # Create event bus
        self.event_bus = LocalEventBus()
        
        # Track events received
        self.events_received = []
        
        # Subscribe to all events
        self.event_bus.subscribe("*", self._on_event)
        
    def _on_event(self, event):
        """Record events received."""
        self.events_received.append(event)
        logger.info(f"Received event: {event.type}")
        
    def test_current_initialization(self):
        """Test that the current initialization pattern doesn't crash."""
        # Create HarmonicIOService with event_bus as first parameter (current pattern)
        io_service = HarmonicIOService(self.event_bus)
        
        # Simply verify that initialization doesn't raise an exception
        self.assertIsNotNone(io_service)
        
        # Check that the service has expected attributes
        self.assertTrue(hasattr(io_service, 'base_frequency'))
        self.assertTrue(hasattr(io_service, 'harmonics'))
        self.assertTrue(hasattr(io_service, 'max_queue_size'))
        self.assertTrue(hasattr(io_service, 'adaptive_timing'))
        
        # Start and stop the service to ensure basic functionality works
        io_service.start()
        self.assertTrue(io_service.running)
        io_service.stop()
        self.assertFalse(io_service.running)
        
    def test_parameter_interpretation(self):
        """Test how the event_bus is being interpreted as a parameter."""
        # Create HarmonicIOService with event_bus as first parameter (current pattern)
        io_service = HarmonicIOService(self.event_bus)
        
        # With our new implementation, the event_bus should be correctly detected
        # and assigned to the event_bus attribute
        self.assertIs(io_service.event_bus, self.event_bus)
        
        # The base_frequency should be set to the default value
        self.assertEqual(io_service.base_frequency, 0.1)
        
        # The other parameters should have their default values
        self.assertEqual(io_service.harmonics, 3)
        self.assertEqual(io_service.max_queue_size, 1000)
        self.assertEqual(io_service.adaptive_timing, True)


    def test_proposed_initialization(self):
        """Test the proposed initialization with explicit event_bus parameter."""
        # Create service with explicit parameters
        io_service = HarmonicIOService(
            event_bus=self.event_bus,
            base_frequency_or_event_bus=0.2,  # Use the new parameter name
            harmonics=4,
            max_queue_size=500,
            adaptive_timing=False
        )
        
        # Check that parameters are set correctly
        self.assertIs(io_service.event_bus, self.event_bus)
        self.assertEqual(io_service.base_frequency, 0.2)
        self.assertEqual(io_service.harmonics, 4)
        self.assertEqual(io_service.max_queue_size, 500)
        self.assertEqual(io_service.adaptive_timing, False)
        
        # Start and stop the service to ensure basic functionality works
        io_service.start()
        self.assertTrue(io_service.running)
        io_service.stop()
        self.assertFalse(io_service.running)
    
    def test_backward_compatibility(self):
        """Test that both initialization patterns work after changes."""
        # Old pattern (positional event_bus)
        io_service1 = HarmonicIOService(self.event_bus)
        
        # New pattern (explicit event_bus parameter)
        io_service2 = HarmonicIOService(event_bus=self.event_bus)
        
        # Both should have the event_bus set correctly
        self.assertIs(io_service1.event_bus, self.event_bus)
        self.assertIs(io_service2.event_bus, self.event_bus)
        
        # io_service1 should have default values for other parameters
        self.assertEqual(io_service1.base_frequency, 0.1)
        
        # Test with mixed parameters
        io_service3 = HarmonicIOService(
            base_frequency_or_event_bus=0.5,
            event_bus=self.event_bus
        )
        self.assertEqual(io_service3.base_frequency, 0.5)
        self.assertIs(io_service3.event_bus, self.event_bus)


    def test_event_bus_integration(self):
        """Test the event bus integration features."""
        # Create service with event bus
        io_service = HarmonicIOService(event_bus=self.event_bus)
        io_service.start()
        
        # Test event subscription
        test_data = {"value": 42}
        received_data = []
        
        def test_handler(event):
            data = event.data if hasattr(event, 'data') else event.get('data', {})
            received_data.append(data)
        
        # Subscribe to test event
        io_service.subscribe_to_event("test.event", test_handler)
        
        # Publish event immediately
        io_service.publish_event("test.event", test_data, immediate=True)
        
        # Wait a short time for event processing
        time.sleep(0.1)
        
        # Verify event was received
        self.assertEqual(len(received_data), 1)
        self.assertEqual(received_data[0], test_data)
        
        # Clean up
        io_service.stop()
    
    def test_harmonic_timing_integration(self):
        """Test the harmonic timing integration with base_frequency."""
        # Create service with custom base_frequency and event bus
        io_service = HarmonicIOService(
            base_frequency_or_event_bus=0.5,  # Higher frequency = faster cycles
            event_bus=self.event_bus
        )
        io_service.start()
        
        # Test that base_frequency affects cycle position
        time.sleep(0.1)  # Sleep briefly
        
        # With base_frequency=0.5, after 0.1 seconds, cycle position should be around 0.05
        # (0.1 seconds * 0.5 Hz = 0.05 cycles)
        cycle_pos = io_service.get_cycle_position()
        
        # Allow some flexibility in timing due to thread scheduling
        self.assertTrue(0 <= cycle_pos <= 0.1, 
                        f"Cycle position {cycle_pos} not in expected range [0, 0.1]")
        
        # Clean up
        io_service.stop()
    
    def test_field_state_integration(self):
        """Test integration with field state updates."""
        # Create service with event bus
        io_service = HarmonicIOService(event_bus=self.event_bus)
        io_service.start()
        
        # Default values
        self.assertEqual(io_service.eigenspace_stability, 0.5)
        self.assertEqual(io_service.pattern_coherence, 0.5)
        
        # Publish field state update
        field_state = {
            "stability": 0.8,
            "coherence": 0.7,
            "resonance": 0.9
        }
        
        # Create and publish event
        event = {
            "type": "field.state.updated",
            "data": field_state
        }
        self.event_bus.publish(event)
        
        # Wait a short time for event processing
        time.sleep(0.1)
        
        # Verify field state was updated
        self.assertEqual(io_service.eigenspace_stability, 0.8)
        self.assertEqual(io_service.pattern_coherence, 0.7)
        self.assertEqual(io_service.resonance_level, 0.9)
        
        # Clean up
        io_service.stop()


if __name__ == "__main__":
    unittest.main()
