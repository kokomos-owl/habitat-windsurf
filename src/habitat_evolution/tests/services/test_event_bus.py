"""Tests for LocalEventBus."""

import pytest
from datetime import datetime
from typing import List, Dict, Any

from habitat_evolution.core.services.event_bus import LocalEventBus, Event, EventFilter
from habitat_evolution.core.services.time_provider import TimeProvider

@pytest.fixture
def event_bus():
    """Create fresh event bus for each test."""
    return LocalEventBus()

def test_simple_publish_subscribe(event_bus):
    """Test basic publish/subscribe functionality."""
    received_events: List[Event] = []
    
    def handler(event: Event):
        received_events.append(event)
    
    event_bus.subscribe("test_event", handler)
    event = Event.create("test_event", {"value": 42})
    event_bus.publish(event)
    
    assert len(received_events) == 1
    assert received_events[0].type == "test_event"
    assert received_events[0].data["value"] == 42

def test_filtered_subscription(event_bus):
    """Test filtered event subscription."""
    received_events: List[Event] = []
    
    def handler(event: Event):
        received_events.append(event)
    
    # Filter for events with value > 50
    def value_matcher(data: Dict[str, Any]) -> bool:
        return data.get("value", 0) > 50
    
    filter = EventFilter(
        type_pattern="test_event",
        data_matcher=value_matcher
    )
    
    event_bus.subscribe_filtered(filter, handler)
    
    # Should match
    event1 = Event.create("test_event", {"value": 100})
    event_bus.publish(event1)
    
    # Should not match
    event2 = Event.create("test_event", {"value": 42})
    event_bus.publish(event2)
    
    assert len(received_events) == 1
    assert received_events[0].data["value"] == 100

def test_multiple_handlers(event_bus):
    """Test multiple handlers for same event type."""
    count1 = 0
    count2 = 0
    
    def handler1(event: Event):
        nonlocal count1
        count1 += 1
    
    def handler2(event: Event):
        nonlocal count2
        count2 += 1
    
    event_bus.subscribe("test_event", handler1)
    event_bus.subscribe("test_event", handler2)
    
    event = Event.create("test_event", {})
    event_bus.publish(event)
    
    assert count1 == 1
    assert count2 == 1

def test_event_history(event_bus):
    """Test event history tracking."""
    events = [
        Event.create("type1", {"value": i})
        for i in range(5)
    ]
    
    for event in events:
        event_bus.publish(event)
    
    history = event_bus.get_history()
    assert len(history) == 5
    
    # Test with limit
    limited = event_bus.get_history(limit=2)
    assert len(limited) == 2
    assert limited[0].data["value"] == 3
    assert limited[1].data["value"] == 4

def test_history_with_filter(event_bus):
    """Test filtered event history."""
    events = [
        Event.create("type1", {"value": i})
        for i in range(5)
    ]
    events.extend([
        Event.create("type2", {"value": i})
        for i in range(5)
    ])
    
    for event in events:
        event_bus.publish(event)
    
    filter = EventFilter(type_pattern="type1")
    filtered = event_bus.get_history(filter=filter)
    assert len(filtered) == 5
    assert all(e.type == "type1" for e in filtered)

def test_unsubscribe(event_bus):
    """Test unsubscribing handlers."""
    received_events: List[Event] = []
    
    def handler(event: Event):
        received_events.append(event)
    
    event_bus.subscribe("test_event", handler)
    event_bus.unsubscribe("test_event", handler)
    
    event = Event.create("test_event", {})
    event_bus.publish(event)
    
    assert len(received_events) == 0

def test_error_handling(event_bus):
    """Test handler error doesn't stop event processing."""
    count = 0
    
    def bad_handler(event: Event):
        raise Exception("Oops")
    
    def good_handler(event: Event):
        nonlocal count
        count += 1
    
    event_bus.subscribe("test_event", bad_handler)
    event_bus.subscribe("test_event", good_handler)
    
    event = Event.create("test_event", {})
    event_bus.publish(event)  # Should not raise
    
    assert count == 1  # Good handler still executed

def test_clear_history(event_bus):
    """Test clearing event history."""
    event = Event.create("test_event", {})
    event_bus.publish(event)
    
    assert len(event_bus.get_history()) == 1
    event_bus.clear_history()
    assert len(event_bus.get_history()) == 0

def test_event_timestamp():
    """Test events are created with current time."""
    before = TimeProvider.now()
    event = Event.create("test_event", {})
    after = TimeProvider.now()
    
    assert before <= event.timestamp <= after
