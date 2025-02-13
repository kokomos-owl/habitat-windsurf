"""Tests for pattern evolution."""

import pytest
from typing import Dict, Any, List
from datetime import datetime

from habitat_evolution.core.pattern.evolution import PatternEvolutionManager, PatternMetrics
from habitat_evolution.core.storage.memory import InMemoryPatternStore, InMemoryRelationshipStore
from habitat_evolution.core.services.event_bus import LocalEventBus, Event

@pytest.fixture(scope="function")
async def event_bus():
    """Create fresh event bus."""
    return LocalEventBus()

@pytest.fixture(scope="function")
async def pattern_store():
    """Create fresh pattern store."""
    return InMemoryPatternStore()

@pytest.fixture(scope="function")
async def relationship_store():
    """Create fresh relationship store."""
    return InMemoryRelationshipStore()

@pytest.fixture(scope="function")
async def evolution_manager(pattern_store, relationship_store, event_bus):
    """Create evolution manager with test dependencies."""
    ps = await pattern_store
    rs = await relationship_store
    eb = await event_bus
    return PatternEvolutionManager(ps, rs, eb)

class TestPatternEvolution:
    """Tests for pattern evolution functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_register_pattern(self, evolution_manager):
        manager = await evolution_manager
        """Test pattern registration."""
        pattern_type = "test"
        content = {"value": 42}
        context = {"source": "test"}
        
        result = await evolution_manager.register_pattern(
            pattern_type, content, context
        )
        
        assert result.success
        assert result.data  # Pattern ID
        
        # Verify pattern was stored
        find_result = await evolution_manager._pattern_store.find_patterns({
            "type": pattern_type
        })
        assert find_result.success
        assert len(find_result.data) == 1
        pattern = find_result.data[0]
        assert pattern["content"] == content
        assert pattern["context"] == context
        
        # Verify initial metrics
        metrics = PatternMetrics.from_dict(pattern["metrics"])
        assert metrics.coherence == 0.0
        assert metrics.stability == 0.0
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_update_pattern(self, evolution_manager):
        manager = await evolution_manager
        """Test pattern updates."""
        # Create pattern
        create_result = await evolution_manager.register_pattern(
            "test",
            {"value": 42}
        )
        assert create_result.success
        pattern_id = create_result.data
        
        # Update pattern
        updates = {
            "content": {"value": 43},
            "metrics": PatternMetrics(
                coherence=0.5,
                emergence_rate=0.3,
                cross_pattern_flow=0.4,
                energy_state=0.6,
                adaptation_rate=0.2,
                stability=0.7
            ).to_dict()
        }
        
        update_result = await evolution_manager.update_pattern(
            pattern_id,
            updates
        )
        assert update_result.success
        
        # Verify updates
        find_result = await evolution_manager._pattern_store.find_patterns({
            "id": pattern_id
        })
        assert find_result.success
        pattern = find_result.data[0]
        assert pattern["content"]["value"] == 43
        metrics = PatternMetrics.from_dict(pattern["metrics"])
        assert metrics.coherence == 0.5
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_pattern_relationships(self, evolution_manager):
        manager = await evolution_manager
        """Test pattern relationship management."""
        # Create two patterns
        p1_result = await evolution_manager.register_pattern(
            "test", {"id": 1}
        )
        p2_result = await evolution_manager.register_pattern(
            "test", {"id": 2}
        )
        assert p1_result.success and p2_result.success
        
        # Create relationship
        rel_result = await evolution_manager.relate_patterns(
            p1_result.data,
            p2_result.data,
            "similar",
            {"weight": 0.5}
        )
        assert rel_result.success
        
        # Get related patterns
        related_result = await evolution_manager.get_related_patterns(
            p1_result.data
        )
        assert related_result.success
        assert len(related_result.data) == 1
        assert related_result.data[0]["content"]["id"] == 2
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_metric_propagation(self, evolution_manager):
        manager = await evolution_manager
        """Test metric updates through relationships."""
        # Create two patterns
        p1_result = await evolution_manager.register_pattern(
            "test", {"id": 1}
        )
        p2_result = await evolution_manager.register_pattern(
            "test", {"id": 2}
        )
        assert p1_result.success and p2_result.success
        
        # Update first pattern metrics
        metrics = PatternMetrics(
            coherence=1.0,
            emergence_rate=1.0,
            cross_pattern_flow=1.0,
            energy_state=1.0,
            adaptation_rate=1.0,
            stability=1.0
        )
        
        await evolution_manager.update_pattern(
            p1_result.data,
            {"metrics": metrics.to_dict()}
        )
        
        # Create relationship
        await evolution_manager.relate_patterns(
            p1_result.data,
            p2_result.data,
            "similar"
        )
        
        # Verify metric propagation
        find_result = await evolution_manager._pattern_store.find_patterns({
            "id": p2_result.data
        })
        assert find_result.success
        p2_metrics = PatternMetrics.from_dict(find_result.data[0]["metrics"])
        
        # Should be average of 0 (initial) and 1 (from p1)
        assert p2_metrics.coherence == 0.5
        assert p2_metrics.stability == 0.5
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_event_handling(self, evolution_manager, event_bus):
        manager = await evolution_manager
        bus = await event_bus
        """Test event handling and notifications."""
        events: List[Event] = []
        
        def handler(event: Event):
            events.append(event)
        
        event_bus.subscribe("pattern.created", handler)
        event_bus.subscribe("pattern.updated", handler)
        event_bus.subscribe("pattern.related", handler)
        
        # Create pattern
        p_result = await evolution_manager.register_pattern(
            "test", {"value": 42}
        )
        assert p_result.success
        
        # Update pattern
        await evolution_manager.update_pattern(
            p_result.data,
            {"content": {"value": 43}}
        )
        
        # Create relationship
        p2_result = await evolution_manager.register_pattern(
            "test", {"value": 44}
        )
        await evolution_manager.relate_patterns(
            p_result.data,
            p2_result.data,
            "similar"
        )
        
        # Verify events
        assert len(events) == 4  # 2 creates + 1 update + 1 relate
        assert events[0].type == "pattern.created"
        assert events[1].type == "pattern.updated"
        assert events[2].type == "pattern.created"
        assert events[3].type == "pattern.related"
