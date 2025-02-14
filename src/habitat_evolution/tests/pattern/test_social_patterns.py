"""Tests for social pattern evolution."""

import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from ...core.pattern.quality import PatternState, SignalMetrics
from ...core.storage.interfaces import PatternStore, RelationshipStore, StorageResult
from ...core.services.event_bus import LocalEventBus, Event
from ...social.services.social_pattern_service import SocialPatternService, SocialMetrics

class MockPatternStore(PatternStore):
    """Mock pattern storage for testing."""
    
    def __init__(self):
        self.patterns: Dict[str, Dict[str, Any]] = {}
    
    async def save_pattern(self, pattern: Dict[str, Any]) -> StorageResult:
        pattern_id = pattern.get("id", str(len(self.patterns)))
        self.patterns[pattern_id] = pattern
        return StorageResult(success=True, data={"id": pattern_id})
    
    async def store_pattern(self, pattern: Dict[str, Any]) -> StorageResult:
        return await self.save_pattern(pattern)
    
    async def find_patterns(self, query: Dict[str, Any], limit: int = 10) -> StorageResult:
        patterns = []
        for pattern in self.patterns.values():
            matches = all(pattern.get(k) == v for k, v in query.items())
            if matches:
                patterns.append(pattern)
                if len(patterns) >= limit:
                    break
        return StorageResult(success=True, data=patterns)
    
    async def get_pattern(self, pattern_id: str) -> StorageResult:
        if pattern_id in self.patterns:
            return StorageResult(success=True, data=self.patterns[pattern_id])
        return StorageResult(success=False, error="Pattern not found")
    
    async def update_pattern(self, pattern_id: str, updates: Dict[str, Any]) -> StorageResult:
        if pattern_id in self.patterns:
            self.patterns[pattern_id].update(updates)
            return StorageResult(success=True, data=self.patterns[pattern_id])
        return StorageResult(success=False, error="Pattern not found")
        
    async def delete_pattern(self, pattern_id: str) -> StorageResult:
        if pattern_id in self.patterns:
            del self.patterns[pattern_id]
            return StorageResult(success=True)
        return StorageResult(success=False, error="Pattern not found")

class MockRelationshipStore(RelationshipStore):
    """Mock relationship storage for testing."""
    
    def __init__(self):
        self.relationships: Dict[str, Dict[str, Any]] = {}
    
    async def save_relationship(self, relationship: Dict[str, Any]) -> StorageResult:
        rel_id = relationship.get("id", str(len(self.relationships)))
        self.relationships[rel_id] = relationship
        return StorageResult(success=True, data={"id": rel_id})
    
    async def store_relationship(self, relationship: Dict[str, Any]) -> StorageResult:
        return await self.save_relationship(relationship)
    
    async def find_relationships(self, query: Dict[str, Any], limit: int = 10) -> StorageResult:
        relationships = []
        for rel in self.relationships.values():
            matches = all(rel.get(k) == v for k, v in query.items())
            if matches:
                relationships.append(rel)
                if len(relationships) >= limit:
                    break
        return StorageResult(success=True, data=relationships)
    
    async def get_related(self, pattern_id: str, relationship_type: Optional[str] = None) -> StorageResult:
        """Get patterns related to given pattern."""
        related = []
        for rel in self.relationships.values():
            if rel.get("source_id") == pattern_id or rel.get("target_id") == pattern_id:
                if relationship_type is None or rel.get("type") == relationship_type:
                    related.append(rel)
        return StorageResult(success=True, data=related)

class TestEventBus(LocalEventBus):
    """Test event bus that tracks emitted events."""
    
    events: List[Event] = []
    
    async def publish(self, event: Event):
        """Track emitted events."""
        TestEventBus.events.append(event)

@pytest.fixture
def pattern_store():
    """Create mock pattern store."""
    return MockPatternStore()

@pytest.fixture
def relationship_store():
    """Create mock relationship store."""
    return MockRelationshipStore()

@pytest.fixture
def event_bus():
    """Create test event bus."""
    return TestEventBus()

@pytest.fixture
def social_service(pattern_store, relationship_store, event_bus):
    """Create social pattern service."""
    return SocialPatternService(pattern_store, relationship_store, event_bus)

@pytest.mark.asyncio
async def test_register_social_pattern(social_service, event_bus):
    """Test registering a new social pattern."""
    # Register pattern
    result = await social_service.register_pattern({
        "type": "social_practice",
        "field_state": {
            "energy": 0.8,
            "coherence": 0.7,
            "flow": 0.6
        }
    })
    
    assert result != ""
    
    # Verify pattern stored
    result = await social_service._pattern_manager._pattern_store.find_patterns({"id": result})
    assert result.success
    assert result.data[0]["state"] == PatternState.EMERGING
    assert result.data[0]["field_state"]["energy"] == 0.8
    
    # Verify event emitted
    events = [e for e in event_bus.events if e.type == "social.pattern.registered"]
    assert len(events) == 1
    assert events[0].data["pattern_id"] == result.data[0]["id"]

@pytest.mark.asyncio
async def test_practice_evolution(social_service, event_bus):
    """Test pattern evolution into practice."""
    # Register initial pattern
    pattern_id = await social_service.register_pattern({
        "type": "social_practice",
        "field_state": {
            "energy": 0.9,
            "coherence": 0.8,
            "flow": 0.7
        }
    })
    
    # Track practice evolution
    await social_service.track_practice_evolution(
        pattern_id,
        {
            "adoption_level": 0.8,
            "institutionalization": 0.7
        }
    )
    
    # Verify pattern updated
    pattern = await social_service._pattern_manager.get_pattern(pattern_id)
    assert pattern.success
    assert pattern.data["state"] == PatternState.STABLE.value
    assert "quality" in pattern.data
    assert pattern.data["quality"]["signal_strength"] > 0.6
    
    # Verify practice event
    practice_events = [e for e in event_bus.events if e.type == "social.practice.emerged"]
    assert len(practice_events) == 1
    assert practice_events[0].data["pattern_id"] == pattern_id
    assert practice_events[0].data["metrics"]["practice_maturity"] > 0.7

@pytest.mark.asyncio
async def test_pattern_relationships(social_service):
    """Test pattern relationship management."""
    # Create two patterns
    pattern1_id = await social_service.register_pattern({
        "type": "social_practice",
        "field_state": {"energy": 0.8, "coherence": 0.7}
    })
    
    pattern2_id = await social_service.register_pattern({
        "type": "social_practice",
        "field_state": {"energy": 0.7, "coherence": 0.8}
    })
    
    # Evolve both to practices
    for pid in [pattern1_id, pattern2_id]:
        await social_service.track_practice_evolution(
            pid,
            {"adoption_level": 0.8, "institutionalization": 0.7}
        )
    
    # Update practice relationships
    await social_service._update_practice_relationships(pattern1_id)
    
    # Verify relationship created
    relationships = await social_service._pattern_manager._relationship_store.find_relationships({
        "source_id": pattern1_id,
        "target_id": pattern2_id,
        "type": "practice_alignment"
    })
    
    assert relationships.success
    assert len(relationships.data) == 1
    assert "alignment_score" in relationships.data[0]["properties"]

@pytest.mark.asyncio
async def test_adaptive_core_integration(social_service):
    """Test integration with adaptive core interface."""
    # Register pattern through adaptive interface
    pattern_id = await social_service.register_pattern({
        "type": "social_practice",
        "field_state": {"energy": 0.8, "coherence": 0.7}
    })
    
    # Get metrics through interface
    metrics = await social_service.get_pattern_metrics(pattern_id)
    assert metrics.coherence == 0.7
    assert metrics.signal_strength == 0.8
    assert "adoption_rate" in metrics.flow_metrics
    
    # Update through interface
    new_state = {"field_state": {"energy": 0.9}}
    await social_service.update_pattern_state(pattern_id, new_state)
    
    # Verify update
    result = await social_service._pattern_manager._pattern_store.find_patterns({"id": pattern_id})
    assert result.success
    assert result.data[0]["field_state"]["energy"] == 0.9

@pytest.mark.asyncio
async def test_quality_metrics_integration(social_service):
    """Test integration with pattern quality analysis."""
    # Register pattern with strong signals
    pattern_id = await social_service.register_pattern({
        "type": "social_practice",
        "field_state": {
            "energy": 0.9,
            "coherence": 0.8,
            "flow": 0.7
        }
    })
    
    # Track evolution with quality metrics
    await social_service.track_practice_evolution(
        pattern_id,
        {
            "adoption_level": 0.9,
            "institutionalization": 0.8
        }
    )
    
    # Verify quality metrics
    pattern = await social_service._pattern_manager.get_pattern(pattern_id)
    assert pattern.success
    assert "quality" in pattern.data
    assert pattern.data["quality"]["signal_strength"] > 0.7
    assert pattern.data["quality"]["persistence"] > 0.7
    
    # Verify metrics affect practice formation
    assert pattern.data["state"] == PatternState.STABLE.value
