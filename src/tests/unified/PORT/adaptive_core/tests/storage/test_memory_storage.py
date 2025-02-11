"""Tests for in-memory storage implementations."""

import pytest
from typing import Dict, Any
from datetime import datetime, timedelta

from ...storage.memory import (
    InMemoryStateStore,
    InMemoryPatternStore,
    InMemoryRelationshipStore
)
from ...services.time_provider import TimeProvider

@pytest.fixture
async def state_store():
    """Create fresh state store for each test."""
    return InMemoryStateStore()

@pytest.fixture
async def pattern_store():
    """Create fresh pattern store for each test."""
    return InMemoryPatternStore()

@pytest.fixture
async def relationship_store():
    """Create fresh relationship store for each test."""
    return InMemoryRelationshipStore()

class TestInMemoryStateStore:
    """Tests for InMemoryStateStore."""
    
    async def test_save_and_load_state(self, state_store):
        """Test basic save and load operations."""
        test_id = "test1"
        test_state = {"key": "value"}
        
        # Save state
        result = await state_store.save_state(test_id, test_state)
        assert result.success
        assert result.data  # Version string
        
        # Load state
        load_result = await state_store.load_state(test_id)
        assert load_result.success
        assert load_result.data == test_state
        assert load_result.metadata.version == result.data
    
    async def test_versioned_state(self, state_store):
        """Test state versioning."""
        test_id = "test1"
        states = [
            {"version": i}
            for i in range(3)
        ]
        
        versions = []
        for state in states:
            result = await state_store.save_state(test_id, state)
            assert result.success
            versions.append(result.data)
        
        # Load specific version
        load_result = await state_store.load_state(test_id, versions[1])
        assert load_result.success
        assert load_result.data == states[1]
        
        # List versions
        list_result = await state_store.list_versions(test_id)
        assert list_result.success
        assert len(list_result.data) == 3
        
        # Verify chronological order
        for i in range(1, len(list_result.data)):
            assert list_result.data[i].created_at > list_result.data[i-1].created_at
    
    async def test_nonexistent_state(self, state_store):
        """Test loading non-existent state."""
        result = await state_store.load_state("nonexistent")
        assert not result.success
        assert "not found" in result.error.lower()

class TestInMemoryPatternStore:
    """Tests for InMemoryPatternStore."""
    
    async def test_save_and_find_pattern(self, pattern_store):
        """Test basic pattern operations."""
        test_pattern = {
            "type": "test",
            "value": 42,
            "metadata": {"tag": "test"}
        }
        
        # Save pattern
        save_result = await pattern_store.save_pattern(test_pattern)
        assert save_result.success
        pattern_id = save_result.data
        
        # Find by exact match
        find_result = await pattern_store.find_patterns({"type": "test"})
        assert find_result.success
        assert len(find_result.data) == 1
        assert find_result.data[0]["value"] == 42
        assert find_result.data[0]["id"] == pattern_id
    
    async def test_pattern_queries(self, pattern_store):
        """Test pattern querying."""
        patterns = [
            {"type": "test", "value": i}
            for i in range(5)
        ]
        
        for pattern in patterns:
            result = await pattern_store.save_pattern(pattern)
            assert result.success
        
        # Test limit
        limited = await pattern_store.find_patterns({"type": "test"}, limit=2)
        assert limited.success
        assert len(limited.data) == 2
        
        # Test offset
        offset = await pattern_store.find_patterns({"type": "test"}, offset=2)
        assert offset.success
        assert len(offset.data) == 3
    
    async def test_delete_pattern(self, pattern_store):
        """Test pattern deletion."""
        # Save pattern
        pattern = {"type": "test"}
        save_result = await pattern_store.save_pattern(pattern)
        assert save_result.success
        
        # Delete pattern
        delete_result = await pattern_store.delete_pattern(save_result.data)
        assert delete_result.success
        
        # Verify deletion
        find_result = await pattern_store.find_patterns({"type": "test"})
        assert find_result.success
        assert len(find_result.data) == 0

class TestInMemoryRelationshipStore:
    """Tests for InMemoryRelationshipStore."""
    
    async def test_save_and_find_relationship(self, relationship_store):
        """Test basic relationship operations."""
        source_id = "source1"
        target_id = "target1"
        rel_type = "test_rel"
        properties = {"weight": 0.5}
        
        # Save relationship
        save_result = await relationship_store.save_relationship(
            source_id, target_id, rel_type, properties
        )
        assert save_result.success
        rel_id = save_result.data
        
        # Find by type
        find_result = await relationship_store.find_relationships({"type": rel_type})
        assert find_result.success
        assert len(find_result.data) == 1
        assert find_result.data[0]["properties"]["weight"] == 0.5
    
    async def test_get_related(self, relationship_store):
        """Test getting related entities."""
        # Create a small graph
        relationships = [
            ("1", "2", "friend", {}),
            ("2", "3", "friend", {}),
            ("3", "1", "enemy", {})
        ]
        
        for source, target, type, props in relationships:
            result = await relationship_store.save_relationship(
                source, target, type, props
            )
            assert result.success
        
        # Test outgoing relationships
        outgoing = await relationship_store.get_related("1", direction="outgoing")
        assert outgoing.success
        assert len(outgoing.data) == 1
        assert outgoing.data[0]["target_id"] == "2"
        
        # Test incoming relationships
        incoming = await relationship_store.get_related("1", direction="incoming")
        assert incoming.success
        assert len(incoming.data) == 1
        assert incoming.data[0]["source_id"] == "3"
        
        # Test filtered by type
        friends = await relationship_store.get_related("2", type="friend")
        assert friends.success
        assert len(friends.data) == 2
    
    async def test_relationship_queries(self, relationship_store):
        """Test relationship querying."""
        # Create test relationships
        for i in range(5):
            result = await relationship_store.save_relationship(
                f"s{i}", f"t{i}", "test",
                {"value": i}
            )
            assert result.success
        
        # Test property queries
        query_result = await relationship_store.find_relationships({
            "type": "test",
            "properties": {"value": 3}
        })
        assert query_result.success
        assert len(query_result.data) == 1
        assert query_result.data[0]["properties"]["value"] == 3

async def test_storage_metadata(state_store, pattern_store, relationship_store):
    """Test metadata handling across stores."""
    # Test state metadata
    state_result = await state_store.save_state(
        "test", {},
        metadata={"tags": ["test"]}
    )
    assert state_result.metadata.tags == ["test"]
    
    # Test pattern metadata
    pattern_result = await pattern_store.save_pattern(
        {},
        metadata={"tags": ["test"]}
    )
    assert pattern_result.metadata.tags == ["test"]
    
    # Test relationship metadata
    rel_result = await relationship_store.save_relationship(
        "s", "t", "test", {},
        metadata={"tags": ["test"]}
    )
    assert rel_result.metadata.tags == ["test"]
