"""
Tests for the field state service implementation.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from habitat_evolution.core.services.field.interfaces import FieldState
from habitat_evolution.core.services.field.field_state_service import ConcreteFieldStateService
from habitat_evolution.core.storage.field_repository import FieldRepository
from habitat_evolution.core.services.event_bus import EventBus

@pytest.fixture
def field_repository():
    """Create a mock field repository"""
    repository = Mock(spec=FieldRepository)
    repository.get_field_state = AsyncMock()
    repository.update_field_state = AsyncMock()
    return repository

@pytest.fixture
def event_bus():
    """Create a mock event bus"""
    bus = Mock(spec=EventBus)
    bus.emit = AsyncMock()
    return bus

@pytest.fixture
def field_service(field_repository, event_bus):
    """Create a field service instance"""
    return ConcreteFieldStateService(field_repository, event_bus)

@pytest.fixture
def sample_field_state():
    """Create a sample field state"""
    return FieldState(
        field_id="test_field",
        timestamp=datetime.now(),
        potential=0.75,
        gradient={"x": 0.5, "y": -0.3},
        stability=0.85,
        metadata={"type": "test"}
    )

@pytest.mark.asyncio
async def test_get_field_state(field_service, field_repository, sample_field_state):
    """Test getting field state"""
    # Setup
    field_repository.get_field_state.return_value = sample_field_state
    
    # Execute
    result = await field_service.get_field_state("test_field")
    
    # Verify
    assert result is not None
    assert result.field_id == sample_field_state.field_id
    assert result.potential == sample_field_state.potential
    assert result.stability == sample_field_state.stability
    field_repository.get_field_state.assert_called_once_with("test_field")

@pytest.mark.asyncio
async def test_update_field_state(field_service, field_repository, event_bus, sample_field_state):
    """Test updating field state"""
    # Setup
    field_repository.get_field_state.return_value = sample_field_state
    
    # Execute
    await field_service.update_field_state("test_field", sample_field_state)
    
    # Verify
    field_repository.update_field_state.assert_called_once()
    event_bus.emit.assert_called_with(
        "field.state.updated",
        {
            "field_id": "test_field",
            "timestamp": pytest.approx(datetime.now(), rel=1),
            "stability": sample_field_state.stability,
            "potential": sample_field_state.potential
        }
    )

@pytest.mark.asyncio
async def test_calculate_field_stability(field_service, field_repository, event_bus, sample_field_state):
    """Test calculating field stability"""
    # Setup
    field_repository.get_field_state.return_value = sample_field_state
    
    # Execute
    stability = await field_service.calculate_field_stability("test_field")
    
    # Verify
    assert stability > 0
    assert stability <= 1.0
    event_bus.emit.assert_called_with(
        "field.stability.calculated",
        {
            "field_id": "test_field",
            "stability": stability,
            "timestamp": pytest.approx(datetime.now(), rel=1)
        }
    )

@pytest.mark.asyncio
async def test_error_handling(field_service, field_repository, event_bus):
    """Test error handling in field service"""
    # Setup
    error_message = "Test error"
    field_repository.get_field_state.side_effect = Exception(error_message)
    
    # Execute and verify
    with pytest.raises(Exception) as exc_info:
        await field_service.get_field_state("test_field")
    
    assert str(exc_info.value) == error_message
    event_bus.emit.assert_called_with(
        "field.state.error",
        {
            "field_id": "test_field",
            "error": error_message,
            "timestamp": pytest.approx(datetime.now(), rel=1)
        }
    )

@pytest.mark.asyncio
async def test_field_not_found(field_service, field_repository):
    """Test handling of non-existent field"""
    # Setup
    field_repository.get_field_state.return_value = None
    
    # Execute
    result = await field_service.get_field_state("nonexistent_field")
    
    # Verify
    assert result is None
    field_repository.get_field_state.assert_called_once_with("nonexistent_field")
