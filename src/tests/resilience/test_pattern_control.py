"""Tests for pattern control functionality."""
import pytest
from datetime import datetime, timedelta
import asyncio
from src.core.resilience.pattern_control import (
    Version,
    VersionState,
    PatternCircuitBreaker,
    BackpressureController,
    VersionedPatternManager,
    CircuitBreakerOpenError
)

@pytest.fixture
def version_sequence():
    """Generate a sequence of versions."""
    base_time = datetime.now()
    return [
        Version(1, 0, 0, base_time),
        Version(1, 1, 0, base_time + timedelta(days=1)),
        Version(2, 0, 0, base_time + timedelta(days=2))
    ]

@pytest.fixture
def mock_pattern():
    """Create a mock pattern."""
    class MockPattern:
        def __init__(self, id: str):
            self.id = id
            self.stability = 1.0
    return MockPattern("test_pattern")

class TestVersionControl:
    @pytest.mark.asyncio
    async def test_version_supersession(self, version_sequence, mock_pattern):
        """Test that newer versions supersede older ones."""
        manager = VersionedPatternManager()
        
        # Register patterns with increasing versions
        for version in version_sequence:
            await manager.register_pattern("test", mock_pattern, version)
        
        # Check latest version is active
        pattern = await manager.get_pattern("test")
        assert pattern.id == mock_pattern.id
        
        # Check older versions are superseded
        stored_versions = manager.patterns["test"]
        assert len(stored_versions) == 3
        assert stored_versions[0][0].state == VersionState.ACTIVE
        assert stored_versions[1][0].state == VersionState.SUPERSEDED
        assert stored_versions[2][0].state == VersionState.SUPERSEDED

class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_circuit_breaker_trip(self):
        """Test circuit breaker trips after failures."""
        breaker = PatternCircuitBreaker(failure_threshold=2)
        
        async def failing_operation():
            raise ValueError("Simulated failure")
        
        # First failure
        with pytest.raises(ValueError):
            await breaker.execute(failing_operation)
        assert breaker.state == CircuitState.CLOSED
        
        # Second failure should trip breaker
        with pytest.raises(ValueError):
            await breaker.execute(failing_operation)
        assert breaker.state == CircuitState.OPEN
        
        # Further attempts should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.execute(failing_operation)

class TestBackpressure:
    @pytest.mark.asyncio
    async def test_backpressure_queue_limit(self, mock_pattern):
        """Test backpressure limits pattern submission."""
        controller = BackpressureController(max_queue_size=2)
        
        # Should accept up to queue size
        assert await controller.submit(mock_pattern)
        assert await controller.submit(mock_pattern)
        
        # Should reject when queue is full
        assert not await controller.submit(mock_pattern)

    @pytest.mark.asyncio
    async def test_pattern_processing(self, mock_pattern):
        """Test pattern processing with backpressure."""
        controller = BackpressureController(max_queue_size=5)
        processed_patterns = []
        
        async def processor(pattern):
            processed_patterns.append(pattern)
            await asyncio.sleep(0.1)  # Simulate processing time
        
        # Start processing in background
        processing_task = asyncio.create_task(
            controller.start_processing(processor)
        )
        
        # Submit patterns
        await controller.submit(mock_pattern)
        await controller.submit(mock_pattern)
        
        # Allow some processing time
        await asyncio.sleep(0.3)
        
        # Stop processing
        controller.stop_processing()
        await processing_task
        
        assert len(processed_patterns) == 2

class TestIntegration:
    @pytest.mark.asyncio
    async def test_versioned_pattern_with_circuit_breaker(
        self,
        version_sequence,
        mock_pattern
    ):
        """Test version control with circuit breaker protection."""
        manager = VersionedPatternManager()
        
        # Register patterns successfully
        await manager.register_pattern("test", mock_pattern, version_sequence[0])
        
        # Simulate failures by trying to register None
        with pytest.raises(Exception):
            await manager.register_pattern("test", None, version_sequence[1])
        with pytest.raises(Exception):
            await manager.register_pattern("test", None, version_sequence[1])
        
        # Circuit should be open now
        with pytest.raises(CircuitBreakerOpenError):
            await manager.register_pattern(
                "test",
                mock_pattern,
                version_sequence[2]
            )
        
        # Original pattern should still be available
        pattern = await manager.get_pattern("test")
        assert pattern.id == mock_pattern.id
