"""Test suite for streaming pattern detection.

This test suite validates the pattern detection system's ability to:
1. Process streaming content while maintaining coherence
2. Track patterns across chunk boundaries
3. Evolve patterns during streaming
4. Maintain natural interfaces between chunks
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime

from habitat_test.core.pattern_detection import (
    PatternDetector,
    DetectionMode,
    PatternCandidate
)
from habitat_test.core.pattern_evolution import (
    PatternEvolutionManager,
    PatternType,
    DocumentStructureType,
    EvolutionState
)
from habitat_test.core.streaming import StreamingContext
from .test_pattern_evolution import MockPatternCore

@pytest.fixture
def stream_context():
    """Create streaming context for testing."""
    return StreamingContext()

@pytest.fixture
def evolution_manager():
    """Create pattern evolution manager for testing."""
    pattern_core = MockPatternCore()
    return PatternEvolutionManager(pattern_core)

@pytest.fixture
def pattern_detector(evolution_manager):
    """Create pattern detector for testing."""
    return PatternDetector(evolution_manager)

@pytest.fixture
def sample_stream():
    """Create sample streaming content."""
    return [
        "System shows periodic behavior in component A",
        "Component A influences component B",
        "Periodic pattern continues in component A",
        "New relationship emerges: B affects C",
        "Component A shows consistent periodicity"
    ]

class TestStreamingDetection:
    """Test streaming pattern detection functionality."""
    
    async def test_chunk_processing(
        self,
        pattern_detector,
        stream_context,
        sample_stream
    ):
        """Test basic chunk processing and pattern detection.
        
        This test verifies that:
        1. Chunks are processed correctly
        2. Patterns are detected within chunks
        3. Pattern confidence evolves appropriately
        4. Context is maintained across chunks
        """
        patterns_by_chunk = []
        
        for chunk in sample_stream:
            patterns = await pattern_detector.process_streaming_chunk(
                chunk,
                stream_context
            )
            patterns_by_chunk.append(patterns)
            
        # Verify pattern evolution
        assert len(patterns_by_chunk) == len(sample_stream)
        assert patterns_by_chunk[-1][0].confidence > patterns_by_chunk[0][0].confidence
        
        # Verify context maintenance
        assert stream_context.is_coherent()
        assert stream_context.chunk_count == len(sample_stream)
        
    async def test_pattern_persistence(
        self,
        pattern_detector,
        stream_context,
        sample_stream
    ):
        """Test pattern persistence across chunks.
        
        This test verifies that:
        1. Patterns persist across chunk boundaries
        2. Pattern relationships are maintained
        3. Pattern evolution continues across chunks
        """
        active_patterns = set()
        
        for chunk in sample_stream:
            patterns = await pattern_detector.process_streaming_chunk(
                chunk,
                stream_context
            )
            
            # Track active patterns
            for pattern in patterns:
                active_patterns.add(pattern.pattern_id)
                
                # Verify pattern evolution
                if pattern.pattern_id in stream_context.active_patterns:
                    assert pattern.confidence >= stream_context.active_patterns[pattern.pattern_id]
                    
        # Verify pattern persistence
        assert len(active_patterns) > 0
        assert len(stream_context.active_patterns) > 0
        
    async def test_coherence_maintenance(
        self,
        pattern_detector,
        stream_context,
        sample_stream
    ):
        """Test coherence maintenance during streaming.
        
        This test verifies that:
        1. Coherence is maintained across chunks
        2. Pattern relationships respect coherence
        3. Evolution maintains system coherence
        """
        for chunk in sample_stream:
            patterns = await pattern_detector.process_streaming_chunk(
                chunk,
                stream_context
            )
            
            # Verify coherence after each chunk
            assert stream_context.is_coherent()
            
            # Verify pattern coherence
            for pattern in patterns:
                assert pattern.confidence >= 0.5  # Minimum confidence
                
        # Verify final state coherence
        assert stream_context.is_coherent()
        assert len(stream_context.active_patterns) > 0
