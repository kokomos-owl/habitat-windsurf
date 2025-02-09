"""Tests for document streaming infrastructure."""

import pytest
import asyncio
from datetime import datetime
from typing import List, AsyncIterator

from src.core.streaming.document_stream import (
    DocumentChunk,
    StreamingPattern,
    DocumentStreamProcessor
)

async def create_test_stream(content: str, chunk_size: int = 100) -> AsyncIterator[DocumentChunk]:
    """Create a test document stream."""
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    total_chunks = len(chunks)
    
    for i, chunk_content in enumerate(chunks):
        chunk = DocumentChunk(
            chunk_id=f"test_chunk_{i}",
            content=chunk_content,
            position=i,
            total_chunks=total_chunks,
            metadata={"test": True},
            timestamp=datetime.utcnow()
        )
        yield chunk
        await asyncio.sleep(0.01)  # Simulate network delay

@pytest.mark.asyncio
async def test_stream_processing():
    """Test basic stream processing functionality."""
    processor = DocumentStreamProcessor(chunk_size=100)
    test_content = "This is a test document with climate patterns. " * 10
    
    patterns: List[StreamingPattern] = []
    async for pattern in processor.process_stream(create_test_stream(test_content)):
        patterns.append(pattern)
        
    # Basic assertions
    assert len(processor.current_chunks) > 0
    assert all(isinstance(p, StreamingPattern) for p in patterns)

@pytest.mark.asyncio
async def test_pattern_evolution():
    """Test pattern evolution across chunks."""
    processor = DocumentStreamProcessor(chunk_size=100)
    test_content = """
    This is a test document that demonstrates pattern evolution.
    We use test content to verify our pattern detection.
    Multiple test documents help validate the system.
    """ * 3
    
    pattern_updates = {}
    async for pattern in processor.process_stream(create_test_stream(test_content)):
        pattern_updates[pattern.pattern_id] = pattern_updates.get(pattern.pattern_id, 0) + 1
        
    # Patterns should be updated multiple times as new chunks arrive
    assert any(updates > 1 for updates in pattern_updates.values())

@pytest.mark.asyncio
async def test_pattern_completion():
    """Test pattern completion detection."""
    processor = DocumentStreamProcessor(chunk_size=100)
    test_content = "Short test content for completion detection."
    
    completed_patterns = []
    async for pattern in processor.process_stream(create_test_stream(test_content)):
        if pattern.is_complete:
            completed_patterns.append(pattern)
            
    # At least some patterns should be marked as complete
    assert len(completed_patterns) > 0
    assert all(p.is_complete for p in completed_patterns)

@pytest.mark.asyncio
async def test_vector_space_updates():
    """Test vector space dimension updates."""
    processor = DocumentStreamProcessor(chunk_size=100)
    test_content = """
    Temperature rise of 2 degrees expected by 2050.
    Sea level rise projections show 0.5m increase.
    Precipitation patterns will change significantly.
    """ * 2
    
    final_patterns = []
    async for pattern in processor.process_stream(create_test_stream(test_content)):
        if pattern.is_complete:
            final_patterns.append(pattern)
            
    # Check vector space dimensions
    for pattern in final_patterns:
        assert 'stability' in pattern.vector_space
        assert 'coherence' in pattern.vector_space
        assert 'emergence_rate' in pattern.vector_space
        assert 'cross_pattern_flow' in pattern.vector_space
        assert 'energy_state' in pattern.vector_space
        assert 'adaptation_rate' in pattern.vector_space
        
        # All dimensions should be between 0 and 1
        assert all(0 <= v <= 1 for v in pattern.vector_space.values())

@pytest.mark.asyncio
async def test_chunk_cleanup():
    """Test cleanup of processed chunks."""
    processor = DocumentStreamProcessor(chunk_size=100)
    test_content = "Test document for verification. " * 10
    
    chunk_counts = []
    async def count_callback(data):
        if isinstance(data, dict) and data.get('type') == 'chunk_count':
            chunk_counts.append(data['count'])
            
    async for _ in processor.process_stream(create_test_stream(test_content), callback=count_callback):
        pass
        
    # Chunk count should not grow indefinitely
    assert max(chunk_counts) < len(test_content) / processor.chunk_size + 1
    
    # After processing, only chunks for incomplete patterns should remain
    incomplete_patterns = [p for p in processor.pattern_buffer.values() if not p.is_complete]
    needed_chunks = set()
    for pattern in incomplete_patterns:
        needed_chunks.update(pattern.chunk_ids)
        
    assert all(chunk_id in needed_chunks for chunk_id in processor.current_chunks)
