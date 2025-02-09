"""Document streaming infrastructure for real-time pattern processing."""

from typing import AsyncIterator, Dict, List, Optional
import asyncio
from logging import getLogger

from ..processor import RiskMetric
from .pattern_extraction import StreamingPatternExtractor
from ..pattern_evolution import EvolutionMetrics
from .types import DocumentChunk, StreamingPattern

logger = getLogger(__name__)

class DocumentStreamProcessor:
    """Processes document streams in real-time."""
    
    def __init__(self, chunk_size: int = 4096):
        self.chunk_size = chunk_size
        self.pattern_buffer: Dict[str, StreamingPattern] = {}
        self.current_chunks: Dict[str, DocumentChunk] = {}
        self.pattern_extractor = StreamingPatternExtractor()
        
    async def process_stream(self, 
                           document_stream: AsyncIterator[DocumentChunk],
                           callback = None) -> AsyncIterator[StreamingPattern]:
        """Process document stream and yield patterns as they emerge.
        
        Args:
            document_stream: AsyncIterator yielding DocumentChunk objects
            callback: Optional callback for pattern updates
            
        Yields:
            StreamingPattern objects as they are detected and updated
        """
        try:
            async for chunk in document_stream:
                # Store chunk immediately
                self.current_chunks[chunk.chunk_id] = chunk
                
                # Extract patterns
                new_patterns = await self.extract_patterns(chunk)
                
                # Update existing patterns
                for pattern in new_patterns:
                    existing = None
                    if pattern.pattern_id in self.pattern_buffer:
                        # Update existing pattern
                        existing = self.pattern_buffer[pattern.pattern_id]
                        if chunk.chunk_id not in existing.chunk_ids:
                            existing.chunk_ids.append(chunk.chunk_id)
                        existing.confidence = max(existing.confidence, pattern.confidence)
                        
                        # Update vector space with weighted average
                        for dim, value in pattern.vector_space.items():
                            existing.vector_space[dim] = (
                                existing.vector_space[dim] * (len(existing.chunk_ids) - 1) + value
                            ) / len(existing.chunk_ids)
                    else:
                        # Add new pattern
                        pattern.chunk_ids = [chunk.chunk_id]
                        self.pattern_buffer[pattern.pattern_id] = pattern
                        existing = pattern
                    
                    # Update pattern state
                    chunk_positions = {self.current_chunks[cid].position 
                                     for cid in existing.chunk_ids}
                    total_chunks = chunk.total_chunks
                    
                    # Calculate completion state
                    has_all_positions = len(chunk_positions) == total_chunks and \
                                      set(range(total_chunks)) == chunk_positions
                    
                    # Calculate pattern evolution metrics
                    stability = min(1.0, len(chunk_positions) / total_chunks)
                    coherence = 0.5 + (0.5 * stability)  # Increases with stability
                    emergence_rate = 1.0 - stability  # Decreases as pattern completes
                    cross_pattern_flow = 0.3 + (0.4 * len(self.pattern_buffer) / 5)
                    energy_state = 0.4 + (0.6 * stability)
                    adaptation_rate = max(0.1, 1.0 - stability)
                    
                    # Update vector space
                    existing.vector_space = {
                        'stability': stability,
                        'coherence': coherence,
                        'emergence_rate': emergence_rate,
                        'cross_pattern_flow': cross_pattern_flow,
                        'energy_state': energy_state,
                        'adaptation_rate': adaptation_rate
                    }
                    
                    # Mark as complete if we have all chunks
                    if has_all_positions:
                        existing.is_complete = True
                    
                    if callback:
                        await callback(existing)
                    yield existing
                
                # Store chunk count for testing
                if callback:
                    await callback({'type': 'chunk_count', 'count': len(self.current_chunks)})
                    
                # Clean up old chunks periodically
                if chunk.position % 3 == 0:  # Clean up more frequently
                    await self._cleanup_old_chunks()
                
        except Exception as e:
            logger.error(f"Error processing document stream: {e}")
            raise
            
    async def extract_patterns(self, chunk: DocumentChunk) -> List[StreamingPattern]:
        """Extract patterns from a document chunk.
        
        Args:
            chunk: DocumentChunk to process
            
        Returns:
            List of StreamingPattern objects detected in the chunk
        """
        return await self.pattern_extractor.extract_patterns(chunk)
        
    async def _cleanup_old_chunks(self):
        """Remove processed chunks that are no longer needed."""
        # Keep only chunks from incomplete patterns
        needed_chunks = set()
        
        # First pass: identify needed chunks
        for pattern in self.pattern_buffer.values():
            if not pattern.is_complete:
                needed_chunks.update(pattern.chunk_ids)
        
        # Second pass: clean up pattern buffer
        completed_patterns = []
        for pattern_id, pattern in self.pattern_buffer.items():
            if pattern.is_complete:
                completed_patterns.append(pattern_id)
        
        # Remove completed patterns
        for pattern_id in completed_patterns:
            del self.pattern_buffer[pattern_id]
        
        # Finally, remove unneeded chunks
        current_chunks = {}
        for chunk_id, chunk in self.current_chunks.items():
            if chunk_id in needed_chunks or any(
                not pattern.is_complete 
                for pattern in self.pattern_buffer.values() 
                if chunk_id in pattern.chunk_ids
            ):
                current_chunks[chunk_id] = chunk
        
        self.current_chunks = current_chunks
