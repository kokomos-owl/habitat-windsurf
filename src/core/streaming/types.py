"""Type definitions for streaming infrastructure."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class DocumentChunk:
    """A chunk of a streaming document."""
    chunk_id: str
    content: str
    position: int
    total_chunks: int
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class StreamingPattern:
    """A pattern detected in a streaming document."""
    pattern_id: str
    confidence: float
    vector_space: Dict[str, float]
    temporal_context: Dict[str, Any]
    chunk_ids: List[str] = field(default_factory=list)
    is_complete: bool = False
