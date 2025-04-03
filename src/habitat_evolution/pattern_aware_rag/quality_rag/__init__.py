"""
Quality-aware RAG with context-aware pattern extraction.

This package extends the pattern-aware RAG system with context-aware
pattern extraction and quality assessment paths.
"""

from .context_aware_rag import ContextAwareRAG
from .quality_enhanced_retrieval import QualityEnhancedRetrieval

__all__ = [
    'ContextAwareRAG',
    'QualityEnhancedRetrieval',
]
