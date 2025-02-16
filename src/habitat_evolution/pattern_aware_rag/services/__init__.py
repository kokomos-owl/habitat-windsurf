"""Service Layer for Pattern-Aware RAG.

This module provides high-level services and integrations:

1. Claude Integration:
   - State handling for Claude
   - Query processing
   - Response management

2. Graph Services:
   - Graph operations
   - Pattern management
   - State persistence

3. LangChain Integration:
   - Embedding management
   - Vector operations
   - Chain configuration

Key Services:
- ClaudeStateHandler: Claude integration service
- GraphService: Graph operation service
- ClaudeLangChainIntegration: LangChain bridge service

Usage:
    from habitat_evolution.pattern_aware_rag.services import (
        ClaudeStateHandler,
        GraphService,
        ClaudeLangChainIntegration
    )
"""

from ..core import *
from ..state.claude_state_handler import ClaudeStateHandler
from ..state.graph_service import GraphService
from ..state.langchain_config import ClaudeLangChainIntegration

__all__ = [
    'ClaudeStateHandler',
    'GraphService',
    'ClaudeLangChainIntegration'
]
