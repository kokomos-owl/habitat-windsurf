"""Pattern-Aware RAG System

This package provides a coherence interface for RAG operations.
Backward compatibility imports are maintained while the new structure
provides better organization.
"""

# Core components
from .core import *

# Services
from .services import *

# Backward compatibility imports
from .state.state_handler import GraphStateHandler as _GraphStateHandler
from .state.state_evolution import StateEvolutionTracker as _StateEvolutionTracker
from .state.claude_state_handler import ClaudeStateHandler as _ClaudeStateHandler
from .state.graph_service import GraphService as _GraphService
from .state.langchain_config import ClaudeLangChainIntegration as _ClaudeLangChainIntegration
from .bridges.adaptive_state_bridge import AdaptiveStateManager as _AdaptiveStateManager
from .learning.learning_control import (
    EventCoordinator as _EventCoordinator,
    LearningWindow as _LearningWindow,
    BackPressureController as _BackPressureController
)

# Maintain old import paths
GraphStateHandler = _GraphStateHandler
StateEvolutionTracker = _StateEvolutionTracker
ClaudeStateHandler = _ClaudeStateHandler
GraphService = _GraphService
ClaudeLangChainIntegration = _ClaudeLangChainIntegration
AdaptiveStateManager = _AdaptiveStateManager
EventCoordinator = _EventCoordinator
LearningWindow = _LearningWindow
BackPressureController = _BackPressureController

__all__ = [
    'GraphStateHandler',
    'StateEvolutionTracker',
    'ClaudeStateHandler',
    'GraphService',
    'ClaudeLangChainIntegration',
    'AdaptiveStateManager',
    'EventCoordinator',
    'LearningWindow',
    'BackPressureController'
]
