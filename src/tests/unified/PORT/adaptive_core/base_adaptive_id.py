"""Base interface for AdaptiveID functionality."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from habitat_test.core.logging.logger import LoggingManager, LogContext

class BaseAdaptiveID(ABC):
    """Abstract base class for AdaptiveID implementations."""
    
    @abstractmethod
    def initialize_logging(self) -> LoggingManager:
        """Initialize logging with proper context.
        
        Returns:
            LoggingManager: Configured logging manager instance
        """
        pass
    
    @abstractmethod
    def get_state_at_time(self, timestamp: str) -> Dict[str, Any]:
        """Retrieve the state at a specific timestamp."""
        pass
        
    @abstractmethod
    def compare_states(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """Compare two states and return differences."""
        pass
        
    @abstractmethod
    def create_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of current state."""
        pass
        
    @abstractmethod
    def restore_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from a snapshot."""
        pass
        
    @abstractmethod
    def update_temporal_context(self, key: str, value: Any, origin: str) -> None:
        """Update temporal context."""
        pass
        
    @abstractmethod
    def update_spatial_context(self, key: str, value: Any, origin: str) -> None:
        """Update spatial context."""
        pass
        
    @abstractmethod
    def get_temporal_context(self, key: str) -> Any:
        """Get temporal context value."""
        pass
        
    @abstractmethod
    def get_spatial_context(self, key: str) -> Any:
        """Get spatial context value."""
        pass
