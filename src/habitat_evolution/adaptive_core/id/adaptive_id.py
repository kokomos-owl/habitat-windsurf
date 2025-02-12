"""
Represents an adaptive concept with versioning, relationships, and context tracking capabilities.
"""

import uuid
import json
import threading
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from .base_adaptive_id import BaseAdaptiveID, LoggingManager

class AdaptiveIDException(Exception):
    """Base exception class for AdaptiveID-related errors."""
    pass

class VersionNotFoundException(AdaptiveIDException):
    """Raised when a specified version is not found."""
    pass

class InvalidValueError(AdaptiveIDException):
    """Raised when an invalid value is provided for an attribute."""
    pass

class Version:
    """Represents a version of a property in AdaptiveID."""
    def __init__(self, version_id: str, data: Dict[str, Any], timestamp: str, origin: str):
        self.version_id = version_id
        self.data = data
        self.timestamp = timestamp
        self.origin = origin

class AdaptiveID(BaseAdaptiveID):
    """
    Represents an adaptive concept in the knowledge base.

    This class allows for flexible, evolving, and traceable representation of concepts.
    It supports versioning, relationships, and integrates with relationship management.
    
    Thread-safety is ensured through the use of threading.Lock for critical operations.
    """

    def __init__(
        self,
        base_concept: str,
        creator_id: str,
        weight: float = 1.0,
        confidence: float = 1.0,
        uncertainty: float = 0.0,
    ):
        """Initialize an AdaptiveID instance.
        
        Args:
            base_concept: The base concept this ID represents
            weight: Initial weight of the concept (default: 1.0)
            confidence: Initial confidence score (default: 1.0)
            uncertainty: Initial uncertainty value (default: 0.0)
        """
        self.id = str(uuid.uuid4())
        self.base_concept = base_concept
        self.creator_id = creator_id
        self.weight = weight
        self.confidence = confidence
        self.uncertainty = uncertainty
        
        # Track user interactions
        self.user_interactions = {
            creator_id: {
                "role": "creator",
                "first_interaction": datetime.now().isoformat(),
                "last_interaction": datetime.now().isoformat(),
                "interaction_count": 1
            }
        }
        
        # Initialize logging
        self.logger = self.initialize_logging()
        
        # Initialize core components
        self._lock = threading.Lock()
        self.versions: Dict[str, Version] = {}
        self.current_version = str(uuid.uuid4())
        self.temporal_context: Dict[str, Dict[str, Any]] = {}
        self.spatial_context: Dict[str, Any] = {
            "latitude": None,
            "longitude": None,
            "geometry": None
        }
        
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "version_count": 0
        }
        
        # Initialize first version
        self._create_initial_version()

    def initialize_logging(self) -> LoggingManager:
        """Initialize logging with proper context."""
        return LoggingManager()  # TODO: Implement proper logging

    def _create_initial_version(self) -> None:
        """Create the initial version of the adaptive ID."""
        initial_data = {
            "base_concept": self.base_concept,
            "weight": self.weight,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty
        }
        
        with self._lock:
            self.versions[self.current_version] = Version(
                self.current_version,
                initial_data,
                self.metadata["created_at"],
                "initialization"
            )
            self.metadata["version_count"] = 1

    def get_state_at_time(self, timestamp: str) -> Dict[str, Any]:
        """Retrieve the state at a specific timestamp."""
        # TODO: Implement version retrieval by timestamp
        return {}

    def compare_states(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """Compare two states and return differences."""
        differences = {}
        all_keys = set(state1.keys()) | set(state2.keys())
        
        for key in all_keys:
            val1 = state1.get(key)
            val2 = state2.get(key)
            if val1 != val2:
                differences[key] = (val1, val2)
                
        return differences

    def create_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of current state."""
        with self._lock:
            return {
                "id": self.id,
                "base_concept": self.base_concept,
                "current_version": self.current_version,
                "versions": self.versions,
                "temporal_context": self.temporal_context,
                "spatial_context": self.spatial_context,
                "metadata": self.metadata,
            "user_interactions": self.user_interactions
            }

    def restore_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from a snapshot."""
        with self._lock:
            self.id = snapshot["id"]
            self.base_concept = snapshot["base_concept"]
            self.current_version = snapshot["current_version"]
            self.versions = snapshot["versions"]
            self.temporal_context = snapshot["temporal_context"]
            self.spatial_context = snapshot["spatial_context"]
            self.metadata = snapshot["metadata"]
            self.user_interactions = snapshot.get("user_interactions", {})

    def update_temporal_context(self, key: str, value: Any, origin: str) -> None:
        """Update temporal context."""
        with self._lock:
            if key not in self.temporal_context:
                self.temporal_context[key] = {}
            
            timestamp = datetime.now().isoformat()
            self.temporal_context[key][timestamp] = {
                "value": value,
                "origin": origin
            }
            self.metadata["last_modified"] = timestamp

    def update_spatial_context(self, key: str, value: Any, origin: str) -> None:
        """Update spatial context."""
        with self._lock:
            if key in self.spatial_context:
                self.spatial_context[key] = value
                self.metadata["last_modified"] = datetime.now().isoformat()
            else:
                raise InvalidValueError(f"Invalid spatial context key: {key}")

    def get_temporal_context(self, key: str) -> Any:
        """Get temporal context value."""
        if key not in self.temporal_context:
            return None
            
        # Return the most recent value
        timestamps = sorted(self.temporal_context[key].keys())
        if not timestamps:
            return None
            
        return self.temporal_context[key][timestamps[-1]]["value"]

    def get_spatial_context(self, key: str) -> Any:
        """Get spatial context value."""
        return self.spatial_context.get(key)
