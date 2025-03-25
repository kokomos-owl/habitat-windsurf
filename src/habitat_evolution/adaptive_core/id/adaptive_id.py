"""
Represents an adaptive concept with versioning, relationships, and context tracking capabilities.
"""

import uuid
import json
import threading
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
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
        old_value = self.get_temporal_context(key)
        
        with self._lock:
            if key not in self.temporal_context:
                self.temporal_context[key] = {}
            
            timestamp = datetime.now().isoformat()
            self.temporal_context[key][timestamp] = {
                "value": value,
                "origin": origin
            }
            self.metadata["last_modified"] = timestamp
            
        # Notify learning windows of the change
        self.notify_state_change(
            "temporal_context", 
            old_value, 
            {"key": key, "value": value}, 
            origin
        )

    def update_spatial_context(self, key: str, value: Any, origin: str) -> None:
        """Update spatial context."""
        # Get old value before updating
        old_value = self.get_spatial_context(key)
        
        with self._lock:
            if key in self.spatial_context:
                self.spatial_context[key] = value
                self.metadata["last_modified"] = datetime.now().isoformat()
            else:
                raise InvalidValueError(f"Invalid spatial context key: {key}")
                
        # Notify learning windows of the change
        self.notify_state_change(
            "spatial_context", 
            old_value, 
            {"key": key, "value": value}, 
            origin
        )

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
        
    def register_with_field_observer(self, field_observer) -> None:
        """Register this ID with a field observer for field-aware tracking.
        
        Args:
            field_observer: The field observer to register with
        """
        with self._lock:
            if not hasattr(self, 'field_observers'):
                self.field_observers = []
            
            # Avoid duplicate registrations
            if field_observer not in self.field_observers:
                self.field_observers.append(field_observer)
                
                # Provide initial state to field observer
                try:
                    # Create context with current state information including vector properties
                    context = {
                        "entity_id": self.id,
                        "entity_type": "adaptive_id",
                        "base_concept": self.base_concept,
                        "current_version": self.current_version,
                        "last_modified": self.metadata["last_modified"],
                        "stability": self.confidence,  # Use confidence as stability proxy
                        "tonic_value": 0.5,  # Default tonic value
                        "vector_properties": {
                            "temporal_context": list(self.temporal_context.keys()),
                            "spatial_context": list(self.spatial_context.keys()),
                            "relationships": list(self.relationships.keys()) if hasattr(self, 'relationships') else []
                        }
                    }
                    
                    # Use asyncio if available, otherwise fall back to direct observation
                    import asyncio
                    try:
                        if asyncio.get_event_loop().is_running():
                            asyncio.create_task(field_observer.observe(context))
                        else:
                            # Direct observation for testing environments
                            field_observer.observations.append({"context": context, "time": datetime.now()})
                    except (RuntimeError, ImportError):
                        # Direct observation as fallback
                        field_observer.observations.append({"context": context, "time": datetime.now()})
                except Exception as e:
                    self.logger.error(f"Error registering with field observer: {e}")

    def register_with_learning_window(self, learning_window) -> None:
        """Register this ID with a learning window for change tracking.
        
        Args:
            learning_window: The learning window to register with
        """
        with self._lock:
            if not hasattr(self, 'learning_windows'):
                self.learning_windows = []
            
            # Avoid duplicate registrations
            if learning_window not in self.learning_windows:
                self.learning_windows.append(learning_window)
    
    def notify_state_change(self, change_type: str, old_value: Any, new_value: Any, origin: str) -> None:
        """Notify all registered learning windows and field observers of state changes.
        
        Args:
            change_type: Type of change (e.g., 'field_metrics', 'relationship')
            old_value: Previous value (can be None)
            new_value: New value
            origin: Origin of the change
        """
        # Calculate tonic value based on change type for tonic-harmonic analysis
        # Temporal context changes have higher tonic values
        tonic_value = 0.7 if change_type == "temporal_context" else 0.5
        
        # Use confidence as stability proxy
        stability = self.confidence
        
        # Notify learning windows
        if hasattr(self, 'learning_windows'):
            for window in self.learning_windows:
                try:
                    # Use the standard parameter order: entity_id, change_type, old_value, new_value, origin
                    window.record_state_change(
                        self.id,
                        change_type,
                        old_value,
                        new_value,
                        origin
                    )
                except Exception as e:
                    # Log error but continue with other windows
                    import logging
                    logging.getLogger(__name__).error(f"Error notifying learning window: {e}")
        
        # Notify field observers with tonic-harmonic context
        if hasattr(self, 'field_observers'):
            for observer in self.field_observers:
                try:
                    # Create context with change information including vector properties
                    context = {
                        "entity_id": self.id,
                        "entity_type": "adaptive_id",
                        "change_type": change_type,
                        "old_value": old_value,
                        "new_value": new_value,
                        "origin": origin,
                        "timestamp": datetime.now().isoformat(),
                        "stability": stability,
                        "tonic_value": tonic_value,
                        "vector_properties": {
                            "temporal_context": list(self.temporal_context.keys()),
                            "spatial_context": list(self.spatial_context.keys()),
                            "relationships": list(self.relationships.keys()) if hasattr(self, 'relationships') else []
                        },
                        "field_properties": {
                            "coherence": self.confidence,  # Use confidence as coherence proxy
                            "navigability": 0.5 + (self.confidence * 0.5),  # Derive navigability from confidence
                            "stability": stability
                        }
                    }
                    
                    # Calculate harmonic value (stability * tonic) for tonic-harmonic analysis
                    context["harmonic_value"] = stability * tonic_value
                    
                    # Use asyncio if available, otherwise fall back to direct observation
                    try:
                        import asyncio
                        if asyncio.get_event_loop().is_running():
                            asyncio.create_task(observer.observe(context))
                        else:
                            # Direct observation for testing environments
                            observer.observations.append({"context": context, "time": datetime.now()})
                    except (RuntimeError, ImportError):
                        # Direct observation as fallback
                        observer.observations.append({"context": context, "time": datetime.now()})
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Error notifying field observer: {e}")
    
    def get_version_history(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[Any]:
        """Get version history within a time window.
        
        Args:
            start_time: ISO format timestamp for start of window (inclusive)
            end_time: ISO format timestamp for end of window (inclusive)
            
        Returns:
            List of versions within the specified time window
        """
        history = []
        with self._lock:
            for version_id, version in self.versions.items():
                # Check if version is within time window
                if (start_time is None or version.timestamp >= start_time) and \
                   (end_time is None or version.timestamp <= end_time):
                    history.append(version)
        
        # Sort by timestamp
        return sorted(history, key=lambda v: v.timestamp)
        
    def get_coherence(self) -> float:
        """Get the coherence score for this adaptive ID.
        
        Coherence represents how well the concept maintains its identity across contexts.
        Higher values indicate stronger semantic stability.
        
        Returns:
            float: Coherence score between 0.0 and 1.0
        """
        # For now, use confidence as a proxy for coherence
        # In a more sophisticated implementation, this would analyze temporal context
        # and relationship stability to derive a true coherence score
        base_coherence = self.confidence
        
        # Adjust based on temporal context stability if available
        if hasattr(self, 'temporal_context') and self.temporal_context:
            # More stable temporal contexts increase coherence
            context_stability = min(1.0, len(self.temporal_context) / 10.0)
            return (base_coherence * 0.7) + (context_stability * 0.3)
        
        return base_coherence
    
    def get_capaciousness(self) -> float:
        """Get the capaciousness score for this adaptive ID.
        
        Capaciousness represents how well the concept can absorb and integrate new information.
        Higher values indicate greater ability to expand while maintaining identity.
        
        Returns:
            float: Capaciousness score between 0.0 and 1.0
        """
        # Base capaciousness is inversely related to uncertainty
        # Lower uncertainty means higher capacity to absorb new information
        base_capaciousness = 1.0 - self.uncertainty
        
        # Adjust based on version history if available
        # More versions indicate higher capaciousness (ability to evolve)
        if hasattr(self, 'versions') and self.versions:
            version_factor = min(1.0, len(self.versions) / 20.0)
            return (base_capaciousness * 0.6) + (version_factor * 0.4)
        
        return base_capaciousness
    
    def get_directionality_dict(self) -> Dict[str, float]:
        """Get the directionality metrics for this adaptive ID.
        
        Directionality represents the tendency of the concept to evolve in particular directions.
        This returns multiple directional metrics to capture different aspects of evolution.
        
        Returns:
            Dict[str, float]: Dictionary of directionality metrics
        """
        # Initialize with default directionality metrics
        directionality = {
            "expansion": 0.5,  # Tendency to expand in scope
            "contraction": 0.5,  # Tendency to become more focused
            "stability": self.confidence,  # Tendency to maintain current state
            "volatility": self.uncertainty  # Tendency to change rapidly
        }
        
        # If we have version history, calculate expansion/contraction based on changes
        if hasattr(self, 'versions') and len(self.versions) > 1:
            # More versions indicate higher expansion potential
            version_count = len(self.versions)
            directionality["expansion"] = min(0.9, 0.3 + (version_count / 20.0))
            directionality["contraction"] = max(0.1, 0.7 - (version_count / 20.0))
        
        # If we have temporal context, adjust stability/volatility
        if hasattr(self, 'temporal_context') and self.temporal_context:
            context_count = len(self.temporal_context)
            # More contexts indicate higher stability
            directionality["stability"] = min(0.9, 0.4 + (context_count / 15.0))
            directionality["volatility"] = max(0.1, 0.6 - (context_count / 15.0))
        
        return directionality
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AdaptiveID to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of this AdaptiveID
        """
        # Create a basic dictionary with essential properties
        result = {
            "id": self.id,
            "base_concept": self.base_concept,
            "weight": self.weight,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "metadata": self.metadata,
            "current_version": self.current_version
        }
        
        # Add temporal context if available
        if hasattr(self, 'temporal_context') and self.temporal_context:
            result["temporal_context"] = {}
            for key, timestamps in self.temporal_context.items():
                result["temporal_context"][key] = {
                    ts: {
                        "value": data["value"],
                        "origin": data["origin"]
                    } for ts, data in timestamps.items()
                }
        
        # Add spatial context if available
        if hasattr(self, 'spatial_context') and self.spatial_context:
            result["spatial_context"] = self.spatial_context
        
        # Add pattern propensities
        result["pattern_propensities"] = {
            "coherence": self.get_coherence(),
            "capaciousness": self.get_capaciousness(),
            "directionality": self.get_directionality_dict()
        }
        
        # Add user interactions if available
        if hasattr(self, 'user_interactions') and self.user_interactions:
            result["user_interactions"] = self.user_interactions
        
        return result
