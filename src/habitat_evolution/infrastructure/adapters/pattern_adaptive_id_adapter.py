"""
Pattern-AdaptiveID adapter for Habitat Evolution.

This module provides a clean adapter between the Pattern class and AdaptiveID,
enabling versioning, relationship tracking, and context management capabilities
for patterns without modifying the original classes directly.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.habitat_evolution.adaptive_core.models.pattern import Pattern
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

logger = logging.getLogger(__name__)


class PatternAdaptiveIDAdapter:
    """
    Adapter that bridges between Pattern and AdaptiveID.
    
    This adapter enables patterns to leverage AdaptiveID's sophisticated versioning,
    relationship tracking, and context management capabilities while maintaining
    backward compatibility with existing code.
    """
    
    def __init__(self, pattern: Pattern):
        """
        Initialize the adapter with a Pattern instance.
        
        Args:
            pattern: The Pattern instance to adapt
        """
        self.pattern = pattern
        self.adaptive_id = self._create_adaptive_id_from_pattern(pattern)
        
    def _create_adaptive_id_from_pattern(self, pattern: Pattern) -> AdaptiveID:
        """
        Create an AdaptiveID instance from a Pattern.
        
        Args:
            pattern: The Pattern to create an AdaptiveID from
            
        Returns:
            An AdaptiveID instance with properties from the Pattern
        """
        adaptive_id = AdaptiveID(
            base_concept=pattern.base_concept,
            creator_id=pattern.creator_id,
            weight=pattern.weight,
            confidence=pattern.confidence,
            uncertainty=pattern.uncertainty
        )
        
        # Sync additional properties
        self._sync_pattern_to_adaptive_id(pattern, adaptive_id)
        
        return adaptive_id
    
    def _sync_pattern_to_adaptive_id(self, pattern: Pattern, adaptive_id: AdaptiveID) -> None:
        """
        Synchronize Pattern properties to AdaptiveID.
        
        Args:
            pattern: The source Pattern
            adaptive_id: The target AdaptiveID
        """
        # Sync temporal context
        for key, value in pattern.temporal_context.items():
            for timestamp, context_value in value.items():
                adaptive_id.update_temporal_context(key, context_value, "pattern_sync")
        
        # Sync spatial context
        for key, value in pattern.spatial_context.items():
            adaptive_id.update_spatial_context(key, value, "pattern_sync")
        
        # Create a version for the current pattern state
        self._create_version_from_pattern(pattern, adaptive_id)
    
    def _create_version_from_pattern(self, pattern: Pattern, adaptive_id: AdaptiveID) -> None:
        """
        Create a version in AdaptiveID based on the current Pattern state.
        
        Args:
            pattern: The source Pattern
            adaptive_id: The target AdaptiveID
        """
        version_data = {
            "base_concept": pattern.base_concept,
            "weight": pattern.weight,
            "confidence": pattern.confidence,
            "uncertainty": pattern.uncertainty,
            "coherence": pattern.coherence,
            "phase_stability": pattern.phase_stability,
            "signal_strength": pattern.signal_strength,
            "properties": pattern.properties,
            "metrics": pattern.metrics,
            "state": pattern.state
        }
        
        # Use the create_version method if it exists, otherwise use a direct approach
        if hasattr(adaptive_id, 'create_version'):
            adaptive_id.create_version(version_data, "pattern_sync")
        else:
            # Direct approach as fallback
            import uuid
            from src.habitat_evolution.adaptive_core.id.adaptive_id import Version
            
            version_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            with adaptive_id._lock:
                adaptive_id.versions[version_id] = Version(
                    version_id,
                    version_data,
                    timestamp,
                    "pattern_sync"
                )
                adaptive_id.current_version = version_id
                adaptive_id.metadata["version_count"] += 1
                adaptive_id.metadata["last_modified"] = timestamp
    
    def _sync_adaptive_id_to_pattern(self) -> None:
        """
        Synchronize AdaptiveID properties back to the Pattern.
        """
        # Get the current version data
        current_version_id = self.adaptive_id.current_version
        if current_version_id in self.adaptive_id.versions:
            version = self.adaptive_id.versions[current_version_id]
            data = version.data
            
            # Update pattern properties from version data
            if "base_concept" in data:
                self.pattern.base_concept = data["base_concept"]
            if "weight" in data:
                self.pattern.weight = data["weight"]
            if "confidence" in data:
                self.pattern.confidence = data["confidence"]
            if "uncertainty" in data:
                self.pattern.uncertainty = data["uncertainty"]
            if "coherence" in data:
                self.pattern.coherence = data["coherence"]
            if "phase_stability" in data:
                self.pattern.phase_stability = data["phase_stability"]
            if "signal_strength" in data:
                self.pattern.signal_strength = data["signal_strength"]
            if "properties" in data:
                self.pattern.properties.update(data["properties"])
            if "metrics" in data:
                self.pattern.metrics.update(data["metrics"])
            if "state" in data:
                self.pattern.state = data["state"]
            
            # Update last_modified
            self.pattern.last_modified = self.adaptive_id.metadata["last_modified"]
    
    # Pattern compatibility methods
    
    def get_pattern(self) -> Pattern:
        """
        Get the adapted Pattern with updated properties from AdaptiveID.
        
        Returns:
            The Pattern instance with synchronized properties
        """
        self._sync_adaptive_id_to_pattern()
        return self.pattern
    
    def get_adaptive_id(self) -> AdaptiveID:
        """
        Get the AdaptiveID instance.
        
        Returns:
            The AdaptiveID instance
        """
        return self.adaptive_id
    
    # AdaptiveID versioning methods
    
    def create_version(self, data: Dict[str, Any], origin: str) -> None:
        """
        Create a new version of the pattern.
        
        Args:
            data: The version data
            origin: The source of the version creation
        """
        # Update the pattern with the new data first
        for key, value in data.items():
            if hasattr(self.pattern, key):
                setattr(self.pattern, key, value)
        
        # Create version in AdaptiveID
        if hasattr(self.adaptive_id, 'create_version'):
            self.adaptive_id.create_version(data, origin)
        else:
            self._create_version_from_pattern(self.pattern, self.adaptive_id)
        
        # Sync back to Pattern to ensure any additional processing in AdaptiveID is reflected
        self._sync_adaptive_id_to_pattern()
    
    def get_version_history(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[Any]:
        """
        Get the version history of the pattern.
        
        Args:
            start_time: ISO format timestamp for start of window (inclusive)
            end_time: ISO format timestamp for end of window (inclusive)
            
        Returns:
            List of versions within the specified time window
        """
        return self.adaptive_id.get_version_history(start_time, end_time)
    
    def get_state_at_time(self, timestamp: str) -> Dict[str, Any]:
        """
        Get the pattern state at a specific time.
        
        Args:
            timestamp: The timestamp to retrieve state for
            
        Returns:
            Dict containing the state at the specified time
        """
        return self.adaptive_id.get_state_at_time(timestamp)
    
    # Context management methods
    
    def update_temporal_context(self, key: str, value: Any) -> None:
        """
        Update temporal context for both Pattern and AdaptiveID.
        
        Args:
            key: Context key to update
            value: New value
        """
        # Update Pattern
        self.pattern.update_temporal_context(key, value)
        
        # Update AdaptiveID
        self.adaptive_id.update_temporal_context(key, value, "pattern_adapter")
    
    def update_spatial_context(self, updates: Dict[str, Any]) -> None:
        """
        Update spatial context for both Pattern and AdaptiveID.
        
        Args:
            updates: Dict of context updates
        """
        # Update Pattern
        self.pattern.update_spatial_context(updates)
        
        # Update AdaptiveID
        for key, value in updates.items():
            self.adaptive_id.update_spatial_context(key, value, "pattern_adapter")
    
    # Relationship management methods
    
    def add_relationship(self, relationship_id: str, rel_type: Optional[str] = None, properties: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a relationship to both Pattern and AdaptiveID.
        
        Args:
            relationship_id: ID of the related pattern/entity
            rel_type: Type of relationship (only for AdaptiveID)
            properties: Additional properties of the relationship (only for AdaptiveID)
        """
        # Update Pattern
        self.pattern.add_relationship(relationship_id)
        
        # Update AdaptiveID relationships if the method exists
        if hasattr(self.adaptive_id, 'add_relationship'):
            if rel_type and properties:
                self.adaptive_id.add_relationship(relationship_id, rel_type, properties)
            else:
                # Fallback to simpler relationship if method signature differs
                self.adaptive_id.add_relationship(relationship_id)
    
    def remove_relationship(self, relationship_id: str) -> None:
        """
        Remove a relationship from both Pattern and AdaptiveID.
        
        Args:
            relationship_id: ID of the relationship to remove
        """
        # Update Pattern
        self.pattern.remove_relationship(relationship_id)
        
        # Update AdaptiveID if the method exists
        if hasattr(self.adaptive_id, 'remove_relationship'):
            self.adaptive_id.remove_relationship(relationship_id)
    
    # Pattern-aware RAG compatibility
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Get the metadata for this pattern.
        
        This property provides compatibility with the pattern-aware RAG components.
        
        Returns:
            A dictionary containing the pattern's metadata
        """
        return {
            "coherence": self.pattern.coherence,
            "quality": self.pattern.confidence,
            "stability": self.pattern.phase_stability,
            "uncertainty": self.pattern.uncertainty,
            "signal_strength": self.pattern.signal_strength,
            "state": self.pattern.state,
            "version": self.pattern.version,
            "creator_id": self.pattern.creator_id,
            "weight": self.pattern.weight,
            "metrics": self.pattern.metrics,
            "properties": self.pattern.properties,
            "text": self.pattern.base_concept,
            "version_count": self.adaptive_id.metadata.get("version_count", 1),
            "created_at": self.adaptive_id.metadata.get("created_at", self.pattern.created_at),
            "last_modified": self.adaptive_id.metadata.get("last_modified", self.pattern.last_modified)
        }
    
    @property
    def text(self) -> str:
        """
        Get the text representation of this pattern.
        
        This property provides compatibility with the quality_enhanced_retrieval.py
        which expects patterns to have a text attribute.
        
        Returns:
            The base concept as the text representation of the pattern
        """
        return self.pattern.base_concept
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation of this pattern with AdaptiveID enhancements
        """
        pattern_dict = self.pattern.to_dict()
        
        # Add AdaptiveID-specific information
        pattern_dict.update({
            "version_history": [v.version_id for v in self.adaptive_id.versions.values()],
            "version_count": self.adaptive_id.metadata.get("version_count", 1),
            "adaptive_id": self.adaptive_id.id
        })
        
        return pattern_dict


class PatternAdaptiveIDFactory:
    """
    Factory for creating PatternAdaptiveIDAdapter instances.
    """
    
    @staticmethod
    def create_adapter(pattern: Pattern) -> PatternAdaptiveIDAdapter:
        """
        Create a PatternAdaptiveIDAdapter for a Pattern.
        
        Args:
            pattern: The Pattern to adapt
            
        Returns:
            A PatternAdaptiveIDAdapter instance
        """
        return PatternAdaptiveIDAdapter(pattern)
    
    @staticmethod
    def create_from_adaptive_id(adaptive_id: AdaptiveID) -> PatternAdaptiveIDAdapter:
        """
        Create a PatternAdaptiveIDAdapter from an AdaptiveID.
        
        Args:
            adaptive_id: The AdaptiveID to create a Pattern from
            
        Returns:
            A PatternAdaptiveIDAdapter instance
        """
        # Create a new Pattern from AdaptiveID
        pattern = Pattern(
            id=adaptive_id.id,
            base_concept=adaptive_id.base_concept,
            creator_id=adaptive_id.creator_id,
            weight=adaptive_id.weight,
            confidence=adaptive_id.confidence,
            uncertainty=adaptive_id.uncertainty
        )
        
        # Create adapter
        adapter = PatternAdaptiveIDAdapter(pattern)
        
        # Replace the auto-created AdaptiveID with the provided one
        adapter.adaptive_id = adaptive_id
        
        # Sync AdaptiveID to Pattern
        adapter._sync_adaptive_id_to_pattern()
        
        return adapter
