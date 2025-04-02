"""
Pattern Observer Interface for Vector-Tonic Persistence Integration.

This module defines the interface for observing pattern events
in the Vector-Tonic Window system. It supports the pattern evolution and 
co-evolution principles of Habitat Evolution by enabling the observation
of semantic change across the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class PatternObserverInterface(ABC):
    """
    Interface for observers of pattern events.
    
    This interface defines the methods that must be implemented by any class
    that wants to observe pattern events in the Vector-Tonic Window system.
    It enables components to react to pattern detection, evolution, and quality changes.
    """
    
    @abstractmethod
    def on_pattern_detected(self, pattern_id: str, pattern_data: Dict[str, Any], 
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when a new pattern is detected.
        
        Args:
            pattern_id: The ID of the detected pattern.
            pattern_data: The data of the detected pattern.
            metadata: Optional metadata about the detection.
        """
        pass
    
    @abstractmethod
    def on_pattern_evolution(self, pattern_id: str, previous_state: Dict[str, Any], 
                            new_state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when a pattern evolves.
        
        Args:
            pattern_id: The ID of the evolving pattern.
            previous_state: The previous state of the pattern.
            new_state: The new state of the pattern.
            metadata: Optional metadata about the evolution.
        """
        pass
    
    @abstractmethod
    def on_pattern_quality_change(self, pattern_id: str, previous_quality: Dict[str, float], 
                                 new_quality: Dict[str, float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when a pattern's quality metrics change.
        
        Args:
            pattern_id: The ID of the pattern.
            previous_quality: The previous quality metrics.
            new_quality: The new quality metrics.
            metadata: Optional metadata about the quality change.
        """
        pass
    
    @abstractmethod
    def on_pattern_relationship_detected(self, source_id: str, target_id: str, 
                                        relationship_data: Dict[str, Any], 
                                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when a relationship between patterns is detected.
        
        Args:
            source_id: The ID of the source pattern.
            target_id: The ID of the target pattern.
            relationship_data: The data of the relationship.
            metadata: Optional metadata about the relationship.
        """
        pass
    
    @abstractmethod
    def on_pattern_merge(self, merged_pattern_id: str, source_pattern_ids: List[str], 
                        merge_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when patterns are merged.
        
        Args:
            merged_pattern_id: The ID of the merged pattern.
            source_pattern_ids: The IDs of the source patterns.
            merge_data: Data about the merge.
            metadata: Optional metadata about the merge.
        """
        pass
    
    @abstractmethod
    def on_pattern_split(self, source_pattern_id: str, result_pattern_ids: List[str], 
                        split_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when a pattern splits.
        
        Args:
            source_pattern_id: The ID of the source pattern.
            result_pattern_ids: The IDs of the resulting patterns.
            split_data: Data about the split.
            metadata: Optional metadata about the split.
        """
        pass
