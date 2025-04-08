"""
Pattern Domain Bridge for Habitat Evolution.

This module provides the PatternDomainBridge class that connects the
vector-tonic statistical pattern domain with the semantic pattern domain
using Habitat's event bus. It enables the observation of co-evolution
between patterns across different domains while maintaining loose coupling.
"""

import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from src.habitat_evolution.core.services.event_bus import EventBus, Event
from src.habitat_evolution.core.pattern.pattern import Pattern, PatternQualityState
from src.habitat_evolution.vector_tonic.bridge.events import (
    StatisticalPatternDetectedEvent,
    StatisticalPatternQualityChangedEvent,
    StatisticalRelationshipDetectedEvent,
    CrossDomainCorrelationDetectedEvent,
    CoPatternCreatedEvent
)
from src.habitat_evolution.vector_tonic.core.time_series_pattern_detector import TimeSeriesPattern

logger = logging.getLogger(__name__)


@dataclass
class CoPattern:
    """
    Represents a correlation between statistical and semantic patterns.
    
    Co-patterns emerge when statistical and semantic patterns show
    significant correlation, suggesting they may represent different
    aspects of the same underlying phenomenon.
    """
    
    id: str
    statistical_pattern_id: str
    semantic_pattern_id: str
    correlation_strength: float
    correlation_type: str
    quality_state: str
    metadata: Dict[str, Any]
    
    @classmethod
    def create(cls, 
              statistical_pattern: Dict[str, Any],
              semantic_pattern: Dict[str, Any],
              correlation_strength: float,
              correlation_type: str) -> 'CoPattern':
        """
        Create a new co-pattern from correlated patterns.
        
        Args:
            statistical_pattern: Statistical pattern data
            semantic_pattern: Semantic pattern data
            correlation_strength: Strength of the correlation (0-1)
            correlation_type: Type of correlation
            
        Returns:
            A new CoPattern instance
        """
        co_pattern_id = f"co_pattern_{uuid.uuid4().hex[:8]}"
        
        # Determine quality state based on component patterns and correlation strength
        quality_state = cls._determine_quality_state(
            statistical_pattern.get("quality_state", "hypothetical"),
            semantic_pattern.get("quality_state", "hypothetical"),
            correlation_strength
        )
        
        # Create metadata combining relevant information from both patterns
        metadata = {
            "statistical_metadata": statistical_pattern.get("metadata", {}),
            "semantic_metadata": semantic_pattern.get("metadata", {}),
            "correlation_data": {
                "strength": correlation_strength,
                "type": correlation_type,
                "created_at": None  # Will be set by TimeProvider in the bridge
            }
        }
        
        return cls(
            id=co_pattern_id,
            statistical_pattern_id=statistical_pattern.get("id", "unknown"),
            semantic_pattern_id=semantic_pattern.get("id", "unknown"),
            correlation_strength=correlation_strength,
            correlation_type=correlation_type,
            quality_state=quality_state,
            metadata=metadata
        )
    
    @staticmethod
    def _determine_quality_state(statistical_quality: str, 
                               semantic_quality: str,
                               correlation_strength: float) -> str:
        """
        Determine the quality state of a co-pattern based on its components.
        
        Args:
            statistical_quality: Quality state of the statistical pattern
            semantic_quality: Quality state of the semantic pattern
            correlation_strength: Strength of the correlation
            
        Returns:
            Quality state: "hypothetical", "emergent", or "stable"
        """
        # Convert quality states to numerical values
        quality_values = {
            "hypothetical": 1,
            "emergent": 2,
            "stable": 3
        }
        
        statistical_value = quality_values.get(statistical_quality.lower(), 1)
        semantic_value = quality_values.get(semantic_quality.lower(), 1)
        
        # Calculate combined quality score
        # Weight: 40% statistical quality, 40% semantic quality, 20% correlation strength
        combined_score = (
            0.4 * statistical_value + 
            0.4 * semantic_value + 
            0.2 * (correlation_strength * 3)
        )
        
        # Determine quality state based on combined score
        if combined_score >= 2.5:
            return "stable"
        elif combined_score >= 1.7:
            return "emergent"
        else:
            return "hypothetical"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the co-pattern to a dictionary representation.
        
        Returns:
            Dictionary representation of the co-pattern
        """
        return {
            "id": self.id,
            "statistical_pattern_id": self.statistical_pattern_id,
            "semantic_pattern_id": self.semantic_pattern_id,
            "correlation_strength": self.correlation_strength,
            "correlation_type": self.correlation_type,
            "quality_state": self.quality_state,
            "metadata": self.metadata
        }


class PatternDomainBridge:
    """
    Bridge between statistical and semantic pattern domains.
    
    This class uses Habitat's event bus to connect the vector-tonic
    statistical pattern domain with the semantic pattern domain,
    enabling the observation of co-evolution between patterns
    across different domains while maintaining loose coupling.
    """
    
    def __init__(self, event_bus: EventBus, time_provider=None):
        """
        Initialize a new pattern domain bridge.
        
        Args:
            event_bus: Habitat's event bus for communication
            time_provider: Optional time provider for timestamps
        """
        self.event_bus = event_bus
        self.time_provider = time_provider
        
        # Pattern storage
        self.statistical_patterns = {}  # id -> pattern data
        self.semantic_patterns = {}     # id -> pattern data
        self.co_patterns = {}           # id -> CoPattern
        
        # Correlation metrics
        self.correlation_thresholds = {
            "temporal": 0.6,
            "semantic": 0.5,
            "combined": 0.65
        }
        
        # Subscribe to events
        self._subscribe_to_events()
        
        logger.info("Pattern Domain Bridge initialized")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events from both domains."""
        # Statistical pattern events
        self.event_bus.subscribe(
            "statistical_pattern_detected", 
            self.on_statistical_pattern_detected
        )
        self.event_bus.subscribe(
            "statistical_pattern_quality_changed",
            self.on_statistical_pattern_quality_changed
        )
        
        # Semantic pattern events
        self.event_bus.subscribe(
            "pattern_detected",  # Habitat's semantic pattern event
            self.on_semantic_pattern_detected
        )
        self.event_bus.subscribe(
            "pattern_quality_changed",  # Habitat's semantic pattern event
            self.on_semantic_pattern_quality_changed
        )
    
    def on_statistical_pattern_detected(self, event: StatisticalPatternDetectedEvent):
        """
        Handle statistical pattern detected events.
        
        Args:
            event: The statistical pattern detected event
        """
        pattern_id = event.pattern_id
        pattern_data = event.pattern_data
        
        # Store the pattern
        self.statistical_patterns[pattern_id] = pattern_data
        
        logger.info(f"Statistical pattern detected: {pattern_id}")
        
        # Look for correlations with existing semantic patterns
        self._detect_cross_domain_correlations(pattern_id, "statistical")
    
    def on_statistical_pattern_quality_changed(self, event: StatisticalPatternQualityChangedEvent):
        """
        Handle statistical pattern quality changed events.
        
        Args:
            event: The statistical pattern quality changed event
        """
        pattern_id = event.pattern_id
        new_state = event.new_state
        
        # Update the pattern
        if pattern_id in self.statistical_patterns:
            self.statistical_patterns[pattern_id]["quality_state"] = new_state
            
            logger.info(f"Statistical pattern quality changed: {pattern_id} -> {new_state}")
            
            # Update affected co-patterns
            self._update_co_patterns_for_statistical_pattern(pattern_id)
    
    def on_semantic_pattern_detected(self, event: Event):
        """
        Handle semantic pattern detected events.
        
        Args:
            event: The semantic pattern detected event
        """
        # Extract pattern data from Habitat's event format
        pattern_id = getattr(event, "pattern_id", None)
        if not pattern_id:
            return
        
        # Convert Habitat's pattern format to our internal format
        pattern_data = self._convert_semantic_pattern(event)
        
        # Store the pattern
        self.semantic_patterns[pattern_id] = pattern_data
        
        logger.info(f"Semantic pattern detected: {pattern_id}")
        
        # Look for correlations with existing statistical patterns
        self._detect_cross_domain_correlations(pattern_id, "semantic")
    
    def on_semantic_pattern_quality_changed(self, event: Event):
        """
        Handle semantic pattern quality changed events.
        
        Args:
            event: The semantic pattern quality changed event
        """
        # Extract pattern data from Habitat's event format
        pattern_id = getattr(event, "pattern_id", None)
        new_state = getattr(event, "new_state", None)
        
        if not pattern_id or not new_state:
            return
        
        # Update the pattern
        if pattern_id in self.semantic_patterns:
            self.semantic_patterns[pattern_id]["quality_state"] = new_state
            
            logger.info(f"Semantic pattern quality changed: {pattern_id} -> {new_state}")
            
            # Update affected co-patterns
            self._update_co_patterns_for_semantic_pattern(pattern_id)
    
    def _convert_semantic_pattern(self, event: Event) -> Dict[str, Any]:
        """
        Convert a semantic pattern event to our internal format.
        
        Args:
            event: The semantic pattern event
            
        Returns:
            Pattern data in our internal format
        """
        # Extract available attributes from the event
        pattern_id = getattr(event, "pattern_id", "unknown")
        pattern_text = getattr(event, "pattern_text", "")
        quality_state = getattr(event, "quality_state", "hypothetical")
        confidence = getattr(event, "confidence", 0.0)
        source = getattr(event, "source", "unknown")
        
        # Additional metadata that might be available
        metadata = {}
        if hasattr(event, "metadata"):
            metadata = event.metadata
        elif hasattr(event, "pattern_metadata"):
            metadata = event.pattern_metadata
        
        # Extract temporal information if available
        temporal_markers = []
        if hasattr(event, "temporal_markers"):
            temporal_markers = event.temporal_markers
        
        # Build our internal representation
        return {
            "id": pattern_id,
            "text": pattern_text,
            "quality_state": quality_state,
            "confidence": confidence,
            "source": source,
            "temporal_markers": temporal_markers,
            "metadata": metadata
        }
    
    def _detect_cross_domain_correlations(self, pattern_id: str, domain: str):
        """
        Detect correlations between patterns across domains.
        
        Args:
            pattern_id: ID of the pattern to check for correlations
            domain: Domain of the pattern ("statistical" or "semantic")
        """
        if domain == "statistical":
            # Look for correlations with semantic patterns
            statistical_pattern = self.statistical_patterns.get(pattern_id)
            if not statistical_pattern:
                return
            
            for semantic_id, semantic_pattern in self.semantic_patterns.items():
                correlation = self._calculate_correlation(
                    statistical_pattern, 
                    semantic_pattern
                )
                
                if correlation[0] >= self.correlation_thresholds["combined"]:
                    self._create_co_pattern(
                        statistical_pattern,
                        semantic_pattern,
                        correlation[0],
                        correlation[1]
                    )
        else:
            # Look for correlations with statistical patterns
            semantic_pattern = self.semantic_patterns.get(pattern_id)
            if not semantic_pattern:
                return
            
            for statistical_id, statistical_pattern in self.statistical_patterns.items():
                correlation = self._calculate_correlation(
                    statistical_pattern, 
                    semantic_pattern
                )
                
                if correlation[0] >= self.correlation_thresholds["combined"]:
                    self._create_co_pattern(
                        statistical_pattern,
                        semantic_pattern,
                        correlation[0],
                        correlation[1]
                    )
    
    def _calculate_correlation(self, 
                             statistical_pattern: Dict[str, Any],
                             semantic_pattern: Dict[str, Any]) -> Tuple[float, str]:
        """
        Calculate correlation between statistical and semantic patterns.
        
        Args:
            statistical_pattern: Statistical pattern data
            semantic_pattern: Semantic pattern data
            
        Returns:
            Tuple of (correlation_strength, correlation_type)
        """
        # Calculate temporal correlation
        temporal_correlation = self._calculate_temporal_correlation(
            statistical_pattern,
            semantic_pattern
        )
        
        # Calculate semantic correlation
        semantic_correlation = self._calculate_semantic_correlation(
            statistical_pattern,
            semantic_pattern
        )
        
        # Determine primary correlation type
        if temporal_correlation > semantic_correlation:
            correlation_type = "temporal"
        else:
            correlation_type = "semantic"
        
        # Calculate combined correlation score
        # Weight: 60% primary correlation, 40% secondary correlation
        combined_correlation = max(
            0.6 * temporal_correlation + 0.4 * semantic_correlation,
            0.6 * semantic_correlation + 0.4 * temporal_correlation
        )
        
        return (combined_correlation, correlation_type)
    
    def _calculate_temporal_correlation(self, 
                                      statistical_pattern: Dict[str, Any],
                                      semantic_pattern: Dict[str, Any]) -> float:
        """
        Calculate temporal correlation between patterns.
        
        Args:
            statistical_pattern: Statistical pattern data
            semantic_pattern: Semantic pattern data
            
        Returns:
            Temporal correlation strength (0-1)
        """
        # Extract temporal information from patterns
        statistical_time = self._extract_time_from_statistical_pattern(statistical_pattern)
        semantic_time = self._extract_time_from_semantic_pattern(semantic_pattern)
        
        if not statistical_time or not semantic_time:
            return 0.0
        
        # Calculate temporal overlap
        # For simplicity, we'll use a basic approach here
        # In a production system, this would use more sophisticated algorithms
        
        # Convert time strings to years for comparison
        stat_start_year = int(statistical_time[0][:4]) if statistical_time[0] else 0
        stat_end_year = int(statistical_time[1][:4]) if statistical_time[1] else 0
        
        sem_start_year = int(semantic_time[0][:4]) if semantic_time[0] else 0
        sem_end_year = int(semantic_time[1][:4]) if semantic_time[1] else 0
        
        # If either pattern doesn't have valid years, no correlation
        if stat_start_year == 0 or sem_start_year == 0:
            return 0.0
        
        # Calculate overlap
        overlap_start = max(stat_start_year, sem_start_year)
        overlap_end = min(stat_end_year, sem_end_year)
        
        if overlap_end < overlap_start:
            return 0.0  # No overlap
        
        # Calculate overlap ratio
        stat_duration = max(1, stat_end_year - stat_start_year)
        sem_duration = max(1, sem_end_year - sem_start_year)
        overlap_duration = overlap_end - overlap_start
        
        overlap_ratio = overlap_duration / min(stat_duration, sem_duration)
        
        return min(1.0, overlap_ratio)
    
    def _calculate_semantic_correlation(self, 
                                      statistical_pattern: Dict[str, Any],
                                      semantic_pattern: Dict[str, Any]) -> float:
        """
        Calculate semantic correlation between patterns.
        
        Args:
            statistical_pattern: Statistical pattern data
            semantic_pattern: Semantic pattern data
            
        Returns:
            Semantic correlation strength (0-1)
        """
        # Extract semantic information
        statistical_keywords = self._extract_keywords_from_statistical_pattern(
            statistical_pattern
        )
        semantic_text = semantic_pattern.get("text", "")
        
        if not statistical_keywords or not semantic_text:
            return 0.0
        
        # Calculate keyword match ratio
        # For simplicity, we'll use a basic approach here
        # In a production system, this would use more sophisticated NLP techniques
        
        match_count = 0
        for keyword in statistical_keywords:
            if keyword.lower() in semantic_text.lower():
                match_count += 1
        
        if not statistical_keywords:
            return 0.0
        
        match_ratio = match_count / len(statistical_keywords)
        
        return min(1.0, match_ratio)
    
    def _extract_time_from_statistical_pattern(self, 
                                             pattern: Dict[str, Any]) -> Tuple[str, str]:
        """
        Extract time information from a statistical pattern.
        
        Args:
            pattern: Statistical pattern data
            
        Returns:
            Tuple of (start_time, end_time)
        """
        start_time = pattern.get("start_time", "")
        end_time = pattern.get("end_time", "")
        
        return (start_time, end_time)
    
    def _extract_time_from_semantic_pattern(self, 
                                          pattern: Dict[str, Any]) -> Tuple[str, str]:
        """
        Extract time information from a semantic pattern.
        
        Args:
            pattern: Semantic pattern data
            
        Returns:
            Tuple of (start_time, end_time)
        """
        # Try to get temporal markers from the pattern
        temporal_markers = pattern.get("temporal_markers", [])
        
        if temporal_markers:
            # Sort markers by time
            sorted_markers = sorted(temporal_markers, key=lambda x: x.get("time", ""))
            
            if sorted_markers:
                start_time = sorted_markers[0].get("time", "")
                end_time = sorted_markers[-1].get("time", "")
                
                return (start_time, end_time)
        
        # If no explicit markers, try to extract from text
        # This is a simplified approach - in production, use NLP
        text = pattern.get("text", "")
        
        # Look for years in the text
        import re
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
        
        if years:
            years = [int(y) for y in years]
            return (str(min(years)), str(max(years)))
        
        return ("", "")
    
    def _extract_keywords_from_statistical_pattern(self, 
                                                 pattern: Dict[str, Any]) -> List[str]:
        """
        Extract keywords from a statistical pattern.
        
        Args:
            pattern: Statistical pattern data
            
        Returns:
            List of keywords
        """
        # For temperature patterns, use relevant climate keywords
        trend = pattern.get("trend", "")
        
        keywords = []
        
        # Add trend-related keywords
        if trend == "increasing":
            keywords.extend([
                "warming", "increase", "rising", "higher", "warmer", 
                "temperature increase", "heat"
            ])
        elif trend == "decreasing":
            keywords.extend([
                "cooling", "decrease", "falling", "lower", "colder",
                "temperature decrease"
            ])
        elif trend == "stable":
            keywords.extend([
                "stable", "consistent", "steady", "unchanged"
            ])
        
        # Add magnitude-related keywords
        magnitude = pattern.get("magnitude", 0.0)
        if magnitude > 1.0:
            keywords.extend([
                "significant", "substantial", "major", "considerable"
            ])
        elif magnitude > 0.5:
            keywords.extend([
                "moderate", "notable", "noticeable"
            ])
        else:
            keywords.extend([
                "slight", "minor", "small", "subtle"
            ])
        
        # Add general climate keywords
        keywords.extend([
            "temperature", "climate", "weather", "thermal", "heat", "warming"
        ])
        
        return keywords
    
    def _create_co_pattern(self, 
                         statistical_pattern: Dict[str, Any],
                         semantic_pattern: Dict[str, Any],
                         correlation_strength: float,
                         correlation_type: str):
        """
        Create a co-pattern from correlated patterns.
        
        Args:
            statistical_pattern: Statistical pattern data
            semantic_pattern: Semantic pattern data
            correlation_strength: Strength of the correlation
            correlation_type: Type of correlation
        """
        # Check if this correlation already exists
        correlation_key = f"{statistical_pattern['id']}_{semantic_pattern['id']}"
        reverse_key = f"{semantic_pattern['id']}_{statistical_pattern['id']}"
        
        if correlation_key in self.co_patterns or reverse_key in self.co_patterns:
            # Update existing co-pattern
            existing_key = correlation_key if correlation_key in self.co_patterns else reverse_key
            existing_co_pattern = self.co_patterns[existing_key]
            
            # Only update if the new correlation is stronger
            if correlation_strength > existing_co_pattern.correlation_strength:
                # Create new co-pattern
                co_pattern = CoPattern.create(
                    statistical_pattern,
                    semantic_pattern,
                    correlation_strength,
                    correlation_type
                )
                
                # Replace existing co-pattern
                self.co_patterns[existing_key] = co_pattern
                
                # Publish event
                self._publish_co_pattern_updated_event(co_pattern)
        else:
            # Create new co-pattern
            co_pattern = CoPattern.create(
                statistical_pattern,
                semantic_pattern,
                correlation_strength,
                correlation_type
            )
            
            # Store co-pattern
            self.co_patterns[correlation_key] = co_pattern
            
            # Publish event
            self._publish_co_pattern_created_event(co_pattern)
    
    def _update_co_patterns_for_statistical_pattern(self, pattern_id: str):
        """
        Update co-patterns affected by a change in a statistical pattern.
        
        Args:
            pattern_id: ID of the statistical pattern
        """
        # Find affected co-patterns
        affected_co_patterns = []
        
        for co_pattern_id, co_pattern in self.co_patterns.items():
            if co_pattern.statistical_pattern_id == pattern_id:
                affected_co_patterns.append(co_pattern)
        
        # Update each affected co-pattern
        for co_pattern in affected_co_patterns:
            statistical_pattern = self.statistical_patterns.get(pattern_id)
            semantic_pattern = self.semantic_patterns.get(co_pattern.semantic_pattern_id)
            
            if statistical_pattern and semantic_pattern:
                # Recalculate correlation
                correlation = self._calculate_correlation(
                    statistical_pattern,
                    semantic_pattern
                )
                
                # Create updated co-pattern
                updated_co_pattern = CoPattern.create(
                    statistical_pattern,
                    semantic_pattern,
                    correlation[0],
                    correlation[1]
                )
                
                # Replace existing co-pattern
                self.co_patterns[co_pattern_id] = updated_co_pattern
                
                # Publish event
                self._publish_co_pattern_updated_event(updated_co_pattern)
    
    def _update_co_patterns_for_semantic_pattern(self, pattern_id: str):
        """
        Update co-patterns affected by a change in a semantic pattern.
        
        Args:
            pattern_id: ID of the semantic pattern
        """
        # Find affected co-patterns
        affected_co_patterns = []
        
        for co_pattern_id, co_pattern in self.co_patterns.items():
            if co_pattern.semantic_pattern_id == pattern_id:
                affected_co_patterns.append(co_pattern)
        
        # Update each affected co-pattern
        for co_pattern in affected_co_patterns:
            statistical_pattern = self.statistical_patterns.get(co_pattern.statistical_pattern_id)
            semantic_pattern = self.semantic_patterns.get(pattern_id)
            
            if statistical_pattern and semantic_pattern:
                # Recalculate correlation
                correlation = self._calculate_correlation(
                    statistical_pattern,
                    semantic_pattern
                )
                
                # Create updated co-pattern
                updated_co_pattern = CoPattern.create(
                    statistical_pattern,
                    semantic_pattern,
                    correlation[0],
                    correlation[1]
                )
                
                # Replace existing co-pattern
                self.co_patterns[co_pattern_id] = updated_co_pattern
                
                # Publish event
                self._publish_co_pattern_updated_event(updated_co_pattern)
    
    def _publish_co_pattern_created_event(self, co_pattern: CoPattern):
        """
        Publish a co-pattern created event.
        
        Args:
            co_pattern: The co-pattern that was created
        """
        event = CoPatternCreatedEvent(
            co_pattern_id=co_pattern.id,
            statistical_pattern_id=co_pattern.statistical_pattern_id,
            semantic_pattern_id=co_pattern.semantic_pattern_id,
            quality_state=co_pattern.quality_state,
            co_pattern_data=co_pattern.to_dict()
        )
        
        self.event_bus.publish(event)
        
        logger.info(f"Co-pattern created: {co_pattern.id} (Quality: {co_pattern.quality_state})")
    
    def _publish_co_pattern_updated_event(self, co_pattern: CoPattern):
        """
        Publish a co-pattern updated event.
        
        Args:
            co_pattern: The co-pattern that was updated
        """
        # For now, we'll reuse the created event
        # In a production system, we'd have a dedicated updated event
        event = CoPatternCreatedEvent(
            co_pattern_id=co_pattern.id,
            statistical_pattern_id=co_pattern.statistical_pattern_id,
            semantic_pattern_id=co_pattern.semantic_pattern_id,
            quality_state=co_pattern.quality_state,
            co_pattern_data=co_pattern.to_dict()
        )
        
        self.event_bus.publish(event)
        
        logger.info(f"Co-pattern updated: {co_pattern.id} (Quality: {co_pattern.quality_state})")
    
    def get_co_patterns(self, quality_state: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all co-patterns, optionally filtered by quality state.
        
        Args:
            quality_state: Optional quality state to filter by
            
        Returns:
            List of co-patterns as dictionaries
        """
        result = []
        
        for co_pattern in self.co_patterns.values():
            if quality_state is None or co_pattern.quality_state == quality_state:
                result.append(co_pattern.to_dict())
        
        return result
    
    def get_co_pattern(self, co_pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific co-pattern by ID.
        
        Args:
            co_pattern_id: ID of the co-pattern
            
        Returns:
            Co-pattern as a dictionary, or None if not found
        """
        co_pattern = self.co_patterns.get(co_pattern_id)
        
        if co_pattern:
            return co_pattern.to_dict()
        
        return None
    
    def get_co_patterns_for_statistical_pattern(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Get all co-patterns for a specific statistical pattern.
        
        Args:
            pattern_id: ID of the statistical pattern
            
        Returns:
            List of co-patterns as dictionaries
        """
        result = []
        
        for co_pattern in self.co_patterns.values():
            if co_pattern.statistical_pattern_id == pattern_id:
                result.append(co_pattern.to_dict())
        
        return result
    
    def get_co_patterns_for_semantic_pattern(self, pattern_id: str) -> List[Dict[str, Any]]:
        """
        Get all co-patterns for a specific semantic pattern.
        
        Args:
            pattern_id: ID of the semantic pattern
            
        Returns:
            List of co-patterns as dictionaries
        """
        result = []
        
        for co_pattern in self.co_patterns.values():
            if co_pattern.semantic_pattern_id == pattern_id:
                result.append(co_pattern.to_dict())
        
        return result
