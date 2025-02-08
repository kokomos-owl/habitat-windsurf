"""
Pattern emergence tracking service.

Observes and tracks patterns as they naturally emerge across multiple documents,
building on the structural understanding from text analysis.
"""

from typing import Dict, Any, List, Set, Optional, DefaultDict
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging

from src.core.analysis.structure_analysis import StructuralElement, StructureContext
from src.core.analysis.text_analysis import TextAnalyzer
from src.core.utils.timestamp_service import TimestampService
from src.core.evolution.temporal_core import TemporalCore
from src.core.types import DensityMetrics

logger = logging.getLogger(__name__)

@dataclass
class EmergentPattern:
    """Pattern that has emerged naturally across documents."""
    pattern_id: str
    pattern_type: str
    elements: List[StructuralElement]
    first_seen: datetime
    last_seen: datetime
    confidence: float = 0.0
    stability: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> float:
        """Pattern age in seconds."""
        return (self.last_seen - self.first_seen).total_seconds()

    @property
    def element_count(self) -> int:
        """Number of elements in pattern."""
        return len(self.elements)

class PatternEmergenceTracker:
    """
    Tracks pattern emergence across documents naturally.
    
    Instead of enforcing patterns, allows them to emerge through
    observation of structural similarities and relationships.
    """
    
    def __init__(
        self,
        timestamp_service: Optional[TimestampService] = None,
        temporal_core: Optional[TemporalCore] = None
    ):
        self.timestamp_service = timestamp_service or TimestampService()
        self.temporal_core = temporal_core or TemporalCore()
        
        # Pattern tracking
        self.emergent_patterns: Dict[str, EmergentPattern] = {}
        self.element_to_patterns: DefaultDict[str, Set[str]] = defaultdict(set)
        
        # Evolution tracking
        self.pattern_history: List[Dict[str, Any]] = []
        self.confidence_threshold = 0.6
        self.stability_threshold = 0.4
        
        logger.info("Initialized PatternEmergenceTracker")
    
    async def observe_elements(
        self,
        elements: List[StructuralElement],
        context: StructureContext
    ) -> Dict[str, Any]:
        """
        Observe structural elements for emerging patterns.
        
        Args:
            elements: List of structural elements to observe
            context: Analysis context
            
        Returns:
            Dict containing newly emerged and evolved patterns
        """
        timestamp = self.timestamp_service.get_timestamp()
        
        # Look for potential patterns
        new_patterns = await self._discover_patterns(elements, timestamp)
        
        # Update existing patterns
        evolved_patterns = self._evolve_patterns(elements, timestamp)
        
        # Track evolution
        self._track_evolution(new_patterns, evolved_patterns, timestamp)
        
        return {
            "new_patterns": new_patterns,
            "evolved_patterns": evolved_patterns,
            "timestamp": timestamp.isoformat()
        }
    
    async def _discover_patterns(
        self,
        elements: List[StructuralElement],
        timestamp: datetime
    ) -> List[EmergentPattern]:
        """Discover new patterns naturally."""
        new_patterns = []
        
        # Group similar elements
        similarity_groups = self._group_by_similarity(elements)
        
        # Look for patterns in each group
        for group_type, group_elements in similarity_groups.items():
            if len(group_elements) < 2:  # Need multiple elements for a pattern
                continue
            
            # Calculate group coherence
            coherence = self._calculate_group_coherence(group_elements)
            if coherence < self.confidence_threshold:
                continue
            
            # Find earliest emergence time
            first_seen = min(e.emergence_time for e in group_elements)
            
            # Create emergent pattern
            pattern = EmergentPattern(
                pattern_id=f"pattern_{timestamp.timestamp()}_{len(new_patterns)}",
                pattern_type=group_type,
                elements=group_elements,
                first_seen=first_seen,
                last_seen=timestamp,
                confidence=coherence,
                stability=0.5  # Initial stability
            )
            
            # Track pattern
            self.emergent_patterns[pattern.pattern_id] = pattern
            for element in group_elements:
                self.element_to_patterns[element.element_id].add(pattern.pattern_id)
            
            new_patterns.append(pattern)
        
        return new_patterns
    
    def _evolve_patterns(
        self,
        elements: List[StructuralElement],
        timestamp: datetime
    ) -> List[EmergentPattern]:
        """Evolve existing patterns naturally."""
        evolved_patterns = []
        
        for pattern in self.emergent_patterns.values():
            # Look for new elements that fit pattern
            new_elements = [
                e for e in elements
                if e.element_id not in self.element_to_patterns or
                pattern.pattern_id not in self.element_to_patterns[e.element_id]
            ]
            
            if not new_elements:
                continue
            
            # Calculate fit with pattern
            fitting_elements = [
                e for e in new_elements
                if self._calculate_element_fit(e, pattern) > self.confidence_threshold
            ]
            
            if fitting_elements:
                # Update pattern
                pattern.elements.extend(fitting_elements)
                pattern.last_seen = timestamp
                pattern.confidence = self._calculate_group_coherence(pattern.elements)
                pattern.stability = self._calculate_pattern_stability(pattern)
                
                # Update tracking
                for element in fitting_elements:
                    self.element_to_patterns[element.element_id].add(pattern.pattern_id)
                
                evolved_patterns.append(pattern)
        
        return evolved_patterns
    
    def _group_by_similarity(
        self,
        elements: List[StructuralElement]
    ) -> Dict[str, List[StructuralElement]]:
        """Group elements by natural similarity."""
        groups: DefaultDict[str, List[StructuralElement]] = defaultdict(list)
        
        for element in elements:
            # Consider multiple factors for grouping
            if element.element_type == 'section':
                depth = element.metadata.get('depth', 0)
                group_key = f"section_depth_{depth}"
            elif element.element_type == 'list':
                group_key = 'list_items'
            else:
                # Group by coherence
                best_group = None
                best_coherence = 0
                
                for group_key, group in groups.items():
                    coherence = self._calculate_group_coherence(group + [element])
                    if coherence > best_coherence and coherence > self.confidence_threshold:
                        best_coherence = coherence
                        best_group = group_key
                
                if best_group:
                    group_key = best_group
                else:
                    group_key = f"group_{len(groups)}"
            
            groups[group_key].append(element)
        
        return groups
    
    def _calculate_group_coherence(
        self,
        elements: List[StructuralElement]
    ) -> float:
        """Calculate natural coherence of a group."""
        if not elements:
            return 0.0
            
        # Consider multiple factors
        type_coherence = len(set(e.element_type for e in elements)) == 1
        
        # Density similarity
        densities = [e.density for e in elements]
        density_range = max(densities) - min(densities)
        density_coherence = 1.0 - min(1.0, density_range)
        
        # Depth similarity
        depths = [e.metadata.get('depth', 0) for e in elements]
        depth_range = max(depths) - min(depths)
        depth_coherence = 1.0 - min(1.0, depth_range / 2)
        
        # Content similarity
        content_sets = [
            set(e.content.lower().split())
            for e in elements
        ]
        common_words = set.intersection(*content_sets)
        all_words = set.union(*content_sets)
        content_coherence = len(common_words) / max(len(all_words), 1)
        
        return (
            0.3 * float(type_coherence) +
            0.2 * density_coherence +
            0.2 * depth_coherence +
            0.3 * content_coherence
        )
    
    def _calculate_element_fit(
        self,
        element: StructuralElement,
        pattern: EmergentPattern
    ) -> float:
        """Calculate how well an element fits a pattern."""
        # Similar to group coherence but comparing one element to pattern
        return self._calculate_group_coherence(pattern.elements + [element])
    
    def _calculate_pattern_stability(
        self,
        pattern: EmergentPattern
    ) -> float:
        """Calculate natural stability of a pattern."""
        # Consider multiple factors
        age_seconds = max(0.1, (pattern.last_seen - pattern.first_seen).total_seconds())
        age_factor = min(1.0, age_seconds / (60 * 60))  # Max age of 1 hour for testing
        
        # Size contribution
        size_factor = min(1.0, pattern.element_count / 5)  # Max size of 5 elements
        
        # Coherence contribution
        coherence_factor = pattern.confidence
        
        # Consistency contribution - how consistently we see this pattern
        consistent_elements = len([
            e for e in pattern.elements
            if abs((e.emergence_time - pattern.first_seen).total_seconds()) < age_seconds
        ])
        consistency_factor = consistent_elements / max(pattern.element_count, 1)
        
        return min(1.0, (
            0.3 * age_factor +
            0.2 * size_factor +
            0.3 * coherence_factor +
            0.2 * consistency_factor
        ))
    
    def _track_evolution(
        self,
        new_patterns: List[EmergentPattern],
        evolved_patterns: List[EmergentPattern],
        timestamp: datetime
    ) -> None:
        """Track pattern evolution."""
        self.pattern_history.append({
            "timestamp": timestamp.isoformat(),
            "new_patterns": [p.pattern_id for p in new_patterns],
            "evolved_patterns": [p.pattern_id for p in evolved_patterns],
            "total_patterns": len(self.emergent_patterns),
            "stable_patterns": len([
                p for p in self.emergent_patterns.values()
                if p.stability >= self.stability_threshold
            ])
        })
