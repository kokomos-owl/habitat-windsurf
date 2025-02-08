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
        elements: List[Any],
        context: Any
    ) -> Dict[str, Any]:
        """
        Observe elements for emerging patterns.
        
        Args:
            elements: List of elements to observe (StructuralElement or PatternEvidence)
            context: Analysis context (StructureContext or TemporalContext)
            
        Returns:
            Dict containing newly emerged and evolved patterns
        """
        timestamp = self.timestamp_service.get_timestamp()
        
        # Convert PatternEvidence to format compatible with pattern tracking
        processed_elements = []
        for element in elements:
            if hasattr(element, 'pattern_type'):
                # Handle PatternEvidence
                element_id = getattr(element, 'evidence_id', str(id(element)))
                pattern_type = element.pattern_type
                metadata = {
                    'density': element.density_metrics.local_density if hasattr(element, 'density_metrics') else 0.0,
                    'coherence': element.evolution_metrics.coherence_level if hasattr(element, 'evolution_metrics') else 0.0,
                    'stability': element.evolution_metrics.stability if hasattr(element, 'evolution_metrics') else 0.0
                }
            else:
                # Handle StructuralElement
                element_id = element.element_id
                pattern_type = element.element_type
                metadata = element.metadata
            
            processed_elements.append(element)
        
        # Look for potential patterns
        new_patterns = await self._discover_patterns(processed_elements, timestamp)
        
        # Update existing patterns
        evolved_patterns = self._evolve_patterns(processed_elements, timestamp)
        
        # Track evolution
        self._track_evolution(new_patterns, evolved_patterns, timestamp)
        
        return {
            "new_patterns": new_patterns,
            "evolved_patterns": evolved_patterns,
            "timestamp": timestamp.isoformat()
        }
    
    async def _discover_patterns(
        self,
        elements: List[Any],
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
            
            # Find earliest emergence time based on element type
            first_seen = timestamp
            if hasattr(group_elements[0], 'pattern_type'):
                # PatternEvidence
                if hasattr(group_elements[0], 'evolution_metrics'):
                    first_seen = min(
                        e.evolution_metrics.first_seen 
                        for e in group_elements 
                        if e.evolution_metrics
                    )
            else:
                # StructuralElement
                first_seen = min(getattr(e, 'emergence_time', timestamp) for e in group_elements)
            
            # Create metadata based on element type
            metadata = {}
            if hasattr(group_elements[0], 'pattern_type'):
                # PatternEvidence metadata
                metadata = {
                    'density': sum(getattr(e.density_metrics, 'local_density', 0.0) for e in group_elements) / len(group_elements),
                    'coherence': sum(getattr(e.evolution_metrics, 'coherence_level', 0.0) for e in group_elements) / len(group_elements),
                    'stability': sum(getattr(e.evolution_metrics, 'stability', 0.0) for e in group_elements) / len(group_elements)
                }
            else:
                # StructuralElement metadata
                for element in group_elements:
                    for key, value in element.metadata.items():
                        if key not in metadata:
                            metadata[key] = []
                        metadata[key].append(value)
                
                # Average numeric metadata
                for key, values in metadata.items():
                    if all(isinstance(v, (int, float)) for v in values):
                        metadata[key] = sum(values) / len(values)
            
            # Create emergent pattern
            pattern = EmergentPattern(
                pattern_id=f"pattern_{timestamp.timestamp()}_{len(new_patterns)}",
                pattern_type=group_type,
                elements=group_elements,
                first_seen=first_seen,
                last_seen=timestamp,
                confidence=coherence,
                stability=metadata.get('stability', 0.5),  # Use stability from metadata if available
                metadata=metadata
            )
            
            # Track pattern
            self.emergent_patterns[pattern.pattern_id] = pattern
            for element in group_elements:
                element_id = getattr(element, 'evidence_id', getattr(element, 'element_id', str(id(element))))
                self.element_to_patterns[element_id].add(pattern.pattern_id)
            
            new_patterns.append(pattern)
        
        return new_patterns
    
    def _evolve_patterns(
        self,
        elements: List[Any],
        timestamp: datetime
    ) -> List[EmergentPattern]:
        """Evolve existing patterns naturally."""
        evolved_patterns = []
        
        for pattern in self.emergent_patterns.values():
            # Look for new elements that fit pattern
            new_elements = []
            for element in elements:
                element_id = getattr(element, 'evidence_id', getattr(element, 'element_id', str(id(element))))
                if element_id not in self.element_to_patterns or \
                   pattern.pattern_id not in self.element_to_patterns[element_id]:
                    new_elements.append(element)
            
            if not new_elements:
                continue
            
            # Calculate fit with pattern
            fitting_elements = []
            for element in new_elements:
                fit = self._calculate_element_fit(element, pattern)
                if fit > self.confidence_threshold:
                    fitting_elements.append(element)
            
            if fitting_elements:
                # Update pattern
                pattern.elements.extend(fitting_elements)
                pattern.last_seen = timestamp
                pattern.confidence = self._calculate_group_coherence(pattern.elements)
                
                # Update pattern metadata based on new elements
                if hasattr(fitting_elements[0], 'pattern_type'):
                    # Update PatternEvidence based metadata
                    all_elements = pattern.elements
                    pattern.metadata.update({
                        'density': sum(getattr(e.density_metrics, 'local_density', 0.0) for e in all_elements) / len(all_elements),
                        'coherence': sum(getattr(e.evolution_metrics, 'coherence_level', 0.0) for e in all_elements) / len(all_elements),
                        'stability': sum(getattr(e.evolution_metrics, 'stability', 0.0) for e in all_elements) / len(all_elements)
                    })
                    pattern.stability = pattern.metadata['stability']
                else:
                    # Update StructuralElement based metadata
                    for element in fitting_elements:
                        for key, value in element.metadata.items():
                            if key not in pattern.metadata:
                                pattern.metadata[key] = []
                            pattern.metadata[key].append(value)
                    
                    # Average numeric metadata
                    for key, values in pattern.metadata.items():
                        if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
                            pattern.metadata[key] = sum(values) / len(values)
                    
                    pattern.stability = self._calculate_pattern_stability(pattern)
                
                # Update tracking
                for element in fitting_elements:
                    element_id = getattr(element, 'evidence_id', getattr(element, 'element_id', str(id(element))))
                    self.element_to_patterns[element_id].add(pattern.pattern_id)
                
                evolved_patterns.append(pattern)
        
        return evolved_patterns
    
    def _group_by_similarity(
        self,
        elements: List[Any]
    ) -> Dict[str, List[Any]]:
        """Group elements by natural similarity."""
        groups: DefaultDict[str, List[Any]] = defaultdict(list)
        
        for element in elements:
            # Handle both StructuralElement and PatternEvidence
            if hasattr(element, 'element_type'):
                # StructuralElement grouping
                depth = element.metadata.get('depth', 0)
                if element.element_type == 'section':
                    group_key = f"section_depth_{depth}"
                elif element.element_type == 'list':
                    group_key = 'list_items'
                else:
                    group_key = element.element_type
            elif hasattr(element, 'pattern_type'):
                # PatternEvidence grouping
                if element.evolution_metrics:
                    coherence = element.evolution_metrics.coherence_level
                    stability = element.evolution_metrics.stability
                    group_key = f"{element.pattern_type}_{coherence:.1f}_{stability:.1f}"
                else:
                    group_key = element.pattern_type
            else:
                # Group by coherence for unknown types
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
        elements: List[Any]
    ) -> float:
        """Calculate natural coherence of a group."""
        if not elements:
            return 0.0
            
        # Handle both StructuralElement and PatternEvidence
        if hasattr(elements[0], 'element_type'):
            # StructuralElement coherence
            type_coherence = len(set(e.element_type for e in elements)) == 1
            depth_coherence = len(set(
                e.metadata.get('depth', 0) for e in elements
            )) == 1
            
            return (
                0.6 * float(type_coherence) +
                0.4 * float(depth_coherence)
            )
        elif hasattr(elements[0], 'pattern_type'):
            # PatternEvidence coherence
            type_coherence = len(set(e.pattern_type for e in elements)) == 1
            
            # Consider evolution metrics if available
            if all(e.evolution_metrics for e in elements):
                coherence_levels = [
                    e.evolution_metrics.coherence_level
                    for e in elements
                ]
                stability_levels = [
                    e.evolution_metrics.stability
                    for e in elements
                ]
                
                return (
                    0.4 * float(type_coherence) +
                    0.3 * (sum(coherence_levels) / len(coherence_levels)) +
                    0.3 * (sum(stability_levels) / len(stability_levels))
                )
            
            return float(type_coherence)
        
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
        element: Any,
        pattern: EmergentPattern
    ) -> float:
        """Calculate how well an element fits a pattern."""
        if hasattr(element, 'pattern_type'):
            # PatternEvidence fit calculation
            if element.pattern_type != pattern.pattern_type:
                return 0.0
            
            # Evolution metrics match
            metrics_match = 0.0
            if hasattr(element, 'evolution_metrics') and 'coherence' in pattern.metadata:
                coherence_diff = abs(
                    element.evolution_metrics.coherence_level - pattern.metadata['coherence']
                )
                stability_diff = abs(
                    element.evolution_metrics.stability - pattern.metadata['stability']
                )
                metrics_match = 1.0 - (coherence_diff + stability_diff) / 2
            
            # Density metrics match
            density_match = 0.0
            if hasattr(element, 'density_metrics') and 'density' in pattern.metadata:
                density_diff = abs(
                    element.density_metrics.local_density - pattern.metadata['density']
                )
                density_match = 1.0 - min(density_diff, 1.0)
            
            # Coherence with pattern elements
            coherence = self._calculate_group_coherence(pattern.elements + [element])
            
            return (
                0.4 * float(element.pattern_type == pattern.pattern_type) +
                0.3 * metrics_match +
                0.2 * density_match +
                0.1 * coherence
            )
        else:
            # Similar to group coherence but comparing one element to pattern
            return self._calculate_group_coherence(pattern.elements + [element])
    
    def _calculate_pattern_stability(
        self,
        pattern: EmergentPattern
    ) -> float:
        """Calculate natural stability of a pattern."""
        # Check if pattern has elements with evolution metrics
        if pattern.elements and hasattr(pattern.elements[0], 'pattern_type'):
            # Use average stability from evolution metrics
            stabilities = []
            for element in pattern.elements:
                if hasattr(element, 'evolution_metrics'):
                    stabilities.append(element.evolution_metrics.stability)
            
            if stabilities:
                return sum(stabilities) / len(stabilities)
        
        # For StructuralElements or mixed patterns
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
