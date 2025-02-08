"""
Natural structure analysis service.

Discovers and tracks structural elements in documents without enforcing
rigid patterns, allowing natural organization to emerge.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
import logging

from src.core.utils.timestamp_service import TimestampService
from src.core.utils.logging_config import get_logger
from src.core.evolution.temporal_core import TemporalCore
from src.core.types import DensityMetrics

logger = get_logger(__name__)

@dataclass
class StructuralElement:
    """Natural structural element discovered in content."""
    element_id: str
    element_type: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    relationships: Set[str] = field(default_factory=set)
    density: float = 0.0
    emergence_time: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate structural element."""
        if not isinstance(self.relationships, set):
            self.relationships = set(self.relationships)
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

@dataclass
class StructureContext:
    """Context for structure analysis."""
    start_time: datetime
    content_type: str
    analysis_depth: int = 1
    density_threshold: float = 0.3
    relationship_strength: float = 0.5

class StructureAnalyzer:
    """
    Natural structure analyzer that discovers organization without enforcing patterns.
    
    Allows structural elements to emerge naturally from content while maintaining
    awareness of temporal evolution and density characteristics.
    """
    
    def __init__(
        self,
        timestamp_service: Optional[TimestampService] = None,
        temporal_core: Optional[TemporalCore] = None
    ):
        self.timestamp_service = timestamp_service or TimestampService()
        self.temporal_core = temporal_core or TemporalCore()
        self._lock = RLock()
        
        # Structure tracking
        self.structural_elements: Dict[str, StructuralElement] = {}
        self.element_relationships: Dict[str, Set[str]] = {}
        self.density_metrics: Dict[str, DensityMetrics] = {}
        
        # Analysis state
        self.active_contexts: Dict[str, StructureContext] = {}
        self.element_history: List[Dict[str, Any]] = []
        
        logger.info("Initialized StructureAnalyzer for natural discovery")
    
    async def analyze_content(
        self,
        content: Any,
        content_type: str,
        analysis_depth: int = 1
    ) -> Dict[str, Any]:
        """
        Analyze content structure naturally.
        
        Args:
            content: Content to analyze
            content_type: Type of content being analyzed
            analysis_depth: How deep to analyze structure
            
        Returns:
            Dict containing discovered structural elements and relationships
        """
        try:
            context = StructureContext(
                start_time=self.timestamp_service.get_timestamp(),
                content_type=content_type,
                analysis_depth=analysis_depth
            )
            
            with self._lock:
                # Discover structural elements
                elements = await self._discover_elements(content, context)
                
                # Observe natural relationships
                relationships = self._observe_relationships(elements)
                
                # Calculate structural density
                densities = self._calculate_densities(elements)
                
                # Track analysis results
                self._track_analysis(elements, relationships, densities)
                
                return {
                    "elements": elements,
                    "relationships": relationships,
                    "densities": densities,
                    "timestamp": context.start_time.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in structure analysis: {str(e)}")
            return {}
    
    async def _discover_elements(
        self,
        content: Any,
        context: StructureContext
    ) -> List[StructuralElement]:
        """Discover structural elements naturally."""
        elements = []
        
        # Allow elements to emerge based on content type
        if context.content_type == "text":
            elements.extend(await self._analyze_text(content, context))
        elif context.content_type == "code":
            elements.extend(await self._analyze_code(content, context))
        elif context.content_type == "data":
            elements.extend(await self._analyze_data(content, context))
            
        return elements
    
    def _observe_relationships(
        self,
        elements: List[StructuralElement]
    ) -> Dict[str, Set[str]]:
        """Observe natural relationships between elements."""
        relationships = {}
        
        for element in elements:
            # Track direct relationships
            relationships[element.element_id] = element.relationships
            
            # Observe indirect relationships
            for other in elements:
                if other.element_id == element.element_id:
                    continue
                    
                # Calculate relationship strength
                strength = self._calculate_relationship_strength(element, other)
                if strength > 0.5:  # Natural threshold
                    relationships[element.element_id].add(other.element_id)
        
        return relationships
    
    def _calculate_densities(
        self,
        elements: List[StructuralElement]
    ) -> Dict[str, DensityMetrics]:
        """Calculate natural density metrics."""
        densities = {}
        
        for element in elements:
            # Calculate local density
            local_density = len(element.relationships) / max(1, len(elements))
            
            # Calculate global density
            total_relationships = sum(
                len(e.relationships) for e in elements
            )
            global_density = total_relationships / max(1, len(elements) ** 2)
            
            densities[element.element_id] = DensityMetrics(
                local_density=local_density,
                global_density=global_density
            )
            
        return densities
    
    def _track_analysis(
        self,
        elements: List[StructuralElement],
        relationships: Dict[str, Set[str]],
        densities: Dict[str, DensityMetrics]
    ) -> None:
        """Track analysis results for temporal awareness."""
        timestamp = self.timestamp_service.get_timestamp()
        
        # Update structural elements
        for element in elements:
            self.structural_elements[element.element_id] = element
            
        # Update relationships
        self.element_relationships.update(relationships)
        
        # Update density metrics
        self.density_metrics.update(densities)
        
        # Record in history
        self.element_history.append({
            "timestamp": timestamp.isoformat(),
            "elements": [e.element_id for e in elements],
            "relationships": relationships,
            "densities": {k: v.__dict__ for k, v in densities.items()}
        })
    
    def _calculate_relationship_strength(
        self,
        element1: StructuralElement,
        element2: StructuralElement
    ) -> float:
        """Calculate natural relationship strength between elements."""
        # Consider multiple factors
        type_similarity = float(element1.element_type == element2.element_type)
        temporal_proximity = 1.0 / (1.0 + abs(
            (element1.emergence_time - element2.emergence_time).total_seconds()
        ))
        density_similarity = 1.0 - abs(element1.density - element2.density)
        
        # Natural strength calculation
        return (
            0.4 * type_similarity +
            0.3 * temporal_proximity +
            0.3 * density_similarity
        )
    
    async def _analyze_text(
        self,
        content: str,
        context: StructureContext
    ) -> List[StructuralElement]:
        """Analyze text content naturally."""
        from src.core.analysis.text_analysis import TextAnalyzer
        
        analyzer = TextAnalyzer()
        return await analyzer.analyze_text(content, context)
    
    async def _analyze_code(
        self,
        content: str,
        context: StructureContext
    ) -> List[StructuralElement]:
        """Analyze code content naturally."""
        # TODO: Implement natural code analysis
        return []
    
    async def _analyze_data(
        self,
        content: Any,
        context: StructureContext
    ) -> List[StructuralElement]:
        """Analyze data content naturally."""
        # TODO: Implement natural data analysis
        return []
