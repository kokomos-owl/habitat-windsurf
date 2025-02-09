"""Metric extraction service with flow-based pattern evolution."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.core.evolution.pattern_core import PatternCore, LearningWindow
from src.core.metrics.flow_metrics import MetricFlowManager
from src.core.validation.document_validator import DocumentSection

@dataclass
class MetricContext:
    """Context for metric extraction."""
    section: DocumentSection
    temporal_range: Optional[Tuple[datetime, datetime]] = None
    source_reliability: float = 1.0
    cross_validation_score: float = 1.0
    
class MetricService:
    """Service for extracting and evolving metrics through flow-based patterns."""
    
    def __init__(self, pattern_core: Optional[PatternCore] = None):
        self.pattern_core = pattern_core or PatternCore()
        self.flow_manager = MetricFlowManager()
        
        # Create initial learning window
        self.learning_window_id = self.pattern_core.create_learning_window()
        
    def extract_metrics(self, section: DocumentSection) -> List[Dict[str, Any]]:
        """Extract metrics from a document section using flow-based pattern matching."""
        # Create context
        context = MetricContext(
            section=section,
            temporal_range=self._extract_temporal_range(section),
            source_reliability=self._calculate_source_reliability(section)
        )
        
        # Extract metrics through flow manager
        metrics = self.flow_manager.extract_metrics(
            section.content,
            {
                'temporal_distance': self._calculate_temporal_distance(context.temporal_range),
                'source_reliability': context.source_reliability,
                'cross_validation_score': context.cross_validation_score
            }
        )
        
        # Process through pattern evolution
        for metric in metrics:
            self._evolve_metric_pattern(metric, context)
            
        return metrics
    
    def _extract_temporal_range(self, section: DocumentSection) -> Optional[Tuple[datetime, datetime]]:
        """Extract temporal range from section markers."""
        if not section.temporal_markers:
            return None
            
        # Convert markers to dates
        dates = []
        for marker in section.temporal_markers:
            try:
                if '-' in marker:
                    start, end = marker.split('-')
                    dates.extend([
                        datetime.strptime(start.strip(), '%Y'),
                        datetime.strptime(end.strip(), '%Y')
                    ])
                else:
                    dates.append(datetime.strptime(marker.strip(), '%Y'))
            except ValueError:
                continue
                
        if not dates:
            return None
            
        return min(dates), max(dates)
    
    def _calculate_temporal_distance(self, temporal_range: Optional[Tuple[datetime, datetime]]) -> float:
        """Calculate temporal distance score."""
        if not temporal_range:
            return 50  # Middle ground when unknown
            
        now = datetime.utcnow()
        start, end = temporal_range
        
        # Calculate average temporal distance in years
        avg_distance = abs((start.year + end.year) / 2 - now.year)
        
        return min(100, avg_distance)
    
    def _calculate_source_reliability(self, section: DocumentSection) -> float:
        """Calculate source reliability score."""
        base_score = 0.8  # Start with good reliability
        
        # Adjust based on section characteristics
        if section.metrics:
            base_score += 0.1  # Bonus for having metrics
            
        if section.temporal_markers:
            base_score += 0.1  # Bonus for temporal context
            
        return min(1.0, base_score)
    
    def _evolve_metric_pattern(self, metric: Dict[str, Any], context: MetricContext) -> None:
        """Evolve metric through pattern core."""
        # Create pattern observation
        observation = {
            'value': metric['value'],
            'type': metric['type'],
            'confidence': metric['confidence'],
            'temporal_context': {
                'range': context.temporal_range,
                'section': context.section.name
            },
            'flow_id': metric['flow_id']
        }
        
        # Observe pattern evolution
        result = self.pattern_core.observe_pattern(
            pattern_id=f"metric_{metric['flow_id']}",
            observation=observation,
            window_id=self.learning_window_id
        )
        
        # Update flow metrics based on evolution
        if result:
            self.flow_manager.update_flow_metrics(
                metric['flow_id'],
                {
                    'confidence': result.get('confidence', metric['confidence']),
                    'viscosity': result.get('viscosity_gradient', 0.35),
                    'density': result.get('density', 1.0),
                    'temporal_stability': result.get('temporal_stability', 1.0),
                    'cross_validation_score': result.get('cross_reference_score', 1.0)
                }
            )
            
    def get_metric_confidence(self, metric_id: str) -> float:
        """Get current confidence for a metric."""
        return self.flow_manager.get_flow_confidence(metric_id)
    
    def get_pattern_confidence(self, pattern: str) -> float:
        """Get confidence for a metric pattern."""
        return self.flow_manager.get_pattern_confidence(pattern)
