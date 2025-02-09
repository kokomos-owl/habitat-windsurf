"""Climate risk document processor using pattern evolution."""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict

from .pattern_evolution import PatternEvolutionTracker, EvolutionMetrics
from .metrics.pattern_recognition import ClimatePatternRecognizer
from .metrics.temporal_stability import TemporalStabilityTracker
from .metrics.flow_metrics import MetricFlowManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class RiskMetric:
    """Represents an extracted climate risk metric with evolution context."""
    value: float
    unit: str
    timeframe: str
    risk_type: str
    confidence: float
    source_text: str
    evolution_metrics: Optional[EvolutionMetrics] = None
    semantic_weight: float = 1.0
    
    def validate(self) -> bool:
        """Validate metric values and structure."""
        try:
            if not (0 <= self.confidence <= 1):
                logger.error(f"Invalid confidence value: {self.confidence}")
                return False
            if not self.value:
                logger.error("Missing metric value")
                return False
            if not self.risk_type:
                logger.error("Missing risk type")
                return False
            return True
        except Exception as e:
            logger.error(f"Metric validation error: {e}")
            return False

@dataclass
class ProcessingResult:
    """Results from document processing with evolution metrics."""
    metrics: List[RiskMetric]
    processing_time: datetime
    validation_errors: List[str]
    evolution_metrics: Optional[EvolutionMetrics] = None
    
    @property
    def coherence_score(self) -> float:
        """Calculate coherence score from evolution metrics."""
        if not self.evolution_metrics:
            return 0.0
        return self.evolution_metrics.coherence
    
    def is_valid(self) -> bool:
        """Check if processing result is valid."""
        return len(self.validation_errors) == 0

class ClimateRiskProcessor:
    """Processes climate risk documents using pattern evolution."""
    
    def __init__(self):
        self.pattern_recognizer = ClimatePatternRecognizer()
        self.stability_tracker = TemporalStabilityTracker()
        self.flow_manager = MetricFlowManager()
        
    async def process_document(self, doc_path: str, doc_id: str = None, adaptive_id: str = None) -> ProcessingResult:
        """Process climate risk document using enhanced pattern recognition and flow metrics.
        
        Args:
            doc_path: Path to document
            doc_id: MongoDB document ID
            adaptive_id: Adaptive ID for evolution tracking
            
        Returns:
            ProcessingResult with extracted metrics and flow data
        """
        start_time = datetime.utcnow()
        validation_errors = []
        
        # Validate input
        if not doc_path:
            validation_errors.append("Missing document path")
            return ProcessingResult([], start_time, validation_errors)
            
        path = Path(doc_path)
        if not path.exists():
            validation_errors.append(f"Document not found: {doc_path}")
            return ProcessingResult([], start_time, validation_errors)
        
        # Generate IDs if not provided
        if not doc_id:
            doc_id = str(hash(path.name))
        if not adaptive_id:
            adaptive_id = f"doc_{doc_id}_{start_time.timestamp()}"
        
        try:
            # Read document
            content = path.read_text()
            logger.debug(f"Read content from {doc_path}")
            
            # Extract patterns using enhanced recognition
            pattern_matches = self.pattern_recognizer.find_patterns(content)
            
            # Convert patterns to risk metrics
            metrics = []
            for match in pattern_matches:
                try:
                    metric = RiskMetric(
                        value=float(match.value),
                        unit=match.unit or '',
                        timeframe=self._extract_timeframe(match.context),
                        risk_type=self._extract_risk_type(match.pattern_type, match.context),
                        confidence=match.confidence,
                        source_text=match.context.get('surrounding_text', ''),
                        semantic_weight=self._calculate_semantic_weight(match)
                    )
                    
                    if metric.validate():
                        # Track temporal stability
                        self.stability_tracker.add_observation(
                            timestamp=start_time,
                            metric_type=metric.risk_type,
                            value=metric.value,
                            context={'confidence': metric.confidence}
                        )
                        
                        # Create metric flow
                        flow = self.flow_manager.create_flow(metric.risk_type)
                        flow_metrics = {
                            'confidence': metric.confidence,
                            'viscosity': 0.5,  # Default starting viscosity
                            'density': metric.semantic_weight
                        }
                        self.flow_manager.update_flow_metrics(flow.flow_id, flow_metrics)
                        
                        # Add stability score
                        stability_report = self.stability_tracker.get_stability_report(metric.risk_type)
                        metric.evolution_metrics = EvolutionMetrics(
                            stability_score=stability_report['stability_score'],
                            trend=stability_report['trend'],
                            confidence=stability_report['trend_confidence']
                        )
                        
                        metrics.append(metric)
                    else:
                        validation_errors.append(f"Invalid metric: {match}")
                        
                except Exception as e:
                    validation_errors.append(f"Error processing metric: {e}")
                    logger.error(f"Metric processing error: {e}")
            
            # Get overall evolution metrics
            evolution_metrics = None
            if metrics:
                avg_stability = sum(m.evolution_metrics.stability_score for m in metrics) / len(metrics)
                evolution_metrics = EvolutionMetrics(
                    stability_score=avg_stability,
                    trend=self._get_dominant_trend(metrics),
                    confidence=self._calculate_aggregate_confidence(metrics)
                )
            
            return ProcessingResult(
                metrics=metrics,
                processing_time=start_time,
                validation_errors=validation_errors,
                evolution_metrics=evolution_metrics
            )
            
        except Exception as e:
            validation_errors.append(f"Error processing document: {e}")
            logger.error(f"Document processing error: {e}")
            return ProcessingResult([], start_time, validation_errors)
    
    def _extract_timeframe(self, context: dict) -> str:
        """Extract timeframe from pattern context."""
        if 'temporal_indicators' in context:
            indicators = context['temporal_indicators']
            if 'year' in indicators:
                return 'specific_year'
            elif 'reference_year' in indicators:
                return 'reference_point'
            elif 'term' in indicators:
                return 'term_based'
            elif 'century_period' in indicators:
                return 'century_period'
        return 'unspecified'
    
    def _extract_risk_type(self, pattern_type: str, context: dict) -> str:
        """Extract risk type from pattern type and context."""
        if 'temperature' in pattern_type.lower():
            return 'temperature'
        elif 'precipitation' in pattern_type.lower():
            return 'precipitation'
        elif 'sea_level' in pattern_type.lower():
            return 'sea_level'
        elif 'frequency' in pattern_type.lower() and 'event' in context:
            return context['event']
        return 'general'
    
    def _calculate_semantic_weight(self, match) -> float:
        """Calculate semantic weight based on pattern context."""
        weight = 1.0
        
        # Adjust for certainty modifiers
        if 'certainty_modifiers' in match.context:
            if 'certainty_terms' in match.context['certainty_modifiers']:
                weight *= 1.2
            if 'data_terms' in match.context['certainty_modifiers']:
                weight *= 1.3
        
        # Adjust for temporal context
        if match.context.get('temporal_indicators'):
            weight *= 1.1
            
        return min(2.0, weight)
    
    def _get_dominant_trend(self, metrics: List[RiskMetric]) -> str:
        """Get dominant trend from metrics."""
        trends = [m.evolution_metrics.trend for m in metrics if m.evolution_metrics]
        if not trends:
            return 'unknown'
            
        trend_counts = defaultdict(int)
        for trend in trends:
            trend_counts[trend] += 1
            
        return max(trend_counts.items(), key=lambda x: x[1])[0]
    
    def _calculate_aggregate_confidence(self, metrics: List[RiskMetric]) -> float:
        """Calculate aggregate confidence across metrics."""
        if not metrics:
            return 0.0
            
        weighted_sum = sum(m.confidence * m.semantic_weight for m in metrics)
        total_weight = sum(m.semantic_weight for m in metrics)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
