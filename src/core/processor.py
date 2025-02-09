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

class SemanticPatternExtractor:
    """Extracts climate risk patterns using evolution tracking."""
    
    def __init__(self):
        # Initialize pattern evolution tracker
        self.evolution_tracker = PatternEvolutionTracker()
        
        # Semantic concept mappings with expanded variations
        self.risk_types = {
            'drought': ['drought', 'dry', 'arid', 'water scarcity'],
            'flood': ['flood', 'flooding', 'inundation', 'water level'],
            'wildfire': ['wildfire', 'fire', 'burn', 'flame', 'wildfire danger'],
            'storm': ['storm', 'hurricane', 'cyclone', 'wind']
        }
        
        self.timeframes = {
            'mid': ['mid-century', 'mid century', 'middle of century', '2050'],
            'late': ['late-century', 'late century', 'end of century', '2100']
        }
        
        self.change_indicators = [
            'increase', 'rise', 'grow', 'escalate', 'surge',
            'from', 'to', 'by', 'reach', 'change', 'likelihood'
        ]
        
    def _normalize_risk_type(self, text: str) -> Optional[str]:
        """Normalize risk type from text variations."""
        text_lower = text.lower()
        for risk_type, variations in self.risk_types.items():
            if any(var in text_lower for var in variations):
                return risk_type
        return None
        
    def _normalize_timeframe(self, text: str) -> Optional[str]:
        """Normalize timeframe from text variations."""
        text_lower = text.lower()
        for timeframe, variations in self.timeframes.items():
            if any(var in text_lower for var in variations):
                return timeframe
        return None
        
    def _extract_value(self, text: str) -> Optional[float]:
        """Extract numerical value from text, handling 'from X to Y' patterns."""
        # First try to find "from X% to Y%" pattern
        from_to_match = re.search(r'from\s*(\d+(?:\.\d+)?)\s*%\s*to\s*(\d+(?:\.\d+)?)\s*%', text.lower())
        if from_to_match:
            return float(from_to_match.group(2))  # Return the target value
            
        # Otherwise look for single percentage
        single_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if single_match:
            return float(single_match.group(1))
            
        return None

    def _extract_semantic_chunks(self, text: str) -> List[dict]:
        """Extract semantic chunks from text based on concept relationships."""
        chunks = []
        
        # Split into sentences but preserve from-to patterns
        sentences = []
        current = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Split by period but be careful with numbers
            parts = re.split(r'(?<!\d)\.(?!\d)', line)
            for part in parts:
                part = part.strip()
                if part:
                    sentences.append(part)
        
        logger.debug(f"Split into sentences: {sentences}")
        
        for sentence in sentences:
            # Initialize chunk with high base confidence
            chunk = {
                'text': sentence,
                'risk_type': None,
                'timeframe': None,
                'value': None,
                'confidence': 1.0
            }
            
            # Find risk type with normalization
            chunk['risk_type'] = self._normalize_risk_type(sentence)
            if chunk['risk_type']:
                logger.debug(f"Found risk type: {chunk['risk_type']}")
            
            # Find timeframe with normalization
            chunk['timeframe'] = self._normalize_timeframe(sentence)
            if chunk['timeframe']:
                logger.debug(f"Found timeframe: {chunk['timeframe']}")
            
            # Extract value with special handling for from-to patterns
            chunk['value'] = self._extract_value(sentence)
            if chunk['value'] is not None:
                logger.debug(f"Found value: {chunk['value']}")
            
            if chunk['risk_type'] and chunk['timeframe'] and chunk['value'] is not None:
                chunks.append(chunk)
                logger.debug(f"Created chunk: {chunk}")
        
        return chunks
        
    async def extract(self, content: str) -> List[RiskMetric]:
        """Extract risk metrics using pattern evolution."""
        try:
            metrics = []
            logger.debug(f"Processing content using pattern evolution")
            
            chunks = self._extract_semantic_chunks(content)
            logger.debug(f"Found {len(chunks)} semantic chunks")
            
            for chunk in chunks:
                try:
                    # Create pattern from chunk
                    pattern = f"{chunk['risk_type']}_{chunk['timeframe']}"
                    
                    # Find related patterns
                    related_patterns = []
                    for other in chunks:
                        if other != chunk:
                            related = f"{other['risk_type']}_{other['timeframe']}"
                            related_patterns.append(related)
                    
                    # Observe pattern evolution
                    evolution_metrics = self.evolution_tracker.observe_pattern(
                        pattern=pattern,
                        confidence=chunk['confidence'],
                        temporal_context={
                            'timeframe': chunk['timeframe'],
                            'timestamp': datetime.utcnow().isoformat()
                        },
                        related_patterns=related_patterns
                    )
                    
                    # Calculate semantic weight based on emphasis words
                    semantic_weight = 1.0
                    emphasis_words = ['significantly', 'dramatically', 'surge', 'extreme']
                    for word in emphasis_words:
                        if word in chunk['text'].lower():
                            semantic_weight *= 1.2
                    
                    metric = RiskMetric(
                        value=chunk['value'],
                        unit='percentage',
                        timeframe=chunk['timeframe'],
                        risk_type=chunk['risk_type'],
                        confidence=chunk['confidence'],
                        source_text=chunk['text'],
                        evolution_metrics=evolution_metrics,
                        semantic_weight=semantic_weight
                    )
                    
                    if metric.validate():
                        metrics.append(metric)
                        logger.info(f"Extracted metric with evolution: {metric}")
                    else:
                        logger.warning(f"Invalid metric: {metric}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process chunk: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Pattern extraction error: {e}")
            return []

class ClimateRiskProcessor:
    """Processes climate risk documents using pattern evolution."""
    
    def __init__(self):
        self.pattern_extractor = SemanticPatternExtractor()
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
        try:
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
                validation_errors.append(f"Error reading document: {e}")
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

            evolution_metrics = None
            if metrics:
                combined = EvolutionMetrics()
                
                # Group metrics by interface context
                interface_groups = defaultdict(list)
                for metric in metrics:
                    # Create interface context
                    interface_key = f"{metric.timeframe}_{metric.risk_type}"
                    interface_groups[interface_key].append(metric)
                
                # Calculate coherence based on interface relationships
                total_coherence = 0.0
                for interface_key, group in interface_groups.items():
                    # Base coherence on interface strength
                    interface_coherence = 0.5  # Base coherence
                    
                    # Adjust for pattern density
                    if len(group) > 1:
                        interface_coherence += 0.1 * (len(group) - 1)
                    
                    # Adjust for semantic relationships
                    semantic_coherence = sum(m.semantic_weight for m in group) / len(group)
                    interface_coherence *= semantic_coherence
                    
                    # Adjust for temporal alignment
                    temporal_groups = defaultdict(list)
                    for m in group:
                        temporal_groups[m.timeframe].append(m)
                    if len(temporal_groups) > 1:
                        interface_coherence *= 1.2  # Bonus for temporal spread
                    
                    total_coherence += interface_coherence
                
                # Normalize coherence
                if interface_groups:
                    combined.coherence = min(total_coherence / len(interface_groups), 1.0)
                else:
                    combined.coherence = 0.3
                
                # Calculate interface-aware metrics
                for metric in metrics:
                    if metric.evolution_metrics:
                        # Weight by interface context
                        interface_weight = len(interface_groups[f"{metric.timeframe}_{metric.risk_type}"]) / len(metrics)
                        combined.stability += metric.evolution_metrics.stability * interface_weight
                        combined.emergence_rate += metric.evolution_metrics.emergence_rate * interface_weight
                        combined.cross_pattern_flow += metric.evolution_metrics.cross_pattern_flow * interface_weight
                
                evolution_metrics = combined
            
            # Validate results
            if not metrics:
                validation_errors.append("No metrics found in document")
            
            result = ProcessingResult(
                metrics=metrics,
                processing_time=start_time,
                validation_errors=validation_errors,
                evolution_metrics=evolution_metrics
            )
            
            # Track document evolution
            if hasattr(self, 'evolution_tracker'):
                evolution_result = self.evolution_tracker.process_document(
                    doc_id=doc_id,
                    adaptive_id=adaptive_id,
                    content=content,
                    processor_result=result
                )
                logger.debug(f"Evolution tracking result: {evolution_result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            return ProcessingResult(
                metrics=[],
                processing_time=datetime.utcnow(),
                validation_errors=[f"Processing error: {e}"],
                evolution_metrics=None
            )
