"""Climate risk document processor using pattern evolution."""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict

from .pattern_evolution import PatternEvolutionTracker, EvolutionMetrics

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
        
    async def process_document(self, doc_path: str) -> ProcessingResult:
        """Process climate risk document using pattern evolution."""
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
            
            # Read document
            try:
                content = path.read_text()
                logger.debug(f"Read content from {doc_path}")
            except Exception as e:
                validation_errors.append(f"Error reading document: {e}")
                return ProcessingResult([], start_time, validation_errors)
            
            # Extract patterns using evolution tracking
            metrics = await self.pattern_extractor.extract(content)
            
            # Calculate overall evolution metrics
            evolution_metrics = None
            if metrics:
                # Calculate coherence based on temporal relationships and semantic weights
                combined = EvolutionMetrics()
                
                # Group metrics by timeframe
                timeframe_groups = defaultdict(list)
                for metric in metrics:
                    timeframe_groups[metric.timeframe].append(metric)
                
                # Calculate coherence for each timeframe group
                total_coherence = 0.0
                for timeframe, group in timeframe_groups.items():
                    # Higher coherence when multiple risks share same timeframe
                    if len(group) > 1:
                        group_coherence = 0.8 + (0.1 * (len(group) - 2))  # Base 0.8, +0.1 for each additional
                        # Adjust by semantic weights
                        avg_weight = sum(m.semantic_weight for m in group) / len(group)
                        group_coherence *= avg_weight
                        total_coherence += group_coherence
                    else:
                        # Single metric in timeframe - base coherence from semantic weight
                        group_coherence = 0.4 * group[0].semantic_weight
                        total_coherence += group_coherence
                
                # Normalize coherence
                if timeframe_groups:
                    combined.coherence = min(total_coherence / len(timeframe_groups), 1.0)
                else:
                    # Ensure minimum coherence when metrics exist
                    combined.coherence = 0.3
                
                # Calculate other metrics
                for metric in metrics:
                    if metric.evolution_metrics:
                        combined.stability += metric.evolution_metrics.stability
                        combined.emergence_rate += metric.evolution_metrics.emergence_rate
                        combined.cross_pattern_flow += metric.evolution_metrics.cross_pattern_flow
                
                # Average stability and other metrics
                n = len(metrics)
                combined.stability /= n
                combined.emergence_rate /= n
                combined.cross_pattern_flow /= n
                evolution_metrics = combined
            
            # Validate results
            if not metrics:
                validation_errors.append("No metrics found in document")
            
            return ProcessingResult(
                metrics=metrics,
                processing_time=start_time,
                validation_errors=validation_errors,
                evolution_metrics=evolution_metrics
            )
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            return ProcessingResult(
                metrics=[],
                processing_time=datetime.utcnow(),
                validation_errors=[f"Processing error: {e}"],
                evolution_metrics=None
            )
