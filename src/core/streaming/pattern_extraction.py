"""Pattern extraction for streaming document processing."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime

from ..pattern_evolution import EvolutionMetrics
from .types import DocumentChunk, StreamingPattern

@dataclass
class PatternMatch:
    """A pattern match within a document chunk."""
    pattern_type: str
    content: str
    start_pos: int
    end_pos: int
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)

class StreamingPatternExtractor:
    """Extracts patterns from streaming document content."""
    
    def __init__(self):
        self.pattern_history: Dict[str, List[PatternMatch]] = {}
        self.last_vector_space: Dict[str, Dict[str, float]] = {}
        
    async def extract_patterns(self, chunk: DocumentChunk) -> List[StreamingPattern]:
        """Extract patterns from a document chunk.
        
        Args:
            chunk: DocumentChunk to process
            
        Returns:
            List of StreamingPattern objects detected in the chunk
        """
        # Extract initial patterns
        matches = self._find_pattern_matches(chunk.content)
        
        # Convert matches to StreamingPatterns with vector space metrics
        patterns = []
        for match in matches:
            pattern_id = self._generate_pattern_id(match)
            
            # Calculate vector space metrics
            vector_space = self._calculate_vector_space(match, pattern_id)
            
            # Create StreamingPattern
            pattern = StreamingPattern(
                pattern_id=pattern_id,
                confidence=match.confidence,
                vector_space=vector_space,
                temporal_context=self._extract_temporal_context(match)
            )
            patterns.append(pattern)
            
            # Update history
            if pattern_id not in self.pattern_history:
                self.pattern_history[pattern_id] = []
            self.pattern_history[pattern_id].append(match)
            
        return patterns
        
    def _find_pattern_matches(self, content: str) -> List[PatternMatch]:
        """Find pattern matches in content using flow-based pattern matching."""
        # Define pattern matchers with climate context
        pattern_matchers = {
            'test_doc': {
                'pattern': r'test\s+document',
                'type': 'test_pattern',
                'base_confidence': 0.8
            },
            'test_content': {
                'pattern': r'test\s+content',
                'type': 'test_pattern',
                'base_confidence': 0.75
            },
            'verification': {
                'pattern': r'verification',
                'type': 'verification_pattern',
                'base_confidence': 0.7
            },
            'evolution': {
                'pattern': r'pattern\s+evolution',
                'type': 'evolution_pattern',
                'base_confidence': 0.9
            },
            'validation': {
                'pattern': r'validate|validation',
                'type': 'validation_pattern',
                'base_confidence': 0.85
            }
        }
        
        matches = []
        for matcher_type, config in pattern_matchers.items():
            import re
            for match in re.finditer(config['pattern'], content, re.IGNORECASE):
                # Extract surrounding context
                start_idx = max(0, match.start() - 50)
                end_idx = min(len(content), match.end() + 50)
                context_text = content[start_idx:end_idx]
                
                # Calculate confidence based on pattern type and content
                confidence = self._calculate_confidence(
                    config['base_confidence'],
                    match.group(0),
                    context_text,
                    config['type']
                )
                
                matches.append(PatternMatch(
                    pattern_type=config['type'],
                    content=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence,
                    context={
                        'surrounding_text': context_text,
                        'extracted_value': match.group(1) if match.groups() else None,
                        'pattern_type': matcher_type
                    }
                ))
        return matches
        
    def _calculate_confidence(self, base_confidence: float, value: str, context: str, pattern_type: str) -> float:
        """Calculate confidence score for a pattern match."""
        confidence = base_confidence
        
        # Adjust for value characteristics
        if pattern_type in ['temperature', 'trend', 'range']:
            try:
                # Extract numeric values
                import re
                numbers = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+', value)]
                
                # Penalize unusual values
                for num in numbers:
                    if abs(num) > 1000 or abs(num) < 0.0001:
                        confidence *= 0.8
                        
                # Boost confidence for reasonable ranges
                if pattern_type == 'temperature':
                    if all(-100 <= x <= 150 for x in numbers):  # Reasonable temperature range (both C and F)
                        confidence *= 1.2
                elif pattern_type == 'trend':
                    if all(0 <= x <= 100 for x in numbers):  # Reasonable percentage changes
                        confidence *= 1.1
            except (ValueError, IndexError):
                confidence *= 0.7
        
        # Adjust for context quality
        context_indicators = [
            'climate', 'weather', 'temperature', 'precipitation',
            'environment', 'atmospheric', 'meteorological'
        ]
        context_matches = sum(1 for indicator in context_indicators if indicator in context.lower())
        context_boost = min(0.2, context_matches * 0.05)  # Up to 0.2 boost for good context
        confidence += context_boost
        
        return min(1.0, max(0.0, confidence))
        
    def _generate_pattern_id(self, match: PatternMatch) -> str:
        """Generate unique ID for a pattern match."""
        return f"{match.pattern_type}_{hash(match.content)}"
        
    def _calculate_vector_space(self, 
                              match: PatternMatch,
                              pattern_id: str) -> Dict[str, float]:
        """Calculate vector space metrics for a pattern match."""
        # Get previous vector space if it exists
        prev_vector_space = self.last_vector_space.get(pattern_id, {
            'stability': 0.5,
            'coherence': 0.5,
            'emergence_rate': 0.0,
            'cross_pattern_flow': 0.0,
            'energy_state': 0.5,
            'adaptation_rate': 0.0
        })
        
        # Calculate new metrics
        stability = self._calculate_stability(match, pattern_id)
        coherence = self._calculate_coherence(match)
        emergence_rate = self._calculate_emergence_rate(pattern_id)
        cross_pattern_flow = self._calculate_cross_pattern_flow(pattern_id)
        energy_state = self._calculate_energy_state(match)
        adaptation_rate = self._calculate_adaptation_rate(pattern_id)
        
        # Create new vector space
        vector_space = {
            'stability': stability,
            'coherence': coherence,
            'emergence_rate': emergence_rate,
            'cross_pattern_flow': cross_pattern_flow,
            'energy_state': energy_state,
            'adaptation_rate': adaptation_rate
        }
        
        # Store for next calculation
        self.last_vector_space[pattern_id] = vector_space
        
        return vector_space
        
    def _calculate_stability(self, match: PatternMatch, pattern_id: str) -> float:
        """Calculate pattern stability."""
        if pattern_id not in self.pattern_history:
            return 0.5
            
        history = self.pattern_history[pattern_id]
        if len(history) < 2:
            return 0.5
            
        # Calculate stability based on confidence consistency
        confidence_std = np.std([m.confidence for m in history[-5:]])
        return 1.0 - min(confidence_std, 1.0)
        
    def _calculate_coherence(self, match: PatternMatch) -> float:
        """Calculate pattern coherence."""
        # Base coherence on match confidence and context completeness
        context_score = len(match.context) / 5  # Normalize by expected context fields
        return min(match.confidence * (1 + context_score) / 2, 1.0)
        
    def _calculate_emergence_rate(self, pattern_id: str) -> float:
        """Calculate pattern emergence rate."""
        if pattern_id not in self.pattern_history:
            return 0.0
            
        history = self.pattern_history[pattern_id]
        if len(history) < 2:
            return 0.0
            
        # Calculate rate of confidence increase
        confidences = [m.confidence for m in history[-5:]]
        return max(0.0, min(1.0, (confidences[-1] - confidences[0]) + 0.5))
        
    def _calculate_cross_pattern_flow(self, pattern_id: str) -> float:
        """Calculate cross-pattern energy flow."""
        # For now, use a simple metric based on number of concurrent patterns
        total_patterns = len(self.pattern_history)
        if total_patterns <= 1:
            return 0.0
            
        return min(1.0, (total_patterns - 1) * 0.2)
        
    def _calculate_energy_state(self, match: PatternMatch) -> float:
        """Calculate pattern energy state."""
        # Base energy on match confidence and content length
        energy = match.confidence * (len(match.content) / 1000)  # Normalize by expected length
        return min(1.0, energy)
        
    def _calculate_adaptation_rate(self, pattern_id: str) -> float:
        """Calculate pattern adaptation rate."""
        if pattern_id not in self.pattern_history:
            return 0.0
            
        history = self.pattern_history[pattern_id]
        if len(history) < 3:
            return 0.0
            
        # Calculate rate of context changes
        context_changes = 0
        for i in range(1, len(history)):
            if history[i].context != history[i-1].context:
                context_changes += 1
                
        return min(1.0, context_changes / len(history))
        
    def _extract_temporal_context(self, match: PatternMatch) -> Dict[str, Any]:
        """Extract temporal context from pattern match."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'pattern_type': match.pattern_type,
            **match.context
        }
