"""Enhanced pattern recognition for climate metrics with improved temporal context."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import re
import logging
from .pattern_learner import PatternLearner

logger = logging.getLogger(__name__)

@dataclass
class TemporalContext:
    """Represents temporal information in text."""
    indicator_type: str
    value: str
    confidence: float = 1.0

@dataclass
class PatternMatch:
    """Represents a matched pattern with context."""
    pattern_type: str
    value: str
    unit: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

class ClimatePatternRecognizer:
    """Advanced pattern recognition for climate data with learning capabilities."""
    
    def __init__(self):
        self.learner = PatternLearner()
        
        # Temporal patterns with named capture groups
        self.temporal_patterns = {
            'year': r'(?:by\s+)?(?P<year>20\d{2})',  # e.g., "by 2050"
            'reference_year': r'(?:compared\s+to|relative\s+to|since|from)\s+(?P<ref_year>\d{4})',  # e.g., "since 2000"
            'term': r'\b(?P<term>short|medium|long)\s*-?\s*term\b',
            'century_period': r'\b(?:by\s+)?(?P<period>mid|late|early)\s*-?\s*century\b'  # e.g., "by mid-century"
        }
        
        # Climate-specific patterns with temporal context integration
        self.climate_patterns = {
            'temperature_change': (
                r'(?P<context_pre>[^.]*?)'  # Capture pre-context
                r'temperature\s+'  # Base word
                r'(?:will\s+)?'  # Optional future tense
                r'(?:increase[ds]?|decrease[ds]?|rise[ds]?|fall[ds]?|change[ds]?)'  # Action
                r'\s+(?:by|to|from|between)?\s*'  # Optional preposition
                r'(?P<value>[-+]?\d*\.?\d+)'  # Value
                r'\s*(?P<unit>°[CF]|degrees?\s*[CF]?)'  # Unit
                r'(?P<context_post>[^.]*)'  # Capture post-context
            ),
            'precipitation_change': (
                r'(?P<context_pre>[^.]*?)'
                r'(?:precipitation|rainfall)\s+'
                r'(?:will\s+)?'
                r'(?:increase[ds]?|decrease[ds]?|change[ds]?)'
                r'\s+(?:by|to|from|between)?\s*'
                r'(?P<value>[-+]?\d*\.?\d+)'
                r'\s*(?P<unit>mm|inches|%)'
                r'(?P<context_post>[^.]*)'
            ),
            'sea_level_rise': (
                r'(?P<context_pre>[^.]*?)'
                r'sea\s+level\s+'
                r'(?:rise|increase)[ds]?\s+'
                r'(?:is\s+projected\s+to\s+be|will\s+be)?\s*'
                r'(?P<value>[-+]?\d*\.?\d+)'
                r'\s*(?P<unit>m|meters|ft|feet|inches)'
                r'(?P<context_post>[^.]*)'
            )
        }
        
        # Register patterns with learner
        for pattern_type, pattern in self.climate_patterns.items():
            self.learner.register_pattern(pattern_type, pattern)
        
        # Confidence modifiers
        self.confidence_modifiers = {
            'uncertainty_terms': {'may', 'might', 'could', 'possibly', 'potentially'},
            'certainty_terms': {'will', 'definitely', 'certainly', 'clearly'},
            'data_terms': {'measured', 'observed', 'recorded', 'analyzed'},
            'model_terms': {'projected', 'modeled', 'simulated', 'predicted'}
        }
    
    def _find_temporal_indicators(self, text: str) -> List[TemporalContext]:
        """Find all temporal indicators in text."""
        indicators = []
        for indicator_type, pattern in self.temporal_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Get the named group value
                value = next((v for k, v in match.groupdict().items() if v is not None), match.group(0))
                indicators.append(TemporalContext(
                    indicator_type=indicator_type,
                    value=value,
                    confidence=1.0
                ))
        return indicators
    
    def find_patterns(self, text: str) -> List[PatternMatch]:
        """Find all climate-related patterns in text."""
        matches = []
        
        # Find climate-specific patterns
        for pattern_type, base_pattern in self.climate_patterns.items():
            logger.debug(f"Searching for pattern type: {pattern_type}")
            
            # Get evolved pattern if available
            pattern = self.learner.get_best_pattern(pattern_type) or base_pattern
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    logger.debug(f"Found match: {match.group(0)}")
                    
                    # Get pre and post context
                    pre_context = match.group('context_pre') if 'context_pre' in match.groupdict() else ""
                    post_context = match.group('context_post') if 'context_post' in match.groupdict() else ""
                    context_text = f"{pre_context} {match.group(0)} {post_context}"
                    
                    # Find temporal indicators in the context
                    temporal_indicators = self._find_temporal_indicators(context_text)
                    
                    context = {
                        'surrounding_text': context_text.strip(),
                        'temporal_indicators': [
                            {
                                'type': indicator.indicator_type,
                                'value': indicator.value,
                                'confidence': indicator.confidence
                            }
                            for indicator in temporal_indicators
                        ],
                        'certainty_modifiers': self._find_certainty_modifiers(context_text)
                    }
                    
                    confidence = self._calculate_confidence(match, context)
                    
                    # Extract value
                    value = match.group('value')
                    
                    match_obj = PatternMatch(
                        pattern_type=pattern_type,
                        value=value,
                        unit=match.group('unit') if 'unit' in match.groupdict() else None,
                        context=context,
                        confidence=confidence
                    )
                    
                    matches.append(match_obj)
                    
                    # Record success/failure based on temporal indicators
                    success = bool(temporal_indicators)
                    self.learner.record_result(pattern_type, text, success)
                    
                except (IndexError, AttributeError) as e:
                    logger.error(f"Error processing match: {e}")
                    self.learner.record_result(pattern_type, text, False)
                    continue
        
        return matches
    
    def _find_certainty_modifiers(self, text: str) -> Dict[str, Set[str]]:
        """Find certainty modifiers in text."""
        modifiers = {}
        for term_type, terms in self.confidence_modifiers.items():
            found_terms = {term for term in terms if term in text.lower()}
            if found_terms:
                modifiers[term_type] = found_terms
        return modifiers
    
    def _calculate_confidence(self, match: re.Match, context: Dict[str, Any]) -> float:
        """Calculate confidence score for a match."""
        confidence = 1.0
        
        # Adjust for certainty modifiers
        if 'certainty_modifiers' in context:
            modifiers = context['certainty_modifiers']
            if 'uncertainty_terms' in modifiers:
                confidence *= 0.8
            if 'certainty_terms' in modifiers:
                confidence *= 1.2
            if 'data_terms' in modifiers:
                confidence *= 1.3
            if 'model_terms' in modifiers:
                confidence *= 0.9
        
        # Adjust for temporal context
        if context.get('temporal_indicators'):
            confidence *= 1.1
            
        # Adjust for pattern quality
        if match.group(0):  # Full match exists
            confidence *= 1.0 + (len(match.groups()) / 10)  # More captured groups = higher confidence
            
        return min(1.0, confidence)  # Cap at 1.0
        
    def get_pattern_stats(self) -> Dict[str, Dict[str, any]]:
        """Get statistics about pattern evolution."""
        return {
            pattern_type: self.learner.get_pattern_stats(pattern_type)
            for pattern_type in self.climate_patterns.keys()
        }
