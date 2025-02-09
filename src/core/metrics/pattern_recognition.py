"""Enhanced pattern recognition for climate metrics."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import re

@dataclass
class PatternMatch:
    """Represents a matched pattern with context."""
    pattern_type: str
    value: str
    unit: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

class ClimatePatternRecognizer:
    """Advanced pattern recognition for climate data."""
    
    def __init__(self):
        # Core numeric patterns
        self.numeric_patterns = {
            'precise_number': r'(?P<value>[-+]?\d*\.?\d+)\s*(?P<unit>\w+)?',
            'range': r'(?P<start>[-+]?\d*\.?\d+)\s*(?:to|-)\s*(?P<end>[-+]?\d*\.?\d+)\s*(?P<unit>\w+)?',
            'percentage': r'(?P<value>[-+]?\d*\.?\d+)\s*%'
        }
        
        # Climate-specific patterns
        self.climate_patterns = {
            'temperature_change': r'(?:temperature|temp\.?)\s+(?:increase[ds]?|decrease[ds]?|rise[ds]?|fall[ds]?|change[ds]?)\s+(?:by|to|from)?\s*(?P<value>[-+]?\d*\.?\d+)\s*(?P<unit>°[CF]|degrees?\s*[CF]?)',
            'precipitation_change': r'(?:precipitation|rainfall)\s+(?:increase[ds]?|decrease[ds]?|change[ds]?)\s+(?:by|to|from)?\s*(?P<value>[-+]?\d*\.?\d+)\s*(?P<unit>mm|inches|%)',
            'sea_level_rise': r'sea\s+level\s+(?:rise|increase)[ds]?\s+(?:by|to|from)?\s*(?P<value>[-+]?\d*\.?\d+)\s*(?P<unit>m|meters|ft|feet|inches)',
            'frequency_pattern': r'(?P<event>flood|storm|drought|heat\s+wave)s?\s+occur\w*\s+(?P<value>\d+(?:\.\d+)?)\s*(?:times|%)?\s*(?:per|a|each)?\s*(?P<unit>year|month|decade)',
            'intensity_pattern': r'(?P<event>flood|storm|drought|heat\s+wave)s?\s+intensit\w+\s+(?:increase[ds]?|decrease[ds]?)\s+(?:by|to)?\s*(?P<value>[-+]?\d*\.?\d+)\s*(?P<unit>%|times)',
            'threshold_pattern': r'(?:above|below|exceeds?|under)\s+(?P<value>[-+]?\d*\.?\d+)\s*(?P<unit>°[CF]|mm|inches)\s*(?:threshold|level|mark)',
            'trend_pattern': r'(?P<trend>upward|downward|increasing|decreasing)\s+trend\s+of\s+(?P<value>[-+]?\d*\.?\d+)\s*(?P<unit>%|°[CF]|mm|inches)?\s*(?:per|a|each)?\s*(?P<period>year|decade)'
        }
        
        # Confidence modifiers
        self.confidence_modifiers = {
            'uncertainty_terms': {'may', 'might', 'could', 'possibly', 'potentially'},
            'certainty_terms': {'will', 'definitely', 'certainly', 'clearly'},
            'data_terms': {'measured', 'observed', 'recorded', 'analyzed'},
            'model_terms': {'projected', 'modeled', 'simulated', 'predicted'}
        }
        
    def find_patterns(self, text: str) -> List[PatternMatch]:
        """Find all climate-related patterns in text."""
        matches = []
        
        # Find climate-specific patterns
        for pattern_type, pattern in self.climate_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                context = self._extract_context(text, match)
                confidence = self._calculate_confidence(match, context)
                
                matches.append(PatternMatch(
                    pattern_type=pattern_type,
                    value=match.group('value'),
                    unit=match.group('unit') if 'unit' in match.groupdict() else None,
                    context=context,
                    confidence=confidence
                ))
        
        # Find general numeric patterns
        for pattern_type, pattern in self.numeric_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if not self._is_duplicate_match(match, matches):
                    context = self._extract_context(text, match)
                    confidence = self._calculate_confidence(match, context)
                    
                    matches.append(PatternMatch(
                        pattern_type=pattern_type,
                        value=match.group('value'),
                        unit=match.group('unit') if 'unit' in match.groupdict() else None,
                        context=context,
                        confidence=confidence
                    ))
        
        return matches
    
    def _extract_context(self, text: str, match: re.Match) -> Dict[str, Any]:
        """Extract contextual information around the match."""
        # Get surrounding text (50 chars before and after)
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        context_text = text[start:end]
        
        context = {
            'surrounding_text': context_text,
            'certainty_modifiers': set(),
            'data_source': None,
            'temporal_indicators': []
        }
        
        # Check for certainty modifiers
        for term_type, terms in self.confidence_modifiers.items():
            found_terms = {term for term in terms if term in context_text.lower()}
            if found_terms:
                context['certainty_modifiers'].add(term_type)
        
        # Look for temporal indicators
        temporal_patterns = [
            (r'\b\d{4}\b', 'year'),
            (r'\b(?:in|by|before|after)\s+\d{4}\b', 'reference_year'),
            (r'\b(?:short|medium|long)-term\b', 'term'),
            (r'\b(?:early|mid|late)-century\b', 'century_period')
        ]
        
        for pattern, indicator_type in temporal_patterns:
            if re.search(pattern, context_text, re.IGNORECASE):
                context['temporal_indicators'].append(indicator_type)
        
        return context
    
    def _calculate_confidence(self, match: re.Match, context: Dict[str, Any]) -> float:
        """Calculate confidence score for a pattern match."""
        base_confidence = 0.7  # Start with moderate confidence
        
        # Adjust for certainty modifiers
        if 'certainty_modifiers' in context:
            if 'certainty_terms' in context['certainty_modifiers']:
                base_confidence += 0.1
            if 'uncertainty_terms' in context['certainty_modifiers']:
                base_confidence -= 0.1
            if 'data_terms' in context['certainty_modifiers']:
                base_confidence += 0.15
            if 'model_terms' in context['certainty_modifiers']:
                base_confidence += 0.05
        
        # Adjust for temporal context
        if context.get('temporal_indicators'):
            base_confidence += 0.1
        
        # Adjust for value reasonableness
        try:
            value = float(match.group('value'))
            if abs(value) > 1000 or abs(value) < 0.0001:
                base_confidence *= 0.8
        except (ValueError, IndexError):
            base_confidence *= 0.9
        
        return min(1.0, max(0.0, base_confidence))
    
    def _is_duplicate_match(self, new_match: re.Match, existing_matches: List[PatternMatch]) -> bool:
        """Check if a match overlaps with existing matches."""
        new_span = new_match.span()
        return any(
            new_span[0] <= match.context.get('match_end', 0) and
            new_span[1] >= match.context.get('match_start', 0)
            for match in existing_matches
        )
