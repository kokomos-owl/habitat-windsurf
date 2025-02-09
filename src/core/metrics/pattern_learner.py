"""Pattern learning system that evolves pattern recognition based on test results."""

from typing import Dict, List, Set, Tuple, Optional
import re
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PatternEvolution:
    """Tracks the evolution of a pattern."""
    original_pattern: str
    success_count: int = 0
    failure_count: int = 0
    variants: List[Tuple[str, float]] = None  # (pattern, success_rate)
    
    def __post_init__(self):
        self.variants = self.variants or [(self.original_pattern, 0.0)]

class PatternLearner:
    """Learns and evolves patterns based on success/failure feedback."""
    
    def __init__(self):
        self.pattern_history: Dict[str, PatternEvolution] = {}
        self.success_threshold = 0.8
        
    def register_pattern(self, pattern_type: str, pattern: str) -> None:
        """Register a new pattern for tracking."""
        if pattern_type not in self.pattern_history:
            self.pattern_history[pattern_type] = PatternEvolution(pattern)
            logger.info(f"Registered new pattern type: {pattern_type}")
    
    def record_result(self, pattern_type: str, text: str, success: bool) -> None:
        """Record the success or failure of a pattern match."""
        if pattern_type not in self.pattern_history:
            logger.warning(f"Unknown pattern type: {pattern_type}")
            return
            
        evolution = self.pattern_history[pattern_type]
        if success:
            evolution.success_count += 1
        else:
            evolution.failure_count += 1
            self._analyze_failure(evolution, text)
    
    def _analyze_failure(self, evolution: PatternEvolution, text: str) -> None:
        """Analyze a failure and propose pattern improvements."""
        # Look for common variations
        variations = [
            self._add_optional_whitespace,
            self._add_optional_words,
            self._relax_word_boundaries,
            self._expand_numeric_formats
        ]
        
        for vary_func in variations:
            new_pattern = vary_func(evolution.original_pattern)
            if new_pattern != evolution.original_pattern:
                # Test if new pattern matches where original failed
                try:
                    if re.search(new_pattern, text, re.IGNORECASE):
                        success_rate = evolution.success_count / (evolution.success_count + evolution.failure_count)
                        evolution.variants.append((new_pattern, success_rate))
                        logger.info(f"Found potential pattern improvement: {new_pattern}")
                except re.error:
                    continue
    
    def _add_optional_whitespace(self, pattern: str) -> str:
        """Add optional whitespace between components."""
        return re.sub(r'\s+', r'\\s*', pattern)
    
    def _add_optional_words(self, pattern: str) -> str:
        """Add optional connecting words."""
        common_words = ['is', 'are', 'was', 'were', 'be', 'being']
        for word in common_words:
            pattern = pattern.replace(f' {word} ', f' (?:{word} )?')
        return pattern
    
    def _relax_word_boundaries(self, pattern: str) -> str:
        """Relax strict word boundary requirements."""
        return pattern.replace(r'\b', r'\b?')
    
    def _expand_numeric_formats(self, pattern: str) -> str:
        """Expand numeric format recognition."""
        return pattern.replace(
            r'\d+',
            r'(?:\d*\.)?\d+(?:e[-+]?\d+)?'
        )
    
    def get_best_pattern(self, pattern_type: str) -> Optional[str]:
        """Get the most successful pattern variant."""
        if pattern_type not in self.pattern_history:
            return None
            
        evolution = self.pattern_history[pattern_type]
        if not evolution.variants:
            return evolution.original_pattern
            
        # Sort by success rate
        sorted_variants = sorted(
            evolution.variants,
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return best performing pattern
        return sorted_variants[0][0]
    
    def get_pattern_stats(self, pattern_type: str) -> Dict[str, any]:
        """Get statistics about pattern evolution."""
        if pattern_type not in self.pattern_history:
            return {}
            
        evolution = self.pattern_history[pattern_type]
        total = evolution.success_count + evolution.failure_count
        success_rate = evolution.success_count / total if total > 0 else 0
        
        return {
            'success_rate': success_rate,
            'total_attempts': total,
            'variant_count': len(evolution.variants),
            'best_pattern': self.get_best_pattern(pattern_type)
        }
