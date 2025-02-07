"""Dynamic semantic pattern learning through density loops."""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SemanticPattern:
    """A learned semantic pattern with confidence and usage metrics."""
    pattern: str
    concept_type: str  # 'risk_type', 'timeframe', etc.
    confidence: float = 0.0
    occurrences: int = 0
    last_seen: datetime = field(default_factory=datetime.utcnow)
    related_patterns: Set[str] = field(default_factory=set)
    
    def update_confidence(self, success: bool):
        """Update pattern confidence based on successful usage."""
        decay = 0.95  # Slight decay over time
        boost = 1.2 if success else 0.8
        self.confidence = min(1.0, self.confidence * decay * boost)
        self.occurrences += 1 if success else 0
        self.last_seen = datetime.utcnow()

@dataclass
class DensityWindow:
    """A time window for observing pattern density and relationships."""
    start_time: datetime
    duration: timedelta
    patterns: Dict[str, SemanticPattern] = field(default_factory=dict)
    relationships: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    def is_active(self) -> bool:
        """Check if window is still active."""
        return datetime.utcnow() - self.start_time < self.duration

class SemanticLearner:
    """Learns and adapts semantic patterns through density loops."""
    
    def __init__(self, patterns_file: Optional[str] = None):
        self.windows: List[DensityWindow] = []
        self.patterns: Dict[str, SemanticPattern] = {}
        self.window_duration = timedelta(hours=1)
        self.max_windows = 24  # Keep 24 hours of history
        self.patterns_file = patterns_file or "learned_patterns.json"
        
        # Load existing patterns
        self._load_patterns()
        
    def _load_patterns(self):
        """Load learned patterns from file."""
        try:
            path = Path(self.patterns_file)
            if path.exists():
                with path.open() as f:
                    data = json.load(f)
                    for p in data.get('patterns', []):
                        pattern = SemanticPattern(**p)
                        self.patterns[pattern.pattern] = pattern
                logger.info(f"Loaded {len(self.patterns)} learned patterns")
        except Exception as e:
            logger.warning(f"Could not load patterns: {e}")
    
    def _save_patterns(self):
        """Save learned patterns to file."""
        try:
            data = {
                'patterns': [vars(p) for p in self.patterns.values()],
                'last_updated': datetime.utcnow().isoformat()
            }
            with open(self.patterns_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved {len(self.patterns)} learned patterns")
        except Exception as e:
            logger.warning(f"Could not save patterns: {e}")
    
    def start_window(self) -> DensityWindow:
        """Start a new density window for pattern learning."""
        # Clean up old windows
        now = datetime.utcnow()
        self.windows = [w for w in self.windows if now - w.start_time < timedelta(hours=24)]
        
        # Create new window
        window = DensityWindow(
            start_time=now,
            duration=self.window_duration
        )
        self.windows.append(window)
        return window
    
    def observe_pattern(self, window: DensityWindow, pattern: str, 
                       concept_type: str, related_to: Optional[str] = None):
        """Observe a pattern in the current window."""
        if not window.is_active():
            logger.warning("Window is no longer active")
            return
            
        # Update pattern
        if pattern not in window.patterns:
            window.patterns[pattern] = SemanticPattern(
                pattern=pattern,
                concept_type=concept_type
            )
        
        # Update relationships
        if related_to:
            window.relationships[pattern].add(related_to)
            window.relationships[related_to].add(pattern)
    
    def learn_from_window(self, window: DensityWindow):
        """Learn from patterns observed in a density window."""
        if window.is_active():
            logger.warning("Window is still active")
            return
            
        # Analyze pattern density
        total_patterns = len(window.patterns)
        if total_patterns == 0:
            return
            
        # Update pattern confidence based on density and relationships
        for pattern, p in window.patterns.items():
            # Calculate relationship strength
            rel_strength = len(window.relationships[pattern]) / total_patterns
            
            # Update global pattern
            if pattern not in self.patterns:
                self.patterns[pattern] = p
            else:
                self.patterns[pattern].update_confidence(True)
                self.patterns[pattern].related_patterns.update(
                    window.relationships[pattern]
                )
            
            # Boost confidence based on relationship strength
            self.patterns[pattern].confidence = min(
                1.0,
                self.patterns[pattern].confidence + (rel_strength * 0.1)
            )
        
        # Save updated patterns
        self._save_patterns()
    
    def suggest_patterns(self, concept_type: str, 
                        min_confidence: float = 0.5) -> List[str]:
        """Suggest patterns for a concept type based on learned confidence."""
        return [
            p.pattern for p in self.patterns.values()
            if p.concept_type == concept_type 
            and p.confidence >= min_confidence
        ]
    
    def get_related_patterns(self, pattern: str, 
                           min_confidence: float = 0.3) -> List[str]:
        """Get related patterns that often occur together."""
        if pattern not in self.patterns:
            return []
            
        return [
            rel for rel in self.patterns[pattern].related_patterns
            if rel in self.patterns 
            and self.patterns[rel].confidence >= min_confidence
        ]
