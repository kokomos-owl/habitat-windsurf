"""
Import adapter for quality-aware pattern context.

This module provides adapter functions to bridge the gap between different import styles.
"""

import sys
from pathlib import Path

# Ensure src directory is in path
src_dir = str(Path(__file__).parent.parent.parent.parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import the modules we need with the src. prefix
try:
    from src.habitat_evolution.pattern_aware_rag.pattern_aware_rag import RAGPatternContext as SrcRAGPatternContext
    from src.habitat_evolution.adaptive_core.models import Pattern as SrcPattern
    from src.habitat_evolution.adaptive_core.models import Relationship as SrcRelationship
    from src.habitat_evolution.core.pattern import PatternState as SrcPatternState
    from src.habitat_evolution.adaptive_core.emergence.context_aware_extraction.quality_assessment import QualityAssessment as SrcQualityAssessment
    
    # Re-export with aliases that match the non-src import style
    RAGPatternContext = SrcRAGPatternContext
    Pattern = SrcPattern
    Relationship = SrcRelationship
    PatternState = SrcPatternState
    QualityAssessment = SrcQualityAssessment
    
except ImportError:
    # Fallback to non-src imports if src imports fail
    from habitat_evolution.pattern_aware_rag.pattern_aware_rag import RAGPatternContext
    from habitat_evolution.adaptive_core.models import Pattern, Relationship
    from habitat_evolution.core.pattern import PatternState
    from habitat_evolution.adaptive_core.emergence.context_aware_extraction.quality_assessment import QualityAssessment
