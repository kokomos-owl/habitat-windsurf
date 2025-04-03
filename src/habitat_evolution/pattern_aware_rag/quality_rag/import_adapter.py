"""
Import adapter for quality-enhanced retrieval.

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
    from src.habitat_evolution.pattern_aware_rag.pattern_aware_rag import PatternAwareRAG as SrcPatternAwareRAG
    from src.habitat_evolution.adaptive_core.emergence.context_aware_extraction.context_aware_extractor import ContextAwareExtractor as SrcContextAwareExtractor
    from src.habitat_evolution.adaptive_core.emergence.context_aware_extraction.quality_assessment import QualityAssessment as SrcQualityAssessment
    from src.habitat_evolution.pattern_aware_rag.context.quality_aware_context import QualityAwarePatternContext as SrcQualityAwarePatternContext
    from src.habitat_evolution.adaptive_core.models import Pattern as SrcPattern
    from src.habitat_evolution.adaptive_core.models import Relationship as SrcRelationship
    from src.habitat_evolution.adaptive_core.persistence.interfaces.pattern_repository import PatternRepository as SrcPatternRepository
    from src.habitat_evolution.core.pattern import PatternState as SrcPatternState
    
    # Re-export with aliases that match the non-src import style
    PatternAwareRAG = SrcPatternAwareRAG
    ContextAwareExtractor = SrcContextAwareExtractor
    QualityAssessment = SrcQualityAssessment
    QualityAwarePatternContext = SrcQualityAwarePatternContext
    Pattern = SrcPattern
    Relationship = SrcRelationship
    PatternRepository = SrcPatternRepository
    PatternState = SrcPatternState
    
except ImportError:
    # Fallback to non-src imports if src imports fail
    from habitat_evolution.pattern_aware_rag.pattern_aware_rag import PatternAwareRAG
    from habitat_evolution.adaptive_core.emergence.context_aware_extraction.context_aware_extractor import ContextAwareExtractor
    from habitat_evolution.adaptive_core.emergence.context_aware_extraction.quality_assessment import QualityAssessment
    from habitat_evolution.pattern_aware_rag.context.quality_aware_context import QualityAwarePatternContext
    from habitat_evolution.adaptive_core.models import Pattern, Relationship
    from habitat_evolution.adaptive_core.persistence.interfaces.pattern_repository import PatternRepository
    from habitat_evolution.core.pattern import PatternState
