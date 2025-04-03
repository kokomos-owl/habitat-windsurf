"""
Import adapter for context-aware extraction.

This module provides adapter functions to bridge the gap between different import styles.
"""

import sys
from pathlib import Path

# Ensure src directory is in path
src_dir = str(Path(__file__).parent.parent.parent.parent.parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import the modules we need with the src. prefix
try:
    from src.habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics as SrcTonicHarmonicMetrics
    from src.habitat_evolution.core.pattern import PatternState as SrcPatternState
    from src.habitat_evolution.core.pattern import SignalMetrics as SrcSignalMetrics
    from src.habitat_evolution.core.pattern import FlowMetrics as SrcFlowMetrics
    
    # Re-export with aliases that match the non-src import style
    TonicHarmonicMetrics = SrcTonicHarmonicMetrics
    PatternState = SrcPatternState
    SignalMetrics = SrcSignalMetrics
    FlowMetrics = SrcFlowMetrics
    
except ImportError:
    # Fallback to non-src imports if src imports fail
    from habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics
    from habitat_evolution.core.pattern import PatternState, SignalMetrics, FlowMetrics
