"""
Import adapter for quality transitions.

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
    from src.habitat_evolution.core.pattern import PatternState as SrcPatternState
    
    # Re-export with aliases that match the non-src import style
    PatternState = SrcPatternState
    
except ImportError:
    # Fallback to non-src imports if src imports fail
    from habitat_evolution.core.pattern import PatternState
