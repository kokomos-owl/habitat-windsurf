"""
Habitat Evolution System
A field-driven pattern evolution framework.

Core Components:
- Pattern Evolution
- Flow Dynamics
- Quality Analysis
- Memory Management
"""

from .core.pattern import FieldDrivenPatternManager, Pattern, FieldState
from .core.flow import GradientFlowController, FlowMetrics
from .core.quality import PatternQualityAnalyzer, QualityMetrics
from .core.memory import PatternMemoryManager

__version__ = '0.1.0'

__all__ = [
    'FieldDrivenPatternManager',
    'GradientFlowController',
    'PatternQualityAnalyzer',
    'PatternMemoryManager',
    'Pattern',
    'FieldState',
    'FlowMetrics',
    'QualityMetrics'
]
