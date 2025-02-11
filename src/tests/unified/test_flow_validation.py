"""
Validates unified test fixtures against actual test patterns.
"""

import pytest
from datetime import datetime
from .flow_fixtures import (
    UnifiedFlowState,
    UnifiedPatternContext,
    UnifiedLearningWindow
)

@pytest.mark.asyncio
class TestFlowValidation:
    """Validates unified fixtures against actual test patterns."""
    
    async def test_flow_state_validation(self, unified_flow_state):
        """Validate flow state matches habitat_poc patterns."""
        # From test_flow_emergence.py
        assert unified_flow_state.energy > 0.6, "Energy threshold"
        assert unified_flow_state.velocity > 0, "Forward movement"
        assert len(unified_flow_state.patterns) >= 2, "Multiple patterns"
        
        # From test_pattern_evolution.py
        assert unified_flow_state.coherence > 0.75, "Coherence threshold"
        assert unified_flow_state.propensity > 0.7, "Evolution potential"
    
    async def test_pattern_context_validation(
        self,
        unified_pattern_context,
        mock_pattern_observer
    ):
        """Validate pattern context matches habitat_poc patterns."""
        # Observe patterns
        result = mock_pattern_observer.observe_patterns(
            unified_pattern_context.flow_state
        )
        
        # From test_pattern_evolution.py
        assert len(result["patterns"]) >= 2, "Pattern emergence"
        assert result["metrics"]["coherence"] > 0.75, "Pattern coherence"
        
        # From test_flow_emergence.py
        assert len(unified_pattern_context.query_patterns) > 0, "Query patterns"
        assert len(unified_pattern_context.retrieval_patterns) > 0, "Retrieval patterns"
    
    async def test_learning_window_validation(
        self,
        unified_learning_window,
        mock_density_analyzer
    ):
        """Validate learning window matches habitat_poc patterns."""
        # Analyze density
        result = mock_density_analyzer.analyze_density(unified_learning_window)
        
        # From test_learning_windows.py
        assert result["local_density"] > 0.8, "Local density"
        assert result["global_density"] > 0.7, "Global density"
        assert result["cross_path_count"] > 0, "Cross-domain paths"
        
        # From window_data structure
        assert unified_learning_window.window_data["score"] > 0.9, "Window score"
        assert unified_learning_window.window_data["potential"] > 0.9, "Window potential"
    
    async def test_bidirectional_validation(
        self,
        unified_flow_state,
        unified_pattern_context,
        unified_learning_window,
        mock_pattern_observer,
        mock_density_analyzer
    ):
        """Validate bidirectional flow matches habitat_poc patterns."""
        # Initial observation
        obs1 = mock_pattern_observer.observe_patterns(unified_flow_state)
        density1 = mock_density_analyzer.analyze_density(unified_learning_window)
        
        # Simulate evolution
        unified_flow_state.energy += 0.1
        unified_flow_state.coherence += 0.05
        unified_flow_state.patterns.add("evolution")
        
        # Second observation
        obs2 = mock_pattern_observer.observe_patterns(unified_flow_state)
        
        # From test_bidirectional_evolution
        assert obs2["metrics"]["energy"] > obs1["metrics"]["energy"], "Energy increase"
        assert len(obs2["patterns"]) > len(obs1["patterns"]), "Pattern growth"
        assert density1["local_density"] > 0.8, "Maintained density"
