"""
Pattern-Aware RAG Testing Suite.

This module provides comprehensive testing for the Pattern-Aware RAG system,
focusing on its role as a coherence interface that manages state alignment
through controlled agreement formation.

Key Test Categories:
1. Graph State Foundation
2. Learning Window Management
3. State Agreement Formation
4. Pattern Evolution
"""

import pytest
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.habitat_evolution.pattern_aware_rag.pattern_aware_rag import PatternAwareRAG
from src.habitat_evolution.core.models.learning_window import LearningWindowState
from src.habitat_evolution.core.models.pattern import Pattern, PatternMetrics
from src.habitat_evolution.core.services.graph_service import GraphService
from src.habitat_evolution.core.services.pattern_evolution import PatternEvolutionService

# Test Data Models
@dataclass
class TestGraphState:
    """Test model for graph state with essential metrics."""
    state_id: str
    timestamp: datetime
    coherence_score: float
    pattern_density: float
    relationship_strengths: Dict[str, float]
    
    @property
    def is_valid(self) -> bool:
        """Validate state metrics are within bounds."""
        return (
            0.0 <= self.coherence_score <= 1.0 and
            0.0 <= self.pattern_density <= 1.0 and
            all(0.0 <= v <= 1.0 for v in self.relationship_strengths.values())
        )

@dataclass
class TestStateAgreement:
    """Test model for state agreement metrics."""
    external_coherence: float
    internal_coherence: float
    agreement_score: float
    stability_index: float
    timestamp: datetime
    
    @property
    def is_stable(self) -> bool:
        """Check if agreement state is stable."""
        return (
            self.agreement_score >= 0.7 and
            self.stability_index >= 0.8
        )

# Fixtures
@pytest.fixture
async def graph_service():
    """Provide configured graph service for testing."""
    class TestGraphService:
        async def get_current_state(self) -> TestGraphState:
            return TestGraphState(
                state_id="test_state_001",
                timestamp=datetime.now(),
                coherence_score=0.85,
                pattern_density=0.75,
                relationship_strengths={"p1->p2": 0.9, "p2->p3": 0.8}
            )
        
        async def update_state(self, new_state: TestGraphState) -> bool:
            return new_state.is_valid
    
    return TestGraphService()

@pytest.fixture
async def pattern_evolution_service():
    """Provide configured pattern evolution service for testing."""
    class TestPatternEvolution:
        async def extract_pattern(self, content: str) -> Pattern:
            return Pattern(
                id="test_pattern_001",
                content=content,
                metrics=PatternMetrics(coherence=0.9),
                relationships=["test_pattern_002"]
            )
    
    return TestPatternEvolution()

@pytest.fixture
async def pattern_aware_rag(graph_service, pattern_evolution_service):
    """Provide configured PatternAwareRAG instance."""
    return PatternAwareRAG(
        graph_service=graph_service,
        pattern_evolution=pattern_evolution_service,
        initial_window_state=LearningWindowState.CLOSED
    )

# Test Suites
class TestGraphStateFoundation:
    """Test suite for graph state foundation."""
    
    async def test_initial_state_loading(self, pattern_aware_rag):
        """Verify initial graph state is properly loaded."""
        state = await pattern_aware_rag.get_current_state()
        assert state.state_id == "test_state_001"
        assert state.coherence_score >= 0.0
        assert state.pattern_density >= 0.0
    
    async def test_prompt_formation(self, pattern_aware_rag):
        """Test prompt formation from graph state."""
        state = await pattern_aware_rag.get_current_state()
        prompt = await pattern_aware_rag.form_prompt_from_state(state)
        assert state.state_id in prompt
        assert "coherence" in prompt.lower()

class TestLearningWindow:
    """Test suite for learning window management."""
    
    async def test_initial_window_state(self, pattern_aware_rag):
        """Verify initial window state is CLOSED."""
        assert pattern_aware_rag.current_window_state == LearningWindowState.CLOSED
    
    async def test_back_pressure_control(self, pattern_aware_rag):
        """Test back pressure control mechanisms."""
        initial_rate = await pattern_aware_rag.get_state_change_rate()
        await pattern_aware_rag.apply_back_pressure()
        final_rate = await pattern_aware_rag.get_state_change_rate()
        assert final_rate < initial_rate

class TestStateAgreement:
    """Test suite for state agreement formation."""
    
    async def test_agreement_formation(self, pattern_aware_rag):
        """Test the agreement formation process."""
        initial = await pattern_aware_rag.measure_state_agreement()
        await pattern_aware_rag.form_agreement()
        final = await pattern_aware_rag.measure_state_agreement()
        assert final.agreement_score > initial.agreement_score
    
    async def test_stability_maintenance(self, pattern_aware_rag):
        """Test stability is maintained during agreement formation."""
        agreement = await pattern_aware_rag.measure_state_agreement()
        assert agreement.stability_index >= 0.0
        assert agreement.is_stable

# Utility Functions
def assert_valid_transition(from_state: LearningWindowState, 
                          to_state: LearningWindowState) -> bool:
    """Validate state transition is allowed."""
    valid_transitions = {
        LearningWindowState.CLOSED: [LearningWindowState.OPENING],
        LearningWindowState.OPENING: [LearningWindowState.OPEN],
        LearningWindowState.OPEN: [LearningWindowState.CLOSED]
    }
    return to_state in valid_transitions[from_state]
