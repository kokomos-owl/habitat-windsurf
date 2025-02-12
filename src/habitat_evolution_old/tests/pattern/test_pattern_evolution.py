"""Test suite for pattern evolution with coherence-based approach.

This test suite validates the pattern evolution system's ability to:
1. Track pattern states (EMERGING -> STABLE -> EVOLVING -> DEGRADING)
2. Detect emergence types (POTENTIAL -> NATURAL -> GUIDED)
3. Calculate accurate evolution scores
4. Maintain field conditions

The tests use a mock pattern core to simulate pattern behavior and verify
that the evolution manager correctly interprets various coherence metrics
to determine pattern states and emergence types.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime

from habitat_evolution.core.pattern.evolution import (
    PatternEvolutionManager,
    EvolutionState,
    PatternType,
    DocumentStructureType,
    EmergenceType,
    EvolutionMetrics
)

class MockPatternCore:
    """Mock pattern core for testing pattern evolution.
    
    This mock simulates the basic functionality of a pattern core,
    allowing tests to verify pattern evolution behavior without
    requiring a full pattern core implementation.
    
    Attributes:
        patterns: Dictionary storing pattern data
    """
    def __init__(self):
        self.patterns = {}

    def get_pattern(self, pattern_id):
        """Retrieve pattern data by ID."""
        return self.patterns.get(pattern_id)

    def update_pattern(self, pattern_id, pattern_data):
        """Update pattern data for given ID."""
        self.patterns[pattern_id] = pattern_data

@pytest.fixture
def evolution_manager():
    """Create pattern evolution manager for testing.
    
    Returns:
        PatternEvolutionManager: Configured with mock pattern core
    """
    pattern_core = MockPatternCore()
    return PatternEvolutionManager(pattern_core=pattern_core)

@pytest.fixture
def sample_evidence():
    """Create sample evidence for testing pattern evolution.
    
    The evidence includes high-quality metrics that should result
    in a STABLE pattern state when properly evaluated:
    - Network structure with high confidence
    - Relational pattern with strong confidence
    - High coherence across all dimensions
    - Strong stability and confidence scores
    
    Returns:
        dict: Sample evidence with various coherence metrics
    """
    return {
        'structure_type': DocumentStructureType.NETWORK,
        'structure_confidence': 0.85,
        'pattern_type': PatternType.RELATIONAL,
        'pattern_confidence': 0.82,
        'temporal_coherence': 0.78,
        'system_coherence': 0.75,
        'knowledge_coherence': 0.80,
        'stability': 0.85,
        'confidence': 0.88,
        'structure_alignment': 0.82,
        'meaning_alignment': 0.79
    }

class TestPatternEvolution:
    """Test pattern evolution with coherence gardening approach."""
    
    @pytest.mark.asyncio
    async def test_natural_emergence(self, evolution_manager, sample_evidence):
        """Test natural pattern emergence without interference.
        
        This test verifies that:
        1. Pattern reaches STABLE state with high-quality evidence
        2. Evolution score exceeds 0.8 threshold
        3. Pattern maintains correct structure and type
        4. Coherence remains above 0.75
        """
        pattern_id = "test_pattern"
        result = await evolution_manager.observe_evolution(pattern_id, sample_evidence)
        
        # Debug metrics
        metrics = result['metrics']
        print(f"\nEvolution Score: {metrics.evolution_score()}")
        print(f"Structure Score: {metrics.structure_confidence * 0.85}")
        print(f"Pattern Score: {metrics.pattern_confidence * 0.9}")
        print(f"Coherence Scores: {metrics.temporal_coherence}, {metrics.system_coherence}, {metrics.knowledge_coherence}")
        print(f"Legacy Scores: {metrics.coherence}, {metrics.stability}, {metrics.confidence}, {metrics.structure_alignment}, {metrics.meaning_alignment}")
        
        assert result['pattern_id'] == pattern_id
        assert result['state'] == EvolutionState.STABLE
        assert isinstance(result['timestamp'], str)
        
        # Verify metrics
        metrics = result['metrics']
        assert metrics.structure_type == DocumentStructureType.NETWORK
        assert metrics.pattern_type == PatternType.RELATIONAL
        assert metrics.coherence > 0.75
        assert metrics.evolution_score() > 0.8
        
    def test_coherence_tending(self, evolution_manager, sample_evidence):
        """Test how manager tends to coherence conditions.
        
        This test verifies that:
        1. New patterns start with POTENTIAL emergence
        2. Patterns transition to NATURAL emergence when stable
        """
        pattern_id = "test_pattern"
        
        # Before evolution
        emergence = evolution_manager.sense_emergence(pattern_id)
        assert emergence == EmergenceType.POTENTIAL
        
        # After evolution
        asyncio.run(evolution_manager.observe_evolution(pattern_id, sample_evidence))
        emergence = evolution_manager.sense_emergence(pattern_id)
        assert emergence == EmergenceType.NATURAL
        
    def test_field_conditions(self, evolution_manager, sample_evidence):
        """Test maintenance of field conditions.
        
        This test verifies that:
        1. Field conditions maintain high stability (>= 0.8)
        2. Coherence remains strong (>= 0.7)
        3. Confidence stays high (>= 0.8)
        """
        pattern_id = "test_pattern"
        result = asyncio.run(evolution_manager.observe_evolution(pattern_id, sample_evidence))
        
        metrics = result['metrics']
        assert metrics.stability >= 0.8
        assert metrics.coherence >= 0.7
        assert metrics.confidence >= 0.8
        
    def test_pattern_enhancement(self, evolution_manager, sample_evidence):
        """Test pattern enhancement under right conditions.
        
        This test verifies that:
        1. Evolution score improves with better metrics
        2. Pattern maintains stability during enhancement
        """
        pattern_id = "test_pattern"
        
        # First evolution
        result = asyncio.run(evolution_manager.observe_evolution(pattern_id, sample_evidence))
        first_score = result['metrics'].evolution_score()
        
        # Second evolution with higher metrics
        enhanced_evidence = sample_evidence.copy()
        enhanced_evidence.update({
            'stability': 0.95,
            'confidence': 0.92,
            'structure_alignment': 0.90,
            'meaning_alignment': 0.88
        })
        
        result = asyncio.run(evolution_manager.observe_evolution(pattern_id, enhanced_evidence))
        second_score = result['metrics'].evolution_score()
        
        assert second_score > first_score
