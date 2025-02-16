"""
Full Cycle Integration Tests for Pattern-Aware RAG.

These tests verify the complete state cycle and integration
with external systems.
"""
import pytest
from habitat_evolution.pattern_aware_rag.core.pattern_processor import PatternProcessor
from habitat_evolution.pattern_aware_rag.core.coherence_interface import CoherenceInterface
from habitat_evolution.pattern_aware_rag.state.graph_state import GraphState
from habitat_evolution.pattern_aware_rag.bridges.adaptive_state_bridge import AdaptiveStateBridge
from habitat_evolution.pattern_aware_rag.services.neo4j_service import Neo4jStateStore
from habitat_evolution.pattern_aware_rag.services.mongo_service import MongoStateStore

@pytest.fixture
def state_stores():
    """Initialize state stores."""
    return {
        "neo4j": Neo4jStateStore(),
        "mongo": MongoStateStore()
    }

@pytest.fixture
def adaptive_bridge():
    """Initialize adaptive state bridge."""
    return AdaptiveStateBridge()

class TestFullCycle:
    """Test the complete state cycle."""
    
    async def test_full_state_cycle(self, pattern_processor, coherence_interface, 
                                  state_stores, adaptive_bridge, sample_document):
        """Test complete state cycle from document to evolution."""
        # 1. Sequential Foundation
        pattern = await pattern_processor.extract_pattern(sample_document)
        adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
        initial_state = await pattern_processor.prepare_graph_state(pattern, adaptive_id)
        
        # 2. Coherence Interface
        alignment = await coherence_interface.align_state(initial_state)
        assert alignment.coherence_score > 0.0
        
        # 3. State Storage
        neo4j_id = await state_stores["neo4j"].store_graph_state(initial_state)
        mongo_id = await state_stores["mongo"].store_state_history(initial_state)
        assert neo4j_id is not None
        assert mongo_id is not None
        
        # 4. Evolution
        evolved_state = await adaptive_bridge.evolve_state(initial_state)
        assert evolved_state.version > initial_state.version
        
        # 5. Verify Evolution
        stored_state = await state_stores["neo4j"].get_graph_state(neo4j_id)
        history = await state_stores["mongo"].get_state_evolution(mongo_id)
        
        assert stored_state.id == evolved_state.id
        assert len(history) > 0
    
    async def test_concurrent_operations(self, pattern_processor, coherence_interface,
                                      state_stores, adaptive_bridge, sample_document):
        """Test concurrent operations after sequential foundation."""
        # 1. Establish Foundation
        pattern = await pattern_processor.extract_pattern(sample_document)
        adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
        initial_state = await pattern_processor.prepare_graph_state(pattern, adaptive_id)
        
        # 2. Verify Concurrent Operations
        tasks = [
            adaptive_bridge.enhance_pattern(initial_state),
            state_stores["neo4j"].store_graph_state(initial_state),
            state_stores["mongo"].store_state_history(initial_state)
        ]
        
        results = await asyncio.gather(*tasks)
        assert all(result is not None for result in results)
