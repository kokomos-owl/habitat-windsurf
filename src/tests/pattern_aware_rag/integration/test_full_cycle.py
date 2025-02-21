"""
Full Cycle Integration Tests for Pattern-Aware RAG.

These tests verify the complete state cycle and integration
with external systems.
"""
import asyncio
import pytest
from habitat_evolution.pattern_aware_rag.core.pattern_processor import PatternProcessor
from habitat_evolution.pattern_aware_rag.core.coherence_interface import CoherenceInterface
from habitat_evolution.pattern_aware_rag.state.test_states import GraphStateSnapshot
from habitat_evolution.pattern_aware_rag.bridges.adaptive_state_bridge import AdaptiveStateBridge
from habitat_evolution.pattern_aware_rag.services.neo4j_service import Neo4jStateStore
from habitat_evolution.pattern_aware_rag.services.mongo_service import MongoStateStore
from habitat_evolution.pattern_aware_rag.learning.learning_control import LearningWindow
from habitat_evolution.pattern_aware_rag.learning.window_manager import LearningWindowManager
from habitat_evolution.pattern_aware_rag.core.pattern_aware_rag import PatternAwareRAG
from habitat_evolution.pattern_aware_rag.llm.prompt_engine import DynamicPromptEngine
from habitat_evolution.pattern_aware_rag.llm.response_analyzer import LLMResponseAnalyzer
from habitat_evolution.pattern_aware_rag.llm.pattern_extractor import ResponsePatternExtractor

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

@pytest.fixture
def window_manager():
    """Initialize learning window manager."""
    return LearningWindowManager()

@pytest.fixture
def pattern_aware_rag():
    """Initialize pattern-aware RAG."""
    return PatternAwareRAG()

@pytest.fixture
def prompt_engine():
    """Initialize dynamic prompt engine."""
    return DynamicPromptEngine()

@pytest.fixture
def response_analyzer():
    """Initialize LLM response analyzer."""
    return LLMResponseAnalyzer()

@pytest.fixture
def pattern_extractor():
    """Initialize response pattern extractor."""
    return ResponsePatternExtractor()

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

    async def test_learning_window_integration(self, pattern_processor, window_manager,
                                           coherence_interface, sample_document):
        """Test learning window integration with pattern processing."""
        # 1. Extract and Process Pattern
        pattern = await pattern_processor.extract_pattern(sample_document)
        adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
        
        # 2. Learning Window Processing
        window = window_manager.current_window
        assert window.state == "CLOSED"  # Initial state
        
        # 3. Process Through Window
        await window_manager.process_pattern(pattern)
        assert window.state in ["OPENING", "OPEN"]  # State progression
        
        # 4. Verify Pattern Processing
        processed_pattern = await window_manager.get_processed_pattern(pattern.id)
        assert processed_pattern is not None
        assert processed_pattern.stability_score > 0
        
    async def test_pattern_aware_rag_cycle(self, pattern_processor, pattern_aware_rag,
                                       window_manager, sample_document):
        """Test complete pattern-aware RAG cycle with learning windows."""
        # 1. Document Processing
        pattern = await pattern_processor.extract_pattern(sample_document)
        
        # 2. RAG Processing
        result = await pattern_aware_rag.process_with_patterns(
            query="test query",
            context={"pattern": pattern}
        )
        assert result is not None
        
        # 3. Verify Window Integration
        window_state = window_manager.current_window.state
        assert window_state in ["CLOSED", "OPENING", "OPEN"]
        
        # 4. Check Pattern Evolution
        evolved_pattern = await pattern_aware_rag.get_evolved_pattern(pattern.id)
        assert evolved_pattern is not None
        assert evolved_pattern.version > pattern.version

    async def test_dynamic_prompt_formation(self, prompt_engine, pattern_processor,
                                        window_manager, sample_document):
        """Test dynamic prompt formation with pattern context."""
        # 1. Extract Initial Pattern
        pattern = await pattern_processor.extract_pattern(sample_document)
        
        # 2. Generate Dynamic Prompt
        prompt = await prompt_engine.generate_prompt(
            query="test query",
            context_pattern=pattern,
            window_state=window_manager.current_window.state
        )
        assert prompt is not None
        assert "test query" in prompt.content
        assert prompt.pattern_references is not None
        
        # 3. Verify Pattern Integration
        assert prompt.has_pattern_context(pattern.id)
        assert prompt.coherence_score > 0.0
        
    async def test_llm_query_cycle(self, pattern_aware_rag, prompt_engine,
                                response_analyzer, sample_document):
        """Test complete LLM query cycle with pattern-aware prompting."""
        # 1. Generate Query Context
        context = await pattern_aware_rag.prepare_query_context(sample_document)
        
        # 2. Form Dynamic Prompt
        prompt = await prompt_engine.generate_prompt(
            query="test query",
            context=context
        )
        
        # 3. Execute LLM Query
        response = await pattern_aware_rag.query_llm(prompt)
        assert response is not None
        
        # 4. Analyze Response
        analysis = await response_analyzer.analyze_response(response)
        assert analysis.coherence_score > 0.0
        assert analysis.pattern_alignment_score > 0.0
        
    async def test_response_pattern_extraction(self, pattern_processor,
                                           response_analyzer, pattern_extractor,
                                           sample_document):
        """Test pattern extraction from LLM responses."""
        # 1. Generate Sample Response
        pattern = await pattern_processor.extract_pattern(sample_document)
        response = await response_analyzer.generate_test_response(pattern)
        
        # 2. Extract Response Patterns
        extracted_patterns = await pattern_extractor.extract_patterns(response)
        assert len(extracted_patterns) > 0
        
        # 3. Verify Pattern Quality
        for pattern in extracted_patterns:
            assert pattern.coherence_score > 0.0
            assert pattern.stability_score > 0.0
            
        # 4. Validate Pattern Relations
        relations = await pattern_extractor.extract_pattern_relations(extracted_patterns)
        assert len(relations) > 0
        assert all(r.confidence_score > 0.0 for r in relations)
