"""
Sequential Foundation Tests for Pattern-Aware RAG.

These tests verify the critical sequential foundation required before
any concurrent operations can begin.
"""
import pytest
from habitat_evolution.pattern_aware_rag.core.pattern_processor import PatternProcessor
from habitat_evolution.pattern_aware_rag.state.test_states import GraphStateSnapshot
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

@pytest.fixture
def pattern_processor():
    """Initialize pattern processor for testing."""
    return PatternProcessor()

@pytest.fixture
def sample_document():
    """Sample document for testing pattern extraction."""
    return {
        "content": "Test pattern content",
        "metadata": {"source": "test", "timestamp": "2025-02-16T12:42:45-05:00"}
    }

class TestSequentialFoundation:
    """Test the sequential foundation requirements."""
    
    # Initial State Loading Tests
    async def test_initial_state_loading(self, pattern_processor):
        """Test initial state loading and validation."""
        # Test empty state initialization
        empty_state = GraphStateSnapshot(
            id="empty_state",
            nodes=[],
            relations=[],
            patterns=[],
            timestamp=datetime.now(),
            version=1
        )
        assert not empty_state.is_graph_ready()
        
        # Test valid state initialization
        valid_node = ConceptNode(
            id="test_node",
            name="Test Node",
            attributes={"type": "test"}
        )
        valid_pattern = PatternState(
            id="test_pattern",
            content="Test content",
            metadata={"source": "test"},
            timestamp=datetime.now()
        )
        valid_state = GraphStateSnapshot(
            id="valid_state",
            nodes=[valid_node],
            relations=[],
            patterns=[valid_pattern],
            timestamp=datetime.now(),
            version=1
        )
        assert valid_state.is_graph_ready()
    
    async def test_state_persistence(self, pattern_processor, sample_document):
        """Test state persistence through transformations."""
        # Create initial state
        pattern = await pattern_processor.extract_pattern(sample_document)
        adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
        initial_state = await pattern_processor.prepare_graph_state(pattern, adaptive_id)
        
        # Verify state persistence
        assert initial_state.id is not None
        assert initial_state.version == 1
        assert initial_state.timestamp is not None
        
        # Verify pattern persistence
        persisted_pattern = initial_state.patterns[0]
        assert persisted_pattern.id == pattern.id
        assert persisted_pattern.content == pattern.content
        
    async def test_invalid_state_handling(self, pattern_processor):
        """Test error handling for invalid states."""
        # Test with invalid node
        with pytest.raises(ValueError):
            invalid_node = ConceptNode(
                id="",  # Invalid empty ID
                name="Invalid Node",
                attributes={}
            )
            GraphStateSnapshot(
                id="invalid_state",
                nodes=[invalid_node],
                relations=[],
                patterns=[],
                timestamp=datetime.now(),
                version=1
            )
        
        # Test with invalid relation
        with pytest.raises(ValueError):
            invalid_relation = ConceptRelation(
                source_id="source",
                target_id="target",
                relation_type="",  # Invalid empty type
                weight=-1.0  # Invalid negative weight
            )
            GraphStateSnapshot(
                id="invalid_state",
                nodes=[],
                relations=[invalid_relation],
                patterns=[],
                timestamp=datetime.now(),
                version=1
            )
    
    # Prompt Formation Tests
    async def test_prompt_template_validation(self, pattern_processor, sample_document):
        """Test prompt template validation."""
        pattern = await pattern_processor.extract_pattern(sample_document)
        adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
        state = await pattern_processor.prepare_graph_state(pattern, adaptive_id)
        
        # Test prompt formation with pattern context
        prompt = await pattern_processor.form_prompt(state, context={"type": "pattern"})
        assert "pattern" in prompt.lower()
        assert pattern.content in prompt
        
        # Test prompt formation with relationship context
        prompt = await pattern_processor.form_prompt(state, context={"type": "relationship"})
        assert "relationship" in prompt.lower()
    
    async def test_context_integration(self, pattern_processor, sample_document):
        """Test context integration in prompt formation."""
        pattern = await pattern_processor.extract_pattern(sample_document)
        adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
        state = await pattern_processor.prepare_graph_state(pattern, adaptive_id)
        
        # Test with temporal context
        temporal_context = {"timestamp": datetime.now().isoformat()}
        prompt = await pattern_processor.form_prompt(state, context=temporal_context)
        assert temporal_context["timestamp"] in prompt
        
        # Test with spatial context
        spatial_context = {"location": "test_location"}
        prompt = await pattern_processor.form_prompt(state, context=spatial_context)
        assert spatial_context["location"] in prompt
    
    # State Agreement Tests
    async def test_consensus_mechanism(self, pattern_processor, sample_document):
        """Test consensus mechanisms for state agreement."""
        pattern = await pattern_processor.extract_pattern(sample_document)
        adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
        state = await pattern_processor.prepare_graph_state(pattern, adaptive_id)
        
        # Test consensus with single pattern
        consensus = await pattern_processor.reach_consensus(state)
        assert consensus.achieved
        assert consensus.confidence > 0.5
        
        # Test consensus with multiple patterns
        second_pattern = PatternState(
            id="test_pattern_2",
            content="Related content",
            metadata={"source": "test"},
            timestamp=datetime.now()
        )
        state.patterns.append(second_pattern)
        consensus = await pattern_processor.reach_consensus(state)
        assert consensus.achieved
        assert consensus.confidence > 0.5
    
    async def test_state_synchronization(self, pattern_processor, sample_document):
        """Test state synchronization during agreement process."""
        pattern = await pattern_processor.extract_pattern(sample_document)
        adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
        state = await pattern_processor.prepare_graph_state(pattern, adaptive_id)
        
        # Create a modified state
        modified_state = GraphStateSnapshot(
            id=state.id,
            nodes=state.nodes,
            relations=state.relations,
            patterns=state.patterns,
            timestamp=datetime.now(),
            version=state.version + 1
        )
        
        # Test state synchronization
        sync_result = await pattern_processor.synchronize_states(state, modified_state)
        assert sync_result.success
        assert sync_result.final_state.version == max(state.version, modified_state.version)
    
    async def test_conflict_resolution(self, pattern_processor, sample_document):
        """Test conflict resolution in state agreement."""
        # Create two conflicting states
        pattern1 = await pattern_processor.extract_pattern(sample_document)
        adaptive_id1 = await pattern_processor.assign_adaptive_id(pattern1)
        state1 = await pattern_processor.prepare_graph_state(pattern1, adaptive_id1)
        
        pattern2 = PatternState(
            id="conflicting_pattern",
            content="Conflicting content",
            metadata={"source": "test"},
            timestamp=datetime.now()
        )
        adaptive_id2 = await pattern_processor.assign_adaptive_id(pattern2)
        state2 = await pattern_processor.prepare_graph_state(pattern2, adaptive_id2)
        
        # Test conflict resolution
        resolution = await pattern_processor.resolve_conflicts([state1, state2])
        assert resolution.success
        assert len(resolution.resolved_state.patterns) >= 1
    
    async def test_pattern_extraction(self, pattern_processor, sample_document):
        """Test pattern extraction from document."""
        pattern = await pattern_processor.extract_pattern(sample_document)
        assert pattern is not None
        assert pattern.content == sample_document["content"]
        assert pattern.metadata["source"] == "test"
    
    async def test_adaptive_id_assignment(self, pattern_processor, sample_document):
        """Test Adaptive ID assignment to pattern."""
        pattern = await pattern_processor.extract_pattern(sample_document)
        adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
        assert isinstance(adaptive_id, AdaptiveID)
        assert adaptive_id.base_concept is not None
    
    async def test_graph_ready_state(self, pattern_processor, sample_document):
        """Test pattern reaches graph-ready state."""
        pattern = await pattern_processor.extract_pattern(sample_document)
        adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
        graph_state = await pattern_processor.prepare_graph_state(pattern, adaptive_id)
        assert isinstance(graph_state, GraphStateSnapshot)
        assert graph_state.is_graph_ready()
    
    async def test_sequential_dependency(self, pattern_processor, sample_document):
        """Test that operations must occur in correct sequence."""
        # Should fail if trying to prepare graph state before ID assignment
        pattern = await pattern_processor.extract_pattern(sample_document)
        with pytest.raises(ValueError):
            await pattern_processor.prepare_graph_state(pattern, None)
    
    async def test_provenance_tracking(self, pattern_processor, sample_document):
        """Test provenance is established and tracked."""
        pattern = await pattern_processor.extract_pattern(sample_document)
        adaptive_id = await pattern_processor.assign_adaptive_id(pattern)
        graph_state = await pattern_processor.prepare_graph_state(pattern, adaptive_id)
        
        provenance = graph_state.get_provenance()
        assert provenance["patterns"][0]["source"] == "test"
        assert provenance["timestamp"] is not None
        assert provenance["patterns"][0]["id"] == pattern.id
