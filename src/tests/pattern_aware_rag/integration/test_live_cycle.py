"""
Live Integration Test for Pattern-Aware RAG with Neo4j and Claude.

This test validates the complete cycle:
1. Neo4j → Pattern-Aware RAG
2. Dynamic Prompt Formation
3. Claude Integration
4. Pattern Extraction
5. Graph Service Integration
"""
import asyncio
import pytest
from datetime import datetime
from typing import Dict, List, Optional

from habitat_evolution.pattern_aware_rag.core.pattern_processor import PatternProcessor
from habitat_evolution.pattern_aware_rag.core.pattern_aware_rag import PatternAwareRAG
from habitat_evolution.pattern_aware_rag.services.neo4j_service import Neo4jStateStore
from habitat_evolution.pattern_aware_rag.learning.learning_control import LearningWindow
from habitat_evolution.pattern_aware_rag.llm.prompt_engine import DynamicPromptEngine
from habitat_evolution.pattern_aware_rag.llm.claude_interface import ClaudeInterface
from habitat_evolution.pattern_aware_rag.state.graph_service import GraphService

class TestLiveCycle:
    """Test the live cycle with real services."""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Initialize test components."""
        self.neo4j = Neo4jStateStore()
        self.rag = PatternAwareRAG()
        self.prompt_engine = DynamicPromptEngine()
        self.claude = ClaudeInterface()
        self.graph_service = GraphService()
        
        # Create test window
        self.window = LearningWindow(
            start_time=datetime.now(),
            end_time=datetime.now(),
            stability_threshold=0.8,
            coherence_threshold=0.7,
            max_changes_per_window=15
        )
        
        yield
        await self.cleanup()
        
    async def cleanup(self):
        """Cleanup test data."""
        # Remove test patterns from Neo4j
        pass

    async def test_pattern_retrieval(self):
        """Test pattern retrieval from Neo4j with relationship reconstruction."""
        # 1. Setup test query and context
        query = "test domain knowledge"
        context = {
            "domain": "test",
            "relevance_threshold": 0.7,
            "relationship_depth": 2
        }
        
        # 2. Retrieve patterns with relationships
        patterns = await self.neo4j.get_relevant_patterns(
            query=query,
            context=context,
            limit=5
        )
        
        # 3. Validate basic pattern structure
        assert len(patterns) > 0
        for pattern in patterns:
            assert pattern.id is not None
            assert pattern.coherence_score >= 0.7
            assert pattern.stability_score >= 0.7
            
        # 4. Verify relationship reconstruction
        for pattern in patterns:
            relationships = await self.neo4j.get_pattern_relationships(pattern.id)
            assert len(relationships) > 0
            for rel in relationships:
                assert rel.source_id is not None
                assert rel.target_id is not None
                assert rel.relationship_type is not None
                assert rel.confidence_score >= 0.7
                
        # 5. Test relevance scoring
        relevance_scores = await self.neo4j.get_pattern_relevance(
            patterns=patterns,
            query=query
        )
        assert len(relevance_scores) == len(patterns)
        assert all(score >= 0.5 for score in relevance_scores.values())
        
        # 6. Validate pattern integrity
        for pattern in patterns:
            validation = await self.neo4j.validate_pattern_integrity(pattern)
            assert validation.is_valid
            assert validation.coherence_valid
            assert validation.relationships_valid
            assert len(validation.errors) == 0

    async def test_live_cycle(self):
        """Test complete live cycle with Claude integration."""
        # 1. Retrieve Existing Patterns from Neo4j
        patterns = await self.neo4j.get_relevant_patterns(
            query="test domain knowledge",
            context={"domain": "test"},
            limit=5
        )
        assert len(patterns) > 0
        
        # 2. Form Dynamic Prompt
        prompt = await self.prompt_engine.generate_prompt(
            query="How does this system handle pattern evolution?",
            context_patterns=patterns,
            window_state=self.window.state
        )
        assert "pattern evolution" in prompt.content
        assert len(prompt.pattern_references) > 0
        
        # 3. Query Claude
        claude_response = await self.claude.query(
            prompt=prompt.content,
            context={
                "patterns": prompt.pattern_references,
                "window_state": self.window.state
            }
        )
        assert claude_response is not None
        
        # 4. Process Response through RAG
        rag_result = await self.rag.process_response(
            response=claude_response,
            original_patterns=patterns,
            window=self.window
        )
        assert rag_result.coherence_score > 0.7
        assert len(rag_result.new_patterns) > 0
        
        # 5. Extract New Patterns
        new_patterns = await self.rag.extract_patterns(
            rag_result,
            context={
                "source": "claude",
                "query_time": datetime.now().isoformat()
            }
        )
        assert all(p.coherence_score > 0.7 for p in new_patterns)
        
        # 6. Store New Patterns in Neo4j
        for pattern in new_patterns:
            # Convert to graph format
            graph_data = await self.graph_service.pattern_to_graph(
                pattern,
                context={"source_patterns": patterns}
            )
            
            # Store in Neo4j
            node_id = await self.neo4j.store_pattern(graph_data)
            assert node_id is not None
            
            # Verify Storage
            stored = await self.neo4j.get_pattern(node_id)
            assert stored.id == pattern.id
            assert stored.coherence_score >= pattern.coherence_score
            
        # 7. Verify Pattern Evolution
        evolved_patterns = await self.neo4j.get_pattern_evolution(
            [p.id for p in patterns]
        )
        assert len(evolved_patterns) > 0
        
        # 8. Check System Stability
        stability = await self.rag.check_system_stability(
            original_patterns=patterns,
            new_patterns=new_patterns,
            window=self.window
        )
        assert stability.score > 0.8
        assert stability.is_stable is True

    async def test_pattern_integration(self):
        """Test how new patterns integrate with existing knowledge."""
        # 1. Get Existing Pattern Network
        network = await self.neo4j.get_pattern_network(
            center_pattern_id="test_pattern",
            depth=2
        )
        assert len(network.nodes) > 0
        
        # 2. Generate Related Query
        query = await self.prompt_engine.generate_network_query(
            network=network,
            focus="pattern integration"
        )
        assert query is not None
        
        # 3. Process Through Claude
        response = await self.claude.query(
            prompt=query,
            context={"network": network}
        )
        assert response is not None
        
        # 4. Extract Integration Patterns
        integration_patterns = await self.rag.extract_integration_patterns(
            response=response,
            existing_network=network
        )
        assert len(integration_patterns) > 0
        
        # 5. Verify Network Enhancement
        enhanced_network = await self.graph_service.enhance_network(
            network=network,
            new_patterns=integration_patterns
        )
        assert len(enhanced_network.nodes) > len(network.nodes)
        assert enhanced_network.coherence_score > network.coherence_score
        
        # 6. Store Enhanced Network
        success = await self.neo4j.store_pattern_network(enhanced_network)
        assert success is True
