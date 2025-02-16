"""
Test suite for LangChain Configuration.

Tests the LangChain integration components, ensuring proper state handling
and Claude-optimized prompting.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from habitat_evolution.pattern_aware_rag.langchain_config import (
    StatePromptConfig,
    ClaudeLangChainIntegration
)

@pytest.fixture
def temp_persist_dir():
    """Provide temporary directory for embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def config():
    """Provide test configuration."""
    return StatePromptConfig(
        min_coherence_score=0.3,
        min_relationship_strength=0.4,
        min_concept_confidence=0.6,
        max_concepts_per_prompt=15,
        max_relations_per_prompt=25,
        max_history_window=5
    )

@pytest.fixture
def sample_metrics():
    """Provide sample state metrics."""
    return {
        "coherence": 0.85,
        "relationship_strength": 0.75,
        "concept_confidence": 0.8,
        "temporal_stability": 0.9
    }

@pytest.fixture
def integration(config, temp_persist_dir):
    """Provide configured LangChain integration."""
    return ClaudeLangChainIntegration(
        config=config,
        persist_dir=temp_persist_dir
    )

class TestStatePromptConfig:
    """Test suite for state prompt configuration."""
    
    def test_metric_validation(self, config, sample_metrics):
        """Test validation of state metrics."""
        assert config.validate_state_metrics(sample_metrics)
        
        # Test below threshold
        invalid_metrics = sample_metrics.copy()
        invalid_metrics["coherence"] = 0.2
        assert not config.validate_state_metrics(invalid_metrics)
    
    def test_context_window_limits(self, config):
        """Test context window limitations."""
        assert config.max_concepts_per_prompt > 0
        assert config.max_relations_per_prompt > 0
        assert config.max_concepts_per_prompt < config.max_relations_per_prompt

class TestClaudeLangChainIntegration:
    """Test suite for Claude LangChain integration."""
    
    def test_embedding_setup(self, integration, temp_persist_dir):
        """Test embedding configuration."""
        # Verify embedding directory
        embed_dir = Path(temp_persist_dir)
        assert embed_dir.exists()
        assert embed_dir.is_dir()
        
        # Verify vector store
        assert integration.vector_store is not None
        assert integration.vector_store.collection_name == "state_embeddings"
    
    def test_prompt_templates(self, integration):
        """Test Claude-optimized prompt templates."""
        template = integration.state_template
        assert "System:" in template.template
        assert "Graph State Overview" in template.template
        assert "Concept Relationships" in template.template
        
        # Test required variables
        assert "graph_state" in template.input_variables
        assert "concept_relations" in template.input_variables
        assert "temporal_context" in template.input_variables
    
    def test_state_persistence(self, integration, sample_metrics):
        """Test state persistence in vector store."""
        # Create test document
        doc = Document(
            page_content="Test state content",
            metadata={
                "metrics": sample_metrics,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Add to vector store
        integration.vector_store.add_documents([doc])
        
        # Verify retrieval
        results = integration.vector_store.similarity_search(
            "Test state",
            k=1
        )
        assert len(results) == 1
        assert results[0].metadata["metrics"] == sample_metrics
