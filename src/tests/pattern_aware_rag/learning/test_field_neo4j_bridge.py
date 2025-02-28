"""
Tests for the field-Neo4j bridge module.

This module tests the integration between pattern-aware RAG input and Neo4j persistence
while maintaining field state awareness and coherence throughout the process.
"""

import pytest
import os
from datetime import datetime, timedelta
from pathlib import Path
import json

from habitat_evolution.pattern_aware_rag.learning.field_neo4j_bridge import (
    FieldStateNeo4jBridge, MockPatternDB
)
from habitat_evolution.pattern_aware_rag.learning.learning_health_integration import (
    FieldObserver, HealthFieldObserver
)
from habitat_evolution.adaptive_core.system_health import SystemHealthService

# Path to test data
TEST_DATA_DIR = Path(__file__).parents[3] / "data" / "climate_risk"
CLIMATE_RISK_FILE = TEST_DATA_DIR / "climate_risk_marthas_vineyard.txt"


@pytest.fixture
def field_observer():
    """Create a field observer for testing."""
    return FieldObserver(field_id="test_field")

@pytest.fixture
def health_service():
    """Create a system health service for testing."""
    return SystemHealthService()

@pytest.fixture
def health_field_observer(health_service):
    """Create a health-aware field observer for testing."""
    return HealthFieldObserver(field_id="test_field", health_service=health_service)

@pytest.fixture
def pattern_db():
    """Create a mock Neo4j pattern DB for testing."""
    return MockPatternDB()

@pytest.fixture
def neo4j_bridge(health_field_observer, pattern_db):
    """Create a bridge with Neo4j persistence for testing."""
    return FieldStateNeo4jBridge(
        field_observer=health_field_observer,
        persistence_mode="neo4j",
        pattern_db=pattern_db
    )


class TestFieldNeo4jBridge:
    """Test the field-Neo4j bridge integration."""
    
    def test_align_incoming_pattern(self, neo4j_bridge):
        """Test aligning incoming pattern data with Neo4j state."""
        # Test pattern data
        pattern_data = {
            "name": "Sea Level Rise",
            "type": "climate_risk",
            "probability": 0.85,
            "impact": "high",
            "location": "Martha's Vineyard",
            "temporal_horizon": "2050"
        }
        
        # Define user ID for provenance tracking
        user_id = "test_user_123"
        
        # Align pattern with Neo4j state
        aligned_pattern = neo4j_bridge.align_incoming_pattern(pattern_data, user_id)
        
        # Verify the pattern has been processed correctly
        assert "adaptive_id" in aligned_pattern
        assert aligned_pattern["name"] == "Sea Level Rise"
        assert aligned_pattern["type"] == "climate_risk"
        assert aligned_pattern["location"] == "Martha's Vineyard"
        
        # Field state might or might not be added depending on the observer state
        if "field_state" in aligned_pattern:
            assert "stability" in aligned_pattern["field_state"]
            assert "coherence" in aligned_pattern["field_state"]

    def test_process_prompt_generated_content_single(self, neo4j_bridge):
        """Test processing single pattern from prompt-generated content."""
        # Single pattern from prompt
        content = {
            "name": "Coastal Erosion",
            "type": "climate_risk",
            "probability": 0.75,
            "impact": "medium",
            "location": "Eastern Shore"
        }
        
        user_id = "test_user_456"
        
        # Process the content
        processed = neo4j_bridge.process_prompt_generated_content(content, user_id)
        
        # Verify processing
        assert isinstance(processed, dict)
        assert "adaptive_id" in processed
        assert processed["name"] == "Coastal Erosion"
        
        # Check neo4j alignment is included
        assert "neo4j_alignment" in processed
        assert "is_aligned" in processed["neo4j_alignment"]
        assert "metrics" in processed["neo4j_alignment"]

    def test_process_prompt_generated_content_list(self, neo4j_bridge):
        """Test processing multiple patterns from prompt-generated content."""
        # List of patterns from prompt
        content = [
            {
                "name": "Storm Surge",
                "type": "climate_risk",
                "probability": 0.9,
                "impact": "high",
                "location": "Coastal Areas"
            },
            {
                "name": "Saltwater Intrusion",
                "type": "climate_risk",
                "probability": 0.65,
                "impact": "medium",
                "location": "Low-lying Aquifers"
            }
        ]
        
        user_id = "test_user_789"
        
        # Process the content
        processed = neo4j_bridge.process_prompt_generated_content(content, user_id)
        
        # Verify processing - the function returns a dict with patterns in a list field
        assert isinstance(processed, dict)
        assert "patterns" in processed
        assert isinstance(processed["patterns"], list)
        assert len(processed["patterns"]) == 2
        
        # Check both patterns have been processed
        assert "adaptive_id" in processed["patterns"][0]
        assert processed["patterns"][0]["name"] == "Storm Surge"
        assert "adaptive_id" in processed["patterns"][1]
        assert processed["patterns"][1]["name"] == "Saltwater Intrusion"
        
        # Verify neo4j alignment information is included
        assert "neo4j_alignment" in processed["patterns"][0]
        assert "is_aligned" in processed["patterns"][0]["neo4j_alignment"]
        assert "neo4j_alignment" in processed["patterns"][1]
        assert "metrics" in processed["patterns"][1]["neo4j_alignment"]
        
        # Verify processed timestamp is included
        assert "processed_at" in processed
