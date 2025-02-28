"""
Tests for the field-Neo4j bridge module.

This module tests the integration between pattern-aware RAG input and Neo4j persistence
while maintaining field state awareness and coherence throughout the process.
"""

import unittest
import os
from datetime import datetime, timedelta
from pathlib import Path
import json

# Use relative imports from the current package
from ...pattern_aware_rag.learning.field_neo4j_bridge import (
    FieldStateNeo4jBridge, MockPatternDB
)
from ...pattern_aware_rag.learning.learning_health_integration import (
    FieldObserver, HealthFieldObserver
)
from ...adaptive_core.system_health import SystemHealthService

# Path to test data
TEST_DATA_DIR = Path(__file__).parents[3] / "data" / "climate_risk"
CLIMATE_RISK_FILE = TEST_DATA_DIR / "climate_risk_marthas_vineyard.txt"


class TestFieldNeo4jBridge(unittest.TestCase):
    """Test the field-Neo4j bridge integration."""
    
    def setUp(self):
        """Set up test resources."""
        # Create a field observer
        self.field_observer = FieldObserver()
        
        # Create system health service
        self.health_service = SystemHealthService()
        
        # Create health-aware field observer
        self.health_field_observer = HealthFieldObserver(
            health_service=self.health_service
        )
        
        # Create mock Neo4j pattern DB
        self.pattern_db = MockPatternDB()
        
        # Create bridge with Neo4j persistence
        self.neo4j_bridge = FieldStateNeo4jBridge(
            field_observer=self.health_field_observer,
            persistence_mode="neo4j",
            pattern_db=self.pattern_db
        )
        
        # Also create a direct mode bridge for testing
        self.direct_bridge = FieldStateNeo4jBridge(
            field_observer=self.health_field_observer,
            persistence_mode="direct"
        )
        
        # Initialize field observer with sample data
        for i in range(10):
            stability = 0.7 + (i * 0.02)  # Small increase over time
            self.field_observer.wave_history.append(stability)
            self.field_observer.tonic_history.append(0.5 + (i * 0.01))
            
            # Health field observer
            self.health_field_observer.wave_history.append(stability)
            self.health_field_observer.tonic_history.append(0.5 + (i * 0.01))
    
    def test_align_incoming_pattern(self):
        """Test alignment of incoming pattern data with field state."""
        # Create a test pattern
        pattern_data = {
            "id": "test_pattern",
            "pattern_type": "climate_risk",
            "hazard_type": "extreme_precipitation",
            "confidence": 0.8,
            "temporal_context": {"current": 2025}
        }
        
        # Align with field state
        aligned_pattern = self.neo4j_bridge.align_incoming_pattern(pattern_data, "test_user")
        
        # Verify field state was added
        self.assertIn("field_state", aligned_pattern, "Field state not added to pattern")
        self.assertIn("stability", aligned_pattern["field_state"], "Stability not in field state")
        self.assertIn("tonic", aligned_pattern["field_state"], "Tonic not in field state")
        self.assertIn("coherence", aligned_pattern["field_state"], "Coherence not in field state")
        
        # Verify adaptive ID was created
        self.assertIn("adaptive_id", aligned_pattern, "AdaptiveID not created")
        
    def test_process_prompt_generated_content(self):
        """Test processing of prompt-generated content."""
        # Create test patterns
        patterns = [
            {
                "id": "pattern1",
                "pattern_type": "climate_risk",
                "hazard_type": "extreme_precipitation",
                "confidence": 0.8,
                "temporal_context": {"current": 2025}
            },
            {
                "id": "pattern2",
                "pattern_type": "climate_risk",
                "hazard_type": "drought",
                "confidence": 0.7,
                "temporal_context": {"mid_century": 2050}
            }
        ]
        
        # Process patterns
        processed_data = self.neo4j_bridge.process_prompt_generated_content(patterns, "test_user")
        
        # Verify processing
        self.assertIn("patterns", processed_data, "Patterns not in processed data")
        self.assertIn("field_state", processed_data, "Field state not in processed data")
        self.assertEqual(len(processed_data["patterns"]), 2, "Incorrect number of processed patterns")
        
        # Verify each pattern has alignment info
        for pattern in processed_data["patterns"]:
            self.assertIn("neo4j_alignment", pattern, "Alignment info not added")
            self.assertIn("is_aligned", pattern["neo4j_alignment"], "Alignment status not present")
            
            # Verify Neo4j pattern preparation
            if self.neo4j_bridge.persistence_mode == "neo4j":
                self.assertIn("neo4j_pattern", pattern, "Neo4j pattern not created")
    
    def test_neo4j_integration(self):
        """Test integration with Neo4j."""
        # Create test pattern
        pattern_data = {
            "id": "test_pattern",
            "pattern_type": "climate_risk",
            "hazard_type": "extreme_precipitation",
            "confidence": 0.8,
            "temporal_context": {"current": 2025},
            "relationships": [
                {
                    "target_id": "other_pattern",
                    "type": "INFLUENCES",
                    "strength": 0.7
                }
            ]
        }
        
        # Process pattern
        processed_pattern = self.neo4j_bridge.process_prompt_generated_content(pattern_data, "test_user")
        
        # Integrate with Neo4j
        success = self.neo4j_bridge.integrate_with_neo4j(processed_pattern)
        
        # Verify integration
        if self.neo4j_bridge.persistence_mode == "neo4j":
            self.assertTrue(success, "Neo4j integration failed")
            
            # Check Neo4j database
            graph_data = self.pattern_db.get_graph()
            self.assertTrue(len(graph_data["nodes"]) > 0, "No nodes created in Neo4j")
            self.assertTrue(len(graph_data["relationships"]) > 0, "No relationships created in Neo4j")
    
    def test_direct_mode(self):
        """Test direct LLM mode operation."""
        # Create test pattern
        pattern_data = {
            "id": "test_pattern",
            "pattern_type": "climate_risk", 
            "hazard_type": "extreme_precipitation",
            "confidence": 0.8
        }
        
        # Process with direct bridge
        processed_pattern = self.direct_bridge.process_prompt_generated_content(pattern_data, "test_user")
        
        # Verify field state was still added
        self.assertIn("field_state", processed_pattern, "Field state not added in direct mode")
        
        # Verify no Neo4j pattern in direct mode
        self.assertNotIn("neo4j_pattern", processed_pattern, "Neo4j pattern should not be created in direct mode")
        
        # Verify alignment is always true in direct mode
        self.assertTrue(
            processed_pattern["neo4j_alignment"]["is_aligned"], 
            "Alignment should always be true in direct mode"
        )


if __name__ == "__main__":
    # Ensure data file exists
    if not CLIMATE_RISK_FILE.exists():
        print(f"Climate risk data file not found: {CLIMATE_RISK_FILE}")
        print("Skipping tests.")
    else:
        unittest.main()
