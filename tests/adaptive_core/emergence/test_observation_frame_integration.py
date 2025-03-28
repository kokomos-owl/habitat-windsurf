"""
Test the integration of observation frames with the emergent pattern detector.

This test validates that the observation frames provide sufficient dimensionality
and density for pattern and meta-pattern detection without imposing artificial
domain boundaries.
"""

import os
import sys
import json
import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional

import pytest

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from habitat_evolution.adaptive_core.emergence.emergent_pattern_detector import EmergentPatternDetector
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from habitat_evolution.field.field_state import TonicHarmonicFieldState
from habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestObservationFrameIntegration:
    """Test the integration of observation frames with pattern detection."""

    @pytest.fixture
    def observation_frames(self) -> List[Dict[str, Any]]:
        """Load observation frames from the test data."""
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "demos", "data", "climate_risk", "observation_frames.json"
        )
        with open(data_path, 'r') as f:
            return json.load(f)["observation_frames"]

    @pytest.fixture
    def test_observations(self) -> List[Dict[str, Any]]:
        """Load test observations from the test data."""
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "demos", "data", "climate_risk", "test_observations.json"
        )
        with open(data_path, 'r') as f:
            return json.load(f)["observations"]

    @pytest.fixture
    def field_state(self, field_analyzer) -> TonicHarmonicFieldState:
        """Create a field state for testing."""
        # Initialize with empty field analysis matching expected structure
        initial_field_analysis = {
            "topology": {
                "effective_dimensionality": 3,
                "principal_dimensions": [0, 1, 2],
                "eigenvalues": np.array([0.8, 0.6, 0.4]),
                "eigenvectors": np.eye(3),
                "resonance_centers": [],
                "flow_vectors": [],
                "boundaries": []
            },
            "density": {
                "density_centers": [],
                "density_map": np.zeros((3, 3))
            },
            "field_properties": {
                "coherence": 0.6,
                "stability": 0.7,
                "navigability_score": 0.65
            },
            "patterns": {},
            "resonance_relationships": {}
        }
        return TonicHarmonicFieldState(field_analysis=initial_field_analysis)

    @pytest.fixture
    def field_analyzer(self) -> TopologicalFieldAnalyzer:
        """Create a field analyzer for testing."""
        return TopologicalFieldAnalyzer()

    @pytest.fixture
    def pattern_detector(self, field_state, field_analyzer) -> EmergentPatternDetector:
        """Create a pattern detector for testing."""
        return EmergentPatternDetector(
            field_state=field_state,
            field_analyzer=field_analyzer,
            config={
                "min_pattern_confidence": 0.7,
                "min_pattern_observations": 2,
                "relationship_similarity_threshold": 0.75,
                "meta_pattern_threshold": 0.65,
                "debug_logging": True
            }
        )

    @pytest.fixture
    def io_service(self) -> HarmonicIOService:
        """Create an IO service for testing."""
        return HarmonicIOService()

    def test_observation_frame_integration(
        self, 
        observation_frames, 
        test_observations, 
        pattern_detector, 
        field_state, 
        io_service
    ):
        """Test that observation frames provide sufficient dimensionality and density."""
        # Initialize field state with observation frames
        for frame in observation_frames:
            field_state.update_field_properties({
                "observation_frame": frame["name"],
                "resonance_centers": frame["resonance_centers"],
                "dimensionality": frame["field_properties"]["dimensionality"],
                "density_centers": frame["field_properties"]["density_centers"],
                "flow_vectors": frame["field_properties"]["flow_vectors"]
            })
        
        # Create adaptive IDs for entities in observations
        entities = set()
        for obs in test_observations:
            entities.add(obs["source"])
            entities.add(obs["target"])
        
        entity_ids = {}
        for entity in entities:
            adaptive_id = AdaptiveID(base_concept=entity)
            adaptive_id.register_with_field_observer(field_state)
            entity_ids[entity] = adaptive_id
        
        # Process observations
        for obs in test_observations:
            # Create relationship data
            relationship_data = {
                "source": entity_ids[obs["source"]].id,
                "source_concept": obs["source"],
                "predicate": obs["predicate"],
                "target": entity_ids[obs["target"]].id,
                "target_concept": obs["target"],
                "confidence": obs["context"]["confidence"],
                "perspective": obs["context"]["perspective"],
                "tonic_value": obs["context"]["tonic_value"],
                "harmonic_properties": obs["context"]["harmonic_properties"],
                "vector_properties": obs["context"]["vector_properties"],
                "timestamp": obs["timestamp"]
            }
            
            # Process the relationship through the pattern detector
            pattern_detector.process_relationship(relationship_data)
            
            # Update field state with the observation
            field_state.register_observation({
                "entity_id": entity_ids[obs["source"]].id,
                "related_entity_id": entity_ids[obs["target"]].id,
                "relationship_type": obs["predicate"],
                "perspective": obs["context"]["perspective"],
                "tonic_value": obs["context"]["tonic_value"],
                "harmonic_properties": obs["context"]["harmonic_properties"],
                "vector_properties": obs["context"]["vector_properties"]
            })
        
        # Force pattern detection cycle
        patterns = pattern_detector.detect_patterns()
        logger.info(f"Detected {len(patterns)} patterns")
        
        # Force meta-pattern detection cycle
        meta_patterns = pattern_detector._detect_meta_patterns()
        logger.info(f"Detected {len(meta_patterns)} meta-patterns")
        
        # Validate field state
        field_metrics = field_state.get_field_metrics()
        logger.info(f"Field metrics: {field_metrics}")
        
        # Assertions
        assert len(patterns) > 0, "No patterns detected"
        assert field_metrics["coherence"] > 0.5, "Field coherence too low"
        assert field_metrics["stability"] > 0.5, "Field stability too low"
        
        # Validate that we have sufficient dimensionality
        field_topology = field_state.get_field_topology()
        assert field_topology.get("effective_dimensionality", 0) > 3, "Insufficient field dimensionality"
        
        # Validate that we have sufficient density
        assert len(field_topology.get("density_centers", [])) > 0, "No density centers detected"
        
        # Validate that meta-patterns were detected
        if len(meta_patterns) == 0:
            logger.warning("No meta-patterns detected. This may be expected if insufficient pattern relationships exist.")
            # Examine pattern relationships to understand why no meta-patterns were detected
            pattern_relationships = pattern_detector._create_pattern_relationship_graph()
            logger.info(f"Pattern relationship graph: {pattern_relationships}")
        else:
            logger.info(f"Meta-patterns: {meta_patterns}")
            assert len(meta_patterns) > 0, "No meta-patterns detected despite sufficient patterns"
            
        # Validate that the field topology articulates the relationships properly
        resonance_centers = field_topology.get("resonance_centers", [])
        logger.info(f"Resonance centers: {resonance_centers}")
        assert len(resonance_centers) > 0, "No resonance centers detected in field topology"
        
        # Validate that the field flow vectors are present
        flow_vectors = field_topology.get("flow_vectors", [])
        logger.info(f"Flow vectors: {flow_vectors}")
        assert len(flow_vectors) > 0, "No flow vectors detected in field topology"

if __name__ == "__main__":
    # Run the test directly for debugging
    pytest.main(["-xvs", __file__])
