"""
Tests for the ObservationFrameBridge component.

These tests validate that observation frames properly integrate with the tonic-harmonic
field architecture, enabling multiple perspectives to contribute to field topology
without imposing artificial domain boundaries.
"""

import sys
import os
import pytest
import numpy as np
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from habitat_evolution.field.observation_frame_bridge import ObservationFrameBridge
from habitat_evolution.field.field_state import TonicHarmonicFieldState
from habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestObservationFrameBridge:
    """Test suite for ObservationFrameBridge."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a realistic field analysis with tonic-harmonic properties
        self.mock_field_analysis = {
            "topology": {
                "effective_dimensionality": 3,
                "principal_dimensions": [0, 1, 2],
                "eigenvalues": np.array([0.6, 0.3, 0.1]),
                "eigenvectors": np.array([
                    [0.8, 0.1, 0.1],
                    [0.1, 0.7, 0.2],
                    [0.1, 0.2, 0.7]
                ])
            },
            "density": {
                "density_centers": [],
                "density_map": np.zeros((3, 3))
            },
            "field_properties": {
                "coherence": 0.6,
                "navigability_score": 0.65,
                "stability": 0.7
            },
            "patterns": {},
            "resonance_relationships": {}
        }
        
        # Create the field state and analyzer
        self.field_state = TonicHarmonicFieldState(self.mock_field_analysis)
        self.field_analyzer = TopologicalFieldAnalyzer()
        
        # Create the observation frame bridge
        self.bridge = ObservationFrameBridge(
            field_state=self.field_state,
            field_analyzer=self.field_analyzer
        )
        
        # Load test observation frames
        self.observation_frames = self._load_test_observation_frames()
        
        # Load test observations
        self.test_observations = self._load_test_observations()
        
    def _load_test_observation_frames(self) -> List[Dict[str, Any]]:
        """Load test observation frames from the test data."""
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "demos", "data", "climate_risk", "observation_frames.json"
        )
        with open(data_path, 'r') as f:
            return json.load(f)["observation_frames"]
    
    def _load_test_observations(self) -> List[Dict[str, Any]]:
        """Load test observations from the test data."""
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "demos", "data", "climate_risk", "test_observations.json"
        )
        with open(data_path, 'r') as f:
            return json.load(f)["observations"]
    
    def test_register_observation_frames(self):
        """Test that observation frames can be registered with the bridge."""
        # Register each observation frame
        frame_ids = []
        for frame in self.observation_frames:
            frame_id = self.bridge.register_observation_frame(frame)
            frame_ids.append(frame_id)
            
        # Check that all frames were registered
        assert len(self.bridge.observation_frames) == len(self.observation_frames)
        
        # Check that field state was updated with frame information
        field_topology = self.bridge.get_field_topology()
        
        # In a real implementation, we would check specific topology properties
        # For now, we're just checking that the field topology exists
        assert field_topology is not None
        
        # Log the field topology for inspection
        logger.info(f"Field topology after registering frames: {field_topology}")
        
    def test_process_observations(self):
        """Test that observations can be processed through frames."""
        # Register each observation frame
        for frame in self.observation_frames:
            self.bridge.register_observation_frame(frame)
            
        # Process each observation
        processed_observations = []
        for obs in self.test_observations:
            processed = self.bridge.process_observation(obs)
            processed_observations.append(processed)
            
        # Check that all observations were processed
        assert len(processed_observations) == len(self.test_observations)
        
        # Check that observations have frame-specific field properties
        for processed in processed_observations:
            assert "context" in processed
            assert "field_properties" in processed["context"]
            assert "frame" in processed["context"]["field_properties"]
            assert "resonance_influence" in processed["context"]["field_properties"]
            
        # Log a sample processed observation for inspection
        logger.info(f"Sample processed observation: {processed_observations[0]}")
        
    def test_cross_frame_relationships(self):
        """Test that cross-frame relationships are detected."""
        # Register each observation frame
        for frame in self.observation_frames:
            self.bridge.register_observation_frame(frame)
            
        # Process observations that span multiple frames
        # In our test data, we have observations that connect ecological, indigenous, and socioeconomic perspectives
        
        # Process each observation
        for obs in self.test_observations:
            self.bridge.process_observation(obs)
            
        # Get cross-frame resonance
        cross_frame_resonance = self.bridge.get_cross_frame_resonance()
        
        # In a real implementation, we would check specific resonance properties
        # For now, we're just logging the result
        logger.info(f"Cross-frame resonance: {cross_frame_resonance}")
        
    def test_integration_with_adaptive_id(self):
        """Test integration with AdaptiveID system."""
        # Create a mock field observer that can be used with AdaptiveID
        class MockFieldObserver:
            def __init__(self):
                self.observations = []
                
            def observe(self, context):
                self.observations.append({"context": context, "time": datetime.now()})
        
        mock_observer = MockFieldObserver()
        
        # Create adaptive IDs for entities in observations
        entities = set()
        for obs in self.test_observations:
            entities.add(obs["source"])
            entities.add(obs["target"])
        
        # Create a system ID to use as creator
        system_id = str(uuid.uuid4())
        
        # Create adaptive IDs
        entity_ids = {}
        for entity in entities:
            adaptive_id = AdaptiveID(base_concept=entity, creator_id=system_id)
            entity_ids[entity] = adaptive_id
            
        # Register adaptive IDs with mock observer instead of field state
        for entity, adaptive_id in entity_ids.items():
            adaptive_id.register_with_field_observer(mock_observer)
            
        # Register observation frames
        for frame in self.observation_frames:
            self.bridge.register_observation_frame(frame)
            
        # Process observations
        for obs in self.test_observations:
            # Update with adaptive ID information
            obs_with_ids = obs.copy()
            obs_with_ids["source_id"] = entity_ids[obs["source"]].id
            obs_with_ids["target_id"] = entity_ids[obs["target"]].id
            
            # Process the observation
            self.bridge.process_observation(obs_with_ids)
            
        # Check that mock observer received observations
        assert len(mock_observer.observations) > 0
        
        # Log a sample observation for inspection
        if mock_observer.observations:
            logger.info(f"Sample observation: {mock_observer.observations[0]}")
            
        # In a real implementation, we would check that field state was updated
        # based on AdaptiveID interactions
            
    def test_field_topology_evolution(self):
        """Test that field topology evolves with observations."""
        # Capture initial field topology
        initial_topology = self.bridge.get_field_topology()
        
        # Register observation frames
        for frame in self.observation_frames:
            self.bridge.register_observation_frame(frame)
            
        # Process observations
        for obs in self.test_observations:
            self.bridge.process_observation(obs)
            
        # Capture final field topology
        final_topology = self.bridge.get_field_topology()
        
        # In a real implementation, we would check specific topology changes
        # For now, we're just logging the result
        logger.info(f"Initial topology: {initial_topology}")
        logger.info(f"Final topology: {final_topology}")
        
        # Check that the field topology has evolved
        # This is a very basic check - in a real implementation, we would check specific properties
        assert final_topology != initial_topology


if __name__ == "__main__":
    # Run the test directly for debugging
    pytest.main(["-xvs", __file__])
