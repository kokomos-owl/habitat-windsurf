"""
Tests for the integration of tonic-harmonic properties with the ID system.

These tests validate that tonic-harmonic properties are properly integrated
with the AdaptiveID system, ensuring equal representation alongside vector-based approaches.
"""

import pytest
import numpy as np
from datetime import datetime
import json
from unittest.mock import MagicMock, patch

# Import the components to test
from habitat_evolution.field.field_state import TonicHarmonicFieldState
from habitat_evolution.field.field_adaptive_id_bridge import FieldAdaptiveIDBridge


class TestTonicHarmonicIntegration:
    """Test suite for tonic-harmonic integration with the ID system."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a realistic field analysis with tonic-harmonic properties
        self.mock_field_analysis = {
            "topology": {
                "effective_dimensionality": 3,
                "principal_dimensions": [
                    {"dimension": 0, "explained_variance": 0.6},
                    {"dimension": 1, "explained_variance": 0.3},
                    {"dimension": 2, "explained_variance": 0.1}
                ],
                "eigenvalues": np.array([0.6, 0.3, 0.1, 0.05, 0.05]),
                "eigenvectors": np.array([
                    [0.8, 0.1, 0.1],
                    [0.1, 0.7, 0.2],
                    [0.1, 0.2, 0.7],
                    [0.5, 0.3, 0.2],
                    [0.3, 0.5, 0.2]
                ])
            },
            "field_properties": {
                "coherence": 0.85,
                "navigability_score": 0.75,
                "stability": 0.9
            },
            "patterns": {
                "pattern1": {"position": [0.1, 0.2, 0.3], "community": 0},
                "pattern2": {"position": [0.2, 0.3, 0.4], "community": 0},
                "pattern3": {"position": [0.7, 0.8, 0.9], "community": 1},
                "pattern4": {"position": [0.8, 0.7, 0.6], "community": 1},
                "pattern5": {"position": [0.4, 0.5, 0.6], "community": 2}
            },
            "pattern_eigenspace": {
                "pattern1": {
                    "projections": {0: 0.8, 1: 0.1, 2: 0.1},
                    "eigenspace_position": {0: 0.8, 1: 0.1, 2: 0.1}
                },
                "pattern2": {
                    "projections": {0: 0.75, 1: 0.15, 2: 0.1},
                    "eigenspace_position": {0: 0.75, 1: 0.15, 2: 0.1}
                },
                "pattern3": {
                    "projections": {0: -0.1, 1: 0.8, 2: 0.1},
                    "eigenspace_position": {0: -0.1, 1: 0.8, 2: 0.1}
                },
                "pattern4": {
                    "projections": {0: -0.2, 1: 0.75, 2: 0.05},
                    "eigenspace_position": {0: -0.2, 1: 0.75, 2: 0.05}
                },
                "pattern5": {
                    "projections": {0: 0.1, 1: 0.1, 2: 0.8},
                    "eigenspace_position": {0: 0.1, 1: 0.1, 2: 0.8}
                }
            },
            "resonance_relationships": {
                "dim_pos_0": {
                    "type": "dimensional_resonance",
                    "direction": "positive",
                    "dimension": 0,
                    "patterns": ["pattern1", "pattern2"],
                    "strength": 0.8
                },
                "dim_pos_1": {
                    "type": "dimensional_resonance",
                    "direction": "positive",
                    "dimension": 1,
                    "patterns": ["pattern3", "pattern4"],
                    "strength": 0.75
                },
                "dim_pos_2": {
                    "type": "dimensional_resonance",
                    "direction": "positive",
                    "dimension": 2,
                    "patterns": ["pattern5"],
                    "strength": 0.7
                },
                "dim_comp_0": {
                    "type": "complementary",
                    "dimension": 0,
                    "patterns": ["pattern1", "pattern2", "pattern3", "pattern4"],
                    "strength": 0.65
                }
            },
            "fuzzy_boundaries": {
                "boundary_fuzziness": {
                    "dim_pos_0__dim_pos_1": 0.7,
                    "dim_pos_0__dim_pos_2": 0.8,
                    "dim_pos_1__dim_pos_2": 0.6
                },
                "transition_zones": {
                    "dim_pos_0__dim_pos_1": ["pattern2", "pattern3"],
                    "dim_pos_0__dim_pos_2": ["pattern2", "pattern5"],
                    "dim_pos_1__dim_pos_2": ["pattern4", "pattern5"]
                }
            }
        }
        
        # Create the TonicHarmonicFieldState instance
        self.field_state = TonicHarmonicFieldState(self.mock_field_analysis)
        
        # Create mock AdaptiveID instance
        self.adaptive_id = MagicMock()
        self.adaptive_id.update_context = MagicMock(return_value={"success": True})
        
        # Create the bridge
        self.bridge = FieldAdaptiveIDBridge(self.field_state, self.adaptive_id)
    
    def test_tonic_harmonic_properties_in_field_state(self):
        """Test that tonic-harmonic properties are properly stored in the field state."""
        # Check eigenspace properties
        assert hasattr(self.field_state, "eigenvalues")
        assert hasattr(self.field_state, "eigenvectors")
        assert hasattr(self.field_state, "principal_dimensions")
        assert hasattr(self.field_state, "effective_dimensionality")
        
        # Check resonance relationships
        assert hasattr(self.field_state, "resonance_relationships")
        assert len(self.field_state.resonance_relationships) >= 3  # At least 3 relationships
        
        # Check pattern eigenspace properties
        assert hasattr(self.field_state, "pattern_eigenspace_properties")
        assert len(self.field_state.pattern_eigenspace_properties) >= 5  # At least 5 patterns
        
        # Verify specific resonance relationships
        assert "dim_pos_0" in self.field_state.resonance_relationships
        assert "pattern1" in self.field_state.resonance_relationships["dim_pos_0"]["patterns"]
        assert "pattern2" in self.field_state.resonance_relationships["dim_pos_0"]["patterns"]
        
        # Verify pattern eigenspace properties
        assert "pattern1" in self.field_state.pattern_eigenspace_properties
        assert "projections" in self.field_state.pattern_eigenspace_properties["pattern1"]
        assert 0 in self.field_state.pattern_eigenspace_properties["pattern1"]["projections"]
        assert self.field_state.pattern_eigenspace_properties["pattern1"]["projections"][0] > 0.7  # Strong positive projection
    
    def test_dimensional_resonance_calculation(self):
        """Test that dimensional resonance is correctly calculated."""
        # Get dimensional resonance for a pattern
        resonance_info = self.field_state._get_dimensional_resonance_for_pattern("pattern1")
        
        # Verify resonance information
        assert "dimension_0" in resonance_info
        assert resonance_info["dimension_0"]["direction"] == "positive"
        assert resonance_info["dimension_0"]["strength"] > 0.7
        
        # Get all dimensional resonance groups
        resonance_groups = self.field_state._get_dimensional_resonance_groups()
        
        # Verify resonance groups
        assert "dim_pos_0" in resonance_groups
        assert "members" in resonance_groups["dim_pos_0"]
        assert "pattern1" in resonance_groups["dim_pos_0"]["members"]
        assert "pattern2" in resonance_groups["dim_pos_0"]["members"]
    
    def test_fuzzy_boundary_calculation(self):
        """Test that fuzzy boundaries are correctly calculated."""
        # Calculate fuzzy boundaries
        boundary_data = self.field_state._get_fuzzy_boundary_data()
        
        # Verify boundary data
        assert "boundary_fuzziness" in boundary_data
        assert "transition_zones" in boundary_data
        
        # There should be at least one boundary
        assert len(boundary_data["boundary_fuzziness"]) > 0
        assert len(boundary_data["transition_zones"]) > 0
    
    def test_eigenspace_distance_calculation(self):
        """Test that eigenspace distances are correctly calculated."""
        # Calculate distance between patterns in the same resonance group
        distance_same_group = self.field_state._calculate_eigenspace_distance("pattern1", "pattern2")
        
        # Calculate distance between patterns in different resonance groups
        distance_diff_group = self.field_state._calculate_eigenspace_distance("pattern1", "pattern3")
        
        # Patterns in the same group should be closer than patterns in different groups
        assert distance_same_group < distance_diff_group
    
    def test_tonic_harmonic_to_adaptive_id_context(self):
        """Test that tonic-harmonic properties are included in AdaptiveID context."""
        # Get the AdaptiveID context from the field state
        context = self.field_state.to_adaptive_id_context()
        
        # Verify that tonic-harmonic properties are included
        assert "field_eigenspace" in context
        assert "pattern_eigenspace" in context["field_eigenspace"]
        assert "resonance_relationships" in context["field_eigenspace"]
        assert "tonic_harmonic_properties" in context["field_eigenspace"]
        
        # Verify specific tonic-harmonic properties
        th_props = context["field_eigenspace"]["tonic_harmonic_properties"]
        assert "eigenvalues" in th_props
        assert "principal_dimensions" in th_props
        assert "effective_dimensionality" in th_props
        assert "dimensional_resonance_groups" in th_props
        
        # Verify pattern eigenspace properties
        pattern_eigenspace = context["field_eigenspace"]["pattern_eigenspace"]
        assert "pattern1" in pattern_eigenspace
        assert "projections" in pattern_eigenspace["pattern1"]
        assert "eigenspace_position" in pattern_eigenspace["pattern1"]
        assert "resonance_groups" in pattern_eigenspace["pattern1"]
        
        # Verify resonance relationships
        resonance_rels = context["field_eigenspace"]["resonance_relationships"]
        assert "dim_pos_0" in resonance_rels
        assert "patterns" in resonance_rels["dim_pos_0"]
        assert "type" in resonance_rels["dim_pos_0"]
        assert "dimension" in resonance_rels["dim_pos_0"]
    
    def test_adaptive_id_to_tonic_harmonic_update(self):
        """Test that AdaptiveID updates properly update tonic-harmonic properties."""
        # Create a mock AdaptiveID context with tonic-harmonic properties
        mock_context = {
            "field_state_id": self.field_state.id,
            "field_version_id": self.field_state.version_id,
            "field_coherence": 0.9,
            "field_navigability": 0.8,
            "field_stability": 0.95,
            "field_dimensionality": 4,
            "field_eigenspace": {
                "pattern_eigenspace": {
                    "pattern1": {
                        "projections": {0: 0.9, 1: 0.05, 2: 0.05},
                        "eigenspace_position": {0: 0.9, 1: 0.05, 2: 0.05},
                        "resonance_groups": ["dim_pos_0", "dim_comp_0"],
                        "dimensional_resonance": {
                            "dimension_0": {
                                "projection": 0.9,
                                "direction": "positive",
                                "strength": 0.9
                            }
                        }
                    }
                },
                "resonance_relationships": {
                    "dim_pos_0": {
                        "type": "dimensional_resonance",
                        "dimension": 0,
                        "patterns": ["pattern1", "pattern2", "pattern6"],
                        "strength": 0.85
                    },
                    "new_resonance_group": {
                        "type": "harmonic_sequence",
                        "patterns": ["pattern3", "pattern4", "pattern6"],
                        "strength": 0.7
                    }
                },
                "tonic_harmonic_properties": {
                    "eigenvalues": [0.65, 0.25, 0.1],
                    "principal_dimensions": [0, 1, 2],
                    "effective_dimensionality": 3,
                    "dimensional_resonance_groups": {
                        "dim_pos_0": {
                            "type": "dimensional_resonance",
                            "dimension": 0,
                            "members": ["pattern1", "pattern2", "pattern6"],
                            "strength": 0.85
                        }
                    },
                    "fuzzy_boundaries": {
                        "boundary_fuzziness": {
                            "dim_pos_0__dim_pos_1": 0.65
                        },
                        "transition_zones": {
                            "dim_pos_0__dim_pos_1": ["pattern2", "pattern3", "pattern6"]
                        }
                    }
                }
            }
        }
        
        # Store original values for comparison
        original_eigenvalues = self.field_state.eigenvalues.copy()
        original_resonance_relationships = self.field_state.resonance_relationships.copy()
        
        # Update the field state from the mock context
        self.field_state.update_from_adaptive_id_context(mock_context)
        
        # Verify that tonic-harmonic properties are updated
        assert not np.array_equal(self.field_state.eigenvalues, original_eigenvalues)
        assert self.field_state.effective_dimensionality == 4
        
        # Verify that resonance relationships are updated
        assert "new_resonance_group" in self.field_state.resonance_relationships
        assert "pattern6" in self.field_state.resonance_relationships["dim_pos_0"]["patterns"]
        
        # Verify that pattern eigenspace properties are updated
        assert self.field_state.pattern_eigenspace_properties["pattern1"]["projections"][0] == 0.9
    
    def test_bridge_propagation_of_tonic_harmonic_properties(self):
        """Test that the bridge properly propagates tonic-harmonic properties."""
        # Call the bridge to propagate field state changes
        result = self.bridge.propagate_field_state_changes()
        
        # Verify that the propagation was successful
        assert result["success"] is True
        
        # Verify that AdaptiveID was called with the correct context
        context_arg = self.adaptive_id.update_context.call_args[0][0]
        
        # Check that tonic-harmonic properties are included
        assert "field_eigenspace" in context_arg
        assert "resonance_relationships" in context_arg["field_eigenspace"]
        assert "tonic_harmonic_properties" in context_arg["field_eigenspace"]
        
        # Check that pattern eigenspace properties are included
        assert "pattern_eigenspace" in context_arg["field_eigenspace"]
        for pattern_id in ["pattern1", "pattern2", "pattern3", "pattern4", "pattern5"]:
            assert pattern_id in context_arg["field_eigenspace"]["pattern_eigenspace"]
            pattern_data = context_arg["field_eigenspace"]["pattern_eigenspace"][pattern_id]
            assert "projections" in pattern_data
            assert "eigenspace_position" in pattern_data
    
    def test_bridge_verification_of_resonance_consistency(self):
        """Test that the bridge verifies resonance consistency."""
        # Test the verification method
        verification_result = self.bridge.verify_resonance_consistency()
        
        # Verify that the verification was successful
        assert verification_result["success"] is True
        assert verification_result["consistent"] is True
        
        # Modify the resonance relationships to create inconsistency
        original_patterns = self.field_state.resonance_relationships["dim_pos_0"]["patterns"].copy()
        self.field_state.resonance_relationships["dim_pos_0"]["patterns"] = ["pattern7", "pattern8"]
        
        # Test the verification method again
        verification_result = self.bridge.verify_resonance_consistency()
        
        # Verify that the verification detected the inconsistency
        assert verification_result["success"] is True
        assert verification_result["consistent"] is False
        
        # Restore the original patterns
        self.field_state.resonance_relationships["dim_pos_0"]["patterns"] = original_patterns
    
    def test_integration_with_neo4j_serialization(self):
        """Test that tonic-harmonic properties are properly serialized for Neo4j."""
        # Serialize the field state for Neo4j
        neo4j_data = self.field_state.to_neo4j()
        
        # Verify that tonic-harmonic properties are included
        assert "resonance_relationships" in neo4j_data
        assert "pattern_eigenspace_properties" in neo4j_data
        assert "eigenvalues" in neo4j_data
        assert "principal_dimensions" in neo4j_data
        
        # Verify that the serialized data is in the correct format
        assert isinstance(neo4j_data["resonance_relationships"], str)  # JSON string
        assert isinstance(neo4j_data["pattern_eigenspace_properties"], str)  # JSON string
        
        # Deserialize and verify content
        resonance_relationships = json.loads(neo4j_data["resonance_relationships"])
        assert "dim_pos_0" in resonance_relationships
        assert "patterns" in resonance_relationships["dim_pos_0"]
        
        pattern_eigenspace = json.loads(neo4j_data["pattern_eigenspace_properties"])
        assert "pattern1" in pattern_eigenspace
        assert "projections" in pattern_eigenspace["pattern1"]


if __name__ == "__main__":
    pytest.main(["-xvs", "test_tonic_harmonic_integration.py"])
