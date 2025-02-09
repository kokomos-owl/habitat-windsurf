"""Tests for pattern evolution system including vector space, anomalies, and flow states."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from src.core.metrics.system_visualization import SystemVisualizer, PatternVector, VectorSpace
from src.core.metrics.anomaly_detection import AnomalyDetector, AnomalyType, AnomalySignal
from src.core.metrics.flow_states import FlowStateManager, FlowState

class TestPatternEvolution:
    """Test suite for pattern evolution system."""
    
    @pytest.fixture
    def visualizer(self):
        """Create system visualizer instance."""
        return SystemVisualizer(dimensions=4)
    
    @pytest.fixture
    def anomaly_detector(self):
        """Create anomaly detector instance."""
        return AnomalyDetector()
    
    @pytest.fixture
    def flow_manager(self):
        """Create flow state manager instance."""
        return FlowStateManager()
    
    def test_pattern_vector_evolution(self, visualizer):
        """Test pattern vector evolution in coherence space."""
        # Initial pattern metrics
        pattern_metrics = {
            'pattern1': {
                'coherence': 0.8,
                'success_rate': 0.75,
                'stability': 0.9,
                'emergence_potential': 0.1
            }
        }
        
        # Update pattern space
        visualizer.update_pattern_space(pattern_metrics, {})
        
        # Get system state
        state = visualizer.get_system_state()
        
        # Verify vector creation
        assert 'pattern1' in state['pattern_space']
        vector = state['pattern_space']['pattern1']
        assert len(vector['coordinates']) == 4
        assert vector['coherence'] == 0.8
        
        # Test vector evolution
        evolved_metrics = {
            'pattern1': {
                'coherence': 0.85,
                'success_rate': 0.8,
                'stability': 0.85,
                'emergence_potential': 0.15
            }
        }
        
        visualizer.update_pattern_space(evolved_metrics, {})
        new_state = visualizer.get_system_state()
        
        # Verify vector evolution
        evolved_vector = new_state['pattern_space']['pattern1']
        assert evolved_vector['coherence'] > vector['coherence']
        assert any(v != 0 for v in evolved_vector['velocity'])  # Should have non-zero velocity
    
    def test_emergence_detection(self, visualizer, flow_manager):
        """Test emergence detection in pattern space."""
        # Create pattern showing emergence signs
        pattern_metrics = {
            'pattern2': {
                'coherence': 0.6,
                'success_rate': 0.7,
                'stability': 0.65,
                'emergence_potential': 0.8,  # Very high emergence potential
                'confidence': 0.75
            }
        }
        
        # Update pattern space
        visualizer.update_pattern_space(pattern_metrics, {})
        state = visualizer.get_system_state()
        
        # Check emergence field
        assert np.any(state['emergence_field'])  # Should have non-zero emergence field
        
        # Test flow state transition
        current_state = FlowState.ACTIVE
        new_state = flow_manager.assess_state(
            current_state,
            pattern_metrics['pattern2'],
            {'success_rate': 0.7}
        )
        
        # Should transition to EMERGING due to high emergence potential
        assert new_state == FlowState.EMERGING
    
    def test_coherence_anomaly_detection(self, anomaly_detector):
        """Test detection of coherence-related anomalies."""
        # Create pattern vectors with sudden coherence drop
        pattern_vectors = {
            'pattern3': {
                'coordinates': [0.8, 0.8, 0.8, 0.1],
                'velocity': [0.1, 0.1, 0.1, 0.1],
                'coherence': 0.3,  # Sudden drop in coherence
                'emergence_potential': 0.1,
                'velocity': [0.1, 0.1, 0.1, 0.1]
            }
        }
        
        # First check - should be stable
        anomalies = anomaly_detector.detect_anomalies(
            {'system_metrics': {'stability': 0.8}},
            pattern_vectors,
            np.array([[1.0]])
        )
        assert not any(a.anomaly_type == AnomalyType.COHERENCE_BREAK for a in anomalies)
        
        # Introduce sudden coherence drop
        pattern_vectors['pattern3']['coherence'] = 0.3
        anomalies = anomaly_detector.detect_anomalies(
            {'system_metrics': {'stability': 0.3}},
            pattern_vectors,
            np.array([[1.0]])
        )
        
        # Should detect coherence break
        assert any(a.anomaly_type == AnomalyType.COHERENCE_BREAK for a in anomalies)
    
    def test_temporal_stability(self, visualizer):
        """Test temporal stability tracking."""
        # Create temporal context
        temporal_context = {
            'time_points': [
                datetime.now() - timedelta(days=i)
                for i in range(5)
            ]
        }
        
        # Update pattern space with temporal context
        pattern_metrics = {
            'pattern4': {
                'coherence': 0.75,
                'success_rate': 0.8,
                'stability': 0.7,
                'emergence_potential': 0.2
            }
        }
        
        visualizer.update_pattern_space(pattern_metrics, temporal_context)
        state = visualizer.get_system_state()
        
        # Verify temporal grid creation
        assert len(state['temporal_grid']) > 0
        assert all(0 <= x <= 1 for x in state['temporal_grid'])  # Should be normalized
    
    def test_pattern_collapse_detection(self, anomaly_detector):
        """Test detection of pattern collapse."""
        # Create unstable pattern vector
        pattern_vectors = {
            'pattern5': {
                'coordinates': [0.3, 0.3, 0.2, 0.8],
                'velocity': [0.2, 0.2, 0.2, 0.2],  # High velocity
                'coherence': 0.3,  # Low coherence
                'emergence_potential': 0.8  # High emergence during collapse
            }
        }
        
        anomalies = anomaly_detector.detect_anomalies(
            {'system_metrics': {'stability': 0.2}},
            pattern_vectors,
            np.array([[1.0]])
        )
        
        # Should detect pattern collapse
        collapse_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.PATTERN_COLLAPSE]
        assert len(collapse_anomalies) > 0
        assert collapse_anomalies[0].severity > 0.7  # Should be high severity
    
    def test_emergence_to_learning_transition(self, flow_manager):
        """Test transition from EMERGING to LEARNING state."""
        # Start with EMERGING state
        current_state = FlowState.EMERGING
        
        # Pattern showing learning need
        metrics = {
            'confidence': 0.4,  # Below learning threshold
            'temporal_stability': 0.6,
            'pattern_matches': 10
        }
        
        # Should transition to LEARNING
        new_state = flow_manager.assess_state(
            current_state,
            metrics,
            {'success_rate': 0.4}
        )
        
        assert new_state == FlowState.LEARNING
    
    def test_structural_shift_detection(self, anomaly_detector):
        """Test detection of structural shifts in pattern relationships."""
        # Create initial coherence matrix
        initial_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        
        # Record initial state
        anomaly_detector.detect_anomalies(
            {'system_metrics': {'stability': 0.8}},
            {'p1': {'coherence': 0.8, 'emergence_potential': 0.1, 'velocity': [0.1, 0.1, 0.1, 0.1]}, 
             'p2': {'coherence': 0.8, 'emergence_potential': 0.1, 'velocity': [0.1, 0.1, 0.1, 0.1]}},
            initial_matrix
        )
        
        # Introduce structural shift
        shifted_matrix = np.array([[1.0, 0.2], [0.2, 1.0]])  # Significant relationship change
        
        anomalies = anomaly_detector.detect_anomalies(
            {'system_metrics': {'stability': 0.8}},
            {'p1': {'coherence': 0.8, 'emergence_potential': 0.1, 'velocity': [0.1, 0.1, 0.1, 0.1]}, 
             'p2': {'coherence': 0.8, 'emergence_potential': 0.1, 'velocity': [0.1, 0.1, 0.1, 0.1]}},
            shifted_matrix
        )
        
        # Should detect structural shift
        shift_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.STRUCTURAL_SHIFT]
        assert len(shift_anomalies) > 0
        
    def test_emergent_structure_detection(self, visualizer):
        """Test detection of emergent structures in pattern space."""
        # Create multiple related patterns
        pattern_metrics = {
            'pattern6': {
                'coherence': 0.8,
                'success_rate': 0.75,
                'stability': 0.8,
                'emergence_potential': 0.3
            },
            'pattern7': {
                'coherence': 0.75,
                'success_rate': 0.7,
                'stability': 0.75,
                'emergence_potential': 0.35
            }
        }
        
        visualizer.update_pattern_space(pattern_metrics, {})
        
        # Detect emergent structures
        structures = visualizer.detect_emergent_structures()
        
        # Should find related patterns
        assert len(structures) > 0
        structure = structures[0]
        assert 'core_pattern' in structure
        assert 'related_patterns' in structure
        assert structure['coherence'] > 0.7
