"""
Eigenspace Window Management Test Suite for Pattern-Aware RAG.

This test suite observes how the system detects and adapts to natural boundaries
in semantic spaces through eigendecomposition analysis.

Testing Philosophy:
- Observe natural boundaries without imposing structure
- Measure boundary properties rather than setting them
- Adapt windows to match the natural semantic structure
- Validate against known semantic patterns
"""

import pytest
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Add src to path if not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import modules using relative imports
sys.path.append(os.path.join(src_path, 'src'))
from habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from habitat_evolution.pattern_aware_rag.learning.window_manager import LearningWindowManager
from habitat_evolution.pattern_aware_rag.learning.learning_control import LearningWindow
from habitat_evolution.pattern_aware_rag.learning.eigenspace_window_manager import EigenspaceWindowManager

class TestEigenspaceWindowManager:
    """Test suite for observing eigenspace-based window management."""
    
    @pytest.fixture
    def field_analyzer(self):
        """Initialize field analyzer for testing."""
        return TopologicalFieldAnalyzer()
    
    @pytest.fixture
    def window_manager(self):
        """Initialize window manager for testing."""
        return LearningWindowManager()
    
    @pytest.fixture
    def eigenspace_manager(self, field_analyzer, window_manager):
        """Initialize eigenspace window manager for testing."""
        return EigenspaceWindowManager(
            field_analyzer=field_analyzer,
            window_manager=window_manager
        )
    
    @pytest.fixture
    def synthetic_data_single_cluster(self):
        """Generate synthetic data with a single coherent cluster."""
        np.random.seed(42)
        return np.random.randn(20, 5) + np.array([5, 0, 0, 0, 0])
    
    @pytest.fixture
    def synthetic_data_multiple_clusters(self):
        """Generate synthetic data with multiple distinct clusters."""
        np.random.seed(42)
        cluster1 = np.random.randn(10, 5) + np.array([5, 0, 0, 0, 0])
        cluster2 = np.random.randn(8, 5) + np.array([0, 5, 0, 0, 0])
        cluster3 = np.random.randn(12, 5) + np.array([0, 0, 5, 0, 0])
        return np.vstack([cluster1, cluster2, cluster3])
    
    def test_detect_natural_boundaries_single_cluster(self, eigenspace_manager, synthetic_data_single_cluster):
        """
        Test detection of natural boundaries in a single coherent cluster.
        
        Expected behavior:
        - Should detect a single window covering the entire cluster
        - No significant spectral gaps should be found
        """
        boundaries = eigenspace_manager.detect_natural_boundaries(synthetic_data_single_cluster)
        
        # Should find a single window covering all data
        assert len(boundaries) == 1
        assert boundaries[0][0] == 0
        assert boundaries[0][1] == len(synthetic_data_single_cluster)
        
        # Eigenvalues should be stored
        assert eigenspace_manager.previous_eigenvalues is not None
        assert eigenspace_manager.previous_eigenvectors is not None
    
    def test_detect_natural_boundaries_multiple_clusters(self, eigenspace_manager, synthetic_data_multiple_clusters):
        """
        Test detection of natural boundaries in multiple distinct clusters.
        
        Expected behavior:
        - Should detect multiple windows corresponding to the natural clusters
        - Significant spectral gaps should be found between clusters
        """
        boundaries = eigenspace_manager.detect_natural_boundaries(synthetic_data_multiple_clusters)
        
        # Should find multiple windows
        assert len(boundaries) > 1
        
        # First window should start at index 0
        assert boundaries[0][0] == 0
        
        # Last window should end at the data length
        assert boundaries[-1][1] == len(synthetic_data_multiple_clusters)
        
        # All windows should meet minimum size requirement
        for start, end in boundaries:
            assert end - start >= eigenspace_manager.min_window_size
    
    def test_adapt_windows_to_semantic_structure(self, eigenspace_manager, synthetic_data_multiple_clusters):
        """
        Test adaptation of learning windows to match semantic structure.
        
        Expected behavior:
        - Should create windows with parameters derived from measurements
        - Windows should align with natural boundaries
        - Parameters should reflect boundary fuzziness and eigenvalue strength
        """
        windows = eigenspace_manager.adapt_windows_to_semantic_structure(
            synthetic_data_multiple_clusters,
            base_duration_minutes=30
        )
        
        # Should create multiple windows
        assert len(windows) > 1
        
        # All windows should have eigenspace properties
        for window in windows:
            assert hasattr(window, 'eigenspace_boundaries')
            assert hasattr(window, 'eigenspace_size')
            assert hasattr(window, 'eigenspace_relative_size')
            assert hasattr(window, 'boundary_fuzziness')
            assert hasattr(window, 'eigenvalue_strength')
            
            # Parameters should be derived from measurements
            assert 0.0 <= window.boundary_fuzziness <= 1.0
            assert 0.0 <= window.eigenvalue_strength <= 1.0
            assert 0.5 <= window.stability_threshold <= 0.9
            assert 0.4 <= window.coherence_threshold <= 0.8
            
            # Window duration should be proportional to size
            duration_minutes = (window.end_time - window.start_time).total_seconds() / 60
            assert duration_minutes >= 5
    
    def test_measure_boundary_fuzziness(self, eigenspace_manager, synthetic_data_multiple_clusters):
        """
        Test measurement of boundary fuzziness.
        
        Expected behavior:
        - Should return higher fuzziness for boundaries between clusters
        - Should return lower fuzziness for coherent regions
        """
        # First detect boundaries
        boundaries = eigenspace_manager.detect_natural_boundaries(synthetic_data_multiple_clusters)
        
        # Measure fuzziness for each boundary
        fuzziness_values = []
        for start, end in boundaries:
            fuzziness = eigenspace_manager._measure_boundary_fuzziness(
                start, end, synthetic_data_multiple_clusters)
            fuzziness_values.append(fuzziness)
        
        # All values should be in valid range
        for fuzziness in fuzziness_values:
            assert 0.0 <= fuzziness <= 1.0
    
    def test_measure_eigenvalue_strength(self, eigenspace_manager, synthetic_data_multiple_clusters):
        """
        Test measurement of eigenvalue strength.
        
        Expected behavior:
        - Should return higher strength for windows with dominant eigenvalues
        - Should return lower strength for windows with weak eigenvalues
        """
        # First detect boundaries to set previous_eigenvalues
        boundaries = eigenspace_manager.detect_natural_boundaries(synthetic_data_multiple_clusters)
        
        # Measure strength for each boundary
        strength_values = []
        for start, end in boundaries:
            strength = eigenspace_manager._measure_eigenvalue_strength(start, end)
            strength_values.append(strength)
        
        # All values should be in valid range
        for strength in strength_values:
            assert 0.0 <= strength <= 1.0