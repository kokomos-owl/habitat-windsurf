"""Test script for demonstrating the topology dashboard functionality.

This script creates sample topology states and demonstrates the visualization
capabilities of the TopologyDashboard class.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import unittest
from typing import Dict, List, Tuple

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from habitat_evolution.pattern_aware_rag.topology.models import (
    FrequencyDomain,
    Boundary,
    ResonancePoint,
    TopologyState,
    FieldMetrics
)
from habitat_evolution.visualization.dashboard.topology_dashboard import TopologyDashboard


class TestTopologyDashboardDemo(unittest.TestCase):
    """Test class for demonstrating the topology dashboard functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample topology states
        self.topology_states = self._create_sample_topology_states()
        
        # Initialize the dashboard
        self.dashboard = TopologyDashboard()
        self.dashboard.load_states(self.topology_states)
    
    def _create_sample_topology_states(self, num_states: int = 5) -> List[TopologyState]:
        """Create sample topology states for testing.
        
        Args:
            num_states: Number of states to create
            
        Returns:
            List of topology states
        """
        states = []
        
        # Create states with evolving properties
        for i in range(num_states):
            # Create timestamp
            timestamp = datetime.now() - timedelta(days=num_states-i-1)
            
            # Create frequency domains
            domains = {}
            for j in range(3):
                # Domain properties evolve over time
                frequency = 0.2 + 0.1 * j + 0.02 * i
                bandwidth = 0.1 + 0.01 * i
                coherence = 0.5 + 0.05 * i
                radius = 0.15 + 0.01 * j
                
                # Create domain
                domain = FrequencyDomain(
                    id=f"domain-{j}",
                    dominant_frequency=frequency,
                    bandwidth=bandwidth,
                    phase_coherence=coherence,
                    center_coordinates=(0.25 + 0.25 * j, 0.5),
                    radius=radius,
                    pattern_ids=[f"pattern-{j}-{k}" for k in range(3)]
                )
                domains[domain.id] = domain
            
            # Create boundaries
            boundaries = {}
            for j in range(2):
                # Boundary properties evolve over time
                sharpness = 0.3 + 0.1 * i
                permeability = 0.7 - 0.05 * i
                stability = 0.4 + 0.1 * i
                
                # Create boundary
                boundary = Boundary(
                    id=f"boundary-{j}",
                    domain_ids=(f"domain-{j}", f"domain-{j+1}"),
                    sharpness=sharpness,
                    permeability=permeability,
                    stability=stability,
                    coordinates=[
                        (0.25 + 0.25 * j + 0.125, 0.3),
                        (0.25 + 0.25 * j + 0.125, 0.7)
                    ]
                )
                boundaries[boundary.id] = boundary
            
            # Create resonance points
            resonance_points = {}
            for j in range(2):
                # Resonance point properties evolve over time
                strength = 0.4 + 0.1 * i
                stability = 0.5 + 0.05 * i
                radius = 0.1 + 0.01 * i
                
                # Create resonance point
                point = ResonancePoint(
                    id=f"resonance-{j}",
                    coordinates=(0.25 + 0.5 * j, 0.5),
                    strength=strength,
                    stability=stability,
                    attractor_radius=radius,
                    contributing_pattern_ids=[f"pattern-{j}-{k}" for k in range(2)]
                )
                resonance_points[point.id] = point
            
            # Create field metrics
            metrics = FieldMetrics(
                coherence=0.5 + 0.1 * i,
                entropy=0.5 - 0.05 * i,
                adaptation_rate=0.3 + 0.05 * i,
                homeostasis_index=0.6 + 0.02 * i
            )
            
            # Create topology state
            state = TopologyState(
                id=f"state-{i}",
                timestamp=timestamp,
                frequency_domains=domains,
                boundaries=boundaries,
                resonance_points=resonance_points,
                field_metrics=metrics
            )
            
            states.append(state)
        
        return states
    
    def test_metrics_panel(self):
        """Test the metrics panel visualization."""
        # Generate metrics panel
        fig = self.dashboard.generate_metrics_panel()
        
        # Save the figure
        output_dir = os.path.join(os.path.dirname(__file__), '../../output/visualization')
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'metrics_panel.png'), dpi=300, bbox_inches='tight')
        
        plt.close(fig)
    
    def test_time_series(self):
        """Test the time series visualization."""
        # Generate time series visualization
        fig = self.dashboard.visualize_time_series()
        
        # Save the figure
        output_dir = os.path.join(os.path.dirname(__file__), '../../output/visualization')
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'time_series.png'), dpi=300, bbox_inches='tight')
        
        plt.close(fig)
    
    def test_domain_evolution(self):
        """Test the domain evolution visualization."""
        # Generate domain evolution visualization
        fig = self.dashboard.track_domain_evolution('domain-0')
        
        # Save the figure
        output_dir = os.path.join(os.path.dirname(__file__), '../../output/visualization')
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'domain_evolution.png'), dpi=300, bbox_inches='tight')
        
        plt.close(fig)
    
    def test_boundary_stability(self):
        """Test the boundary stability visualization."""
        # Generate boundary stability visualization
        fig = self.dashboard.visualize_boundary_stability()
        
        # Save the figure
        output_dir = os.path.join(os.path.dirname(__file__), '../../output/visualization')
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'boundary_stability.png'), dpi=300, bbox_inches='tight')
        
        plt.close(fig)
    
    def test_resonance_strength(self):
        """Test the resonance strength visualization."""
        # Generate resonance strength visualization
        fig = self.dashboard.visualize_resonance_strength()
        
        # Save the figure
        output_dir = os.path.join(os.path.dirname(__file__), '../../output/visualization')
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'resonance_strength.png'), dpi=300, bbox_inches='tight')
        
        plt.close(fig)
    
    def test_dashboard(self):
        """Test the complete dashboard visualization."""
        # Generate complete dashboard
        fig = self.dashboard.generate_dashboard()
        
        # Save the figure
        output_dir = os.path.join(os.path.dirname(__file__), '../../output/visualization')
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'complete_dashboard.png'), dpi=300, bbox_inches='tight')
        
        plt.close(fig)


if __name__ == '__main__':
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '../../output/visualization')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the tests
    unittest.main()
