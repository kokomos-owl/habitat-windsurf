"""Tests for topology visualization components.

This module tests the visualization of topology components including frequency domains,
boundaries, and resonance points.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import os
import tempfile

from habitat_evolution.pattern_aware_rag.topology.models import (
    FrequencyDomain,
    Boundary,
    ResonancePoint,
    TopologyState,
    FieldMetrics
)
from habitat_evolution.visualization.topology_visualizer import TopologyVisualizer


class TestTopologyVisualizer:
    """Test suite for topology visualization."""
    
    @pytest.fixture
    def sample_topology_state(self) -> TopologyState:
        """Create a sample topology state for testing."""
        # Create frequency domains
        domains = {
            "fd-1": FrequencyDomain(
                id="fd-1",
                dominant_frequency=0.3,
                bandwidth=0.1,
                phase_coherence=0.8,
                center_coordinates=(0.2, 0.3),
                radius=0.2,
                pattern_ids={"p-1", "p-2", "p-3"}
            ),
            "fd-2": FrequencyDomain(
                id="fd-2",
                dominant_frequency=0.5,
                bandwidth=0.15,
                phase_coherence=0.7,
                center_coordinates=(0.6, 0.4),
                radius=0.25,
                pattern_ids={"p-4", "p-5", "p-6"}
            ),
            "fd-3": FrequencyDomain(
                id="fd-3",
                dominant_frequency=0.7,
                bandwidth=0.12,
                phase_coherence=0.75,
                center_coordinates=(0.4, 0.7),
                radius=0.18,
                pattern_ids={"p-7", "p-8"}
            )
        }
        
        # Create boundaries
        boundaries = {
            "b-1": Boundary(
                id="b-1",
                domain_ids=("fd-1", "fd-2"),
                sharpness=0.6,
                permeability=0.4,
                stability=0.7,
                dimensionality=2,
                coordinates=[
                    (0.4, 0.3),
                    (0.4, 0.4)
                ]
            ),
            "b-2": Boundary(
                id="b-2",
                domain_ids=("fd-2", "fd-3"),
                sharpness=0.5,
                permeability=0.6,
                stability=0.65,
                dimensionality=2,
                coordinates=[
                    (0.5, 0.5),
                    (0.5, 0.6)
                ]
            ),
            "b-3": Boundary(
                id="b-3",
                domain_ids=("fd-3", "fd-1"),
                sharpness=0.7,
                permeability=0.3,
                stability=0.8,
                dimensionality=2,
                coordinates=[
                    (0.3, 0.5),
                    (0.3, 0.6)
                ]
            )
        }
        
        # Create resonance points
        resonance_points = {
            "r-1": ResonancePoint(
                id="r-1",
                coordinates=(0.25, 0.25),
                strength=0.9,
                stability=0.85,
                attractor_radius=0.1,
                contributing_pattern_ids={"p-1": 0.7, "p-2": 0.8, "p-3": 0.6}
            ),
            "r-2": ResonancePoint(
                id="r-2",
                coordinates=(0.65, 0.45),
                strength=0.8,
                stability=0.75,
                attractor_radius=0.12,
                contributing_pattern_ids={"p-4": 0.6, "p-5": 0.9, "p-6": 0.7}
            ),
            "r-3": ResonancePoint(
                id="r-3",
                coordinates=(0.45, 0.75),
                strength=0.85,
                stability=0.8,
                attractor_radius=0.08,
                contributing_pattern_ids={"p-7": 0.8, "p-8": 0.75}
            )
        }
        
        # Create field metrics
        field_metrics = FieldMetrics(
            coherence=0.75,
            energy_density={"fd-1": 0.6, "fd-2": 0.7, "fd-3": 0.65},
            adaptation_rate=0.4,
            homeostasis_index=0.8,
            entropy=0.3
        )
        
        # Create topology state
        topology_state = TopologyState(
            id="ts-1",
            frequency_domains=domains,
            boundaries=boundaries,
            resonance_points=resonance_points,
            field_metrics=field_metrics
        )
        
        return topology_state
    
    @pytest.fixture
    def topology_visualizer(self) -> TopologyVisualizer:
        """Create a topology visualizer instance."""
        return TopologyVisualizer()
    
    def test_frequency_domain_visualization(self, topology_visualizer, sample_topology_state):
        """Test visualization of frequency domains."""
        # Visualize frequency domains
        fig = topology_visualizer.visualize_frequency_domains(sample_topology_state)
        
        # Check that the figure is created
        assert isinstance(fig, Figure)
        
        # Check that all domains are visualized
        for domain_id in sample_topology_state.frequency_domains:
            assert domain_id in topology_visualizer.visualization_data["domains"]
        
        # Save to temporary file to verify output
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            os.unlink(tmp.name)
    
    def test_boundary_visualization(self, topology_visualizer, sample_topology_state):
        """Test visualization of boundaries between domains."""
        # Visualize boundaries
        fig = topology_visualizer.visualize_boundaries(sample_topology_state)
        
        # Check that the figure is created
        assert isinstance(fig, Figure)
        
        # Check that all boundaries are visualized
        for boundary_id in sample_topology_state.boundaries:
            assert boundary_id in topology_visualizer.visualization_data["boundaries"]
        
        # Save to temporary file to verify output
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            os.unlink(tmp.name)
    
    def test_resonance_point_visualization(self, topology_visualizer, sample_topology_state):
        """Test visualization of resonance points."""
        # Visualize resonance points
        fig = topology_visualizer.visualize_resonance_points(sample_topology_state)
        
        # Check that the figure is created
        assert isinstance(fig, Figure)
        
        # Check that all resonance points are visualized
        for point_id in sample_topology_state.resonance_points:
            assert point_id in topology_visualizer.visualization_data["resonance_points"]
        
        # Save to temporary file to verify output
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            os.unlink(tmp.name)
    
    def test_complete_topology_visualization(self, topology_visualizer, sample_topology_state):
        """Test visualization of complete topology state."""
        # Visualize complete topology
        fig = topology_visualizer.visualize_topology(sample_topology_state)
        
        # Check that the figure is created
        assert isinstance(fig, Figure)
        
        # Check that all components are visualized
        for domain_id in sample_topology_state.frequency_domains:
            assert domain_id in topology_visualizer.visualization_data["domains"]
        
        for boundary_id in sample_topology_state.boundaries:
            assert boundary_id in topology_visualizer.visualization_data["boundaries"]
        
        for point_id in sample_topology_state.resonance_points:
            assert point_id in topology_visualizer.visualization_data["resonance_points"]
        
        # Save to temporary file to verify output
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            os.unlink(tmp.name)
    
    def test_topology_evolution_visualization(self, topology_visualizer, sample_topology_state):
        """Test visualization of topology evolution over time."""
        # Create a sequence of topology states
        states = [sample_topology_state]
        
        # Create a modified state
        modified_state = sample_topology_state.to_dict()
        modified_state["id"] = "ts-2"
        modified_state["timestamp"] = datetime.now()
        
        # Modify a domain
        modified_state["frequency_domains"]["fd-1"]["radius"] = 0.25
        modified_state["frequency_domains"]["fd-1"]["phase_coherence"] = 0.85
        
        # Add a new domain
        modified_state["frequency_domains"]["fd-4"] = {
            "id": "fd-4",
            "dominant_frequency": 0.4,
            "bandwidth": 0.08,
            "phase_coherence": 0.9,
            "center_coordinates": (0.8, 0.8),
            "radius": 0.15,
            "pattern_ids": ["p-9", "p-10"],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "metadata": {}
        }
        
        # Convert back to TopologyState
        second_state = TopologyState.from_dict(modified_state)
        states.append(second_state)
        
        # Visualize evolution
        fig = topology_visualizer.visualize_topology_evolution(states)
        
        # Check that the figure is created
        assert isinstance(fig, Figure)
        
        # Save to temporary file to verify output
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            os.unlink(tmp.name)
    
    def test_interactive_visualization(self, topology_visualizer, sample_topology_state):
        """Test interactive visualization capabilities."""
        # This test is a placeholder for interactive visualization features
        # In a real implementation, we would test the interactive elements
        
        # For now, just ensure the method exists and returns a figure
        fig = topology_visualizer.create_interactive_visualization(sample_topology_state)
        
        # Check that the figure is created
        assert isinstance(fig, Figure)
        
        # Save to temporary file to verify output
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            os.unlink(tmp.name)
