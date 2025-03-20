"""Tests for topology dashboard functionality.

This module tests the dashboard for monitoring topology evolution, including metrics
display and interactive elements.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime, timedelta
import os
import tempfile
from typing import Dict, List, Tuple, Any, Optional

from habitat_evolution.pattern_aware_rag.topology.models import (
    FrequencyDomain,
    Boundary,
    ResonancePoint,
    TopologyState,
    FieldMetrics,
    TopologyDiff
)
from habitat_evolution.visualization.dashboard.topology_dashboard import TopologyDashboard


class TestTopologyDashboard:
    """Test suite for topology dashboard."""
    
    @pytest.fixture
    def sample_topology_states(self) -> List[TopologyState]:
        """Create a sequence of topology states for testing."""
        states = []
        
        # Create base state
        base_state = TopologyState(
            id="ts-1",
            frequency_domains={
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
                )
            },
            boundaries={
                "b-1": Boundary(
                    id="b-1",
                    domain_ids=("fd-1", "fd-2"),
                    sharpness=0.6,
                    permeability=0.4,
                    stability=0.7,
                    dimensionality=2,
                    coordinates=[(0.4, 0.3), (0.4, 0.4)]
                )
            },
            resonance_points={
                "r-1": ResonancePoint(
                    id="r-1",
                    coordinates=(0.25, 0.25),
                    strength=0.9,
                    stability=0.85,
                    attractor_radius=0.1,
                    contributing_pattern_ids={"p-1": 0.7, "p-2": 0.8}
                )
            },
            field_metrics=FieldMetrics(
                coherence=0.75,
                energy_density={"fd-1": 0.6, "fd-2": 0.7},
                adaptation_rate=0.4,
                homeostasis_index=0.8,
                entropy=0.3
            ),
            timestamp=datetime.now() - timedelta(days=3)
        )
        states.append(base_state)
        
        # Create state 2 (day 2)
        state2 = TopologyState(
            id="ts-2",
            frequency_domains={
                "fd-1": FrequencyDomain(
                    id="fd-1",
                    dominant_frequency=0.32,
                    bandwidth=0.12,
                    phase_coherence=0.82,
                    center_coordinates=(0.22, 0.32),
                    radius=0.22,
                    pattern_ids={"p-1", "p-2", "p-3", "p-9"}
                ),
                "fd-2": FrequencyDomain(
                    id="fd-2",
                    dominant_frequency=0.48,
                    bandwidth=0.14,
                    phase_coherence=0.72,
                    center_coordinates=(0.58, 0.42),
                    radius=0.24,
                    pattern_ids={"p-4", "p-5", "p-6"}
                )
            },
            boundaries={
                "b-1": Boundary(
                    id="b-1",
                    domain_ids=("fd-1", "fd-2"),
                    sharpness=0.62,
                    permeability=0.42,
                    stability=0.72,
                    dimensionality=2,
                    coordinates=[(0.4, 0.32), (0.4, 0.42)]
                )
            },
            resonance_points={
                "r-1": ResonancePoint(
                    id="r-1",
                    coordinates=(0.26, 0.26),
                    strength=0.92,
                    stability=0.86,
                    attractor_radius=0.11,
                    contributing_pattern_ids={"p-1": 0.72, "p-2": 0.82, "p-9": 0.65}
                )
            },
            field_metrics=FieldMetrics(
                coherence=0.78,
                energy_density={"fd-1": 0.65, "fd-2": 0.68},
                adaptation_rate=0.42,
                homeostasis_index=0.82,
                entropy=0.28
            ),
            timestamp=datetime.now() - timedelta(days=2)
        )
        states.append(state2)
        
        # Create state 3 (day 1)
        state3 = TopologyState(
            id="ts-3",
            frequency_domains={
                "fd-1": FrequencyDomain(
                    id="fd-1",
                    dominant_frequency=0.34,
                    bandwidth=0.13,
                    phase_coherence=0.84,
                    center_coordinates=(0.24, 0.34),
                    radius=0.23,
                    pattern_ids={"p-1", "p-2", "p-3", "p-9"}
                ),
                "fd-2": FrequencyDomain(
                    id="fd-2",
                    dominant_frequency=0.46,
                    bandwidth=0.13,
                    phase_coherence=0.74,
                    center_coordinates=(0.56, 0.44),
                    radius=0.23,
                    pattern_ids={"p-4", "p-5", "p-6", "p-10"}
                ),
                "fd-3": FrequencyDomain(
                    id="fd-3",
                    dominant_frequency=0.7,
                    bandwidth=0.1,
                    phase_coherence=0.9,
                    center_coordinates=(0.8, 0.8),
                    radius=0.15,
                    pattern_ids={"p-11", "p-12"}
                )
            },
            boundaries={
                "b-1": Boundary(
                    id="b-1",
                    domain_ids=("fd-1", "fd-2"),
                    sharpness=0.64,
                    permeability=0.44,
                    stability=0.74,
                    dimensionality=2,
                    coordinates=[(0.4, 0.34), (0.4, 0.44)]
                ),
                "b-2": Boundary(
                    id="b-2",
                    domain_ids=("fd-2", "fd-3"),
                    sharpness=0.7,
                    permeability=0.3,
                    stability=0.8,
                    dimensionality=2,
                    coordinates=[(0.68, 0.62), (0.68, 0.72)]
                )
            },
            resonance_points={
                "r-1": ResonancePoint(
                    id="r-1",
                    coordinates=(0.27, 0.27),
                    strength=0.94,
                    stability=0.87,
                    attractor_radius=0.12,
                    contributing_pattern_ids={"p-1": 0.74, "p-2": 0.84, "p-9": 0.7}
                ),
                "r-2": ResonancePoint(
                    id="r-2",
                    coordinates=(0.82, 0.82),
                    strength=0.95,
                    stability=0.9,
                    attractor_radius=0.08,
                    contributing_pattern_ids={"p-11": 0.9, "p-12": 0.85}
                )
            },
            field_metrics=FieldMetrics(
                coherence=0.82,
                energy_density={"fd-1": 0.7, "fd-2": 0.65, "fd-3": 0.85},
                adaptation_rate=0.45,
                homeostasis_index=0.84,
                entropy=0.25
            ),
            timestamp=datetime.now() - timedelta(days=1)
        )
        states.append(state3)
        
        # Create state 4 (current)
        state4 = TopologyState(
            id="ts-4",
            frequency_domains={
                "fd-1": FrequencyDomain(
                    id="fd-1",
                    dominant_frequency=0.36,
                    bandwidth=0.14,
                    phase_coherence=0.86,
                    center_coordinates=(0.26, 0.36),
                    radius=0.24,
                    pattern_ids={"p-1", "p-2", "p-3", "p-9", "p-13"}
                ),
                "fd-2": FrequencyDomain(
                    id="fd-2",
                    dominant_frequency=0.44,
                    bandwidth=0.12,
                    phase_coherence=0.76,
                    center_coordinates=(0.54, 0.46),
                    radius=0.22,
                    pattern_ids={"p-4", "p-5", "p-6", "p-10"}
                ),
                "fd-3": FrequencyDomain(
                    id="fd-3",
                    dominant_frequency=0.72,
                    bandwidth=0.11,
                    phase_coherence=0.92,
                    center_coordinates=(0.78, 0.78),
                    radius=0.16,
                    pattern_ids={"p-11", "p-12", "p-14"}
                )
            },
            boundaries={
                "b-1": Boundary(
                    id="b-1",
                    domain_ids=("fd-1", "fd-2"),
                    sharpness=0.66,
                    permeability=0.46,
                    stability=0.76,
                    dimensionality=2,
                    coordinates=[(0.4, 0.36), (0.4, 0.46)]
                ),
                "b-2": Boundary(
                    id="b-2",
                    domain_ids=("fd-2", "fd-3"),
                    sharpness=0.72,
                    permeability=0.32,
                    stability=0.82,
                    dimensionality=2,
                    coordinates=[(0.66, 0.62), (0.66, 0.72)]
                ),
                "b-3": Boundary(
                    id="b-3",
                    domain_ids=("fd-3", "fd-1"),
                    sharpness=0.6,
                    permeability=0.5,
                    stability=0.7,
                    dimensionality=2,
                    coordinates=[(0.52, 0.57), (0.52, 0.67)]
                )
            },
            resonance_points={
                "r-1": ResonancePoint(
                    id="r-1",
                    coordinates=(0.28, 0.28),
                    strength=0.96,
                    stability=0.88,
                    attractor_radius=0.13,
                    contributing_pattern_ids={"p-1": 0.76, "p-2": 0.86, "p-9": 0.75, "p-13": 0.8}
                ),
                "r-2": ResonancePoint(
                    id="r-2",
                    coordinates=(0.8, 0.8),
                    strength=0.97,
                    stability=0.92,
                    attractor_radius=0.09,
                    contributing_pattern_ids={"p-11": 0.92, "p-12": 0.87, "p-14": 0.85}
                ),
                "r-3": ResonancePoint(
                    id="r-3",
                    coordinates=(0.5, 0.5),
                    strength=0.85,
                    stability=0.8,
                    attractor_radius=0.1,
                    contributing_pattern_ids={"p-5": 0.7, "p-10": 0.75, "p-13": 0.65}
                )
            },
            field_metrics=FieldMetrics(
                coherence=0.85,
                energy_density={"fd-1": 0.75, "fd-2": 0.7, "fd-3": 0.9},
                adaptation_rate=0.48,
                homeostasis_index=0.86,
                entropy=0.22
            ),
            timestamp=datetime.now()
        )
        states.append(state4)
        
        return states
    
    @pytest.fixture
    def topology_dashboard(self) -> TopologyDashboard:
        """Create a topology dashboard instance."""
        return TopologyDashboard()
    
    def test_dashboard_metrics(self, topology_dashboard, sample_topology_states):
        """Test that metrics display correctly in the dashboard."""
        # Load states into dashboard
        topology_dashboard.load_states(sample_topology_states)
        
        # Generate metrics panel
        fig = topology_dashboard.generate_metrics_panel()
        
        # Check that the figure is created
        assert isinstance(fig, Figure)
        
        # Check that metrics data is populated
        assert len(topology_dashboard.metrics_data["timestamps"]) == len(sample_topology_states)
        assert len(topology_dashboard.metrics_data["coherence"]) == len(sample_topology_states)
        assert len(topology_dashboard.metrics_data["entropy"]) == len(sample_topology_states)
        
        # Save to temporary file to verify output
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            os.unlink(tmp.name)
    
    def test_dashboard_interactive_elements(self, topology_dashboard, sample_topology_states):
        """Test interactive elements of the dashboard."""
        # Load states into dashboard
        topology_dashboard.load_states(sample_topology_states)
        
        # Generate interactive dashboard
        fig = topology_dashboard.generate_interactive_dashboard()
        
        # Check that the figure is created
        assert isinstance(fig, Figure)
        
        # Check that interactive elements are created
        assert topology_dashboard.interactive_elements is not None
        assert len(topology_dashboard.interactive_elements) > 0
        
        # Save to temporary file to verify output
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            os.unlink(tmp.name)
    
    def test_time_series_visualization(self, topology_dashboard, sample_topology_states):
        """Test time-series visualization of topology evolution."""
        # Load states into dashboard
        topology_dashboard.load_states(sample_topology_states)
        
        # Generate time-series visualization
        fig = topology_dashboard.visualize_time_series()
        
        # Check that the figure is created
        assert isinstance(fig, Figure)
        
        # Check that time-series data is populated
        assert len(topology_dashboard.time_series_data["domain_count"]) == len(sample_topology_states)
        assert len(topology_dashboard.time_series_data["boundary_count"]) == len(sample_topology_states)
        assert len(topology_dashboard.time_series_data["resonance_count"]) == len(sample_topology_states)
        
        # Save to temporary file to verify output
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            os.unlink(tmp.name)
    
    def test_domain_evolution_tracking(self, topology_dashboard, sample_topology_states):
        """Test tracking of domain evolution over time."""
        # Load states into dashboard
        topology_dashboard.load_states(sample_topology_states)
        
        # Track evolution of a specific domain
        domain_id = "fd-1"
        fig = topology_dashboard.track_domain_evolution(domain_id)
        
        # Check that the figure is created
        assert isinstance(fig, Figure)
        
        # Check that domain evolution data is populated
        assert domain_id in topology_dashboard.domain_evolution_data
        assert len(topology_dashboard.domain_evolution_data[domain_id]["timestamps"]) > 0
        
        # Save to temporary file to verify output
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            os.unlink(tmp.name)
    
    def test_boundary_stability_visualization(self, topology_dashboard, sample_topology_states):
        """Test visualization of boundary stability over time."""
        # Load states into dashboard
        topology_dashboard.load_states(sample_topology_states)
        
        # Visualize boundary stability
        fig = topology_dashboard.visualize_boundary_stability()
        
        # Check that the figure is created
        assert isinstance(fig, Figure)
        
        # Check that boundary stability data is populated
        assert len(topology_dashboard.boundary_stability_data) > 0
        
        # Save to temporary file to verify output
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            os.unlink(tmp.name)
    
    def test_resonance_strength_visualization(self, topology_dashboard, sample_topology_states):
        """Test visualization of resonance point strength over time."""
        # Load states into dashboard
        topology_dashboard.load_states(sample_topology_states)
        
        # Visualize resonance strength
        fig = topology_dashboard.visualize_resonance_strength()
        
        # Check that the figure is created
        assert isinstance(fig, Figure)
        
        # Check that resonance strength data is populated
        assert len(topology_dashboard.resonance_strength_data) > 0
        
        # Save to temporary file to verify output
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            os.unlink(tmp.name)
    
    def test_full_dashboard(self, topology_dashboard, sample_topology_states):
        """Test generation of the full dashboard with all components."""
        # Load states into dashboard
        topology_dashboard.load_states(sample_topology_states)
        
        # Generate full dashboard
        fig = topology_dashboard.generate_dashboard()
        
        # Check that the figure is created
        assert isinstance(fig, Figure)
        
        # Save to temporary file to verify output
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            os.unlink(tmp.name)
