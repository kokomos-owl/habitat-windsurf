"""Topology Dashboard for monitoring topology evolution.

This module provides a dashboard for visualizing topology metrics and evolution
over time, including interactive elements for exploring topology states.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.widgets import Slider, Button, RadioButtons
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime, timedelta
import os

from ...pattern_aware_rag.topology.models import (
    FrequencyDomain,
    Boundary,
    ResonancePoint,
    TopologyState,
    FieldMetrics,
    TopologyDiff
)
from ..topology_visualizer import TopologyVisualizer


class TopologyDashboard:
    """Dashboard for monitoring topology evolution and metrics.
    
    This class provides methods for visualizing topology metrics, evolution,
    and interactive exploration of topology states.
    """
    
    def __init__(self):
        """Initialize the topology dashboard."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize topology visualizer
        self.topology_visualizer = TopologyVisualizer()
        
        # Store topology states
        self.topology_states = []
        
        # Store metrics data for visualization
        self.metrics_data = {
            "timestamps": [],
            "coherence": [],
            "entropy": [],
            "adaptation_rate": [],
            "homeostasis_index": []
        }
        
        # Store time-series data for visualization
        self.time_series_data = {
            "domain_count": [],
            "boundary_count": [],
            "resonance_count": []
        }
        
        # Store domain evolution data
        self.domain_evolution_data = {}
        
        # Store boundary stability data
        self.boundary_stability_data = {}
        
        # Store resonance strength data
        self.resonance_strength_data = {}
        
        # Store interactive elements
        self.interactive_elements = {}
        
        # Default figure size
        self.figsize = (14, 10)
    
    def load_states(self, topology_states: List[TopologyState]):
        """Load topology states into the dashboard.
        
        Args:
            topology_states: List of topology states to load
        """
        # Store states
        self.topology_states = sorted(topology_states, key=lambda state: state.timestamp)
        
        # Extract metrics data
        self._extract_metrics_data()
        
        # Extract time-series data
        self._extract_time_series_data()
        
        # Extract domain evolution data
        self._extract_domain_evolution_data()
        
        # Extract boundary stability data
        self._extract_boundary_stability_data()
        
        # Extract resonance strength data
        self._extract_resonance_strength_data()
        
        self.logger.info(f"Loaded {len(topology_states)} topology states into dashboard")
    
    def _extract_metrics_data(self):
        """Extract metrics data from topology states."""
        # Clear existing data
        self.metrics_data = {
            "timestamps": [],
            "coherence": [],
            "entropy": [],
            "adaptation_rate": [],
            "homeostasis_index": []
        }
        
        # Extract metrics from each state
        for state in self.topology_states:
            if state.field_metrics:
                self.metrics_data["timestamps"].append(state.timestamp)
                self.metrics_data["coherence"].append(state.field_metrics.coherence)
                self.metrics_data["entropy"].append(state.field_metrics.entropy)
                self.metrics_data["adaptation_rate"].append(state.field_metrics.adaptation_rate)
                self.metrics_data["homeostasis_index"].append(state.field_metrics.homeostasis_index)
    
    def _extract_time_series_data(self):
        """Extract time-series data from topology states."""
        # Clear existing data
        self.time_series_data = {
            "timestamps": [],
            "domain_count": [],
            "boundary_count": [],
            "resonance_count": []
        }
        
        # Extract counts from each state
        for state in self.topology_states:
            self.time_series_data["timestamps"].append(state.timestamp)
            self.time_series_data["domain_count"].append(len(state.frequency_domains))
            self.time_series_data["boundary_count"].append(len(state.boundaries))
            self.time_series_data["resonance_count"].append(len(state.resonance_points))
    
    def _extract_domain_evolution_data(self):
        """Extract domain evolution data from topology states."""
        # Clear existing data
        self.domain_evolution_data = {}
        
        # Collect all domain IDs
        all_domain_ids = set()
        for state in self.topology_states:
            all_domain_ids.update(state.frequency_domains.keys())
        
        # Initialize data for each domain
        for domain_id in all_domain_ids:
            self.domain_evolution_data[domain_id] = {
                "timestamps": [],
                "dominant_frequency": [],
                "bandwidth": [],
                "phase_coherence": [],
                "radius": [],
                "pattern_count": []
            }
        
        # Extract data for each domain
        for state in self.topology_states:
            for domain_id in all_domain_ids:
                if domain_id in state.frequency_domains:
                    domain = state.frequency_domains[domain_id]
                    self.domain_evolution_data[domain_id]["timestamps"].append(state.timestamp)
                    self.domain_evolution_data[domain_id]["dominant_frequency"].append(domain.dominant_frequency)
                    self.domain_evolution_data[domain_id]["bandwidth"].append(domain.bandwidth)
                    self.domain_evolution_data[domain_id]["phase_coherence"].append(domain.phase_coherence)
                    self.domain_evolution_data[domain_id]["radius"].append(domain.radius)
                    self.domain_evolution_data[domain_id]["pattern_count"].append(
                        len(domain.pattern_ids) if domain.pattern_ids else 0
                    )
    
    def _extract_boundary_stability_data(self):
        """Extract boundary stability data from topology states."""
        # Clear existing data
        self.boundary_stability_data = {}
        
        # Collect all boundary IDs
        all_boundary_ids = set()
        for state in self.topology_states:
            all_boundary_ids.update(state.boundaries.keys())
        
        # Initialize data for each boundary
        for boundary_id in all_boundary_ids:
            self.boundary_stability_data[boundary_id] = {
                "timestamps": [],
                "sharpness": [],
                "permeability": [],
                "stability": []
            }
        
        # Extract data for each boundary
        for state in self.topology_states:
            for boundary_id in all_boundary_ids:
                if boundary_id in state.boundaries:
                    boundary = state.boundaries[boundary_id]
                    self.boundary_stability_data[boundary_id]["timestamps"].append(state.timestamp)
                    self.boundary_stability_data[boundary_id]["sharpness"].append(boundary.sharpness)
                    self.boundary_stability_data[boundary_id]["permeability"].append(boundary.permeability)
                    self.boundary_stability_data[boundary_id]["stability"].append(boundary.stability)
    
    def _extract_resonance_strength_data(self):
        """Extract resonance strength data from topology states."""
        # Clear existing data
        self.resonance_strength_data = {}
        
        # Collect all resonance point IDs
        all_resonance_ids = set()
        for state in self.topology_states:
            all_resonance_ids.update(state.resonance_points.keys())
        
        # Initialize data for each resonance point
        for resonance_id in all_resonance_ids:
            self.resonance_strength_data[resonance_id] = {
                "timestamps": [],
                "strength": [],
                "stability": [],
                "attractor_radius": []
            }
        
        # Extract data for each resonance point
        for state in self.topology_states:
            for resonance_id in all_resonance_ids:
                if resonance_id in state.resonance_points:
                    point = state.resonance_points[resonance_id]
                    self.resonance_strength_data[resonance_id]["timestamps"].append(state.timestamp)
                    self.resonance_strength_data[resonance_id]["strength"].append(point.strength)
                    self.resonance_strength_data[resonance_id]["stability"].append(point.stability)
                    self.resonance_strength_data[resonance_id]["attractor_radius"].append(point.attractor_radius)
    
    def generate_metrics_panel(self) -> Figure:
        """Generate a panel showing topology metrics over time.
        
        Returns:
            Figure: Matplotlib figure with the metrics panel
        """
        # Check if we have data
        if not self.metrics_data["timestamps"]:
            self.logger.warning("No metrics data available for visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No metrics data available", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        axes = axes.flatten()
        
        # Plot coherence
        axes[0].plot(self.metrics_data["timestamps"], self.metrics_data["coherence"], 
                    'o-', color='blue', linewidth=2)
        axes[0].set_title("Field Coherence")
        axes[0].set_ylabel("Coherence")
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot entropy
        axes[1].plot(self.metrics_data["timestamps"], self.metrics_data["entropy"], 
                    'o-', color='red', linewidth=2)
        axes[1].set_title("Field Entropy")
        axes[1].set_ylabel("Entropy")
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        # Plot adaptation rate
        axes[2].plot(self.metrics_data["timestamps"], self.metrics_data["adaptation_rate"], 
                    'o-', color='green', linewidth=2)
        axes[2].set_title("Adaptation Rate")
        axes[2].set_ylabel("Rate")
        axes[2].set_xlabel("Time")
        axes[2].grid(True, linestyle='--', alpha=0.7)
        
        # Plot homeostasis index
        axes[3].plot(self.metrics_data["timestamps"], self.metrics_data["homeostasis_index"], 
                    'o-', color='purple', linewidth=2)
        axes[3].set_title("Homeostasis Index")
        axes[3].set_ylabel("Index")
        axes[3].set_xlabel("Time")
        axes[3].grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis as dates
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add overall title
        fig.suptitle("Topology Field Metrics Over Time", fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig
    
    def visualize_time_series(self) -> Figure:
        """Visualize time-series data for topology components.
        
        Returns:
            Figure: Matplotlib figure with the time-series visualization
        """
        # Check if we have data
        if not self.time_series_data["timestamps"]:
            self.logger.warning("No time-series data available for visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No time-series data available", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot component counts
        ax.plot(self.time_series_data["timestamps"], self.time_series_data["domain_count"], 
               'o-', color='blue', linewidth=2, label="Frequency Domains")
        ax.plot(self.time_series_data["timestamps"], self.time_series_data["boundary_count"], 
               'o-', color='red', linewidth=2, label="Boundaries")
        ax.plot(self.time_series_data["timestamps"], self.time_series_data["resonance_count"], 
               'o-', color='green', linewidth=2, label="Resonance Points")
        
        # Set axis properties
        ax.set_title("Topology Component Counts Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def track_domain_evolution(self, domain_id: str) -> Figure:
        """Track the evolution of a specific frequency domain over time.
        
        Args:
            domain_id: ID of the domain to track
            
        Returns:
            Figure: Matplotlib figure with the domain evolution visualization
        """
        # Check if we have data for this domain
        if domain_id not in self.domain_evolution_data:
            self.logger.warning(f"No evolution data available for domain {domain_id}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"No evolution data available for domain {domain_id}", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Get domain data
        domain_data = self.domain_evolution_data[domain_id]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        axes = axes.flatten()
        
        # Plot dominant frequency
        axes[0].plot(domain_data["timestamps"], domain_data["dominant_frequency"], 
                    'o-', color='blue', linewidth=2)
        axes[0].set_title("Dominant Frequency")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot bandwidth
        axes[1].plot(domain_data["timestamps"], domain_data["bandwidth"], 
                    'o-', color='red', linewidth=2)
        axes[1].set_title("Bandwidth")
        axes[1].set_ylabel("Bandwidth")
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        # Plot phase coherence
        axes[2].plot(domain_data["timestamps"], domain_data["phase_coherence"], 
                    'o-', color='green', linewidth=2)
        axes[2].set_title("Phase Coherence")
        axes[2].set_ylabel("Coherence")
        axes[2].set_xlabel("Time")
        axes[2].grid(True, linestyle='--', alpha=0.7)
        
        # Plot pattern count
        axes[3].plot(domain_data["timestamps"], domain_data["pattern_count"], 
                    'o-', color='purple', linewidth=2)
        axes[3].set_title("Pattern Count")
        axes[3].set_ylabel("Count")
        axes[3].set_xlabel("Time")
        axes[3].grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis as dates
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add overall title
        fig.suptitle(f"Evolution of Frequency Domain {domain_id}", fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig
    
    def visualize_boundary_stability(self) -> Figure:
        """Visualize boundary stability over time.
        
        Returns:
            Figure: Matplotlib figure with the boundary stability visualization
        """
        # Check if we have data
        if not self.boundary_stability_data:
            self.logger.warning("No boundary stability data available for visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No boundary stability data available", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot stability for each boundary
        for boundary_id, data in self.boundary_stability_data.items():
            if data["timestamps"] and data["stability"]:
                ax.plot(data["timestamps"], data["stability"], 
                       'o-', linewidth=2, label=f"Boundary {boundary_id}")
        
        # Set axis properties
        ax.set_title("Boundary Stability Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Stability")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def visualize_resonance_strength(self) -> Figure:
        """Visualize resonance point strength over time.
        
        Returns:
            Figure: Matplotlib figure with the resonance strength visualization
        """
        # Check if we have data
        if not self.resonance_strength_data:
            self.logger.warning("No resonance strength data available for visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No resonance strength data available", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot strength for each resonance point
        for resonance_id, data in self.resonance_strength_data.items():
            if data["timestamps"] and data["strength"]:
                ax.plot(data["timestamps"], data["strength"], 
                       'o-', linewidth=2, label=f"Resonance {resonance_id}")
        
        # Set axis properties
        ax.set_title("Resonance Point Strength Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Strength")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def generate_interactive_dashboard(self) -> Figure:
        """Generate an interactive dashboard for exploring topology states.
        
        Note: This is a placeholder for interactive dashboard capabilities.
        In a real implementation, this would use interactive libraries
        like Plotly, Bokeh, or Panel.
        
        Returns:
            Figure: Matplotlib figure with the interactive dashboard
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Add placeholder text
        ax.text(0.5, 0.5, "Interactive Dashboard Placeholder", 
               ha='center', va='center', fontsize=14)
        
        # Store interactive elements
        self.interactive_elements = {
            "figure": fig,
            "axes": ax
        }
        
        return fig
    
    def generate_dashboard(self) -> Figure:
        """Generate a complete dashboard with all components.
        
        Returns:
            Figure: Matplotlib figure with the complete dashboard
        """
        # Check if we have states
        if not self.topology_states:
            self.logger.warning("No topology states available for dashboard generation")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No topology states available", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Create figure with grid layout
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # Add metrics panel
        ax_metrics = fig.add_subplot(gs[0, :])
        self._plot_metrics_summary(ax_metrics)
        
        # Add time-series visualization
        ax_time_series = fig.add_subplot(gs[1, 0])
        self._plot_time_series_summary(ax_time_series)
        
        # Add boundary stability visualization
        ax_boundary = fig.add_subplot(gs[1, 1])
        self._plot_boundary_summary(ax_boundary)
        
        # Add resonance strength visualization
        ax_resonance = fig.add_subplot(gs[1, 2])
        self._plot_resonance_summary(ax_resonance)
        
        # Add topology state visualization
        ax_topology = fig.add_subplot(gs[2, :])
        self._plot_current_topology(ax_topology)
        
        # Add overall title
        fig.suptitle("Topology Dashboard", fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig
    
    def _plot_metrics_summary(self, ax):
        """Plot a summary of metrics on the given axis."""
        # Get the latest metrics
        if self.metrics_data["timestamps"]:
            latest_coherence = self.metrics_data["coherence"][-1]
            latest_entropy = self.metrics_data["entropy"][-1]
            latest_adaptation = self.metrics_data["adaptation_rate"][-1]
            latest_homeostasis = self.metrics_data["homeostasis_index"][-1]
            
            # Create bar chart
            metrics = ["Coherence", "Entropy", "Adaptation", "Homeostasis"]
            values = [latest_coherence, latest_entropy, latest_adaptation, latest_homeostasis]
            colors = ['blue', 'red', 'green', 'purple']
            
            ax.bar(metrics, values, color=colors, alpha=0.7)
            ax.set_title("Current Field Metrics")
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        else:
            ax.text(0.5, 0.5, "No metrics data available", 
                   ha='center', va='center', fontsize=14)
    
    def _plot_time_series_summary(self, ax):
        """Plot a summary of time-series data on the given axis."""
        if self.time_series_data["timestamps"]:
            # Plot component counts
            ax.plot(self.time_series_data["timestamps"], self.time_series_data["domain_count"], 
                   'o-', color='blue', linewidth=2, label="Domains")
            ax.plot(self.time_series_data["timestamps"], self.time_series_data["boundary_count"], 
                   'o-', color='red', linewidth=2, label="Boundaries")
            ax.plot(self.time_series_data["timestamps"], self.time_series_data["resonance_count"], 
                   'o-', color='green', linewidth=2, label="Resonances")
            
            ax.set_title("Component Counts")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, "No time-series data available", 
                   ha='center', va='center', fontsize=14)
    
    def _plot_boundary_summary(self, ax):
        """Plot a summary of boundary stability on the given axis."""
        if self.boundary_stability_data:
            # Plot average stability for all boundaries
            for boundary_id, data in self.boundary_stability_data.items():
                if data["timestamps"] and data["stability"]:
                    ax.plot(data["timestamps"], data["stability"], 
                           'o-', linewidth=2, label=f"B{boundary_id.split('-')[-1]}")
            
            ax.set_title("Boundary Stability")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, "No boundary data available", 
                   ha='center', va='center', fontsize=14)
    
    def _plot_resonance_summary(self, ax):
        """Plot a summary of resonance strength on the given axis."""
        if self.resonance_strength_data:
            # Plot strength for each resonance point
            for resonance_id, data in self.resonance_strength_data.items():
                if data["timestamps"] and data["strength"]:
                    ax.plot(data["timestamps"], data["strength"], 
                           'o-', linewidth=2, label=f"R{resonance_id.split('-')[-1]}")
            
            ax.set_title("Resonance Strength")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, "No resonance data available", 
                   ha='center', va='center', fontsize=14)
    
    def _plot_current_topology(self, ax):
        """Plot the current topology state on the given axis."""
        if self.topology_states:
            # Get the latest state
            latest_state = self.topology_states[-1]
            
            # Visualize the topology state
            self.topology_visualizer.visualize_topology(latest_state, ax=ax, 
                                                      show_labels=True, 
                                                      title="Current Topology State")
        else:
            ax.text(0.5, 0.5, "No topology state available", 
                   ha='center', va='center', fontsize=14)
