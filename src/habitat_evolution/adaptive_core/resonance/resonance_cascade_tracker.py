"""
Resonance Cascade Tracker for monitoring resonance propagation.

This module provides functionality to track how resonance propagates through
the semantic network as wave-like phenomena, detecting cascade paths and
analyzing their properties.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from datetime import datetime
import math
import networkx as nx
from collections import defaultdict

# Import AdaptiveID for pattern identification
from habitat_evolution.pattern_aware_rag.adaptive_core.id.adaptive_id import AdaptiveID

logger = logging.getLogger(__name__)

class ResonanceCascadeTracker:
    """
    Tracks how resonance propagates through the semantic network as wave-like phenomena.
    
    This class observes resonance cascades, identifying paths of strong resonance
    through the semantic network and analyzing their properties over time.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ResonanceCascadeTracker with configuration parameters.
        
        Args:
            config: Configuration dictionary with the following optional parameters:
                - propagation_threshold: Minimum strength for propagation (default: 0.6)
                - max_cascade_depth: Maximum depth for cascade tracking (default: 5)
                - temporal_window: Time window for cascade observation (default: 10)
                - decay_factor: Decay factor for propagation strength (default: 0.2)
                - min_cascade_size: Minimum size for a valid cascade (default: 3)
        """
        default_config = {
            "propagation_threshold": 0.6,      # Minimum strength for propagation
            "max_cascade_depth": 5,            # Maximum depth for cascade tracking
            "temporal_window": 10,             # Time window for cascade observation
            "decay_factor": 0.2,               # Decay factor for propagation strength
            "min_cascade_size": 3              # Minimum size for a valid cascade
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
    
    def detect_cascade_paths(self, resonance_network: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect cascade paths in a resonance network.
        
        Args:
            resonance_network: Dictionary with nodes (domains) and edges (resonance connections)
            
        Returns:
            List of detected cascade paths with their properties
        """
        # Create a directed graph from the resonance network
        G = nx.DiGraph()
        
        # Add nodes
        for node in resonance_network["nodes"]:
            G.add_node(node["id"], **node)
        
        # Add edges with strength above threshold
        for edge in resonance_network["edges"]:
            if edge["strength"] >= self.config["propagation_threshold"]:
                G.add_edge(edge["source"], edge["target"], strength=edge["strength"])
        
        # Find all simple paths up to max_depth
        cascades = []
        for source in G.nodes():
            for target in G.nodes():
                if source != target:
                    for path in nx.all_simple_paths(G, source, target, cutoff=self.config["max_cascade_depth"]):
                        if len(path) >= self.config["min_cascade_size"]:
                            # Calculate path strength as product of edge strengths
                            strengths = [G[path[i]][path[i+1]]["strength"] for i in range(len(path)-1)]
                            path_strength = np.prod(strengths)
                            
                            # Apply decay factor based on path length
                            decayed_strength = path_strength * (1 - self.config["decay_factor"] * (len(path) - 1))
                            
                            cascades.append({
                                "path": path,
                                "strength": decayed_strength,
                                "length": len(path),
                                "avg_strength": np.mean(strengths)
                            })
        
        # Sort cascades by strength (descending)
        cascades.sort(key=lambda x: x["strength"], reverse=True)
        
        return cascades
    
    def detect_temporal_cascades(self, temporal_resonance_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect cascades in temporal resonance events.
        
        Args:
            temporal_resonance_events: List of resonance events with timestamps
            
        Returns:
            List of detected temporal cascades
        """
        # Sort events by timestamp
        sorted_events = sorted(temporal_resonance_events, key=lambda x: x["timestamp"])
        
        # Group events by source domain
        source_events = defaultdict(list)
        for event in sorted_events:
            source_events[event["source"]].append(event)
        
        # Find temporal cascades
        cascades = []
        for start_event in sorted_events:
            # Skip events with strength below threshold
            if start_event["strength"] < self.config["propagation_threshold"]:
                continue
            
            # Start a new cascade
            cascade = {
                "events": [start_event],
                "path": [start_event["source"], start_event["target"]],
                "timestamps": [start_event["timestamp"]],
                "strengths": [start_event["strength"]]
            }
            
            # Track the current end of the cascade
            current_target = start_event["target"]
            current_time = start_event["timestamp"]
            
            # Extend the cascade
            for _ in range(self.config["max_cascade_depth"] - 1):
                # Find next events from current target within time window
                next_events = [
                    e for e in source_events[current_target]
                    if e["timestamp"] > current_time and 
                       e["timestamp"] <= current_time + self.config["temporal_window"] and
                       e["strength"] >= self.config["propagation_threshold"] and
                       e["target"] not in cascade["path"]  # Avoid cycles
                ]
                
                if not next_events:
                    break
                
                # Select the strongest next event
                next_event = max(next_events, key=lambda x: x["strength"])
                
                # Add to cascade
                cascade["events"].append(next_event)
                cascade["path"].append(next_event["target"])
                cascade["timestamps"].append(next_event["timestamp"])
                cascade["strengths"].append(next_event["strength"])
                
                # Update current state
                current_target = next_event["target"]
                current_time = next_event["timestamp"]
            
            # Only keep cascades that meet minimum size
            if len(cascade["path"]) >= self.config["min_cascade_size"]:
                # Calculate overall strength
                cascade["overall_strength"] = np.prod(cascade["strengths"])
                cascade["avg_strength"] = np.mean(cascade["strengths"])
                cascade["duration"] = cascade["timestamps"][-1] - cascade["timestamps"][0]
                
                cascades.append(cascade)
        
        # Sort cascades by overall strength (descending)
        cascades.sort(key=lambda x: x["overall_strength"], reverse=True)
        
        return cascades
    
    def detect_cascade_convergence(self, resonance_network: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect convergence points in resonance cascades.
        
        Args:
            resonance_network: Dictionary with nodes (domains) and edges (resonance connections)
            
        Returns:
            List of detected convergence points with their properties
        """
        # Create a directed graph from the resonance network
        G = nx.DiGraph()
        
        # Add nodes
        for node in resonance_network["nodes"]:
            G.add_node(node["id"], **node)
        
        # Add edges with strength above threshold
        for edge in resonance_network["edges"]:
            if edge["strength"] >= self.config["propagation_threshold"]:
                G.add_edge(edge["source"], edge["target"], strength=edge["strength"])
        
        # Find nodes with high in-degree (convergence points)
        convergence_points = []
        for node_id, in_degree in G.in_degree():
            if in_degree >= 2:  # At least two incoming connections
                # Get incoming edges
                incoming_edges = [(u, v, d) for u, v, d in G.in_edges(node_id, data=True)]
                
                # Calculate average incoming strength
                avg_strength = np.mean([d["strength"] for _, _, d in incoming_edges])
                
                # Find source nodes that are not directly connected
                source_nodes = [u for u, _, _ in incoming_edges]
                independent_sources = 0
                for i, source1 in enumerate(source_nodes):
                    independent = True
                    for j, source2 in enumerate(source_nodes):
                        if i != j and (G.has_edge(source1, source2) or G.has_edge(source2, source1)):
                            independent = False
                            break
                    if independent:
                        independent_sources += 1
                
                # Only consider as convergence point if there are independent sources
                if independent_sources >= 2:
                    convergence_points.append({
                        "node_id": node_id,
                        "in_degree": in_degree,
                        "avg_incoming_strength": avg_strength,
                        "independent_sources": independent_sources,
                        "source_nodes": source_nodes
                    })
        
        # Sort by number of independent sources (descending)
        convergence_points.sort(key=lambda x: (x["independent_sources"], x["avg_incoming_strength"]), reverse=True)
        
        return convergence_points
    
    def generate_cascade_id(self, cascade: Dict[str, Any]) -> AdaptiveID:
        """
        Generate an AdaptiveID for a resonance cascade.
        
        Args:
            cascade: Dictionary with cascade information
            
        Returns:
            AdaptiveID for the cascade
        """
        # Generate a unique ID based on the cascade path
        cascade_path_str = "-".join(cascade["path"])
        cascade_id = f"cascade_{hash(cascade_path_str) % 10000}"
        
        # Determine the base concept from the strongest node in the cascade
        base_concept = cascade["path"][0]  # Default to first node
        
        # Create and return an AdaptiveID
        return AdaptiveID(
            id=cascade_id,
            base_concept=base_concept,
            timestamp=datetime.now()
        )
    
    def generate_cascade_visualization(self, resonance_network: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate data for visualizing resonance cascades.
        
        Args:
            resonance_network: Dictionary with nodes (domains) and edges (resonance connections)
            
        Returns:
            Visualization data structure
        """
        # Detect cascade paths
        cascades = self.detect_cascade_paths(resonance_network)
        
        # Detect convergence points
        convergence_points = self.detect_cascade_convergence(resonance_network)
        
        # Prepare node data
        nodes = []
        for node in resonance_network["nodes"]:
            # Check if node is a convergence point
            is_convergence = any(cp["node_id"] == node["id"] for cp in convergence_points)
            
            # Count cascade participation
            cascade_count = sum(1 for c in cascades if node["id"] in c["path"])
            
            nodes.append({
                "id": node["id"],
                "frequency": node.get("frequency", 0.0),
                "amplitude": node.get("amplitude", 0.0),
                "is_convergence_point": is_convergence,
                "cascade_participation": cascade_count
            })
        
        # Prepare link data
        links = []
        for edge in resonance_network["edges"]:
            if edge["strength"] >= self.config["propagation_threshold"]:
                # Count cascade participation
                cascade_count = sum(1 for c in cascades 
                                   if edge["source"] in c["path"] and edge["target"] in c["path"] and
                                   c["path"].index(edge["source"]) + 1 == c["path"].index(edge["target"]))
                
                links.append({
                    "source": edge["source"],
                    "target": edge["target"],
                    "strength": edge["strength"],
                    "cascade_count": cascade_count
                })
        
        # Prepare cascade data
        cascade_data = []
        for i, cascade in enumerate(cascades[:10]):  # Limit to top 10 cascades
            # Generate AdaptiveID for the cascade
            adaptive_id = self.generate_cascade_id(cascade)
            
            cascade_data.append({
                "id": adaptive_id.id,
                "base_concept": adaptive_id.base_concept,
                "timestamp": adaptive_id.timestamp.isoformat(),
                "path": cascade["path"],
                "strength": cascade["strength"],
                "length": cascade["length"]
            })
        
        return {
            "nodes": nodes,
            "links": links,
            "cascades": cascade_data,
            "convergence_points": convergence_points
        }
    
    def calculate_cascade_metrics(self, resonance_network: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics for resonance cascades.
        
        Args:
            resonance_network: Dictionary with nodes (domains) and edges (resonance connections)
            
        Returns:
            Dictionary of cascade metrics
        """
        # Detect cascade paths
        cascades = self.detect_cascade_paths(resonance_network)
        
        # Detect convergence points
        convergence_points = self.detect_cascade_convergence(resonance_network)
        
        # Calculate metrics
        if cascades:
            avg_cascade_length = np.mean([c["length"] for c in cascades])
            max_cascade_length = max([c["length"] for c in cascades])
            avg_cascade_strength = np.mean([c["strength"] for c in cascades])
            max_cascade_strength = max([c["strength"] for c in cascades])
        else:
            avg_cascade_length = 0.0
            max_cascade_length = 0
            avg_cascade_strength = 0.0
            max_cascade_strength = 0.0
        
        # Calculate network metrics
        G = nx.DiGraph()
        for node in resonance_network["nodes"]:
            G.add_node(node["id"])
        for edge in resonance_network["edges"]:
            if edge["strength"] >= self.config["propagation_threshold"]:
                G.add_edge(edge["source"], edge["target"], weight=edge["strength"])
        
        # Calculate average path length if graph is connected
        try:
            avg_path_length = nx.average_shortest_path_length(G, weight='weight')
        except nx.NetworkXError:
            avg_path_length = 0.0
        
        # Calculate clustering coefficient
        try:
            clustering_coefficient = nx.average_clustering(G.to_undirected())
        except:
            clustering_coefficient = 0.0
        
        return {
            "cascade_count": len(cascades),
            "average_cascade_length": float(avg_cascade_length),
            "max_cascade_length": int(max_cascade_length),
            "average_cascade_strength": float(avg_cascade_strength),
            "max_cascade_strength": float(max_cascade_strength),
            "convergence_point_count": len(convergence_points),
            "average_path_length": float(avg_path_length),
            "clustering_coefficient": float(clustering_coefficient),
            "network_density": float(nx.density(G))
        }
    
    def analyze_actant_cascade_participation(self, actants: List[Dict[str, Any]], 
                                           resonance_network: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze actant participation in resonance cascades.
        
        Args:
            actants: List of actant dictionaries with domain participation
            resonance_network: Dictionary with nodes (domains) and edges (resonance connections)
            
        Returns:
            List of actant participation analyses
        """
        # Detect cascade paths
        cascades = self.detect_cascade_paths(resonance_network)
        
        # Analyze each actant
        analyses = []
        for actant in actants:
            actant_id = actant["id"]
            actant_domains = set(actant["domains"])
            
            # Count cascade participation
            cascade_participation = []
            for cascade in cascades:
                cascade_domains = set(cascade["path"])
                overlap = actant_domains.intersection(cascade_domains)
                
                if overlap:
                    participation = {
                        "cascade_path": cascade["path"],
                        "overlap_domains": list(overlap),
                        "overlap_count": len(overlap),
                        "overlap_percentage": len(overlap) / len(cascade["path"]) * 100,
                        "cascade_strength": cascade["strength"]
                    }
                    # Generate AdaptiveID for this participation
                    adaptive_id = AdaptiveID(
                        id=f"participation_{actant_id}_{len(cascade_participation)}",
                        base_concept=actant_id,
                        timestamp=datetime.now()
                    )
                    
                    participation["adaptive_id"] = {
                        "id": adaptive_id.id,
                        "base_concept": adaptive_id.base_concept,
                        "timestamp": adaptive_id.timestamp.isoformat()
                    }
                    
                    cascade_participation.append(participation)
            
            # Calculate cross-cascade influence
            cross_cascade_count = 0
            for i, cascade1 in enumerate(cascades):
                for j, cascade2 in enumerate(cascades):
                    if i < j:  # Only compare unique pairs
                        cascade1_domains = set(cascade1["path"])
                        cascade2_domains = set(cascade2["path"])
                        
                        # Check if actant spans both cascades
                        if (actant_domains.intersection(cascade1_domains) and 
                            actant_domains.intersection(cascade2_domains)):
                            cross_cascade_count += 1
            
            # Calculate overall metrics
            if cascade_participation:
                avg_overlap_percentage = np.mean([p["overlap_percentage"] for p in cascade_participation])
                max_overlap_percentage = max([p["overlap_percentage"] for p in cascade_participation])
                avg_cascade_strength = np.mean([p["cascade_strength"] for p in cascade_participation])
            else:
                avg_overlap_percentage = 0.0
                max_overlap_percentage = 0.0
                avg_cascade_strength = 0.0
            
            # Calculate cross-cascade influence score
            total_cascade_pairs = len(cascades) * (len(cascades) - 1) / 2
            cross_cascade_influence = cross_cascade_count / total_cascade_pairs if total_cascade_pairs > 0 else 0.0
            
            analyses.append({
                "actant_id": actant_id,
                "domain_count": len(actant_domains),
                "cascade_participation": cascade_participation,
                "cascade_participation_count": len(cascade_participation),
                "avg_overlap_percentage": float(avg_overlap_percentage),
                "max_overlap_percentage": float(max_overlap_percentage),
                "avg_cascade_strength": float(avg_cascade_strength),
                "cross_cascade_count": cross_cascade_count,
                "cross_cascade_influence": float(cross_cascade_influence)
            })
        
        # Sort by cross-cascade influence (descending)
        analyses.sort(key=lambda x: x["cross_cascade_influence"], reverse=True)
        
        return analyses
