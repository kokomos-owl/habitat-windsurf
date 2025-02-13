"""Medical pattern visualization and graph generation."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import networkx as nx
from dataclasses import dataclass

@dataclass
class MedicalVisualizationConfig:
    """Configuration for medical pattern visualization."""
    
    vital_ranges = {
        "heart_rate": (60, 100),
        "systolic_bp": (90, 140),
        "temperature": (36.5, 37.5),
        "respiratory_rate": (12, 20)
    }
    
    lab_ranges = {
        "wbc": (4.5, 11.0),
        "lactate": (0.5, 2.0),
        "creatinine": (0.6, 1.2)
    }
    
    # Sepsis criteria thresholds
    sepsis_thresholds = {
        "sofa_score": 2,
        "qsofa_score": 2,
        "lactate": 2.0
    }
    
    # Visualization parameters
    colors = {
        "normal": "#2ecc71",
        "warning": "#f1c40f",
        "critical": "#e74c3c",
        "background": "#ecf0f1"
    }

class MedicalPatternVisualizer:
    """Visualizer for medical pattern evolution and relationships."""
    
    def __init__(self):
        """Initialize medical pattern visualizer."""
        self.config = MedicalVisualizationConfig()
        self.test_results = []
        
    def visualize_temporal_evolution(self,
                                   case_data: Dict[str, Any],
                                   field_states: List[Dict[str, float]],
                                   patterns: List[Dict[str, Any]]) -> plt.Figure:
        """Visualize temporal evolution of clinical patterns.
        
        Args:
            case_data: Clinical case data
            field_states: List of field states over time
            patterns: Detected patterns
            
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), height_ratios=[2, 2, 3])
        
        # 1. Vital Signs Evolution
        ax_vitals = axes[0]
        base_time = min(v.timestamp for v in case_data["vitals"])
        
        for vital in ["heart_rate", "systolic_bp", "respiratory_rate"]:
            vital_data = [v for v in case_data["vitals"] if v.name == vital]
            times = [v.timestamp for v in vital_data]
            hours = [(t - base_time).total_seconds() / 3600 for t in times]
            values = [v.value for v in vital_data]
            ax_vitals.plot(hours, values, label=vital, marker='o')
            
        ax_vitals.set_xlabel("Hours from Presentation")
        ax_vitals.set_ylabel("Value")
        ax_vitals.set_title("Vital Signs Evolution")
        ax_vitals.legend()
        ax_vitals.grid(True)
        
        # 2. Field State Evolution
        ax_field = axes[1]
        field_times = range(len(field_states))
        
        for metric in ["temporal", "vitals", "labs", "events"]:
            values = [state[metric] for state in field_states]
            ax_field.plot(field_times, values, label=metric, marker='o')
            
        ax_field.set_xlabel("Time Steps")
        ax_field.set_ylabel("Field State")
        ax_field.set_title("Field State Evolution")
        ax_field.legend()
        ax_field.grid(True)
        
        # 3. Pattern Network
        ax_network = axes[2]
        self._draw_pattern_network(patterns, ax_network)
        
        plt.tight_layout()
        return fig
    
    def visualize_pattern_coherence(self,
                                  patterns: List[Dict[str, Any]],
                                  field_state: Dict[str, float]) -> plt.Figure:
        """Visualize pattern coherence and relationships.
        
        Args:
            patterns: List of detected patterns
            field_state: Current field state
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create graph
        G = nx.Graph()
        
        # Add pattern nodes
        for i, pattern in enumerate(patterns):
            G.add_node(i,
                      coherence=pattern.get("coherence", 0),
                      energy=pattern.get("energy_state", 0),
                      pattern_type=pattern.get("pattern_type", "unknown"))
        
        # Add edges based on pattern relationships
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                # Calculate relationship strength
                coherence_sim = 1 - abs(pattern1.get("coherence", 0) - 
                                     pattern2.get("coherence", 0))
                if coherence_sim > 0.7:  # Threshold for relationship
                    G.add_edge(i, j, weight=coherence_sim)
        
        # Draw network
        pos = nx.spring_layout(G)
        
        # Draw nodes
        node_colors = [G.nodes[n]["coherence"] for n in G.nodes()]
        node_sizes = [G.nodes[n]["energy"] * 1000 for n in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos,
                             node_color=node_colors,
                             node_size=node_sizes,
                             cmap=plt.cm.viridis,
                             ax=ax)
        
        # Draw edges
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos,
                             width=edge_weights,
                             alpha=0.5,
                             ax=ax)
        
        # Add labels
        labels = {n: f"P{n}\n{G.nodes[n]['pattern_type']}" for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        ax.set_title("Pattern Coherence Network")
        return fig
    
    def export_to_neo4j(self,
                       case_data: Dict[str, Any],
                       patterns: List[Dict[str, Any]],
                       driver) -> None:
        """Export patterns and relationships to Neo4j.
        
        Args:
            case_data: Clinical case data
            patterns: Detected patterns
            driver: Neo4j driver instance
        """
        with driver.session(database="neo4j") as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create case node
            session.run("""
                CREATE (c:Case {
                    case_id: $case_id,
                    onset_time: $onset_time
                })
            """, case_id=case_data["case_id"],
                 onset_time=case_data["onset_time"].isoformat())
            
            # Create pattern nodes
            for i, pattern in enumerate(patterns):
                session.run("""
                    MATCH (c:Case {case_id: $case_id})
                    CREATE (p:Pattern {
                        pattern_id: $pattern_id,
                        type: $pattern_type,
                        coherence: $coherence,
                        energy_state: $energy_state
                    })-[:BELONGS_TO]->(c)
                """,
                    case_id=case_data["case_id"],
                    pattern_id=f"P{i}",
                    pattern_type=pattern.get("pattern_type", "unknown"),
                    coherence=pattern.get("coherence", 0),
                    energy_state=pattern.get("energy_state", 0))
            
            # Create relationships between patterns
            for i, pattern1 in enumerate(patterns):
                for j, pattern2 in enumerate(patterns[i+1:], i+1):
                    coherence_sim = 1 - abs(pattern1.get("coherence", 0) - 
                                         pattern2.get("coherence", 0))
                    if coherence_sim > 0.7:
                        session.run("""
                            MATCH (p1:Pattern {pattern_id: $pid1})
                            MATCH (p2:Pattern {pattern_id: $pid2})
                            CREATE (p1)-[:RELATED_TO {
                                similarity: $similarity
                            }]->(p2)
                        """,
                            pid1=f"P{i}",
                            pid2=f"P{j}",
                            similarity=coherence_sim)
    
    def _draw_pattern_network(self,
                            patterns: List[Dict[str, Any]],
                            ax: plt.Axes) -> None:
        """Draw pattern network on given axes."""
        G = nx.Graph()
        
        # Add nodes
        for i, pattern in enumerate(patterns):
            G.add_node(i,
                      coherence=pattern.get("coherence", 0),
                      energy=pattern.get("energy_state", 0),
                      pattern_type=pattern.get("pattern_type", "unknown"))
        
        # Add edges
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                coherence_sim = 1 - abs(pattern1.get("coherence", 0) - 
                                     pattern2.get("coherence", 0))
                if coherence_sim > 0.7:
                    G.add_edge(i, j, weight=coherence_sim)
        
        # Draw network
        pos = nx.spring_layout(G)
        
        node_colors = [G.nodes[n]["coherence"] for n in G.nodes()]
        node_sizes = [G.nodes[n]["energy"] * 1000 for n in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos,
                             node_color=node_colors,
                             node_size=node_sizes,
                             cmap=plt.cm.viridis,
                             ax=ax)
        
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos,
                             width=edge_weights,
                             alpha=0.5,
                             ax=ax)
        
        labels = {n: f"P{n}\n{G.nodes[n]['pattern_type']}" for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        ax.set_title("Pattern Evolution Network")
