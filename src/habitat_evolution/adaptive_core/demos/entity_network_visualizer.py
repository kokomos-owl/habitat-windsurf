"""
Entity Network Visualizer

This module provides enhanced visualization capabilities for entity networks
created by the context-aware NER evolution system.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import Counter, defaultdict

# Configure logging
logger = logging.getLogger(__name__)

class EntityNetworkVisualizer:
    """
    Provides enhanced visualization capabilities for entity networks.
    """
    
    def __init__(self, network: nx.DiGraph, output_dir: str = None):
        """
        Initialize the visualizer with an entity network.
        
        Args:
            network: NetworkX DiGraph containing entity nodes and relationship edges
            output_dir: Directory to save visualizations (defaults to current directory)
        """
        self.network = network
        self.output_dir = output_dir or Path.cwd() / "visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract entity categories and relationship types
        self.entity_categories = self._extract_entity_categories()
        self.relationship_types = self._extract_relationship_types()
        
        # Define color schemes
        self.quality_colors = {
            'good': '#2ecc71',      # Green
            'uncertain': '#f39c12',  # Orange
            'poor': '#e74c3c',       # Red
            'unknown': '#95a5a6'     # Gray
        }
        
        self.category_colors = {
            'CLIMATE_HAZARD': '#3498db',       # Blue
            'ECOSYSTEM': '#2ecc71',            # Green
            'INFRASTRUCTURE': '#e74c3c',       # Red
            'ADAPTATION_STRATEGY': '#9b59b6',  # Purple
            'ASSESSMENT_COMPONENT': '#f1c40f'  # Yellow
        }
        
        logger.info(f"Initialized EntityNetworkVisualizer with {len(network.nodes)} nodes and {len(network.edges)} edges")
    
    def _extract_entity_categories(self) -> Dict[str, List[str]]:
        """Extract entity categories from the network."""
        categories = defaultdict(list)
        
        for node, data in self.network.nodes(data=True):
            category = data.get('category')
            if category:
                categories[category].append(node)
        
        return dict(categories)
    
    def _extract_relationship_types(self) -> Dict[str, List[Tuple[str, str]]]:
        """Extract relationship types from the network."""
        rel_types = defaultdict(list)
        
        for source, target, data in self.network.edges(data=True):
            rel_type = data.get('type')
            if rel_type:
                rel_types[rel_type].append((source, target))
        
        return dict(rel_types)
    
    def visualize_by_quality(self, filename: str = "entity_network_by_quality.png"):
        """
        Visualize the entity network colored by quality.
        
        Args:
            filename: Output filename
        """
        plt.figure(figsize=(16, 12))
        
        # Create position layout
        pos = nx.spring_layout(self.network, seed=42)
        
        # Draw nodes by quality
        for quality, color in self.quality_colors.items():
            nodes = [n for n, d in self.network.nodes(data=True) if d.get('quality') == quality]
            nx.draw_networkx_nodes(
                self.network, 
                pos, 
                nodelist=nodes,
                node_color=color,
                node_size=100,
                alpha=0.8,
                label=f"{quality} ({len(nodes)})"
            )
        
        # Draw edges with alpha based on density
        nx.draw_networkx_edges(
            self.network,
            pos,
            width=0.3,
            alpha=0.2,
            arrows=False
        )
        
        # Draw labels for high-degree nodes only
        high_degree_nodes = [n for n, d in self.network.degree() if d > 10]
        nx.draw_networkx_labels(
            self.network,
            pos,
            labels={n: n for n in high_degree_nodes},
            font_size=8,
            font_family='sans-serif'
        )
        
        plt.title(f"Entity Network by Quality (Entities: {len(self.network.nodes)}, Relationships: {len(self.network.edges)})")
        plt.legend(scatterpoints=1, loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved quality visualization to {output_path}")
        return output_path
    
    def visualize_by_category(self, filename: str = "entity_network_by_category.png"):
        """
        Visualize the entity network colored by category.
        
        Args:
            filename: Output filename
        """
        plt.figure(figsize=(16, 12))
        
        # Create position layout
        pos = nx.spring_layout(self.network, seed=42)
        
        # Draw nodes by category
        for category, color in self.category_colors.items():
            nodes = self.entity_categories.get(category, [])
            if nodes:
                nx.draw_networkx_nodes(
                    self.network, 
                    pos, 
                    nodelist=nodes,
                    node_color=color,
                    node_size=100,
                    alpha=0.8,
                    label=f"{category} ({len(nodes)})"
                )
        
        # Draw nodes with unknown category
        unknown_nodes = [n for n in self.network.nodes() if not any(n in nodes for nodes in self.entity_categories.values())]
        if unknown_nodes:
            nx.draw_networkx_nodes(
                self.network, 
                pos, 
                nodelist=unknown_nodes,
                node_color='gray',
                node_size=50,
                alpha=0.5,
                label=f"Unknown ({len(unknown_nodes)})"
            )
        
        # Draw edges with alpha based on density
        nx.draw_networkx_edges(
            self.network,
            pos,
            width=0.3,
            alpha=0.2,
            arrows=False
        )
        
        # Draw labels for high-degree nodes only
        high_degree_nodes = [n for n, d in self.network.degree() if d > 10]
        nx.draw_networkx_labels(
            self.network,
            pos,
            labels={n: n for n in high_degree_nodes},
            font_size=8,
            font_family='sans-serif'
        )
        
        plt.title(f"Entity Network by Category (Entities: {len(self.network.nodes)}, Relationships: {len(self.network.edges)})")
        plt.legend(scatterpoints=1, loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved category visualization to {output_path}")
        return output_path
    
    def visualize_category_subgraphs(self, filename_prefix: str = "category_subgraph"):
        """
        Visualize subgraphs for each entity category.
        
        Args:
            filename_prefix: Prefix for output filenames
        """
        output_paths = []
        
        for category, nodes in self.entity_categories.items():
            if len(nodes) < 5:  # Skip categories with too few nodes
                continue
                
            # Create subgraph
            subgraph = self.network.subgraph(nodes)
            
            plt.figure(figsize=(12, 10))
            
            # Create position layout
            pos = nx.spring_layout(subgraph, seed=42)
            
            # Draw nodes by quality
            for quality, color in self.quality_colors.items():
                quality_nodes = [n for n, d in subgraph.nodes(data=True) if d.get('quality') == quality]
                if quality_nodes:
                    nx.draw_networkx_nodes(
                        subgraph, 
                        pos, 
                        nodelist=quality_nodes,
                        node_color=color,
                        node_size=150,
                        alpha=0.8,
                        label=f"{quality} ({len(quality_nodes)})"
                    )
            
            # Draw edges
            nx.draw_networkx_edges(
                subgraph,
                pos,
                width=0.5,
                alpha=0.4,
                arrows=True
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                subgraph,
                pos,
                font_size=8,
                font_family='sans-serif'
            )
            
            plt.title(f"{category} Entity Network (Entities: {len(subgraph.nodes)}, Relationships: {len(subgraph.edges)})")
            plt.legend(scatterpoints=1, loc='upper right')
            plt.axis('off')
            plt.tight_layout()
            
            output_path = os.path.join(self.output_dir, f"{filename_prefix}_{category.lower()}.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logger.info(f"Saved {category} subgraph visualization to {output_path}")
            output_paths.append(output_path)
        
        return output_paths
    
    def visualize_cross_category_relationships(self, filename: str = "cross_category_relationships.png"):
        """
        Visualize relationships between different entity categories.
        
        Args:
            filename: Output filename
        """
        # Count relationships between categories
        category_matrix = defaultdict(lambda: defaultdict(int))
        
        for source, target, data in self.network.edges(data=True):
            source_category = self.network.nodes[source].get('category', 'Unknown')
            target_category = self.network.nodes[target].get('category', 'Unknown')
            
            if source_category and target_category:
                category_matrix[source_category][target_category] += 1
        
        # Convert to DataFrame
        categories = sorted(set(self.entity_categories.keys()))
        df = pd.DataFrame(0, index=categories, columns=categories)
        
        for source_cat, targets in category_matrix.items():
            for target_cat, count in targets.items():
                if source_cat in categories and target_cat in categories:
                    df.loc[source_cat, target_cat] = count
        
        # Visualize as heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(df, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5)
        plt.title("Cross-Category Relationships")
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved cross-category relationship visualization to {output_path}")
        return output_path
    
    def visualize_quality_distribution(self, filename: str = "quality_distribution.png"):
        """
        Visualize the distribution of entity qualities by category.
        
        Args:
            filename: Output filename
        """
        # Count entities by category and quality
        quality_counts = defaultdict(lambda: defaultdict(int))
        
        for node, data in self.network.nodes(data=True):
            category = data.get('category', 'Unknown')
            quality = data.get('quality', 'unknown')
            
            quality_counts[category][quality] += 1
        
        # Convert to DataFrame
        categories = sorted(set(self.entity_categories.keys()))
        qualities = ['good', 'uncertain', 'poor', 'unknown']
        
        data = []
        for category in categories:
            for quality in qualities:
                count = quality_counts[category][quality]
                data.append({
                    'Category': category,
                    'Quality': quality,
                    'Count': count
                })
        
        df = pd.DataFrame(data)
        
        # Pivot for plotting
        pivot_df = df.pivot(index='Category', columns='Quality', values='Count')
        pivot_df = pivot_df.fillna(0)
        
        # Ensure all qualities are present
        for quality in qualities:
            if quality not in pivot_df.columns:
                pivot_df[quality] = 0
        
        # Visualize as stacked bar chart
        plt.figure(figsize=(12, 8))
        pivot_df.plot(
            kind='bar', 
            stacked=True, 
            color=[self.quality_colors.get(q, 'gray') for q in pivot_df.columns],
            figsize=(12, 8)
        )
        
        plt.title("Entity Quality Distribution by Category")
        plt.xlabel("Category")
        plt.ylabel("Number of Entities")
        plt.legend(title="Quality")
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved quality distribution visualization to {output_path}")
        return output_path
    
    def visualize_relationship_types(self, filename: str = "relationship_types.png"):
        """
        Visualize the distribution of relationship types.
        
        Args:
            filename: Output filename
        """
        # Count relationships by type
        rel_counts = Counter()
        
        for _, _, data in self.network.edges(data=True):
            rel_type = data.get('type', 'unknown')
            rel_counts[rel_type] += 1
        
        # Sort by count
        sorted_rels = sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 15 for readability
        top_rels = sorted_rels[:15]
        
        # Visualize as bar chart
        plt.figure(figsize=(12, 8))
        
        rel_types, counts = zip(*top_rels) if top_rels else ([], [])
        
        plt.bar(rel_types, counts, color='steelblue')
        plt.title("Top Relationship Types")
        plt.xlabel("Relationship Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved relationship types visualization to {output_path}")
        return output_path
    
    def visualize_central_entities(self, filename: str = "central_entities.png", top_n: int = 20):
        """
        Visualize the most central entities in the network.
        
        Args:
            filename: Output filename
            top_n: Number of top entities to show
        """
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(self.network)
        betweenness_centrality = nx.betweenness_centrality(self.network, k=100)  # Use k for approximation in large networks
        
        # Combine centrality measures
        combined_centrality = {}
        for node in self.network.nodes():
            combined_centrality[node] = (
                degree_centrality.get(node, 0) * 0.5 + 
                betweenness_centrality.get(node, 0) * 0.5
            )
        
        # Get top N central entities
        top_entities = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Extract subgraph of central entities
        central_nodes = [node for node, _ in top_entities]
        central_graph = self.network.subgraph(central_nodes)
        
        # Visualize central entities
        plt.figure(figsize=(14, 12))
        
        # Create position layout
        pos = nx.spring_layout(central_graph, seed=42)
        
        # Draw nodes by category
        for category, color in self.category_colors.items():
            category_nodes = [n for n in central_nodes if self.network.nodes[n].get('category') == category]
            if category_nodes:
                nx.draw_networkx_nodes(
                    central_graph, 
                    pos, 
                    nodelist=category_nodes,
                    node_color=color,
                    node_size=[combined_centrality[n] * 5000 for n in category_nodes],
                    alpha=0.8,
                    label=category
                )
        
        # Draw edges
        nx.draw_networkx_edges(
            central_graph,
            pos,
            width=0.8,
            alpha=0.4,
            arrows=True
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            central_graph,
            pos,
            font_size=10,
            font_family='sans-serif'
        )
        
        plt.title(f"Top {top_n} Central Entities in the Network")
        plt.legend(scatterpoints=1, loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved central entities visualization to {output_path}")
        return output_path
    
    def generate_all_visualizations(self) -> Dict[str, str]:
        """
        Generate all visualizations.
        
        Returns:
            Dictionary mapping visualization types to file paths
        """
        visualizations = {}
        
        visualizations['quality'] = self.visualize_by_quality()
        visualizations['category'] = self.visualize_by_category()
        visualizations['category_subgraphs'] = self.visualize_category_subgraphs()
        visualizations['cross_category'] = self.visualize_cross_category_relationships()
        visualizations['quality_distribution'] = self.visualize_quality_distribution()
        visualizations['relationship_types'] = self.visualize_relationship_types()
        visualizations['central_entities'] = self.visualize_central_entities()
        
        return visualizations


def load_network_from_json(json_path: str) -> nx.DiGraph:
    """
    Load an entity network from a JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        NetworkX DiGraph
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    if 'entities' in data:
        for entity_id, entity_data in data['entities'].items():
            G.add_node(
                entity_data.get('text', entity_id),
                id=entity_id,
                quality=entity_data.get('quality', 'unknown'),
                category=entity_data.get('category', None),
                type='entity'
            )
    
    # Add edges
    if 'relationships' in data:
        for rel_id, rel_data in data['relationships'].items():
            source = rel_data.get('source', '')
            target = rel_data.get('target', '')
            
            if source and target:
                G.add_edge(
                    source,
                    target,
                    id=rel_id,
                    type=rel_data.get('relation_type', 'unknown'),
                    quality=rel_data.get('quality', 'unknown')
                )
    
    return G


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python entity_network_visualizer.py <results_json_path> [output_dir]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "visualizations"
    
    # Load network
    network = load_network_from_json(json_path)
    
    # Create visualizer
    visualizer = EntityNetworkVisualizer(network, output_dir)
    
    # Generate all visualizations
    visualizations = visualizer.generate_all_visualizations()
    
    print(f"Generated {len(visualizations)} visualizations in {output_dir}")
    for viz_type, path in visualizations.items():
        print(f"- {viz_type}: {path}")
