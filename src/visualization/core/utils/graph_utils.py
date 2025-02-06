"""Utility functions for graph visualization."""

from typing import Dict, Any, List, Tuple
import networkx as nx
from pydantic import BaseModel

class GraphMetrics(BaseModel):
    """Graph metrics container."""
    node_count: int
    edge_count: int
    density: float
    avg_degree: float
    clustering_coefficient: float

def calculate_graph_metrics(graph: nx.Graph) -> GraphMetrics:
    """Calculate basic graph metrics.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        GraphMetrics object with calculated metrics
    """
    return GraphMetrics(
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        density=nx.density(graph),
        avg_degree=sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
        clustering_coefficient=nx.average_clustering(graph)
    )

def extract_subgraph(
    graph: nx.Graph,
    center_node: str,
    max_distance: int = 2
) -> nx.Graph:
    """Extract subgraph centered on a node.
    
    Args:
        graph: Input graph
        center_node: Central node for subgraph
        max_distance: Maximum distance from center node
        
    Returns:
        Subgraph containing nodes within max_distance of center_node
    """
    nodes = {center_node}
    for distance in range(max_distance):
        boundary = set()
        for node in nodes:
            boundary.update(graph.neighbors(node))
        nodes.update(boundary)
    
    return graph.subgraph(nodes)

def find_communities(
    graph: nx.Graph,
    algorithm: str = "louvain"
) -> Dict[str, int]:
    """Detect communities in graph.
    
    Args:
        graph: Input graph
        algorithm: Community detection algorithm to use
        
    Returns:
        Dictionary mapping node ids to community ids
    """
    if algorithm == "louvain":
        communities = nx.community.louvain_communities(graph)
    else:
        communities = nx.community.label_propagation_communities(graph)
        
    community_map = {}
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i
            
    return community_map

def calculate_centrality_metrics(
    graph: nx.Graph
) -> Dict[str, Dict[str, float]]:
    """Calculate various centrality metrics.
    
    Args:
        graph: Input graph
        
    Returns:
        Dictionary mapping node ids to their centrality metrics
    """
    metrics = {
        "degree": nx.degree_centrality(graph),
        "betweenness": nx.betweenness_centrality(graph),
        "closeness": nx.closeness_centrality(graph),
        "eigenvector": nx.eigenvector_centrality(graph, max_iter=1000)
    }
    
    # Reorganize by node
    result = {}
    for node in graph.nodes():
        result[node] = {
            metric: values[node]
            for metric, values in metrics.items()
        }
        
    return result
