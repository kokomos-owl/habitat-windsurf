#!/usr/bin/env python3
"""
Runner script for Entity Network Visualizer.

This script generates enhanced visualizations from the context-aware NER
evolution test results to better understand the entity network evolution.
"""

import os
import sys
import json
import glob
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import the visualizer
from habitat_evolution.adaptive_core.demos.entity_network_visualizer_fixed import EntityNetworkVisualizer, load_network_from_json

def find_latest_results():
    """Find the latest results JSON file."""
    results_dir = Path(__file__).parent / "src" / "habitat_evolution" / "adaptive_core" / "demos" / "analysis_results"
    json_files = list(results_dir.glob("context_aware_ner_evolution_*.json"))
    
    if not json_files:
        print("No results files found. Please run the context-aware NER evolution test first.")
        sys.exit(1)
    
    # Sort by modification time (newest first)
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    return latest_file

def create_network_from_results(results_file):
    """Create a network from the results file."""
    print(f"Loading results from {results_file}")
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create graph
    G = nx.DiGraph()
    
    # We need to reconstruct the network from the results
    # This is a simplified version - the actual network would be more complex
    
    # Process document results to extract entities and relationships
    for doc_result in results.get('document_results', []):
        doc_name = doc_result.get('name', 'unknown')
        print(f"Processing entities from document: {doc_name}")
        
        # Since we don't have the actual entities and relationships in the results JSON,
        # we'll need to look for the raw data files
        
        # Try to find the corresponding raw data file
        doc_base = Path(results_file).parent
        raw_data_files = list(doc_base.glob(f"*{doc_name.replace('.txt', '')}*.json"))
        
        if raw_data_files:
            raw_data_file = raw_data_files[0]
            print(f"Found raw data file: {raw_data_file}")
            
            # Load raw data
            try:
                network = load_network_from_json(raw_data_file)
                print(f"Loaded network with {len(network.nodes)} nodes and {len(network.edges)} edges")
                return network
            except Exception as e:
                print(f"Error loading raw data: {e}")
    
    # If we couldn't find raw data, create a mock network from the results
    print("Creating mock network from results summary")
    
    # Add some example nodes based on entity categories
    categories = results.get('entities', {}).get('by_category', {})
    for category, count in categories.items():
        for i in range(min(count, 10)):  # Limit to 10 per category for simplicity
            node_name = f"{category}_{i+1}"
            G.add_node(
                node_name,
                category=category,
                quality='good' if i < 7 else 'uncertain',  # 70% good, 30% uncertain
                type='entity'
            )
    
    # Add some example edges
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, min(i+5, len(nodes))):
            G.add_edge(
                nodes[i],
                nodes[j],
                type='related_to',
                quality='uncertain'
            )
    
    print(f"Created mock network with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

def main():
    """Main function."""
    # Find latest results
    results_file = find_latest_results()
    print(f"Found latest results file: {results_file}")
    
    # Create output directory
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Create network
    network = create_network_from_results(results_file)
    
    # Create visualizer
    visualizer = EntityNetworkVisualizer(network, str(output_dir))
    
    # Generate all visualizations
    print("Generating visualizations...")
    visualizations = visualizer.generate_all_visualizations()
    
    print(f"\nGenerated {len(visualizations)} visualizations in {output_dir}:")
    for viz_type, path in visualizations.items():
        if isinstance(path, list):
            for p in path:
                print(f"- {viz_type}: {p}")
        else:
            print(f"- {viz_type}: {path}")
    
    print("\nThese visualizations provide different perspectives on the entity network:")
    print("1. Quality: Shows entities colored by their quality state (good, uncertain, poor)")
    print("2. Category: Shows entities colored by their category (climate hazard, ecosystem, etc.)")
    print("3. Category Subgraphs: Shows detailed views of each category's internal structure")
    print("4. Cross-Category: Shows relationships between different entity categories")
    print("5. Quality Distribution: Shows the distribution of entity qualities by category")
    print("6. Relationship Types: Shows the most common relationship types in the network")
    print("7. Central Entities: Shows the most central entities in the network")

if __name__ == "__main__":
    main()
