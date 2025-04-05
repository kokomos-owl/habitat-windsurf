#!/usr/bin/env python3
"""
Run the Topological-Temporal Expression Visualizer.

This script creates and runs the visualization interface for the
topological-temporal potential framework.
"""
import sys
import os
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.habitat_evolution.visualization.topological_temporal_visualizer import create_visualizer


def main():
    """Run the topological-temporal expression visualizer."""
    parser = argparse.ArgumentParser(description='Run the Topological-Temporal Expression Visualizer')
    parser.add_argument('--host', default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    print(f"Starting Topological-Temporal Expression Visualizer on {args.host}:{args.port}")
    print("This visualizer demonstrates the Habitat Evolution system's ability to:")
    print("  - Visualize the semantic field and its potential gradients")
    print("  - Generate expressions from areas of high potential")
    print("  - Explore the co-evolutionary space of concepts and predicates")
    print("  - Detect and highlight areas of constructive dissonance")
    print("\nPress Ctrl+C to stop the server.")
    
    # Create and run the visualizer
    visualizer = create_visualizer()
    visualizer.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
