"""Agentic Workshop Builder for Habitat Windsurf UI Course.

This script demonstrates how Windsurf can build and tear down the workshop environment,
showing the power of agentic development in action.
"""

import os
import sys
import time
import shutil
from pathlib import Path
from typing import List, Dict, Optional

class WorkshopBuilder:
    """Agentic builder for workshop environment."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.components_dir = self.base_dir / "src" / "core" / "visualization"
        self.notebooks_dir = self.base_dir / "notebooks"
        self.tests_dir = self.base_dir / "tests"
        
    def build_step(self, message: str, delay: float = 1.0):
        """Display build step with animation."""
        print(f"\n🔨 {message}", end="")
        for _ in range(3):
            time.sleep(delay/3)
            print(".", end="", flush=True)
        print(" ✓")
        
    def create_directory_structure(self):
        """Create core directory structure."""
        self.build_step("Creating directory structure")
        
        dirs = [
            self.components_dir,
            self.notebooks_dir / "lesson_01",
            self.tests_dir
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def implement_visualization_components(self):
        """Create visualization components."""
        self.build_step("Implementing GraphVisualizer")
        
        graph_viz_content = '''"""Graph visualization module adapted from habitat_poc."""

import os
from typing import Dict, Any, Optional
import networkx as nx
import plotly.graph_objects as go

class GraphVisualizer:
    """Visualizer for graph components using plotly."""

    def __init__(self, layout_engine=None):
        """Initialize visualizer."""
        self.layout_engine = layout_engine
        self.default_node_color = '#1f77b4'
        self.default_edge_color = '#7f7f7f'
'''
        
        with open(self.components_dir / "graph.py", "w") as f:
            f.write(graph_viz_content)
            
        self.build_step("Implementing LayoutEngine")
        
        layout_content = '''"""Layout engine for graph visualization."""

from typing import Dict, Any, Optional, Tuple
import networkx as nx

class LayoutEngine:
    """Engine for calculating graph layouts."""
    
    def __init__(self, default_layout='spring'):
        """Initialize layout engine."""
        self.default_layout = default_layout
        self._layout_funcs = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout
        }
'''
        
        with open(self.components_dir / "layout.py", "w") as f:
            f.write(layout_content)
            
    def create_notebook(self):
        """Create example notebook."""
        self.build_step("Creating example notebook")
        
        notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Getting Started with Graph Visualization\\n",
    "This notebook introduces basic graph visualization concepts."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "import networkx as nx\\n",
    "from src.core.visualization.graph import GraphVisualizer\\n",
    "from src.core.visualization.layout import LayoutEngine"
   ]
  }
 ],
 "metadata": {"kernelspec": {"name": "python3"}},
 "nbformat": 4,
 "nbformat_minor": 4
}'''
        
        with open(self.notebooks_dir / "lesson_01" / "basics.ipynb", "w") as f:
            f.write(notebook_content)
            
    def create_tests(self):
        """Create test suite."""
        self.build_step("Implementing test suite")
        
        test_content = '''"""Test suite for visualization components."""

import pytest
import networkx as nx
from src.core.visualization.graph import GraphVisualizer
from src.core.visualization.layout import LayoutEngine

def test_graph_visualizer():
    """Test basic visualization creation."""
    viz = GraphVisualizer()
    assert viz is not None
'''
        
        with open(self.tests_dir / "test_graph_visualization.py", "w") as f:
            f.write(test_content)
            
    def build_workshop(self):
        """Build complete workshop environment."""
        print("\n🚀 Building Habitat Windsurf UI Workshop\n")
        
        steps = [
            self.create_directory_structure,
            self.implement_visualization_components,
            self.create_notebook,
            self.create_tests
        ]
        
        for step in steps:
            step()
            
        print("\n✨ Workshop environment built successfully!")
        
    def tear_down(self):
        """Remove workshop environment."""
        print("\n🧹 Cleaning up workshop environment")
        
        dirs = [
            self.components_dir,
            self.notebooks_dir,
            self.tests_dir
        ]
        
        for dir_path in dirs:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"Removed {dir_path}")
                
        print("\n🏁 Environment cleaned successfully!")

def main():
    """Run workshop builder."""
    # Get repository root
    repo_root = Path(__file__).parent.parent.parent
    
    builder = WorkshopBuilder(repo_root)
    
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        builder.tear_down()
    else:
        builder.build_workshop()

if __name__ == "__main__":
    main()
