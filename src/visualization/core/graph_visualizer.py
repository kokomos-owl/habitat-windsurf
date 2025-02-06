"""Graph visualization module for knowledge evolution tracking."""

from typing import Dict, List, Any, Optional
import os
import json
import logging
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class VisualizationConfig(BaseModel):
    """Configuration for visualization settings."""
    output_dir: str = Field(default="visualizations")
    max_nodes: int = Field(default=100)
    layout_algorithm: str = Field(default="force_directed")
    node_size: int = Field(default=20)
    edge_width: float = Field(default=1.0)
    timeline_height: int = Field(default=400)
    network_height: int = Field(default=600)
    coherence_height: int = Field(default=300)

class GraphVisualizer:
    """Visualizer for knowledge graph evolution with timeline and network views."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize visualizer.
        
        Args:
            config: Optional configuration settings
        """
        self.config = config or VisualizationConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def create_evolution_view(
        self,
        temporal_stages: List[str],
        concept_evolution: Dict[str, List[Dict[str, Any]]],
        relationship_changes: List[Dict[str, str]],
        coherence_metrics: Dict[str, float]
    ) -> Dict[str, str]:
        """Create visualizations showing knowledge evolution.
        
        Args:
            temporal_stages: List of stage names in temporal order
            concept_evolution: Dictionary mapping concepts to their evolution data
            relationship_changes: List of relationship changes between stages
            coherence_metrics: Dictionary of coherence metrics
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        # Validate inputs
        if not temporal_stages:
            raise ValueError("No temporal stages provided")
        if not concept_evolution:
            raise ValueError("No concept evolution data provided")
        if not relationship_changes:
            raise ValueError("No relationship changes provided")
        if not coherence_metrics:
            raise ValueError("No coherence metrics provided")
        """Create visualizations showing knowledge evolution.
        
        Args:
            temporal_stages: List of stage names in temporal order
            concept_evolution: Dictionary mapping concepts to their evolution data
            relationship_changes: List of relationship changes between stages
            coherence_metrics: Dictionary of coherence metrics
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        try:
            # Create timeline visualization
            timeline_data = await self._create_timeline_data(temporal_stages, concept_evolution)
            timeline_path = self.output_dir / "timeline.json"
            timeline_path.write_text(json.dumps(timeline_data))
            
            # Create network visualization
            network_data = await self._create_network_data(relationship_changes)
            network_path = self.output_dir / "network.json"
            network_path.write_text(json.dumps(network_data))
            
            # Create coherence visualization
            coherence_data = self._create_coherence_data(coherence_metrics)
            coherence_path = self.output_dir / "coherence.json"
            coherence_path.write_text(json.dumps(coherence_data))
            
            return {
                "timeline": str(timeline_path),
                "network": str(network_path),
                "coherence": str(coherence_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to create evolution view: {e}")
            raise
            
    async def _create_timeline_data(
        self,
        temporal_stages: List[str],
        concept_evolution: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Create timeline visualization data."""
        return {
            "stages": temporal_stages,
            "evolution": concept_evolution
        }
        
    async def _create_network_data(
        self,
        relationship_changes: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Create network visualization data."""
        graph = nx.DiGraph()
        
        # Add relationships
        for rel in relationship_changes:
            graph.add_edge(
                rel["from"],
                rel["to"],
                type=rel.get("type", "default"),
                weight=rel.get("weight", 1.0)
            )
            
        # Convert to visualization format
        return nx.node_link_data(graph)
        
    def _create_coherence_data(
        self,
        coherence_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create coherence visualization data."""
        return {
            "metrics": coherence_metrics,
            "timestamp": str(self.output_dir.stat().st_ctime)
        }
