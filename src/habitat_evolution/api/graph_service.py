"""Graph service for processing text and generating NetworkX visualizations."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import networkx as nx
import json
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import io
import base64
from ..visualization.test_visualization import TestPatternVisualizer
from ..visualization.pattern_id import PatternAdaptiveID

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextSelectionRequest(BaseModel):
    text: str
    
class GraphResponse(BaseModel):
    graph_image: str  # Base64 encoded image
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

@app.post("/api/process-text", response_model=GraphResponse)
async def process_text(request: TextSelectionRequest):
    """Process selected text and return graph visualization."""
    try:
        # Initialize visualizer
        visualizer = TestPatternVisualizer()
        
        # Create test patterns for hazard types found in text
        patterns = []
        hazard_types = ["extreme_precipitation", "drought", "wildfire"]
        
        for hazard_type in hazard_types:
            if hazard_type.lower() in request.text.lower():
                pattern = PatternAdaptiveID(
                    pattern_type="event",
                    hazard_type=hazard_type
                )
                patterns.append(pattern)
        
        # Create relationships between patterns
        for i in range(len(patterns)-1):
            patterns[i].add_relationship(
                patterns[i+1].id,
                "INTERACTS_WITH",
                {"spatial_distance": 1.0, "coherence_similarity": 0.8}
            )
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        nodes = []
        edges = []
        
        for pattern in patterns:
            node_data = pattern.to_dict()
            G.add_node(pattern.id, **node_data)
            nodes.append(node_data)
            
            for target_id, relationships in pattern.relationships.items():
                latest = sorted(relationships, key=lambda x: x['timestamp'])[-1]
                G.add_edge(pattern.id, target_id, **latest)
                edges.append({
                    "source": pattern.id,
                    "target": target_id,
                    "type": latest["type"],
                    **latest["metrics"]
                })
        
        # Generate visualization
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, arrowsize=20)
        
        # Convert plot to base64 image
        img_stream = io.BytesIO()
        plt.savefig(img_stream, format='png', bbox_inches='tight')
        plt.close()
        img_stream.seek(0)
        img_base64 = base64.b64encode(img_stream.read()).decode()
        
        return GraphResponse(
            graph_image=img_base64,
            nodes=nodes,
            edges=edges
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
