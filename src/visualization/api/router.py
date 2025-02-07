"""FastAPI router for visualization endpoints."""

from typing import Dict, Any, Optional
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from ..core.graph_visualizer import GraphVisualizer, VisualizationConfig
from ..websocket.manager import ConnectionManager
from ..core.db.mongo_client import MongoClient, MongoConfig
from ..core.db.neo4j_client import Neo4jClient, Neo4jConfig

router = APIRouter()
manager = ConnectionManager()

class VisualizationRequest(BaseModel):
    """Visualization request structure."""
    doc_id: str
    temporal_stages: list
    concept_evolution: Dict[str, list]
    relationship_changes: list
    coherence_metrics: Dict[str, float]

class VisualizationResponse(BaseModel):
    """Visualization response structure."""
    doc_id: str
    network_data: Dict[str, Any]
    timeline_data: Dict[str, Any]
    coherence_data: Dict[str, Any]

@router.post("/visualize", response_model=VisualizationResponse)
async def create_visualization(
    request: VisualizationRequest,
    mongo_client: MongoClient = Depends(lambda: MongoClient()),
    neo4j_client: Neo4jClient = Depends(lambda: Neo4jClient())
) -> Dict[str, Any]:
    """Create new visualization.
    
    Args:
        request: Visualization request
        mongo_client: MongoDB client
        neo4j_client: Neo4j client
        
    Returns:
        Visualization metadata
    """
    try:
        # Initialize clients
        await mongo_client.connect()
        await neo4j_client.connect()
        
        # Create visualizations
        visualizer = GraphVisualizer()
        vis_paths = await visualizer.create_evolution_view(
            request.temporal_stages,
            request.concept_evolution,
            request.relationship_changes,
            request.coherence_metrics
        )
        
        # Store in MongoDB
        doc = {
            "doc_id": request.doc_id,
            "file_paths": vis_paths,
            "metadata": {
                "temporal_stages": request.temporal_stages,
                "concept_evolution": request.concept_evolution,
                "relationship_changes": request.relationship_changes,
                "coherence_metrics": request.coherence_metrics
            }
        }
        
        # Note: We preserve _id and doc_id for habitat_evolution compatibility
        vis_id = await mongo_client.store_visualization(request.doc_id, doc)
        
        # Notify WebSocket clients
        await manager.broadcast(
            {"type": "visualization_created", "doc_id": request.doc_id}
        )
        
        return VisualizationResponse(
            doc_id=request.doc_id,
            network_data=network_data,
            timeline_data=timeline_data,
            coherence_data=coherence_data
        )
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await mongo_client.disconnect()
        await neo4j_client.disconnect()

@router.get("/visualize/{doc_id}", response_model=VisualizationResponse)
async def get_visualization(doc_id: str) -> VisualizationResponse:
    """Get visualization data by document ID."""
    try:
        # For testing, generate sample data
        if doc_id == "latest":
            # Create sample network data
            network_data = {
                "directed": True,
                "multigraph": False,
                "graph": {},
                "nodes": [
                    {"id": "Power Grid", "type": "concept", "weight": 0.95, "confidence": 0.95},
                    {"id": "Transportation", "type": "concept", "weight": 0.90, "confidence": 0.95},
                    {"id": "Storage", "type": "concept", "weight": 0.85, "confidence": 0.95},
                    {"id": "Infrastructure", "type": "concept", "weight": 0.92, "confidence": 0.95},
                    {"id": "Resilience", "type": "concept", "weight": 0.88, "confidence": 0.95}
                ],
                "links": [
                    {"source": "Power Grid", "target": "Infrastructure", "type": "default", "weight": 0.9, "stage": "stage1", "confidence": 0.95},
                    {"source": "Transportation", "target": "Infrastructure", "type": "default", "weight": 0.85, "stage": "stage1", "confidence": 0.95},
                    {"source": "Storage", "target": "Power Grid", "type": "default", "weight": 0.8, "stage": "stage2", "confidence": 0.95},
                    {"source": "Storage", "target": "Transportation", "type": "default", "weight": 0.75, "stage": "stage2", "confidence": 0.95},
                    {"source": "Resilience", "target": "Infrastructure", "type": "default", "weight": 0.95, "stage": "stage3", "confidence": 0.95}
                ],
                "metadata": {
                    "confidence_threshold": 0.95,
                    "relationship_types": ["concept", "temporal", "causal"],
                    "node_types": ["concept", "event", "entity"]
                }
            }
            
            return VisualizationResponse(
                doc_id=doc_id,
                network_data=network_data,
                timeline_data={},
                coherence_data={}
            )
            
        raise HTTPException(
            status_code=404,
            detail=f"Visualization not found for document {doc_id}"
        )
        
    except Exception as e:
        logger.error(f"Error getting visualization: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Error getting visualization: {str(e)}"
        )

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str
):
    """WebSocket endpoint for real-time updates.
    
    Args:
        websocket: WebSocket connection
        client_id: Client identifier
    """
    await manager.connect(websocket, client_id)
    try:
        while True:
            message = await websocket.receive_json()
            await manager.handle_message(websocket, client_id, message)
    except WebSocketDisconnect:
        await manager.disconnect(websocket, client_id)
    except Exception as e:
        await manager.disconnect(websocket, client_id)
        raise HTTPException(
            status_code=500,
            detail=f"WebSocket error: {str(e)}"
        )
