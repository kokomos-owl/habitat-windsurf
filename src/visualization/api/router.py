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
    visualization_id: str
    file_paths: Dict[str, str]

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
            visualization_id=vis_id,
            file_paths=vis_paths
        )
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await mongo_client.disconnect()
        await neo4j_client.disconnect()

@router.get("/visualize/{doc_id}")
async def get_visualization(
    doc_id: str,
    mongo_client: MongoClient = Depends(lambda: MongoClient())
) -> Dict[str, Any]:
    # Initialize client
    await mongo_client.connect()
    """Retrieve visualization data.
    
    Args:
        doc_id: Document identifier
        mongo_client: MongoDB client
        
    Returns:
        Visualization data
    """
    doc = await mongo_client.get_visualization(doc_id)
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Visualization {doc_id} not found"
        )
    return doc

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
