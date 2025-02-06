"""FastAPI router for visualization endpoints."""

from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel

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
    # Initialize clients
    await mongo_client.connect()
    await neo4j_client.connect()
    """Create new visualization.
    
    Args:
        request: Visualization request
        mongo_client: MongoDB client
        neo4j_client: Neo4j client
        
    Returns:
        Visualization metadata
    """
    try:
        # Initialize visualization
        visualizer = GraphVisualizer()
        
        # Create visualizations
        file_paths = await visualizer.create_evolution_view(
            request.temporal_stages,
            request.concept_evolution,
            request.relationship_changes,
            request.coherence_metrics
        )
        
        # Store in MongoDB
        doc_id = await mongo_client.store_visualization(
            request.doc_id,
            {
                "file_paths": file_paths,
                "metadata": request.dict()
            }
        )
        
        # Broadcast update
        await manager.broadcast(
            {
                "type": "visualization_created",
                "data": {"doc_id": doc_id}
            }
        )
        
        return {
            "visualization_id": doc_id,
            "file_paths": file_paths
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create visualization: {str(e)}"
        )

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
