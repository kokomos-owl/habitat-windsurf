"""WebSocket connection manager."""

from typing import Dict, Set, Any
import json
import logging
import asyncio
from fastapi import WebSocket
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class WebSocketMessage(BaseModel):
    """WebSocket message structure."""
    type: str
    data: Dict[str, Any]

class ConnectionManager:
    """Manager for WebSocket connections."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.lock = asyncio.Lock()
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Handle new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            client_id: Client identifier
        """
        await websocket.accept()
        async with self.lock:
            if client_id not in self.active_connections:
                self.active_connections[client_id] = set()
            self.active_connections[client_id].add(websocket)
        logger.info(f"Client {client_id} connected")
        
    async def disconnect(self, websocket: WebSocket, client_id: str):
        """Handle WebSocket disconnection.
        
        Args:
            websocket: WebSocket connection
            client_id: Client identifier
        """
        async with self.lock:
            self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
        logger.info(f"Client {client_id} disconnected")
        
    async def send_personal_message(
        self,
        message: Dict[str, Any],
        websocket: WebSocket
    ):
        """Send message to specific client.
        
        Args:
            message: Message to send
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            
    async def broadcast(
        self,
        message: Dict[str, Any],
        client_id: str = None
    ):
        """Broadcast message to all connected clients.
        
        Args:
            message: Message to broadcast
            client_id: Optional client ID to restrict broadcast
        """
        async with self.lock:
            if client_id and client_id in self.active_connections:
                connections = self.active_connections[client_id]
            else:
                connections = [
                    ws for conns in self.active_connections.values()
                    for ws in conns
                ]
                
            for websocket in connections:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to broadcast message: {e}")
                    
    async def handle_message(
        self,
        websocket: WebSocket,
        client_id: str,
        message: Dict[str, Any]
    ):
        """Handle incoming WebSocket message.
        
        Args:
            websocket: WebSocket connection
            client_id: Client identifier
            message: Received message
        """
        try:
            msg = WebSocketMessage(**message)
            
            if msg.type == "subscribe":
                # Handle subscription to updates
                pass
            elif msg.type == "unsubscribe":
                # Handle unsubscription
                pass
            elif msg.type == "update":
                # Handle graph updates
                await self.broadcast(
                    {"type": "update", "data": msg.data},
                    client_id
                )
            else:
                logger.warning(f"Unknown message type: {msg.type}")
                
        except Exception as e:
            logger.error(f"Failed to handle message: {e}")
            await self.send_personal_message(
                {"type": "error", "data": {"message": str(e)}},
                websocket
            )
