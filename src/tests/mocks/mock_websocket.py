"""Mock WebSocket client for testing."""

from typing import Dict, Any, List, Optional
import json
import asyncio
from fastapi import WebSocket

class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        """Initialize mock websocket."""
        self.sent_messages: List[Dict[str, Any]] = []
        self.received_messages: List[Dict[str, Any]] = []
        self.connected = False
        
    async def accept(self):
        """Accept connection."""
        self.connected = True
        
    async def close(self):
        """Close connection."""
        self.connected = False
        
    async def send_json(self, data: Dict[str, Any]):
        """Send JSON message."""
        self.sent_messages.append(data)
        
    async def receive_json(self) -> Dict[str, Any]:
        """Receive JSON message."""
        if self.received_messages:
            return self.received_messages.pop(0)
        await asyncio.sleep(0.1)  # Simulate network delay
        raise RuntimeError("No messages available")
        
    def inject_message(self, message: Dict[str, Any]):
        """Inject message for testing."""
        self.received_messages.append(message)

class MockConnectionManager:
    """Mock connection manager for testing."""
    
    def __init__(self):
        """Initialize mock manager."""
        self.active_connections: Dict[str, List[MockWebSocket]] = {}
        self.broadcast_messages: List[Dict[str, Any]] = []
        
    async def connect(self, websocket: MockWebSocket, client_id: str):
        """Handle mock connection."""
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)
        
    async def disconnect(self, websocket: MockWebSocket, client_id: str):
        """Handle mock disconnection."""
        if client_id in self.active_connections:
            if websocket in self.active_connections[client_id]:
                self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
                
    async def broadcast(
        self,
        message: Dict[str, Any],
        client_id: Optional[str] = None
    ):
        """Mock message broadcast."""
        self.broadcast_messages.append({
            "message": message,
            "client_id": client_id
        })
        
        if client_id and client_id in self.active_connections:
            connections = self.active_connections[client_id]
        else:
            connections = [
                ws for conns in self.active_connections.values()
                for ws in conns
            ]
            
        for websocket in connections:
            await websocket.send_json(message)
