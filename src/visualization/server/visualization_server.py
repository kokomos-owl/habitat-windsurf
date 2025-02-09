"""WebSocket server for topology visualization."""

import asyncio
import websockets
import json
from typing import Dict, Set
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import asdict

from src.core.metrics.flow_metrics import MetricFlowManager
from src.core.processor import ClimateRiskProcessor
from src.visualization.topology.TopologyConnector import TopologyConnector

logger = logging.getLogger(__name__)

class VisualizationServer:
    """WebSocket server for real-time topology visualization."""
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.flow_manager = MetricFlowManager()
        self.processor = ClimateRiskProcessor()
        self.connector = TopologyConnector(self.flow_manager)
        
    async def register(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
    async def unregister(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client."""
        self.clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def send_update(self, message: Dict):
        """Send update to all connected clients."""
        if not self.clients:
            return
        
        message_str = json.dumps(message)
        await asyncio.gather(
            *[client.send(message_str) for client in self.clients]
        )
        
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle client connection."""
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get('type') == 'request_update':
                        state = self.connector.get_visualization_state()
                        await websocket.send(json.dumps(state))
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        finally:
            await self.unregister(websocket)
            
    async def start_server(self):
        """Start the WebSocket server."""
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"Visualization server started at ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever
