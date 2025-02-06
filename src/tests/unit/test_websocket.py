"""Tests for WebSocket functionality."""

import pytest
from typing import Dict, Any

from src.tests.mocks.mock_websocket import MockWebSocket, MockConnectionManager

pytestmark = pytest.mark.asyncio

async def test_websocket_connection(
    mock_websocket: MockWebSocket,
    mock_manager: MockConnectionManager
):
    """Test WebSocket connection management."""
    client_id = "test_client"
    
    # Test connection
    await mock_manager.connect(mock_websocket, client_id)
    assert mock_websocket.connected
    assert client_id in mock_manager.active_connections
    assert mock_websocket in mock_manager.active_connections[client_id]
    
    # Test disconnection
    await mock_manager.disconnect(mock_websocket, client_id)
    assert client_id not in mock_manager.active_connections

async def test_message_broadcast(
    mock_websocket: MockWebSocket,
    mock_manager: MockConnectionManager
):
    """Test message broadcasting."""
    client_id = "test_client"
    await mock_manager.connect(mock_websocket, client_id)
    
    # Test broadcasting message
    test_message = {
        "type": "update",
        "data": {"test": "data"}
    }
    await mock_manager.broadcast(test_message, client_id)
    
    # Verify broadcast was recorded
    assert len(mock_manager.broadcast_messages) == 1
    assert mock_manager.broadcast_messages[0]["message"] == test_message
    assert mock_manager.broadcast_messages[0]["client_id"] == client_id
    
    # Verify message was sent to websocket
    assert len(mock_websocket.sent_messages) == 1
    assert mock_websocket.sent_messages[0] == test_message

async def test_message_handling(
    mock_websocket: MockWebSocket,
    mock_manager: MockConnectionManager
):
    """Test message handling."""
    client_id = "test_client"
    await mock_manager.connect(mock_websocket, client_id)
    
    # Test subscribe message
    subscribe_msg = {
        "type": "subscribe",
        "data": {"topic": "test_topic"}
    }
    mock_websocket.inject_message(subscribe_msg)
    received = await mock_websocket.receive_json()
    assert received == subscribe_msg
    
    # Test update message
    update_msg = {
        "type": "update",
        "data": {"test": "update"}
    }
    mock_websocket.inject_message(update_msg)
    received = await mock_websocket.receive_json()
    assert received == update_msg

async def test_multiple_clients(mock_manager: MockConnectionManager):
    """Test handling multiple WebSocket clients."""
    client1 = MockWebSocket()
    client2 = MockWebSocket()
    
    # Connect both clients
    await mock_manager.connect(client1, "client1")
    await mock_manager.connect(client2, "client2")
    
    # Broadcast to all clients
    broadcast_msg = {
        "type": "broadcast",
        "data": {"message": "all"}
    }
    await mock_manager.broadcast(broadcast_msg)
    
    # Verify both clients received the message
    assert len(client1.sent_messages) == 1
    assert len(client2.sent_messages) == 1
    assert client1.sent_messages[0] == broadcast_msg
    assert client2.sent_messages[0] == broadcast_msg
    
    # Broadcast to specific client
    client_msg = {
        "type": "personal",
        "data": {"message": "one"}
    }
    await mock_manager.broadcast(client_msg, "client1")
    
    # Verify only client1 received the message
    assert len(client1.sent_messages) == 2
    assert len(client2.sent_messages) == 1
    assert client1.sent_messages[1] == client_msg
