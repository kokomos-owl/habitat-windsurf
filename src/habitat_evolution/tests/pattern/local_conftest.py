"""Local test configuration."""

import pytest
from typing import Dict, Any

class MockDB:
    """Simple mock database."""
    def __init__(self):
        self.data = {}
        
    async def store(self, key: str, value: Dict[str, Any]) -> None:
        self.data[key] = value
        
    async def get(self, key: str) -> Dict[str, Any]:
        return self.data.get(key, {})

class MockWebSocket:
    """Simple mock websocket."""
    def __init__(self):
        self.messages = []
        
    async def send(self, message: str) -> None:
        self.messages.append(message)
        
    async def receive(self) -> str:
        return self.messages.pop(0) if self.messages else ""

@pytest.fixture
def mock_db():
    """Provide mock database."""
    return MockDB()

@pytest.fixture
def mock_websocket():
    """Provide mock websocket."""
    return MockWebSocket()
