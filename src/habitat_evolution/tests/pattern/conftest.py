"""Local test configuration."""

import os
import pytest
from typing import Dict, Any, List
from unittest.mock import patch
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

# Mock Embeddings for Testing
class MockEmbeddings(Embeddings):
    """Mock embeddings for testing."""
    def __init__(self):
        self.embed_count = 0
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Mock document embedding."""
        self.embed_count += len(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Mock query embedding."""
        self.embed_count += 1
        return [0.1, 0.2, 0.3]

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Setup mock environment variables for testing."""
    with patch.dict(os.environ, {
        'COHERE_API_KEY': 'mock-cohere-key',
        'LANGCHAIN_API_KEY': 'mock-langchain-key',
        'ANTHROPIC_API_KEY': 'mock-anthropic-key'
    }):
        yield

@pytest.fixture
def mock_embeddings():
    """Provide mock embeddings."""
    return MockEmbeddings()

@pytest.fixture
def mock_db():
    """Simple mock database."""
    class MockDB:
        def __init__(self):
            self.data = {}
            
        async def store(self, key: str, value: Dict[str, Any]) -> None:
            self.data[key] = value
            
        async def get(self, key: str) -> Dict[str, Any]:
            return self.data.get(key, {})
    
    return MockDB()

@pytest.fixture
def mock_websocket():
    """Simple mock websocket."""
    class MockWebSocket:
        def __init__(self):
            self.messages = []
            
        async def send(self, message: str) -> None:
            self.messages.append(message)
            
        async def receive(self) -> str:
            return self.messages.pop(0) if self.messages else ""
    
    return MockWebSocket()

@pytest.fixture
def sample_document() -> Document:
    """Provide a sample document for testing."""
    return Document(
        page_content="Test content for embeddings",
        metadata={
            "source": "test",
            "timestamp": "2025-02-16T11:20:36-05:00"
        }
    )
