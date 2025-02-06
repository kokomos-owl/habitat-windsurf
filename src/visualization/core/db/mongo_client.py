"""MongoDB client for visualization data."""

from typing import Dict, Any, Optional, List
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class MongoConfig(BaseModel):
    """MongoDB connection configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=27017)
    username: str = Field(default="admin")
    password: str = Field(default="password")
    database: str = Field(default="visualization")

class MongoClient:
    """Async MongoDB client for visualization data."""
    
    def __init__(self, config: Optional[MongoConfig] = None):
        """Initialize MongoDB client.
        
        Args:
            config: MongoDB configuration
        """
        self.config = config or MongoConfig()
        self.client = None
        self.db = None
        
    async def connect(self):
        """Establish database connection."""
        try:
            connection_url = (
                f"mongodb://{self.config.username}:{self.config.password}"
                f"@{self.config.host}:{self.config.port}"
            )
            self.client = AsyncIOMotorClient(connection_url)
            self.db = self.client[self.config.database]
            logger.info("Connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
            
    async def disconnect(self):
        """Close database connection."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
            
    async def store_visualization(
        self,
        doc_id: str,
        visualization_data: Dict[str, Any]
    ) -> str:
        """Store visualization data.
        
        Args:
            doc_id: Document identifier
            visualization_data: Visualization data to store
            
        Returns:
            Stored document ID
        """
        collection = self.db.visualizations
        result = await collection.update_one(
            {"doc_id": doc_id},
            {"$set": visualization_data},
            upsert=True
        )
        return str(result.upserted_id or doc_id)
        
    async def get_visualization(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve visualization data.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Visualization data if found
        """
        collection = self.db.visualizations
        doc = await collection.find_one({"doc_id": doc_id})
        return doc if doc else None
        
    async def list_visualizations(
        self,
        limit: int = 10,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """List stored visualizations.
        
        Args:
            limit: Maximum number of documents to return
            skip: Number of documents to skip
            
        Returns:
            List of visualization documents
        """
        collection = self.db.visualizations
        cursor = collection.find().skip(skip).limit(limit)
        return await cursor.to_list(length=limit)
