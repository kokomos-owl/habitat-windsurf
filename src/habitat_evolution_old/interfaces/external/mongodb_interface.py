"""
MongoDB interface for pattern storage and field states.
"""
from typing import Dict, Any, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

from ...core.pattern.types import Pattern, FieldState
from ...core.pattern.metrics import PatternMetrics

class MongoDBFieldStore:
    """MongoDB-based field state storage."""
    
    def __init__(self, uri: str, database: str):
        self._client = AsyncIOMotorClient(uri)
        self._db = self._client[database]
        self._fields = self._db.fields
        self._patterns = self._db.patterns
    
    async def store_field_state(self, field_state: FieldState) -> str:
        """Store field state in MongoDB.
        
        Args:
            field_state: Field state to store
            
        Returns:
            Field state ID
        """
        document = {
            'gradients': {
                'coherence': field_state.gradients.coherence,
                'energy': field_state.gradients.energy,
                'density': field_state.gradients.density,
                'turbulence': field_state.gradients.turbulence
            },
            'timestamp': field_state.timestamp,
            'created_at': datetime.now()
        }
        
        result = await self._fields.insert_one(document)
        return str(result.inserted_id)
    
    async def get_field_state(self, field_id: str) -> Optional[FieldState]:
        """Retrieve field state from MongoDB."""
        document = await self._fields.find_one({'_id': field_id})
        if not document:
            return None
            
        return FieldState(
            gradients=document['gradients'],
            patterns=[],  # Patterns loaded separately
            timestamp=document['timestamp']
        )
    
    async def store_pattern(self, pattern: Pattern) -> str:
        """Store pattern in MongoDB."""
        document = {
            'id': pattern['id'],
            'coherence': pattern['coherence'],
            'energy': pattern['energy'],
            'state': pattern['state'],
            'metrics': pattern['metrics'],
            'relationships': pattern.get('relationships', []),
            'created_at': datetime.now()
        }
        
        result = await self._patterns.insert_one(document)
        return str(result.inserted_id)
    
    async def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Retrieve pattern from MongoDB."""
        document = await self._patterns.find_one({'id': pattern_id})
        if not document:
            return None
            
        return {
            'id': document['id'],
            'coherence': document['coherence'],
            'energy': document['energy'],
            'state': document['state'],
            'metrics': document['metrics'],
            'relationships': document.get('relationships', [])
        }
    
    async def get_patterns_in_timeframe(self,
                                      start_time: float,
                                      end_time: float) -> List[Pattern]:
        """Get patterns within a timeframe."""
        cursor = self._patterns.find({
            'created_at': {
                '$gte': datetime.fromtimestamp(start_time),
                '$lte': datetime.fromtimestamp(end_time)
            }
        })
        
        patterns = []
        async for document in cursor:
            patterns.append({
                'id': document['id'],
                'coherence': document['coherence'],
                'energy': document['energy'],
                'state': document['state'],
                'metrics': document['metrics'],
                'relationships': document.get('relationships', [])
            })
            
        return patterns
    
    async def update_pattern(self, pattern_id: str,
                           updates: Dict[str, Any]) -> bool:
        """Update pattern in MongoDB."""
        result = await self._patterns.update_one(
            {'id': pattern_id},
            {'$set': updates}
        )
        
        return result.modified_count > 0
