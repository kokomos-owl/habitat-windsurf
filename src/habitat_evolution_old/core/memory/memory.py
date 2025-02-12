"""
In-memory implementations of storage interfaces.

These implementations provide lightweight, in-memory storage suitable for
testing and development. They implement the full interface contracts while
maintaining data only for the lifetime of the process.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import copy
import uuid

from .interfaces import (
    StateStore,
    PatternStore,
    RelationshipStore,
    StorageResult,
    StorageMetadata
)
from ..services.time_provider import TimeProvider

class InMemoryStateStore(StateStore):
    """In-memory implementation of StateStore."""
    
    def __init__(self):
        self._states: Dict[str, List[Dict[str, Any]]] = {}  # id -> [versions]
        self._metadata: Dict[str, List[StorageMetadata]] = {}  # id -> [metadata]
    
    async def save_state(self,
                        id: str,
                        state: Dict[str, Any],
                        metadata: Optional[Dict[str, Any]] = None) -> StorageResult[str]:
        """Save state to memory."""
        try:
            if id not in self._states:
                self._states[id] = []
                self._metadata[id] = []
            
            # Deep copy to prevent mutations
            state_copy = copy.deepcopy(state)
            self._states[id].append(state_copy)
            
            # Create metadata
            meta = StorageMetadata(
                created_at=TimeProvider.now(),
                updated_at=TimeProvider.now(),
                version=str(uuid.uuid4()),
                tags=metadata.get('tags', []) if metadata else []
            )
            self._metadata[id].append(meta)
            
            return StorageResult(True, meta.version, metadata=meta)
            
        except Exception as e:
            return StorageResult(False, error=str(e))
    
    async def load_state(self,
                        id: str,
                        version: Optional[str] = None) -> StorageResult[Dict[str, Any]]:
        """Load state from memory."""
        try:
            if id not in self._states:
                return StorageResult(False, error="State not found")
                
            if version is None:
                # Get latest
                state = self._states[id][-1]
                metadata = self._metadata[id][-1]
            else:
                # Find specific version
                for i, meta in enumerate(self._metadata[id]):
                    if meta.version == version:
                        state = self._states[id][i]
                        metadata = meta
                        break
                else:
                    return StorageResult(False, error="Version not found")
            
            return StorageResult(True, copy.deepcopy(state), metadata=metadata)
            
        except Exception as e:
            return StorageResult(False, error=str(e))
    
    async def list_versions(self,
                          id: str) -> StorageResult[List[StorageMetadata]]:
        """List versions from memory."""
        try:
            if id not in self._metadata:
                return StorageResult(False, error="State not found")
            
            return StorageResult(True, copy.deepcopy(self._metadata[id]))
            
        except Exception as e:
            return StorageResult(False, error=str(e))

class InMemoryPatternStore(PatternStore):
    """In-memory implementation of PatternStore with history tracking."""
    
    def __init__(self):
        self._patterns: Dict[str, Dict[str, Any]] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}  # pattern_id -> [historical states]
        self._metadata: Dict[str, StorageMetadata] = {}
    
    async def save_pattern(self,
                          pattern: Dict[str, Any],
                          metadata: Optional[Dict[str, Any]] = None) -> StorageResult[str]:
        """Save pattern to memory and update history."""
        try:
            pattern_id = pattern.get("id", str(uuid.uuid4()))
            pattern_copy = copy.deepcopy(pattern)
            pattern_copy["id"] = pattern_id
            
            # Store current state in history
            if pattern_id in self._patterns:
                if pattern_id not in self._history:
                    self._history[pattern_id] = []
                self._history[pattern_id].append(self._patterns[pattern_id])
            
            # Update current state
            self._patterns[pattern_id] = pattern_copy
            
            meta = StorageMetadata(
                created_at=TimeProvider.now(),
                updated_at=TimeProvider.now(),
                version="1",  # Patterns are not versioned
                tags=metadata.get('tags', []) if metadata else []
            )
            self._metadata[pattern_id] = meta
            
            return StorageResult(True, pattern_id, metadata=meta)
            
        except Exception as e:
            return StorageResult(False, error=str(e))
    
    async def find_patterns(self,
                          query: Dict[str, Any],
                          limit: Optional[int] = None,
                          offset: Optional[int] = None) -> StorageResult[List[Dict[str, Any]]]:
        """Find patterns in memory."""
        try:
            # Simple matching
            matches = []
            for id, pattern in self._patterns.items():
                if all(pattern.get(k) == v for k, v in query.items()):
                    result = pattern.copy()
                    result['id'] = id
                    # Add history to pattern
                    if id in self._history:
                        result['history'] = copy.deepcopy(self._history[id])
                    matches.append(result)
            
            # Apply offset and limit
            if offset:
                matches = matches[offset:]
            if limit:
                matches = matches[:limit]
                
            return StorageResult(True, matches)
            
        except Exception as e:
            return StorageResult(False, error=str(e))
    
    async def delete_pattern(self, id: str) -> StorageResult[bool]:
        """Delete pattern from memory."""
        try:
            if id not in self._patterns:
                return StorageResult(False, error="Pattern not found")
                
            del self._patterns[id]
            del self._metadata[id]
            return StorageResult(True, True)
            
        except Exception as e:
            return StorageResult(False, error=str(e))

class InMemoryRelationshipStore(RelationshipStore):
    """In-memory implementation of RelationshipStore."""
    
    def __init__(self):
        self._relationships: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, StorageMetadata] = {}
        self._index: Dict[str, List[str]] = {}  # entity_id -> [relationship_ids]
    
    async def save_relationship(self,
                              source_id: str,
                              target_id: str,
                              type: str,
                              properties: Dict[str, Any],
                              metadata: Optional[Dict[str, Any]] = None) -> StorageResult[str]:
        """Save relationship to memory."""
        try:
            rel_id = str(uuid.uuid4())
            relationship = {
                'source_id': source_id,
                'target_id': target_id,
                'type': type,
                'properties': copy.deepcopy(properties)
            }
            self._relationships[rel_id] = relationship
            
            # Update indices
            if source_id not in self._index:
                self._index[source_id] = []
            self._index[source_id].append(rel_id)
            
            if target_id not in self._index:
                self._index[target_id] = []
            self._index[target_id].append(rel_id)
            
            meta = StorageMetadata(
                created_at=TimeProvider.now(),
                updated_at=TimeProvider.now(),
                version="1",
                tags=metadata.get('tags', []) if metadata else []
            )
            self._metadata[rel_id] = meta
            
            return StorageResult(True, rel_id, metadata=meta)
            
        except Exception as e:
            return StorageResult(False, error=str(e))
    
    async def find_relationships(self,
                               query: Dict[str, Any],
                               limit: Optional[int] = None,
                               offset: Optional[int] = None) -> StorageResult[List[Dict[str, Any]]]:
        """Find relationships in memory."""
        try:
            matches = []
            for id, rel in self._relationships.items():
                # Check main properties
                if all(rel.get(k) == v for k, v in query.items() if k != 'properties'):
                    # Check nested properties
                    if 'properties' in query:
                        if not all(rel['properties'].get(k) == v 
                                 for k, v in query['properties'].items()):
                            continue
                    
                    result = copy.deepcopy(rel)
                    result['id'] = id
                    matches.append(result)
            
            if offset:
                matches = matches[offset:]
            if limit:
                matches = matches[:limit]
                
            return StorageResult(True, matches)
            
        except Exception as e:
            return StorageResult(False, error=str(e))
    
    async def get_related(self,
                         id: str,
                         type: Optional[str] = None,
                         direction: Optional[str] = None) -> StorageResult[List[Dict[str, Any]]]:
        """Get related entities from memory."""
        try:
            if id not in self._index:
                return StorageResult(True, [])
            
            results = []
            for rel_id in self._index[id]:
                rel = self._relationships[rel_id]
                
                # Check type filter
                if type and rel['type'] != type:
                    continue
                    
                # Check direction filter
                if direction:
                    if direction == 'outgoing' and rel['source_id'] != id:
                        continue
                    if direction == 'incoming' and rel['target_id'] != id:
                        continue
                
                result = copy.deepcopy(rel)
                result['id'] = rel_id
                results.append(result)
                
            return StorageResult(True, results)
            
        except Exception as e:
            return StorageResult(False, error=str(e))
