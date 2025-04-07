"""
Caching mechanism for Claude API responses.

This module provides a caching system for Claude API responses to reduce
API costs and improve response times for common queries.
"""

import hashlib
import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ClaudeCache:
    """
    Cache for Claude API responses.
    
    This class provides a caching mechanism for Claude API responses,
    reducing API costs and improving response times for common queries.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, ttl_hours: int = 24):
        """
        Initialize the Claude cache.
        
        Args:
            cache_dir: Optional directory to store cache data (if None, will use default)
            ttl_hours: Time-to-live for cache entries in hours (default: 24)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parents[3] / "data" / "cache" / "claude"
        self.ttl_hours = ttl_hours
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "expired": 0
        }
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Claude cache (cache_dir: {self.cache_dir}, ttl_hours: {ttl_hours})")
    
    def get_query_cache_key(self, query: str, context: Dict[str, Any], patterns: List[Dict[str, Any]]) -> str:
        """
        Generate a cache key for a query.
        
        Args:
            query: The query text
            context: The context for the query
            patterns: The patterns for the query
            
        Returns:
            A unique cache key
        """
        # Create a deterministic representation of the input
        cache_input = {
            "query": query,
            "context": context,
            "patterns": patterns
        }
        
        # Convert to JSON and hash
        cache_json = json.dumps(cache_input, sort_keys=True)
        return hashlib.md5(cache_json.encode()).hexdigest()
    
    def get_document_cache_key(self, document: Dict[str, Any]) -> str:
        """
        Generate a cache key for a document.
        
        Args:
            document: The document to process
            
        Returns:
            A unique cache key
        """
        # Use document ID if available, otherwise hash the content
        doc_id = document.get("id")
        if doc_id:
            content_hash = hashlib.md5(document.get("content", "").encode()).hexdigest()[:8]
            return f"{doc_id}-{content_hash}"
        else:
            # Create a deterministic representation of the document
            cache_input = {
                "content": document.get("content", ""),
                "metadata": document.get("metadata", {})
            }
            
            # Convert to JSON and hash
            cache_json = json.dumps(cache_input, sort_keys=True)
            return hashlib.md5(cache_json.encode()).hexdigest()
    
    def get_cache_path(self, cache_key: str) -> Path:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key: The cache key
            
        Returns:
            Path to the cache file
        """
        return self.cache_dir / f"{cache_key}.pickle"
    
    def get_from_cache(self, cache_key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Get a response from the cache.
        
        Args:
            cache_key: The cache key
            
        Returns:
            Tuple of (cache_hit, response)
        """
        cache_path = self.get_cache_path(cache_key)
        
        if not cache_path.exists():
            self.cache_stats["misses"] += 1
            return False, None
        
        try:
            with open(cache_path, "rb") as f:
                cache_entry = pickle.load(f)
            
            # Check if the cache entry has expired
            timestamp = cache_entry.get("timestamp", datetime.min)
            expiry = timestamp + timedelta(hours=self.ttl_hours)
            
            if datetime.now() > expiry:
                self.cache_stats["expired"] += 1
                return False, None
            
            self.cache_stats["hits"] += 1
            return True, cache_entry.get("response")
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            self.cache_stats["misses"] += 1
            return False, None
    
    def save_to_cache(self, cache_key: str, response: Dict[str, Any]):
        """
        Save a response to the cache.
        
        Args:
            cache_key: The cache key
            response: The response to cache
        """
        cache_path = self.get_cache_path(cache_key)
        
        try:
            cache_entry = {
                "timestamp": datetime.now(),
                "response": response
            }
            
            with open(cache_path, "wb") as f:
                pickle.dump(cache_entry, f)
            
            logger.debug(f"Saved response to cache: {cache_key}")
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def clear_cache(self, older_than_hours: Optional[int] = None):
        """
        Clear the cache.
        
        Args:
            older_than_hours: If provided, only clear entries older than this many hours
        """
        if older_than_hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            cleared_count = 0
            
            for cache_file in self.cache_dir.glob("*.pickle"):
                try:
                    with open(cache_file, "rb") as f:
                        cache_entry = pickle.load(f)
                    
                    timestamp = cache_entry.get("timestamp", datetime.min)
                    if timestamp < cutoff_time:
                        cache_file.unlink()
                        cleared_count += 1
                except Exception:
                    # If we can't read the file, just delete it
                    cache_file.unlink()
                    cleared_count += 1
            
            logger.info(f"Cleared {cleared_count} cache entries older than {older_than_hours} hours")
        else:
            # Clear all cache entries
            for cache_file in self.cache_dir.glob("*.pickle"):
                cache_file.unlink()
            
            logger.info("Cleared all cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests) if total_requests > 0 else 0
        
        # Count cache entries
        cache_entries = list(self.cache_dir.glob("*.pickle"))
        cache_size_bytes = sum(f.stat().st_size for f in cache_entries)
        
        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "expired": self.cache_stats["expired"],
            "hit_rate": hit_rate,
            "cache_entries": len(cache_entries),
            "cache_size_bytes": cache_size_bytes,
            "cache_size_mb": cache_size_bytes / (1024 * 1024)
        }


# Global cache instance
claude_cache = ClaudeCache()
