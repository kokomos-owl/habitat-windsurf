"""
Time provider service for adaptive core.

This module provides a simple, testable interface for time-based operations.
It replaces the more complex TimestampService with a streamlined approach
that maintains UTC consistency and ISO format compatibility.
"""

from datetime import datetime, UTC
from typing import Optional

class TimeProvider:
    """Simple interface for time-based operations."""
    
    @staticmethod
    def now() -> datetime:
        """Get current UTC time.
        
        Returns:
            datetime: Current time in UTC
        """
        return datetime.now(UTC)
    
    @staticmethod
    def parse(timestamp_str: str) -> datetime:
        """Parse ISO format timestamp string to datetime.
        
        Args:
            timestamp_str: ISO format timestamp string
            
        Returns:
            datetime: Parsed datetime object
            
        Raises:
            ValueError: If timestamp string is invalid
        """
        return datetime.fromisoformat(timestamp_str)
    
    @staticmethod
    def format(dt: datetime) -> str:
        """Format datetime to ISO format string.
        
        Args:
            dt: Datetime object to format
            
        Returns:
            str: ISO format timestamp string
        """
        return dt.isoformat()
    
    @staticmethod
    def compare(dt1: datetime, dt2: datetime) -> int:
        """Compare two datetimes, handling timezone differences.
        
        Args:
            dt1: First datetime
            dt2: Second datetime
            
        Returns:
            int: -1 if dt1 < dt2, 0 if equal, 1 if dt1 > dt2
        """
        # Ensure both times are in UTC
        if dt1.tzinfo is None:
            dt1 = dt1.replace(tzinfo=UTC)
        if dt2.tzinfo is None:
            dt2 = dt2.replace(tzinfo=UTC)
            
        if dt1 < dt2:
            return -1
        elif dt1 > dt2:
            return 1
        return 0
