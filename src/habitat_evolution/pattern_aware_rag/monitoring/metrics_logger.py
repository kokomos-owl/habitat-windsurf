"""Simple metrics logger for vector attention monitoring."""

import logging
from datetime import datetime
from typing import Dict, Any

class MetricsLogger:
    """Log metrics for monitoring and navigation."""
    
    def __init__(self, name: str = "vector_attention"):
        """Initialize metrics logger."""
        self.logger = logging.getLogger(name)
    
    def log_metrics(self, context: Dict[str, Any], metrics: Dict[str, Any]):
        """Log metrics with context."""
        self.logger.info(f"Context: {context}, Metrics: {metrics}")
        
    def log_error(self, error: str, context: Dict[str, Any] = None):
        """Log error with optional context."""
        self.logger.error(f"Error: {error}, Context: {context}")
