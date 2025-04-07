"""
Metrics collection for Claude API usage.

This module provides utilities for tracking and analyzing Claude API usage,
including response times, token consumption, and pattern extraction quality.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ClaudeAPIMetrics:
    """
    Metrics collection for Claude API usage.
    
    This class tracks API usage, response times, token consumption, and
    pattern extraction quality to help optimize API usage and monitor costs.
    """
    
    def __init__(self, metrics_dir: Optional[str] = None):
        """
        Initialize the metrics collector.
        
        Args:
            metrics_dir: Optional directory to store metrics data (if None, will use default)
        """
        self.metrics_dir = metrics_dir or Path(__file__).parents[3] / "data" / "metrics"
        self.metrics_file = self.metrics_dir / "claude_api_metrics.jsonl"
        
        # Create metrics directory if it doesn't exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics counters
        self.reset_session_metrics()
        
        logger.info(f"Initialized Claude API metrics collector (metrics_file: {self.metrics_file})")
    
    def reset_session_metrics(self):
        """Reset session metrics counters."""
        self.session_metrics = {
            "query_count": 0,
            "document_count": 0,
            "total_tokens": 0,
            "total_response_time_ms": 0,
            "error_count": 0,
            "pattern_count": 0,
            "session_start": datetime.now().isoformat()
        }
    
    def track_query(self, query: str, response: Dict[str, Any], response_time_ms: float):
        """
        Track a query to the Claude API.
        
        Args:
            query: The query text
            response: The response from Claude
            response_time_ms: Response time in milliseconds
        """
        # Extract metrics from the response
        tokens_used = response.get("tokens_used", 0)
        model = response.get("model", "unknown")
        
        # Update session metrics
        self.session_metrics["query_count"] += 1
        self.session_metrics["total_tokens"] += tokens_used
        self.session_metrics["total_response_time_ms"] += response_time_ms
        
        # Create metrics entry
        metrics_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "query",
            "query_length": len(query),
            "tokens_used": tokens_used,
            "response_time_ms": response_time_ms,
            "model": model,
            "avg_token_per_second": (tokens_used / (response_time_ms / 1000)) if response_time_ms > 0 else 0
        }
        
        # Log the metrics
        self._log_metrics(metrics_entry)
    
    def track_document(self, document: Dict[str, Any], response: Dict[str, Any], response_time_ms: float):
        """
        Track a document processed by the Claude API.
        
        Args:
            document: The document processed
            response: The response from Claude
            response_time_ms: Response time in milliseconds
        """
        # Extract metrics from the response
        tokens_used = response.get("tokens_used", 0)
        model = response.get("model", "unknown")
        patterns = response.get("patterns", [])
        
        # Update session metrics
        self.session_metrics["document_count"] += 1
        self.session_metrics["total_tokens"] += tokens_used
        self.session_metrics["total_response_time_ms"] += response_time_ms
        self.session_metrics["pattern_count"] += len(patterns)
        
        # Create metrics entry
        metrics_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "document",
            "document_id": document.get("id", "unknown"),
            "document_length": len(document.get("content", "")),
            "tokens_used": tokens_used,
            "response_time_ms": response_time_ms,
            "model": model,
            "pattern_count": len(patterns),
            "avg_token_per_second": (tokens_used / (response_time_ms / 1000)) if response_time_ms > 0 else 0
        }
        
        # Log the metrics
        self._log_metrics(metrics_entry)
    
    def track_error(self, operation_type: str, error_message: str, context: Dict[str, Any]):
        """
        Track an error in Claude API usage.
        
        Args:
            operation_type: Type of operation (query or document)
            error_message: Error message
            context: Additional context for the error
        """
        # Update session metrics
        self.session_metrics["error_count"] += 1
        
        # Create metrics entry
        metrics_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "operation_type": operation_type,
            "error_message": error_message,
            "context": context
        }
        
        # Log the metrics
        self._log_metrics(metrics_entry)
    
    def track_pattern_quality(self, pattern_id: str, quality_state: str, confidence: float):
        """
        Track pattern quality metrics.
        
        Args:
            pattern_id: ID of the pattern
            quality_state: Quality state of the pattern
            confidence: Confidence score for the quality assessment
        """
        # Create metrics entry
        metrics_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "pattern_quality",
            "pattern_id": pattern_id,
            "quality_state": quality_state,
            "confidence": confidence
        }
        
        # Log the metrics
        self._log_metrics(metrics_entry)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current session metrics.
        
        Returns:
            Dictionary containing session metrics summary
        """
        # Calculate derived metrics
        avg_response_time = 0
        if self.session_metrics["query_count"] + self.session_metrics["document_count"] > 0:
            avg_response_time = (
                self.session_metrics["total_response_time_ms"] / 
                (self.session_metrics["query_count"] + self.session_metrics["document_count"])
            )
        
        avg_tokens_per_query = 0
        if self.session_metrics["query_count"] > 0:
            avg_tokens_per_query = self.session_metrics["total_tokens"] / self.session_metrics["query_count"]
        
        avg_patterns_per_document = 0
        if self.session_metrics["document_count"] > 0:
            avg_patterns_per_document = self.session_metrics["pattern_count"] / self.session_metrics["document_count"]
        
        # Create summary
        summary = {
            **self.session_metrics,
            "session_end": datetime.now().isoformat(),
            "avg_response_time_ms": avg_response_time,
            "avg_tokens_per_query": avg_tokens_per_query,
            "avg_patterns_per_document": avg_patterns_per_document,
            "error_rate": (
                self.session_metrics["error_count"] / 
                (self.session_metrics["query_count"] + self.session_metrics["document_count"])
            ) if (self.session_metrics["query_count"] + self.session_metrics["document_count"]) > 0 else 0
        }
        
        return summary
    
    def _log_metrics(self, metrics_entry: Dict[str, Any]):
        """
        Log metrics to the metrics file.
        
        Args:
            metrics_entry: Metrics entry to log
        """
        try:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(metrics_entry) + "\n")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")


# Global metrics collector instance
claude_metrics = ClaudeAPIMetrics()
