"""
Climate Risk Query Service

This service provides a high-level interface for querying climate risk information
using the pattern-aware RAG system. It handles the bidirectional flow of information,
ensuring that pattern usage is tracked and quality metrics are updated based on query results.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from src.habitat_evolution.infrastructure.interfaces.services.pattern_aware_rag_interface import PatternAwareRAGInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService

logger = logging.getLogger(__name__)


class ClimateRiskQueryService:
    """
    Service for querying climate risk information using pattern-aware RAG.
    
    This service provides a high-level interface for querying climate risk
    information using the pattern-aware RAG system. It handles the bidirectional
    flow of information, ensuring that pattern usage is tracked and quality
    metrics are updated based on query results.
    """
    
    def __init__(
        self,
        pattern_aware_rag_service: PatternAwareRAGInterface,
        pattern_evolution_service: PatternEvolutionService,
        event_service: Optional[EventServiceInterface] = None
    ):
        """
        Initialize the climate risk query service.
        
        Args:
            pattern_aware_rag_service: The pattern-aware RAG service
            pattern_evolution_service: The pattern evolution service
            event_service: Optional event service for publishing events
        """
        self.pattern_aware_rag_service = pattern_aware_rag_service
        self.pattern_evolution_service = pattern_evolution_service
        self.event_service = event_service
        logger.info("ClimateRiskQueryService initialized")
    
    async def query(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query climate risk information using pattern-aware RAG.
        
        Args:
            query_text: The query text
            context: Optional context for the query
            
        Returns:
            Query results including relevant patterns and generated response
        """
        # Add query metadata
        query_id = str(uuid.uuid4())
        query_timestamp = datetime.now().isoformat()
        query_context = context or {}
        query_context.update({
            "query_id": query_id,
            "timestamp": query_timestamp,
            "source": "climate_risk_query_service"
        })
        
        # Process query through RAG
        logger.info(f"Processing query: {query_text}")
        result = await self.pattern_aware_rag_service.query(query_text, query_context)
        
        # Track pattern usage for each pattern in the result
        patterns = result.get("patterns", [])
        logger.info(f"Query returned {len(patterns)} patterns")
        
        for pattern in patterns:
            pattern_id = pattern.get("id")
            if not pattern_id:
                continue
                
            # Calculate usage metrics based on pattern's role in the query
            usage_metrics = {
                "used_in_query": True,
                "relevance_score": pattern.get("score", 0.5),
                "coherence_score": pattern.get("coherence", 0.5),
                "query_context": query_text
            }
            
            # Track pattern usage
            self._track_pattern_usage(pattern_id, usage_metrics)
            
            # Provide feedback on pattern quality
            self._provide_pattern_feedback(pattern_id, usage_metrics)
        
        # Publish query event
        if self.event_service:
            try:
                self.event_service.publish(
                    "climate_risk.query",
                    {
                        "query_id": query_id,
                        "query": query_text,
                        "context": context,
                        "result_summary": {
                            "pattern_count": len(patterns),
                            "has_response": "response" in result
                        },
                        "timestamp": query_timestamp
                    }
                )
                logger.info(f"Published climate_risk.query event for query: {query_id}")
            except Exception as event_error:
                logger.warning(f"Error publishing query event: {event_error}")
        
        # Add query metadata to result
        result.update({
            "query_id": query_id,
            "timestamp": query_timestamp,
            "pattern_count": len(patterns)
        })
        
        return result
    
    def _track_pattern_usage(self, pattern_id: str, usage_data: Dict[str, Any]):
        """
        Track pattern usage for quality feedback.
        
        Args:
            pattern_id: The ID of the pattern
            usage_data: Usage data for the pattern
        """
        try:
            # Track usage through pattern evolution service
            self.pattern_evolution_service.track_pattern_usage(
                pattern_id=pattern_id,
                context={
                    "usage_type": "query",
                    "usage_data": usage_data,
                    "source": "climate_risk_query_service",
                    "timestamp": datetime.now().isoformat()
                }
            )
            logger.info(f"Tracked usage for pattern: {pattern_id}")
            
            # Publish pattern usage event
            if self.event_service:
                self.event_service.publish(
                    "pattern.usage",
                    {
                        "pattern_id": pattern_id,
                        "usage_data": usage_data,
                        "source": "climate_risk_query_service",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                logger.info(f"Published pattern.usage event for pattern: {pattern_id}")
                
        except Exception as e:
            logger.warning(f"Error tracking pattern usage: {e}")
    
    def _provide_pattern_feedback(self, pattern_id: str, usage_metrics: Dict[str, Any]):
        """
        Provide feedback to pattern evolution service.
        
        Args:
            pattern_id: The ID of the pattern
            usage_metrics: Usage metrics for the pattern
        """
        try:
            # Calculate quality metrics based on usage
            quality_metrics = self._calculate_quality_metrics(usage_metrics)
            
            # Update pattern quality
            self.pattern_evolution_service.update_pattern_quality(
                pattern_id=pattern_id,
                quality_metrics=quality_metrics,
                context={
                    "source": "climate_risk_query_service",
                    "usage_metrics": usage_metrics,
                    "timestamp": datetime.now().isoformat()
                }
            )
            logger.info(f"Updated quality for pattern: {pattern_id}")
            
            # Publish feedback event
            if self.event_service:
                self.event_service.publish(
                    "pattern.quality.feedback",
                    {
                        "pattern_id": pattern_id,
                        "quality_metrics": quality_metrics,
                        "source": "climate_risk_query_service",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                logger.info(f"Published pattern.quality.feedback event for pattern: {pattern_id}")
                
        except Exception as e:
            logger.warning(f"Error providing pattern feedback: {e}")
    
    def _calculate_quality_metrics(self, usage_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate quality metrics based on usage metrics.
        
        Args:
            usage_metrics: Usage metrics for the pattern
            
        Returns:
            Quality metrics for the pattern
        """
        # Extract metrics from usage data
        relevance_score = usage_metrics.get("relevance_score", 0.5)
        coherence_score = usage_metrics.get("coherence_score", 0.5)
        
        # Calculate quality metrics
        quality_metrics = {
            "confidence": min(relevance_score + 0.1, 1.0),  # Increase confidence slightly
            "coherence": coherence_score,
            "signal_strength": relevance_score,
            "phase_stability": coherence_score * 0.8,  # Phase stability is related to coherence
            "uncertainty": max(1.0 - relevance_score, 0.1)  # Inverse of relevance
        }
        
        return quality_metrics
