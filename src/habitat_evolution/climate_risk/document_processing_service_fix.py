"""
Document processing service for climate risk documents.

This module provides a service for processing climate risk documents,
extracting patterns, and storing them in the database.
"""

import os
import re
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.habitat_evolution.infrastructure.services.claude_pattern_extraction_service import ClaudePatternExtractionService
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.pattern_aware_rag.pattern_aware_rag import PatternAwareRAG
from src.habitat_evolution.infrastructure.services.event_service import EventService

logger = logging.getLogger(__name__)

class DocumentProcessingService:
    """
    Service for processing climate risk documents and extracting patterns.
    
    This service uses the Claude API to extract patterns from climate risk documents
    and stores them in the database. It also provides a fallback extraction method
    when the Claude API is not available or fails.
    """
    
    def __init__(self, pattern_evolution_service: PatternEvolutionService,
                 arangodb_connection: ArangoDBConnection,
                 claude_api_key: Optional[str] = None,
                 pattern_aware_rag_service: Optional[PatternAwareRAG] = None,
                 event_service: Optional[EventService] = None):
        """
        Initialize the document processing service.
        
        Args:
            pattern_evolution_service: Pattern evolution service
            arangodb_connection: ArangoDB connection
            claude_api_key: Optional Claude API key
            pattern_aware_rag_service: Optional PatternAwareRAG service
            event_service: Optional event service for publishing events
        """
        self.pattern_evolution_service = pattern_evolution_service
        self.arangodb_connection = arangodb_connection
        self.claude_extraction_service = ClaudePatternExtractionService(api_key=claude_api_key)
        self.pattern_aware_rag_service = pattern_aware_rag_service
        self.event_service = event_service
        logger.info("DocumentProcessingService initialized with Claude integration and PatternAwareRAG")
        
    def process_document(self, document_path: str = None, document_id: str = None, content: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a climate risk document and extract patterns.

        This method can be called in two ways:
        1. With a document_path parameter, which will read the file from disk
        2. With document_id, content, and metadata parameters for direct processing

        Args:
            document_path: Path to the document (optional)
            document_id: ID of the document (optional)
            content: Document content (optional)
            metadata: Document metadata (optional)

        Returns:
            Dictionary with processing status and extracted patterns
        """
        if document_path:
            logger.info(f"Processing document from path: {document_path}")
            # Read the document from file
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            document_id = os.path.basename(document_path)
            metadata = metadata or {"source": "file", "path": document_path}
        elif content and document_id:
            logger.info(f"Processing document with ID: {document_id}")
            metadata = metadata or {}
        else:
            error_msg = "Either document_path or both document_id and content must be provided"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
            
        logger.info(f"Document content length: {len(content)}")
        
        # Extract patterns from the document
        patterns = self._extract_patterns(content, document_id, document_path)
        logger.info(f"Extracted {len(patterns)} patterns from document")
        
        # Store the patterns in the database
        stored_patterns = []
        processing_timestamp = datetime.now().isoformat()
        
        for pattern in patterns:
            # Add metadata
            pattern["metadata"] = pattern.get("metadata", {})
            pattern["metadata"].update({
                "document_id": document_id,
                "processing_timestamp": processing_timestamp,
                "source": metadata.get("source", "unknown")
            })
            
            # Store the pattern
            if self.pattern_evolution_service:
                try:
                    # Use pattern evolution service to store the pattern
                    stored_pattern = self.pattern_evolution_service.create_pattern(pattern)
                    stored_patterns.append(stored_pattern)
                    logger.info(f"Processed pattern: {pattern.get('id')}")
                    
                    # Publish event if event service is available
                    if self.event_service:
                        try:
                            self.event_service.publish("document.processed", {
                                "document_id": document_id,
                                "pattern_id": pattern.get("id"),
                                "timestamp": processing_timestamp
                            })
                            logger.info(f"Published document.processed event for pattern: {pattern.get('id')}")
                        except Exception as e:
                            logger.error(f"Error publishing event: {e}")
                except Exception as e:
                    logger.error(f"Error storing pattern: {e}")
            
            # Add to pattern-aware RAG if available
            if self.pattern_aware_rag_service:
                try:
                    self.pattern_aware_rag_service.add_pattern(pattern)
                except Exception as e:
                    logger.error(f"Error adding pattern to RAG: {e}")
        
        return {
            "status": "success",
            "document_id": document_id,
            "patterns": patterns,
            "stored_patterns": stored_patterns
        }
    
    def _extract_patterns(self, content: str, document_id: str, document_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract patterns from document content.
        
        Args:
            content: Document content
            document_id: Document ID
            document_path: Optional path to the document
            
        Returns:
            List of extracted patterns
        """
        # Try to extract patterns using Claude
        logger.info("Using Claude for pattern extraction")
        patterns = self.claude_extraction_service.extract_patterns(content, document_id)
        
        # If Claude extraction failed or returned no patterns, use fallback extraction
        if not patterns:
            logger.info("Claude extraction returned no patterns, using fallback extraction")
            patterns = self._fallback_extract_patterns(content, document_id, document_path)
            
        return patterns
        
    def _fallback_extract_patterns(self, content: str, document_id: str, document_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fallback method for extracting patterns when Claude extraction fails.
        
        Args:
            content: Document content
            document_id: Document ID
            document_path: Optional path to the document
            
        Returns:
            List of extracted patterns
        """
        patterns = []
        document_name = document_path if document_path else document_id
        if document_path:
            document_name = os.path.basename(document_path)
        timestamp = datetime.now().isoformat()
        
        # Extract location from content
        location_match = re.search(r'CLIMATE RISK ASSESSMENT – ([^,]+)', content)
        location = location_match.group(1).strip() if location_match else "Unknown"
        
        # Extract key concepts and create patterns
        if "sea level rise" in content.lower() or "flood risk" in content.lower():
            patterns.append({
                "id": f"sea-level-rise-{uuid.uuid4()}",
                "base_concept": "sea_level_rise",
                "creator_id": "document_processor",
                "weight": 1.0,
                "confidence": 0.85,
                "uncertainty": 0.15,
                "coherence": 0.8,
                "phase_stability": 0.7,
                "signal_strength": 0.9,
                "quality_state": "hypothetical",
                "properties": {
                    "location": location,
                    "risk_type": "flooding",
                    "timeframe": "2050",
                    "source_document": document_name
                }
            })
            
        if "drought" in content.lower():
            # Extract drought probability from content
            drought_prob_match = re.search(r'experienced extreme drought between ([0-9.]+)% and ([0-9.]+)% of the time', content)
            drought_prob = f"{drought_prob_match.group(1)}% to {drought_prob_match.group(2)}% of the time" if drought_prob_match else "unknown"
            
            patterns.append({
                "id": f"drought-risk-{uuid.uuid4()}",
                "base_concept": "drought_risk",
                "creator_id": "document_processor",
                "weight": 0.8,
                "confidence": 0.75,
                "uncertainty": 0.25,
                "coherence": 0.7,
                "phase_stability": 0.6,
                "signal_strength": 0.8,
                "properties": {
                    "location": location,
                    "risk_type": "drought",
                    "probability": drought_prob,
                    "source_document": document_name
                }
            })
            
        if "storm" in content.lower() or "hurricane" in content.lower():
            patterns.append({
                "id": f"storm-risk-{uuid.uuid4()}",
                "base_concept": "storm_risk",
                "creator_id": "document_processor",
                "weight": 0.9,
                "confidence": 0.8,
                "uncertainty": 0.2,
                "coherence": 0.75,
                "phase_stability": 0.65,
                "signal_strength": 0.85,
                "properties": {
                    "location": location,
                    "risk_type": "storms",
                    "source_document": document_name
                }
            })
            
        if "temperature" in content.lower() or "heat" in content.lower():
            patterns.append({
                "id": f"temperature-risk-{uuid.uuid4()}",
                "base_concept": "temperature_risk",
                "creator_id": "document_processor",
                "weight": 0.85,
                "confidence": 0.8,
                "uncertainty": 0.2,
                "coherence": 0.75,
                "phase_stability": 0.7,
                "signal_strength": 0.8,
                "properties": {
                    "location": location,
                    "risk_type": "extreme_heat",
                    "source_document": document_name
                }
            })
            
        return patterns
