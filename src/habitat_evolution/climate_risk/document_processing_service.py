"""
Climate Risk Document Processing Service

This service processes climate risk documents, extracts patterns, and stores them in ArangoDB
using the PatternEvolutionService with AdaptiveID integration.
"""

import os
import logging
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.infrastructure.adapters.pattern_adaptive_id_adapter import PatternAdaptiveIDAdapter
from src.habitat_evolution.infrastructure.services.claude_pattern_extraction_service import ClaudePatternExtractionService
from src.habitat_evolution.adaptive_core.models.pattern import Pattern

logger = logging.getLogger(__name__)

class DocumentProcessingService:
    """
    Service for processing climate risk documents and extracting patterns.
    
    This service reads climate risk documents, extracts patterns, and stores them
    in ArangoDB using the PatternEvolutionService. It leverages the AdaptiveID
    integration for pattern versioning and relationship tracking.
    """
    
    def __init__(self, 
                 pattern_evolution_service: PatternEvolutionService,
                 arangodb_connection: Optional[ArangoDBConnection] = None,
                 claude_api_key: Optional[str] = None):
        """
        Initialize the document processing service.
        
        Args:
            pattern_evolution_service: Service for pattern evolution
            arangodb_connection: Optional ArangoDB connection
            claude_api_key: Optional Claude API key
        """
        self.pattern_evolution_service = pattern_evolution_service
        self.arangodb_connection = arangodb_connection
        self.claude_extraction_service = ClaudePatternExtractionService(api_key=claude_api_key)
        logger.info("DocumentProcessingService initialized with Claude integration")
        
    def process_document(self, document_path: str) -> List[Dict[str, Any]]:
        """
        Process a climate risk document and extract patterns.
        
        Args:
            document_path: Path to the document
            
        Returns:
            List of extracted patterns
        """
        logger.info(f"Processing document: {document_path}")
        
        # Read the document
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        logger.info(f"Document content length: {len(content)}")
        
        # Extract patterns from the document
        patterns = self._extract_patterns(content, document_path)
        logger.info(f"Extracted {len(patterns)} patterns from document")
        
        # Store the patterns in the database
        stored_patterns = []
        for pattern in patterns:
            try:
                # Store the pattern using the PatternEvolutionService
                pattern_id = self.pattern_evolution_service.store_pattern(pattern)
                pattern["id"] = pattern_id
                stored_patterns.append(pattern)
                logger.info(f"Stored pattern: {pattern_id}")
            except Exception as e:
                logger.error(f"Error storing pattern: {e}")
                
        return stored_patterns
        
    def _extract_patterns(self, content: str, document_path: str) -> List[Dict[str, Any]]:
        """
        Extract patterns from document content using Claude or fallback to rule-based extraction.
        
        Args:
            content: Document content
            document_path: Path to the document
            
        Returns:
            List of extracted patterns
        """
        document_name = os.path.basename(document_path)
        
        # Use Claude for pattern extraction
        logger.info("Using Claude for pattern extraction")
        patterns = self.claude_extraction_service.extract_patterns(content, document_name)
        
        # If Claude extraction failed or returned no patterns, use fallback extraction
        if not patterns:
            logger.info("Claude extraction returned no patterns, using fallback extraction")
            patterns = self._fallback_extract_patterns(content, document_path)
            
        return patterns
        
    def _fallback_extract_patterns(self, content: str, document_path: str) -> List[Dict[str, Any]]:
        """
        Fallback method for extracting patterns when Claude extraction fails.
        
        Args:
            content: Document content
            document_path: Path to the document
            
        Returns:
            List of extracted patterns
        """
        patterns = []
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
                "id": f"extreme-drought-{uuid.uuid4()}",
                "base_concept": "extreme_drought",
                "creator_id": "document_processor",
                "weight": 1.0,
                "confidence": 0.78,
                "uncertainty": 0.22,
                "coherence": 0.75,
                "phase_stability": 0.65,
                "signal_strength": 0.8,
                "quality_state": "hypothetical",
                "properties": {
                    "location": location,
                    "risk_type": "drought",
                    "timeframe": "present",
                    "frequency": drought_prob,
                    "source_document": document_name
                }
            })
            
        if "wildfire" in content.lower():
            # Extract wildfire data from content
            wildfire_match = re.search(r'wildfire days is expected to increase ([0-9]+)% by mid-century and ([0-9]+)% by late-century', content)
            mid_century_increase = f"{wildfire_match.group(1)}%" if wildfire_match else "unknown"
            late_century_increase = f"{wildfire_match.group(2)}%" if wildfire_match else "unknown"
            
            patterns.append({
                "id": f"wildfire-risk-{uuid.uuid4()}",
                "base_concept": "wildfire_risk",
                "creator_id": "document_processor",
                "weight": 1.0,
                "confidence": 0.75,
                "uncertainty": 0.25,
                "coherence": 0.7,
                "phase_stability": 0.6,
                "signal_strength": 0.8,
                "quality_state": "hypothetical",
                "properties": {
                    "location": location,
                    "risk_type": "wildfire",
                    "mid_century_increase": mid_century_increase,
                    "late_century_increase": late_century_increase,
                    "source_document": document_name
                }
            })
            
        if "storm" in content.lower() or "cyclone" in content.lower() or "nor'easter" in content.lower():
            patterns.append({
                "id": f"storm-risk-{uuid.uuid4()}",
                "base_concept": "storm_risk",
                "creator_id": "document_processor",
                "weight": 1.0,
                "confidence": 0.72,
                "uncertainty": 0.28,
                "coherence": 0.7,
                "phase_stability": 0.6,
                "signal_strength": 0.75,
                "quality_state": "hypothetical",
                "properties": {
                    "location": location,
                    "risk_type": "storm",
                    "storm_type": "extratropical_cyclone",
                    "trend": "increasing intensity",
                    "source_document": document_name
                }
            })
            
        return patterns
        
    def query_patterns(self, base_concept: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query patterns from ArangoDB.
        
        Args:
            base_concept: Optional base concept to filter by
            
        Returns:
            List of patterns
        """
        if not self.arangodb_connection:
            logger.error("ArangoDB connection not available")
            return []
            
        # Build query
        query = """
        FOR p IN patterns
        """
        
        if base_concept:
            query += f"""
            FILTER p.base_concept == @base_concept
            """
            bind_vars = {"base_concept": base_concept}
        else:
            bind_vars = {}
            
        query += """
        RETURN p
        """
        
        # Execute query
        try:
            result = list(self.arangodb_connection.execute_aql(query, bind_vars))
            logger.info(f"Retrieved {len(result)} patterns from ArangoDB")
            return result
        except Exception as e:
            logger.error(f"Error querying patterns: {e}")
            return []
