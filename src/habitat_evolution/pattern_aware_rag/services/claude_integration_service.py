"""
Claude Integration Service for Pattern-Aware RAG

This service integrates Claude LLM with the Pattern-Aware RAG system,
providing enhanced pattern extraction, query understanding, and response generation.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid
import asyncio

from src.habitat_evolution.infrastructure.services.claude_pattern_extraction_service import ClaudePatternExtractionService
from src.habitat_evolution.pattern_aware_rag.core.coherence_interface import CoherenceInterface, StateAlignment

logger = logging.getLogger(__name__)


class ClaudeRAGService:
    """
    Service for integrating Claude with Pattern-Aware RAG.
    
    This service provides enhanced pattern extraction, query understanding,
    and response generation using Claude's advanced language understanding.
    It serves as the LLM component in the bidirectional flow architecture.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Claude RAG service.
        
        Args:
            api_key: Optional Claude API key, will use environment variable if not provided
        """
        self.api_key = api_key or os.environ.get("CLAUDE_API_KEY")
        self.extraction_service = ClaudePatternExtractionService(api_key=self.api_key)
        if not self.api_key:
            logger.warning("Claude API key not provided, using fallback methods")
        
        logger.info("ClaudeRAGService initialized")
    
    async def extract_patterns_from_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Extract patterns from a query using Claude.
        
        Args:
            query: The query to extract patterns from
            
        Returns:
            List of extracted patterns
        """
        logger.info(f"Extracting patterns from query: {query}")
        
        # Create a specialized prompt for query pattern extraction
        prompt = self._create_query_pattern_prompt(query)
        
        try:
            # Call Claude API
            response = await self._call_claude_api(prompt)
            
            # Parse Claude's response
            patterns = self._parse_claude_response(response, "query")
            
            logger.info(f"Extracted {len(patterns)} patterns from query")
            return patterns
        except Exception as e:
            logger.error(f"Error extracting patterns from query: {e}")
            # Return basic pattern as fallback
            return [{
                "id": f"query-pattern-{uuid.uuid4()}",
                "base_concept": query,
                "confidence": 0.7,
                "coherence": 0.7,
                "signal_strength": 0.7,
                "phase_stability": 0.6,
                "uncertainty": 0.3,
                "type": "query_pattern"
            }]
    
    async def enhance_retrieval(self, query: str, patterns: List[Dict[str, Any]], documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance retrieval results using Claude.
        
        Args:
            query: The original query
            patterns: The patterns extracted from the query
            documents: The retrieved documents
            
        Returns:
            Enhanced retrieval results
        """
        logger.info(f"Enhancing retrieval for query: {query}")
        
        # Create a specialized prompt for retrieval enhancement
        prompt = self._create_retrieval_enhancement_prompt(query, patterns, documents)
        
        try:
            # Call Claude API
            response = await self._call_claude_api(prompt)
            
            # Parse Claude's response
            enhanced_results = self._parse_enhancement_response(response)
            
            logger.info(f"Enhanced retrieval with {len(enhanced_results)} results")
            return enhanced_results
        except Exception as e:
            logger.error(f"Error enhancing retrieval: {e}")
            # Return original documents as fallback
            return documents
    
    async def generate_response(self, query: str, patterns: List[Dict[str, Any]], documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a response using Claude.
        
        Args:
            query: The original query
            patterns: The patterns extracted from the query and retrieval
            documents: The retrieved documents
            
        Returns:
            Generated response with metadata
        """
        logger.info(f"Generating response for query: {query}")
        
        # Create a specialized prompt for response generation
        prompt = self._create_response_generation_prompt(query, patterns, documents)
        
        try:
            # Call Claude API
            response = await self._call_claude_api(prompt)
            
            # Parse Claude's response
            generated_response = self._parse_response(response)
            
            logger.info(f"Generated response for query: {query}")
            return generated_response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Return basic response as fallback
            return {
                "response": f"I'm sorry, I couldn't generate a detailed response for your query about {query}. Please try again or rephrase your question.",
                "confidence": 0.3,
                "patterns_used": [p.get("id") for p in patterns]
            }
    
    async def analyze_coherence(self, pattern_context: Any, content: str) -> StateAlignment:
        """
        Analyze coherence of patterns in content.
        
        Args:
            pattern_context: The pattern context
            content: The content to analyze
            
        Returns:
            Coherence analysis results
        """
        logger.info("Analyzing coherence of patterns in content")
        
        # Extract patterns from context
        query_patterns = pattern_context.query_patterns
        retrieval_patterns = pattern_context.retrieval_patterns
        augmentation_patterns = pattern_context.augmentation_patterns
        
        # Create a specialized prompt for coherence analysis
        prompt = self._create_coherence_analysis_prompt(
            query_patterns,
            retrieval_patterns,
            augmentation_patterns,
            content
        )
        
        try:
            # Call Claude API
            response = await self._call_claude_api(prompt)
            
            # Parse Claude's response
            coherence_result = self._parse_coherence_response(response)
            
            logger.info(f"Analyzed coherence with flow state: {coherence_result.flow_state}")
            return coherence_result
        except Exception as e:
            logger.error(f"Error analyzing coherence: {e}")
            # Return basic coherence result as fallback
            return StateAlignment(
                flow_state="moderate",
                patterns=query_patterns,
                confidence=0.5,
                emergence_potential=0.3
            )
    
    def _create_query_pattern_prompt(self, query: str) -> str:
        """
        Create a prompt for query pattern extraction.
        
        Args:
            query: The query to extract patterns from
            
        Returns:
            Prompt for Claude
        """
        return f"""
        Human: I need you to extract semantic patterns from the following query. 
        A pattern consists of a base concept, relationships, and contextual attributes.
        
        Query: {query}
        
        For each pattern, provide:
        1. A base_concept (the core semantic concept)
        2. A confidence score (0.0-1.0)
        3. A coherence score (0.0-1.0)
        4. A signal_strength score (0.0-1.0)
        5. A phase_stability score (0.0-1.0)
        6. An uncertainty score (0.0-1.0)
        7. Any relevant properties as key-value pairs
        
        Format your response as a JSON array of pattern objects.
        
        Assistant:
        """
    
    def _create_retrieval_enhancement_prompt(self, query: str, patterns: List[Dict[str, Any]], documents: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for retrieval enhancement.
        
        Args:
            query: The original query
            patterns: The patterns extracted from the query
            documents: The retrieved documents
            
        Returns:
            Prompt for Claude
        """
        # Format patterns for prompt
        patterns_text = json.dumps(patterns, indent=2)
        
        # Format documents for prompt (limit to prevent token overflow)
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc.get('content', '')[:500]}..." 
            for i, doc in enumerate(documents[:5])
        ])
        
        return f"""
        Human: I need you to enhance the relevance ranking of retrieved documents based on the query and extracted patterns.
        
        Query: {query}
        
        Extracted Patterns:
        {patterns_text}
        
        Retrieved Documents:
        {docs_text}
        
        For each document, provide:
        1. A relevance score (0.0-1.0)
        2. A coherence score with the patterns (0.0-1.0)
        3. Key concepts that align with the patterns
        4. Any additional patterns discovered in the document
        
        Format your response as a JSON array of enhanced document objects.
        
        Assistant:
        """
    
    def _create_response_generation_prompt(self, query: str, patterns: List[Dict[str, Any]], documents: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for response generation.
        
        Args:
            query: The original query
            patterns: The patterns extracted from the query and retrieval
            documents: The retrieved documents
            
        Returns:
            Prompt for Claude
        """
        # Format patterns for prompt
        patterns_text = json.dumps(patterns, indent=2)
        
        # Format documents for prompt (limit to prevent token overflow)
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc.get('content', '')[:300]}..." 
            for i, doc in enumerate(documents[:3])
        ])
        
        return f"""
        Human: I need you to generate a comprehensive response to the following query using the provided patterns and documents.
        
        Query: {query}
        
        Relevant Patterns:
        {patterns_text}
        
        Retrieved Documents:
        {docs_text}
        
        Please generate a response that:
        1. Directly answers the query
        2. Incorporates information from the patterns and documents
        3. Maintains coherence with the identified patterns
        4. Provides specific details and examples where possible
        
        Format your response as a JSON object with:
        1. "response": The generated response text
        2. "confidence": A confidence score (0.0-1.0)
        3. "patterns_used": IDs of patterns that influenced the response
        4. "coherence": A coherence score (0.0-1.0)
        
        Assistant:
        """
    
    def _create_coherence_analysis_prompt(self, query_patterns: List[str], retrieval_patterns: List[str], augmentation_patterns: List[str], content: str) -> str:
        """
        Create a prompt for coherence analysis.
        
        Args:
            query_patterns: Patterns from the query
            retrieval_patterns: Patterns from retrieval
            augmentation_patterns: Patterns from augmentation
            content: The content to analyze
            
        Returns:
            Prompt for Claude
        """
        return f"""
        Human: I need you to analyze the coherence between different sets of patterns and the generated content.
        
        Query Patterns:
        {', '.join(query_patterns)}
        
        Retrieval Patterns:
        {', '.join(retrieval_patterns)}
        
        Augmentation Patterns:
        {', '.join(augmentation_patterns)}
        
        Generated Content:
        {content[:1000]}...
        
        Please analyze:
        1. The flow state between patterns (low, moderate, high, optimal)
        2. The confidence in this assessment (0.0-1.0)
        3. The emergence potential for new patterns (0.0-1.0)
        4. Any emergent patterns not in the original sets
        
        Format your response as a JSON object.
        
        Assistant:
        """
    
    async def _call_claude_api(self, prompt: str) -> str:
        """
        Call Claude API with the given prompt.
        
        Args:
            prompt: The prompt for Claude
            
        Returns:
            Claude's response
        """
        # This is a placeholder for the actual API call
        # In a real implementation, this would use the Claude API client
        
        # Simulate API call with a delay
        await asyncio.sleep(0.5)
        
        # For testing purposes, return a mock response
        if "extract patterns from query" in prompt.lower():
            return json.dumps([
                {
                    "id": f"query-pattern-{uuid.uuid4()}",
                    "base_concept": "climate_risk_assessment",
                    "confidence": 0.85,
                    "coherence": 0.8,
                    "signal_strength": 0.9,
                    "phase_stability": 0.75,
                    "uncertainty": 0.2,
                    "properties": {
                        "location": "coastal",
                        "timeframe": "2050",
                        "risk_types": ["flooding", "sea_level_rise"]
                    }
                }
            ])
        elif "enhance the relevance ranking" in prompt.lower():
            return json.dumps([
                {
                    "document_id": "doc1",
                    "relevance_score": 0.9,
                    "coherence_score": 0.85,
                    "key_concepts": ["sea_level_rise", "coastal_flooding"],
                    "additional_patterns": ["infrastructure_vulnerability"]
                }
            ])
        elif "generate a comprehensive response" in prompt.lower():
            return json.dumps({
                "response": "Based on the climate risk assessment, coastal areas face significant flooding risks due to sea level rise projected by 2050. The data indicates that infrastructure vulnerability is a key concern that should be addressed in adaptation planning.",
                "confidence": 0.85,
                "patterns_used": ["climate_risk_assessment", "sea_level_rise"],
                "coherence": 0.8
            })
        elif "analyze the coherence" in prompt.lower():
            return json.dumps({
                "flow_state": "high",
                "confidence": 0.8,
                "emergence_potential": 0.7,
                "emergent_patterns": ["adaptation_planning", "resilience_measures"]
            })
        else:
            return json.dumps({
                "error": "Unrecognized prompt type",
                "response": "I'm not sure how to process this request."
            })
    
    def _parse_claude_response(self, response: str, source: str) -> List[Dict[str, Any]]:
        """
        Parse Claude's response for pattern extraction.
        
        Args:
            response: Claude's response
            source: Source of the patterns (query, document, etc.)
            
        Returns:
            List of extracted patterns
        """
        try:
            # Parse JSON response
            patterns = json.loads(response)
            
            # Ensure each pattern has an ID
            for pattern in patterns:
                if "id" not in pattern:
                    pattern["id"] = f"{source}-pattern-{uuid.uuid4()}"
            
            return patterns
        except Exception as e:
            logger.error(f"Error parsing Claude response: {e}")
            return []
    
    def _parse_enhancement_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse Claude's response for retrieval enhancement.
        
        Args:
            response: Claude's response
            
        Returns:
            Enhanced retrieval results
        """
        try:
            # Parse JSON response
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error parsing enhancement response: {e}")
            return []
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse Claude's response for response generation.
        
        Args:
            response: Claude's response
            
        Returns:
            Generated response with metadata
        """
        try:
            # Parse JSON response
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                "response": "I apologize, but I couldn't generate a proper response.",
                "confidence": 0.3,
                "patterns_used": [],
                "coherence": 0.3
            }
    
    def _parse_coherence_response(self, response: str) -> StateAlignment:
        """
        Parse Claude's response for coherence analysis.
        
        Args:
            response: Claude's response
            
        Returns:
            Coherence analysis results
        """
        try:
            # Parse JSON response
            data = json.loads(response)
            
            return StateAlignment(
                flow_state=data.get("flow_state", "moderate"),
                patterns=data.get("emergent_patterns", []),
                confidence=data.get("confidence", 0.5),
                emergence_potential=data.get("emergence_potential", 0.5)
            )
        except Exception as e:
            logger.error(f"Error parsing coherence response: {e}")
            return StateAlignment(
                flow_state="moderate",
                patterns=[],
                confidence=0.5,
                emergence_potential=0.3
            )
