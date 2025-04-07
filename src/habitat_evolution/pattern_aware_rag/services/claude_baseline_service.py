"""
Claude Baseline Service for Habitat Evolution

This service provides baseline query enhancement using Claude LLM,
focusing on minimal enhancement rather than pattern extraction.
This supports the relational accretion model where queries gradually
accrete significance through interactions rather than having patterns
projected onto them.
"""

import logging
import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter

logger = logging.getLogger(__name__)

class ClaudeBaselineService:
    """
    Service that provides baseline query enhancement using Claude LLM.
    """
    
    def __init__(self, claude_adapter: Optional[ClaudeAdapter] = None, api_key: Optional[str] = None):
        """
        Initialize the Claude baseline service.
        
        Args:
            claude_adapter: Optional pre-configured Claude adapter
            api_key: API key for Claude (used if claude_adapter is not provided)
        """
        self.api_key = api_key
        self.claude_adapter = claude_adapter or ClaudeAdapter(api_key=api_key)
        logger.info("Initialized ClaudeBaselineService")
    
    async def enhance_query_baseline(self, query: str) -> Dict[str, Any]:
        """
        Provide minimal baseline enhancement to a query without projecting patterns.
        
        Args:
            query: The query to enhance
            
        Returns:
            Enhanced query with minimal semantic structure
        """
        query_id = str(uuid.uuid4())
        
        # Try to use Claude API for enhancement if available
        try:
            # Create a system prompt for minimal enhancement
            system_prompt = """
            You are an expert system that provides minimal semantic enhancement to queries.
            Your task is to analyze the query and provide basic semantic dimensions without changing the query itself.
            Do not add any additional information or context to the query.
            Format your response as JSON with the following structure:
            {
                "enhanced_query": "<original query with minimal refinement>",
                "semantic_dimensions": {
                    "specificity": <float between 0 and 1>,
                    "complexity": <float between 0 and 1>,
                    "domain_relevance": <float between 0 and 1>
                },
                "potential_domains": ["<domain1>", "<domain2>", ...]
            }
            """
            
            # Process the query with Claude
            response = await self.claude_adapter.process_query(query, {}, [])
            
            if not self.claude_adapter.use_mock and response and "error" not in response:
                # Parse the response
                try:
                    # Try to extract JSON from the response
                    response_text = response.get("response", "")
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        claude_data = json.loads(json_str)
                        
                        # Create enhanced query with Claude's analysis
                        enhanced_query = {
                            "query_id": query_id,
                            "original_query": query,
                            "enhanced_query": claude_data.get("enhanced_query", query),
                            "semantic_dimensions": claude_data.get("semantic_dimensions", {
                                "specificity": self._calculate_specificity(query),
                                "complexity": self._calculate_complexity(query),
                                "domain_relevance": 0.7
                            }),
                            "potential_domains": claude_data.get("potential_domains", self._extract_potential_domains(query)),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        logger.info(f"Enhanced query with Claude API: {query[:50]}...")
                        return enhanced_query
                except Exception as e:
                    logger.error(f"Error parsing Claude response: {e}")
        except Exception as e:
            logger.error(f"Error using Claude API for query enhancement: {e}")
        
        # Fallback to heuristic enhancement
        logger.info("Using heuristic query enhancement")
        
        # Create baseline enhancement with minimal structure
        enhanced_query = {
            "query_id": query_id,
            "original_query": query,
            "enhanced_query": query,  # Minimal change at this stage
            "semantic_dimensions": {
                "specificity": self._calculate_specificity(query),
                "complexity": self._calculate_complexity(query),
                "domain_relevance": 0.7  # Placeholder
            },
            "potential_domains": self._extract_potential_domains(query),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Enhanced query with baseline semantic structure: {query[:50]}...")
        return enhanced_query
    
    def _calculate_specificity(self, query: str) -> float:
        """
        Calculate the specificity of a query.
        
        Args:
            query: The query to analyze
            
        Returns:
            Specificity score (0.0 to 1.0)
        """
        # Simple heuristic based on query length and presence of specific terms
        words = query.split()
        word_count = len(words)
        
        # More words generally means more specific
        length_factor = min(1.0, word_count / 15)
        
        # Check for specific terms that indicate specificity
        specific_terms = ["specifically", "exactly", "precisely", "particular", "detailed"]
        specific_term_count = sum(1 for word in words if word.lower() in specific_terms)
        term_factor = min(1.0, specific_term_count / 2)
        
        # Combine factors
        return (length_factor * 0.7) + (term_factor * 0.3)
    
    def _calculate_complexity(self, query: str) -> float:
        """
        Calculate the complexity of a query.
        
        Args:
            query: The query to analyze
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        # Simple heuristic based on sentence structure and vocabulary
        words = query.split()
        word_count = len(words)
        
        # More words generally means more complex
        length_factor = min(1.0, word_count / 20)
        
        # Check for complex sentence structures
        complex_indicators = ["however", "although", "nevertheless", "therefore", "consequently"]
        complex_count = sum(1 for word in words if word.lower() in complex_indicators)
        structure_factor = min(1.0, complex_count / 2)
        
        # Combine factors
        return (length_factor * 0.6) + (structure_factor * 0.4)
    
    def _extract_potential_domains(self, query: str) -> Dict[str, float]:
        """
        Extract potential domains relevant to the query.
        
        Args:
            query: The query to analyze
            
        Returns:
            Dictionary of domain relevance scores
        """
        # Simple keyword matching for domains
        domains = {
            "climate_risk": 0.0,
            "sea_level_rise": 0.0,
            "extreme_weather": 0.0,
            "drought": 0.0,
            "wildfire": 0.0,
            "infrastructure": 0.0,
            "adaptation": 0.0,
            "economic_impact": 0.0
        }
        
        query_lower = query.lower()
        
        # Check for domain keywords
        if "climate" in query_lower or "risk" in query_lower:
            domains["climate_risk"] = 0.8
            
        if "sea" in query_lower and "level" in query_lower:
            domains["sea_level_rise"] = 0.9
            
        if "storm" in query_lower or "hurricane" in query_lower or "flood" in query_lower:
            domains["extreme_weather"] = 0.8
            
        if "drought" in query_lower or "water" in query_lower or "dry" in query_lower:
            domains["drought"] = 0.7
            
        if "fire" in query_lower or "wildfire" in query_lower or "burn" in query_lower:
            domains["wildfire"] = 0.8
            
        if "infrastructure" in query_lower or "building" in query_lower or "road" in query_lower:
            domains["infrastructure"] = 0.7
            
        if "adapt" in query_lower or "mitigate" in query_lower or "prepare" in query_lower:
            domains["adaptation"] = 0.8
            
        if "economic" in query_lower or "cost" in query_lower or "financial" in query_lower:
            domains["economic_impact"] = 0.7
        
        return domains
    
    async def observe_interactions(
        self,
        enhanced_query: Dict[str, Any],
        retrieval_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Observe interactions between an enhanced query and retrieval results.
        
        Args:
            enhanced_query: The enhanced query
            retrieval_results: Results from retrieval
            
        Returns:
            Interaction metrics
        """
        # Extract query information
        query_id = enhanced_query["query_id"]
        query_text = enhanced_query["original_query"]
        semantic_dimensions = enhanced_query["semantic_dimensions"]
        
        # Extract retrieval information
        document_count = len(retrieval_results)
        
        # Calculate pattern relevance
        pattern_relevance = {}
        for result in retrieval_results:
            for pattern in result.get("patterns", []):
                pattern_id = pattern.get("id")
                if not pattern_id:
                    continue
                    
                relevance = pattern.get("relevance", 0.5)
                
                if pattern_id in pattern_relevance:
                    # Take maximum relevance across documents
                    pattern_relevance[pattern_id] = max(pattern_relevance[pattern_id], relevance)
                else:
                    pattern_relevance[pattern_id] = relevance
        
        # Calculate coherence score
        coherence_scores = [result.get("coherence", 0.5) for result in retrieval_results]
        avg_coherence = sum(coherence_scores) / max(1, len(coherence_scores))
        
        # Calculate retrieval quality
        relevance_scores = [result.get("relevance", 0.5) for result in retrieval_results]
        avg_relevance = sum(relevance_scores) / max(1, len(relevance_scores))
        
        # Create interaction metrics
        interaction_metrics = {
            "query_id": query_id,
            "timestamp": datetime.now().isoformat(),
            "document_count": document_count,
            "pattern_relevance": pattern_relevance,
            "pattern_count": len(pattern_relevance),
            "coherence_score": avg_coherence,
            "retrieval_quality": avg_relevance,
            "semantic_dimensions": semantic_dimensions
        }
        
        logger.info(f"Observed interactions for query {query_id} with {len(pattern_relevance)} patterns")
        return interaction_metrics
    
    async def generate_response_with_significance(
        self,
        query: str,
        significance_vector: Dict[str, Any],
        retrieval_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a response that incorporates query significance.
        
        Args:
            query: The original query
            significance_vector: The query's significance vector
            retrieval_results: Results from retrieval
            
        Returns:
            Generated response
        """
        # This would normally call Claude API for response generation
        # For now, we'll simulate the response
        
        # Simulate API call with a delay
        await asyncio.sleep(0.5)
        
        # Extract significance information
        accretion_level = significance_vector.get("accretion_level", 0.1)
        semantic_stability = significance_vector.get("semantic_stability", 0.1)
        relational_density = significance_vector.get("relational_density", 0.0)
        
        # Create response with confidence based on significance
        confidence = (accretion_level * 0.4) + (semantic_stability * 0.4) + (relational_density * 0.2)
        
        # Simulate different response quality based on significance
        if accretion_level > 0.7:
            quality_prefix = "Based on extensive analysis and established patterns,"
        elif accretion_level > 0.4:
            quality_prefix = "Based on emerging patterns in the data,"
        else:
            quality_prefix = "Based on preliminary analysis,"
        
        # Create response
        response = {
            "response": f"{quality_prefix} Martha's Vineyard faces significant climate risks including sea level rise, increased storm intensity, and changing precipitation patterns. The data indicates that infrastructure vulnerability is a key concern that should be addressed in adaptation planning.",
            "confidence": confidence,
            "significance_level": accretion_level,
            "semantic_stability": semantic_stability,
            "relational_density": relational_density,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Generated response with confidence {confidence:.2f} based on significance level {accretion_level:.2f}")
        return response
