"""
Enhanced Claude Baseline Service for the Habitat Evolution system.

This service extends the ClaudeBaselineService to incorporate constructive dissonance
detection during interactions and include dissonance metrics in interaction observations.
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from habitat_evolution.infrastructure.adapters.claude_adapter import ClaudeAdapter

logger = logging.getLogger(__name__)

class EnhancedClaudeBaselineService:
    """Service for providing minimal baseline enhancement with dissonance awareness."""
    
    def __init__(self, claude_adapter=None, db_connection=None, event_service=None, 
                 constructive_dissonance_service=None, api_key=None):
        self.claude_adapter = claude_adapter or ClaudeAdapter(api_key=api_key)
        self.db_connection = db_connection
        self.event_service = event_service
        self.constructive_dissonance_service = constructive_dissonance_service
        self.use_real_claude = not self.claude_adapter.use_mock
        logger.info(f"Initialized Enhanced Claude Baseline Service (use_real_claude: {self.use_real_claude})")
    
    async def enhance_query(self, query_id, query_text, significance_vector=None):
        """Enhance a query with minimal baseline and dissonance awareness."""
        # If significance vector is provided, use it to influence enhancement
        if significance_vector and isinstance(significance_vector, dict):
            # Get the actual vector if it's nested
            if "significance_vector" in significance_vector:
                significance_vector = significance_vector.get("significance_vector", {})
            
            # Log the size of the significance vector
            logger.info(f"Enhancing query with significance vector of size {len(significance_vector)}")
        else:
            significance_vector = {}
            logger.info("Enhancing query without significance vector")
        
        # Determine query domains
        domains = self._determine_query_domains(query_text)
        
        # If using real Claude API
        if self.use_real_claude:
            try:
                # Prepare significance context and patterns for Claude
                patterns_for_claude = []
                if significance_vector:
                    # Get top patterns by significance
                    top_patterns = sorted(
                        significance_vector.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                    
                    # Prepare patterns for Claude
                    for pattern_id, score in top_patterns:
                        # Get pattern details if available
                        pattern_details = await self._get_pattern_details(pattern_id)
                        if pattern_details:
                            patterns_for_claude.append({
                                "id": pattern_id,
                                "name": pattern_details.get('base_concept', pattern_id),
                                "description": pattern_details.get('description', ''),
                                "quality_state": pattern_details.get('quality_state', 'hypothetical'),
                                "significance": score
                            })
                        else:
                            patterns_for_claude.append({
                                "id": pattern_id,
                                "name": pattern_id,
                                "significance": score
                            })
                
                # Create context for Claude
                context = {
                    "domains": domains,
                    "significance_vector": significance_vector,
                    "query_id": query_id
                }
                
                # Process the query with Claude
                response = await self.claude_adapter.process_query(
                    query_text, 
                    context, 
                    patterns_for_claude
                )
                
                # Extract the enhanced query
                if response and "error" not in response:
                    # Try to extract just the enhanced query (remove any explanations)
                    enhanced_query = response.get("response", "").strip()
                    
                    # If the response has multiple lines, take the first line as the enhanced query
                    if "\n" in enhanced_query:
                        enhanced_query = enhanced_query.split("\n")[0]
                    
                    logger.info(f"Enhanced query with Claude API: {enhanced_query}")
                    return enhanced_query
                else:
                    logger.error(f"Error in Claude response: {response.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error using Claude API for query enhancement: {e}")
                logger.info("Falling back to mock implementation")
        
        # Mock implementation for testing or when Claude API is not available
        # Simple enhancement based on domains
        enhanced_query = query_text
        
        # Add domain-specific enhancements
        if "sea_level" in domains:
            enhanced_query = f"{enhanced_query} (considering coastal flooding, erosion, and infrastructure impacts)"
        elif "extreme_weather" in domains:
            enhanced_query = f"{enhanced_query} (considering storm intensity, precipitation patterns, and resilience)"
        elif "adaptation" in domains:
            enhanced_query = f"{enhanced_query} (considering resilience strategies, infrastructure planning, and community preparedness)"
        
        logger.info(f"Enhanced query with mock implementation: {enhanced_query}")
        return enhanced_query
    
    def _determine_query_domains(self, query):
        """Determine relevant domains for a query."""
        domains = []
        
        # Sea level related terms
        sea_level_terms = ["sea level", "coastal", "flooding", "erosion", "inundation"]
        # Extreme weather related terms
        extreme_weather_terms = ["storm", "hurricane", "precipitation", "rainfall", "drought", "heat"]
        # Adaptation related terms
        adaptation_terms = ["adapt", "resilience", "planning", "strategy", "prepare", "mitigate"]
        
        # Check for domain terms
        if any(term in query.lower() for term in sea_level_terms):
            domains.append("sea_level")
        if any(term in query.lower() for term in extreme_weather_terms):
            domains.append("extreme_weather")
        if any(term in query.lower() for term in adaptation_terms):
            domains.append("adaptation")
        
        # If no specific domains found, use general domain
        if not domains:
            domains.append("general")
        
        return domains
    
    async def observe_interactions(self, enhanced_query, retrieval_results):
        """Observe interactions between a query and retrieved patterns with dissonance awareness."""
        # Count patterns
        pattern_count = len(retrieval_results)
        
        # Extract patterns for dissonance analysis
        patterns = []
        for result in retrieval_results:
            pattern = result.get("pattern", {})
            if pattern:
                patterns.append(pattern)
        
        # Calculate dissonance metrics if service is available
        dissonance_metrics = {}
        if self.constructive_dissonance_service and patterns:
            # Get dissonance potential for the pattern set
            try:
                # Extract pattern IDs for significance vector format
                pattern_ids = {p.get("id", ""): result.get("relevance", 0.5) for p, result in zip(patterns, retrieval_results)}
                dissonance_metrics = await self.constructive_dissonance_service.get_dissonance_potential_for_query(
                    "temp-query-id", pattern_ids
                )
                logger.info(f"Calculated dissonance metrics: {dissonance_metrics}")
            except Exception as e:
                logger.error(f"Error calculating dissonance metrics: {e}")
                # Default dissonance metrics
                dissonance_metrics = {
                    "dissonance_potential": 0.3,
                    "pattern_diversity": 0.4,
                    "emergence_probability": 0.2
                }
        else:
            # Default dissonance metrics
            dissonance_metrics = {
                "dissonance_potential": 0.3,
                "pattern_diversity": 0.4,
                "emergence_probability": 0.2
            }
        
        if self.use_real_claude and pattern_count > 0:
            try:
                # Prepare patterns for Claude
                patterns_for_claude = []
                for i, result in enumerate(retrieval_results, 1):
                    pattern = result.get("pattern", {})
                    patterns_for_claude.append({
                        "id": pattern.get("id", f"pattern-{i}"),
                        "name": pattern.get("base_concept", "Unknown concept"),
                        "description": pattern.get("description", ""),
                        "properties": pattern.get("properties", {}),
                        "confidence": pattern.get("confidence", 0.0),
                        "coherence": pattern.get("coherence", 0.0),
                        "quality_state": pattern.get("quality_state", "hypothetical"),
                        "relevance": result.get("relevance", 0.5)
                    })
                
                # Create context for Claude
                context = {
                    "query_id": query_id,
                    "dissonance_metrics": dissonance_metrics,
                    "task": "interaction_analysis"
                }
                
                # Process the query with Claude
                response = await self.claude_adapter.process_query(
                    enhanced_query, 
                    context, 
                    patterns_for_claude
                )
                
                # Parse the response
                if response and "error" not in response:
                    try:
                        # Extract JSON from the response
                        response_text = response.get("response", "")
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            json_str = response_text[json_start:json_end]
                            interaction_data = json.loads(json_str)
                            
                            logger.info(f"Analyzed interactions with Claude API: {interaction_data}")
                            
                            # Combine with basic metrics
                            observation = {
                                "query_id": query_id,
                                "pattern_count": pattern_count,
                                "timestamp": datetime.now().isoformat(),
                                "dissonance_metrics": dissonance_metrics,
                                "interaction_metrics": interaction_data
                            }
                            
                            return observation
                    except Exception as e:
                        logger.error(f"Error parsing interaction JSON: {e}")
                else:
                    logger.error(f"Error in Claude response: {response.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error using Claude API for interaction analysis: {e}")
                logger.info("Falling back to mock implementation")
        
        # Enhanced mock implementation for testing or when Claude API is not available
        # Calculate interaction strength with improved scaling for larger chunks
        interaction_strength = 0.15 * pattern_count  # Increased base interaction strength
        if pattern_count > 0:
            interaction_strength = max(0.3, interaction_strength)  # Higher minimum interaction strength
        
        # Extract pattern relevance with enhanced values
        pattern_relevance = {}
        quality_transitions = {}
        for result in retrieval_results:
            pattern = result.get("pattern", {})
            pattern_id = pattern.get("id", "")
            # Enhanced relevance calculation
            relevance = result.get("relevance", 0.0) * 1.2  # Boost relevance by 20%
            relevance = min(1.0, relevance)  # Cap at 1.0
            
            if pattern_id:
                pattern_relevance[pattern_id] = relevance
                # Simulate quality transitions based on relevance
                if relevance < 0.4:
                    quality_transitions[pattern_id] = "stable"
                elif relevance < 0.7:
                    quality_transitions[pattern_id] = "poor_to_uncertain"
                else:
                    quality_transitions[pattern_id] = "uncertain_to_good"
                    
                logger.info(f"Observed interaction with pattern {pattern_id} (relevance: {relevance:.2f}, transition: {quality_transitions[pattern_id]})")
        
        # If no patterns were found, create enhanced mock pattern relevance for testing
        if not pattern_relevance:
            logger.info("No pattern interactions found, creating enhanced mock interactions for testing")
            # We'll add mock patterns in the significance service with quality transitions
        
        # Create enhanced interaction metrics with dissonance awareness
        interaction_metrics = {
            "pattern_count": pattern_count,
            "interaction_strength": interaction_strength,
            "pattern_relevance": pattern_relevance,
            "quality_transitions": quality_transitions,
            "semantic_chunk_size": "large",  # Default to large chunks
            "transition_confidence": 0.65,    # Moderate-high confidence
            "coherence_score": 0.7,          # Enhanced coherence
            "retrieval_quality": 0.7,        # Enhanced retrieval quality
            "dissonance_metrics": dissonance_metrics
        }
        
        logger.info(f"Observed interactions with {pattern_count} patterns using large semantic chunks")
        logger.info(f"Dissonance potential: {dissonance_metrics.get('dissonance_potential', 0):.2f}")
        return interaction_metrics
    
    async def generate_response_with_significance(self, query, significance_vector, retrieval_results):
        """Generate a response based on query and significance with dissonance awareness."""
        # Log significance vector size
        if isinstance(significance_vector, dict):
            pattern_count = len(significance_vector.get("significance_vector", {}))
            logger.info(f"Significance vector has {pattern_count} patterns")
            
            # Extract dissonance metrics if available
            dissonance_potential = significance_vector.get("dissonance_potential", 0.0)
            pattern_diversity = significance_vector.get("pattern_diversity", 0.0)
            emergence_probability = significance_vector.get("emergence_probability", 0.0)
            logger.info(f"Dissonance metrics: potential={dissonance_potential:.2f}, diversity={pattern_diversity:.2f}, emergence={emergence_probability:.2f}")
        
        # Extract pattern information for context
        pattern_context = ""
        if retrieval_results:
            pattern_context = "\n\nRelevant patterns:\n"
            for i, result in enumerate(retrieval_results, 1):
                pattern = result.get("pattern", {})
                relevance = result.get("relevance", 0)
                pattern_context += f"{i}. {pattern.get('base_concept', 'Unknown concept')} (relevance: {relevance:.2f})\n"
        
        if self.use_real_claude:
            try:
                # Extract dissonance information for prompt
                dissonance_context = ""
                if isinstance(significance_vector, dict):
                    dissonance_potential = significance_vector.get("dissonance_potential", 0.0)
                    if dissonance_potential > 0.3:
                        dissonance_context = f"\n\nThis query has significant constructive dissonance potential ({dissonance_potential:.2f}), indicating productive tension between patterns that may lead to emergent insights."
                
                # Create prompt for Claude with dissonance awareness
                prompt = f"""
                You are an expert system that generates responses to queries based on retrieved patterns and their significance.
                Generate a comprehensive response to the query below, using the provided patterns and their significance.
                
                {dissonance_context}
                
                Query: {query}
                {pattern_context}
                
                Response:
                """
                
                # Use Claude API to generate response
                response = self.claude_adapter.generate_text(prompt, max_tokens=500)
                
                # Calculate confidence based on significance and dissonance
                confidence = 0.1  # Start with minimal confidence
                
                if isinstance(significance_vector, dict):
                    # Get actual significance vector if nested
                    actual_vector = significance_vector.get("significance_vector", {})
                    if actual_vector:
                        # Calculate confidence based on significance values
                        avg_significance = sum(actual_vector.values()) / len(actual_vector)
                        confidence = min(0.9, avg_significance * 2)  # Scale up but cap at 0.9
                    
                    # Adjust confidence based on dissonance potential
                    dissonance_potential = significance_vector.get("dissonance_potential", 0.0)
                    if dissonance_potential > 0.5:
                        # High dissonance potential can increase confidence for innovative responses
                        confidence = min(0.95, confidence * 1.1)
                
                # Create response data
                response_data = {
                    "response": response.strip(),
                    "confidence": confidence,
                    "patterns_used": [result.get("pattern", {}).get("id", "") for result in retrieval_results] if retrieval_results else [],
                    "dissonance_potential": significance_vector.get("dissonance_potential", 0.0) if isinstance(significance_vector, dict) else 0.0,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"Generated response with Claude API, confidence: {confidence:.2f}, dissonance: {response_data['dissonance_potential']:.2f}")
                return response_data
            except Exception as e:
                logger.error(f"Error using Claude API for response generation: {e}")
                logger.info("Falling back to mock implementation")
                # Fall back to mock implementation if Claude API fails
        
        # Mock implementation for testing or when Claude API is not available
        # Simple response generation
        response = f"Response to: {query}"
        
        # Calculate confidence based on significance and dissonance
        confidence = 0.09  # Start with minimal confidence
        
        # Adjust confidence based on dissonance potential if available
        dissonance_potential = 0.0
        if isinstance(significance_vector, dict):
            dissonance_potential = significance_vector.get("dissonance_potential", 0.0)
            if dissonance_potential > 0.5:
                confidence = min(0.9, confidence * 1.2)  # Boost confidence for high dissonance potential
        
        # Generate simulated response
        if "sea level" in query.lower():
            response = "Based on the patterns related to sea level rise, Martha's Vineyard faces significant risks from coastal flooding and erosion. By 2050, sea levels are projected to rise by 1.5 to 3.1 feet, threatening coastal properties and infrastructure."
            # Add dissonance-aware insight if applicable
            if dissonance_potential > 0.5:
                response += " Interestingly, the constructive tension between sea level rise patterns and adaptation strategies reveals emerging opportunities for innovative coastal management approaches that balance protection with managed retreat."
        elif "wildfire" in query.lower():
            response = "The patterns indicate that wildfire risk on Martha's Vineyard is increasing. The number of wildfire days is expected to increase 40% by mid-century and 70% by late-century, with extended dry seasons increasing combustible vegetation."
            # Add dissonance-aware insight if applicable
            if dissonance_potential > 0.5:
                response += " The productive dissonance between wildfire risk patterns and changing precipitation patterns suggests an emerging understanding of complex seasonal risk windows that require dynamic rather than static management strategies."
        elif "adaptation" in query.lower():
            response = "Adaptation strategies recommended in the patterns include implementing coastal buffer zones, beach nourishment programs, elevation of critical infrastructure, and managed retreat from highest-risk areas."
            # Add dissonance-aware insight if applicable
            if dissonance_potential > 0.5:
                response += " The constructive tension between immediate protection strategies and long-term retreat options is generating innovative hybrid approaches that phase interventions based on risk thresholds and community values."
        else:
            response = "The analysis of climate risk patterns for Martha's Vineyard shows multiple interconnected risks including sea level rise, increased storm intensity, and changing precipitation patterns. These risks require comprehensive adaptation strategies."
            # Add dissonance-aware insight if applicable
            if dissonance_potential > 0.5:
                response += " The productive dissonance between these interconnected patterns is revealing emergent properties of the climate risk system that suggest non-linear intervention points with high leverage potential."
        
        # Create response data with dissonance awareness
        response_data = {
            "response": response,
            "confidence": confidence,
            "patterns_used": [result.get("pattern", {}).get("id", "") for result in retrieval_results] if retrieval_results else [],
            "dissonance_potential": dissonance_potential,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Generated response with confidence: {confidence:.2f}, dissonance: {dissonance_potential:.2f}")
        return response_data
        
    async def _get_pattern_details(self, pattern_id):
        """Get pattern details from the database."""
        if hasattr(self.db_connection, 'execute_query'):
            query = f"""
            FOR p IN patterns
            FILTER p.id == '{pattern_id}'
            RETURN p
            """
            result = await self.db_connection.execute_query(query)
            if result and len(result) > 0:
                return result[0]
        return None
