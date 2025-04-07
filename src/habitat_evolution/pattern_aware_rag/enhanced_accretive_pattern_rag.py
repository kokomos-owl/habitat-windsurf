"""
Enhanced Accretive Pattern RAG for the Habitat Evolution system.

This module extends the AccretivePatternRAG to incorporate constructive dissonance
and accretive weeding, enhancing the system's ability to discern semantic topology
over time and enabling a more adaptive and coherent pattern evolution process.
"""

import logging
import uuid
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class EnhancedAccretivePatternRAG:
    """
    Enhanced Pattern-aware RAG that uses relational accretion with constructive dissonance
    and accretive weeding to model queries as actants in the semantic field.
    """
    
    def __init__(self, 
                 pattern_evolution_service,
                 claude_baseline_service, 
                 significance_accretion_service,
                 constructive_dissonance_service=None,
                 accretive_weeding_service=None,
                 event_service=None):
        self.pattern_evolution_service = pattern_evolution_service
        self.claude_baseline_service = claude_baseline_service
        self.significance_accretion_service = significance_accretion_service
        self.constructive_dissonance_service = constructive_dissonance_service
        self.accretive_weeding_service = accretive_weeding_service
        self.event_service = event_service
        logger.info("Initialized Enhanced Accretive Pattern RAG with dissonance and weeding capabilities")
    
    async def process_query(self, query_text):
        """Process a query using relational accretion with dissonance awareness and weeding."""
        # Generate query ID
        query_id = f"query-{str(uuid.uuid4())}"
        
        # Initialize significance
        await self.significance_accretion_service.initialize_query_significance(query_id, query_text)
        
        # Process query
        logger.info(f"Processing query: {query_text}")
        
        # Enhance query with minimal baseline
        enhanced_query = await self.claude_baseline_service.enhance_query(query_id, query_text)
        
        # Retrieve relevant patterns
        retrieval_results = await self._retrieve_patterns(query_text)
        
        # Observe interactions with dissonance awareness
        interaction_metrics = await self.claude_baseline_service.observe_interactions(enhanced_query, retrieval_results)
        
        # Apply accretive weeding if available to prune low-value patterns
        if self.accretive_weeding_service and len(retrieval_results) > 5:
            # Identify patterns to prune
            pruning_candidates = []
            for result in retrieval_results:
                pattern = result.get("pattern", {})
                pattern_id = pattern.get("id", "")
                if pattern_id:
                    # Evaluate pattern value
                    evaluation = await self.accretive_weeding_service.evaluate_pattern_value(pattern_id)
                    if evaluation["pruning_recommendation"]:
                        pruning_candidates.append(pattern_id)
            
            # Remove pruned patterns from retrieval results
            if pruning_candidates:
                logger.info(f"Pruning {len(pruning_candidates)} low-value patterns from retrieval results")
                retrieval_results = [
                    result for result in retrieval_results 
                    if result.get("pattern", {}).get("id", "") not in pruning_candidates
                ]
        
        # Calculate accretion rate
        accretion_rate = await self.significance_accretion_service.calculate_accretion_rate(interaction_metrics)
        
        # Update significance with dissonance awareness
        updated_significance = await self.significance_accretion_service.update_significance(
            query_id, interaction_metrics, accretion_rate
        )
        
        # Generate response with significance and dissonance awareness
        response_data = await self.claude_baseline_service.generate_response_with_significance(
            query_text, updated_significance, retrieval_results
        )
        
        # Check for pattern emergence from constructive dissonance
        if self.constructive_dissonance_service:
            await self._check_pattern_emergence(query_id, updated_significance, retrieval_results)
        
        # Add query metadata to response
        response_data["query_id"] = query_id
        response_data["enhanced_query"] = enhanced_query
        response_data["significance"] = {
            "accretion_level": updated_significance.get("accretion_level", 0),
            "semantic_stability": updated_significance.get("semantic_stability", 0),
            "relational_density": updated_significance.get("relational_density", 0),
            "dissonance_potential": updated_significance.get("dissonance_potential", 0),
            "pattern_diversity": updated_significance.get("pattern_diversity", 0),
            "emergence_probability": updated_significance.get("emergence_probability", 0)
        }
        
        # Publish query processed event
        if self.event_service:
            self.event_service.publish(
                "query.processed",
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "enhanced_query": enhanced_query,
                    "significance": response_data["significance"],
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        logger.info(f"Processed query with ID: {query_id}")
        logger.info(f"Significance metrics: accretion={updated_significance.get('accretion_level', 0):.2f}, stability={updated_significance.get('semantic_stability', 0):.2f}")
        logger.info(f"Dissonance metrics: potential={updated_significance.get('dissonance_potential', 0):.2f}, diversity={updated_significance.get('pattern_diversity', 0):.2f}")
        return response_data
    
    async def _retrieve_patterns(self, query_text):
        """Retrieve patterns relevant to the query."""
        # Get all patterns
        patterns = await self.pattern_evolution_service.get_patterns()
        
        # Log the patterns we're searching through
        logger.info(f"Searching through {len(patterns)} patterns for query relevance")
        
        results = []
        for pattern in patterns:
            # Calculate relevance based on simple keyword matching
            relevance = self._calculate_pattern_relevance(query_text, pattern)
            
            # Lower the threshold to ensure we get some patterns
            if relevance > 0.05:  # Include patterns with even minimal relevance
                pattern_copy = pattern.copy()
                pattern_copy["relevance"] = relevance
                results.append({
                    "pattern": pattern_copy,
                    "relevance": relevance,
                    "patterns": [pattern_copy]
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Take top results
        top_results = results[:5]
        
        # Log the patterns we found
        if top_results:
            logger.info(f"Retrieved {len(top_results)} patterns for query:")
            for i, result in enumerate(top_results):
                pattern = result["pattern"]
                logger.info(f"  {i+1}. {pattern.get('base_concept', 'unknown')} (relevance: {result['relevance']:.2f})")
        else:
            logger.info(f"No relevant patterns found for query")
            
            # If no patterns found, create a fallback result with all patterns at low relevance
            # This ensures the significance vector will still grow
            for pattern in patterns[:3]:  # Take first 3 patterns
                pattern_copy = pattern.copy()
                pattern_copy["relevance"] = 0.1  # Low relevance
                results.append({
                    "pattern": pattern_copy,
                    "relevance": 0.1,
                    "patterns": [pattern_copy]
                })
            
            top_results = results[:3]
            logger.info(f"Using {len(top_results)} fallback patterns with low relevance")
        
        return top_results
    
    def _calculate_pattern_relevance(self, query_text, pattern):
        """Calculate relevance of a pattern to a query with dissonance awareness."""
        # Simple relevance calculation based on keyword matching
        query_words = set(query_text.lower().split())
        
        # Check pattern base concept
        base_concept = pattern.get("base_concept", "").lower()
        base_concept_words = set(base_concept.split("_"))
        
        # Check pattern properties
        properties = pattern.get("properties", {})
        property_words = set()
        for key, value in properties.items():
            if isinstance(value, str):
                property_words.update(value.lower().split())
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                for item in value:
                    property_words.update(item.lower().split())
        
        # Calculate matches
        base_matches = len(query_words.intersection(base_concept_words))
        property_matches = len(query_words.intersection(property_words))
        
        # Calculate relevance score
        relevance = 0.0
        if len(query_words) > 0:
            base_relevance = base_matches / len(query_words) * 0.7
            property_relevance = property_matches / len(query_words) * 0.3
            relevance = base_relevance + property_relevance
        
        # Adjust for constructive dissonance potential
        # Patterns with moderate coherence (not too high, not too low) get a boost
        coherence = pattern.get("coherence", 0.5)
        coherence_factor = 1.0
        if 0.4 <= coherence <= 0.7:
            # Boost patterns in the "productive dissonance" range
            coherence_factor = 1.2
        
        # Apply coherence factor
        relevance *= coherence_factor
        
        # Cap at 1.0
        return min(relevance, 1.0)
    
    async def _check_pattern_emergence(self, query_id, significance, retrieval_results):
        """Check for pattern emergence from constructive dissonance."""
        # Only check if significance has reached a threshold
        accretion_level = significance.get("accretion_level", 0)
        dissonance_potential = significance.get("dissonance_potential", 0)
        emergence_probability = significance.get("emergence_probability", 0)
        
        if accretion_level < 0.4 or dissonance_potential < 0.5 or emergence_probability < 0.6:
            logger.info(f"Not checking for pattern emergence: accretion={accretion_level:.2f}, dissonance={dissonance_potential:.2f}, emergence={emergence_probability:.2f}")
            return
        
        logger.info(f"Checking for pattern emergence from constructive dissonance: accretion={accretion_level:.2f}, dissonance={dissonance_potential:.2f}, emergence={emergence_probability:.2f}")
        
        # Get patterns from retrieval results
        patterns = [result.get("pattern", {}) for result in retrieval_results]
        
        # Identify dissonance zones
        if self.constructive_dissonance_service:
            dissonance_zones = await self.constructive_dissonance_service.identify_dissonance_zones(
                patterns, threshold=0.6
            )
            
            if dissonance_zones:
                logger.info(f"Identified {len(dissonance_zones)} dissonance zones with emergence potential")
                
                # Generate emergent pattern from query and dissonance zones
                emergent_pattern = await self._generate_pattern_from_query(
                    query_id, 
                    significance.get("query_text", ""), 
                    significance, 
                    dissonance_zones
                )
                
                if emergent_pattern:
                    logger.info(f"Generated emergent pattern: {emergent_pattern.get('base_concept', 'unknown')}")
                    
                    # Add pattern to evolution service
                    if self.pattern_evolution_service:
                        pattern_id = await self.pattern_evolution_service.add_pattern(emergent_pattern)
                        logger.info(f"Added emergent pattern to evolution service: {pattern_id}")
                        
                        # Publish pattern emergence event
                        if self.event_service:
                            self.event_service.publish(
                                "pattern.emerged",
                                {
                                    "pattern_id": pattern_id,
                                    "query_id": query_id,
                                    "dissonance_potential": dissonance_potential,
                                    "emergence_probability": emergence_probability,
                                    "timestamp": datetime.now().isoformat()
                                }
                            )
    
    async def _generate_pattern_from_query(self, query_id, query_text, significance, dissonance_zones):
        """Generate a pattern from a query with sufficient significance and dissonance."""
        # Extract significance vector
        significance_vector = significance.get("significance_vector", {})
        
        # Get top patterns by significance
        top_patterns = sorted(
            significance_vector.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        if not top_patterns:
            logger.info("No significant patterns found for pattern generation")
            return None
        
        # Get pattern details
        pattern_details = []
        for pattern_id, score in top_patterns:
            pattern = await self._get_pattern(pattern_id)
            if pattern:
                pattern_details.append({
                    "pattern": pattern,
                    "significance": score
                })
        
        if not pattern_details:
            logger.info("No pattern details found for pattern generation")
            return None
        
        # Extract dissonance zone information
        dissonance_info = []
        for zone in dissonance_zones:
            dissonance_info.append({
                "central_pattern_id": zone.get("central_pattern_id", ""),
                "dissonance_metrics": zone.get("dissonance_metrics", {}),
                "emergence_probability": zone.get("emergence_probability", 0)
            })
        
        # Generate pattern properties
        # Enhanced to utilize quality transitions and larger semantic chunks
        properties = {}
        
        # Add properties based on significance
        for detail in pattern_details:
            pattern = detail["pattern"]
            pattern_properties = pattern.get("properties", {})
            
            # Extract key properties based on significance
            for key, value in pattern_properties.items():
                if key not in properties:
                    properties[key] = value
        
        # Add properties based on dissonance zones
        for zone_info in dissonance_info:
            # Add dissonance-specific properties
            properties["dissonance_emergence"] = True
            properties["emergence_probability"] = zone_info.get("emergence_probability", 0)
            
            # Add related patterns
            if "related_patterns" not in properties:
                properties["related_patterns"] = []
            
            properties["related_patterns"].append(zone_info.get("central_pattern_id", ""))
        
        # Calculate confidence and coherence based on significance and dissonance
        # Enhanced to adjust based on chunk size and dissonance
        semantic_chunk_size = significance.get("semantic_chunk_size", "medium")
        dissonance_potential = significance.get("dissonance_potential", 0)
        
        # Base confidence from significance
        confidence = significance.get("accretion_level", 0.1) * 0.8
        
        # Adjust confidence based on chunk size
        if semantic_chunk_size == "large":
            confidence *= 1.2  # 20% boost for large chunks
        elif semantic_chunk_size == "small":
            confidence *= 0.9  # 10% reduction for small chunks
        
        # Adjust confidence based on dissonance
        if dissonance_potential > 0.6:
            # High dissonance can reduce confidence slightly
            confidence *= 0.9
        
        # Calculate coherence
        coherence = significance.get("semantic_stability", 0.1) * 0.7
        
        # Adjust coherence based on dissonance
        if dissonance_potential > 0.6:
            # High dissonance reduces coherence
            coherence *= 0.8
        elif dissonance_potential > 0.3:
            # Moderate dissonance slightly reduces coherence
            coherence *= 0.9
        
        # Cap values
        confidence = min(confidence, 0.8)  # Cap confidence
        coherence = min(coherence, 0.7)    # Cap coherence
        
        # Generate base concept from query
        base_concept = query_text.lower().replace(" ", "_")[:50]
        
        # Create pattern
        pattern = {
            "id": f"pattern-{str(uuid.uuid4())}",
            "base_concept": base_concept,
            "properties": properties,
            "confidence": confidence,
            "coherence": coherence,
            "quality_state": "emergent",  # Start as emergent
            "semantic_stability": significance.get("semantic_stability", 0.1),
            "origin": {
                "type": "query_emergence",
                "query_id": query_id,
                "dissonance_potential": dissonance_potential,
                "emergence_probability": significance.get("emergence_probability", 0)
            },
            "created_at": datetime.now().isoformat()
        }
        
        return pattern
        
    async def _get_pattern(self, pattern_id):
        """Get pattern details."""
        if self.pattern_evolution_service:
            return await self.pattern_evolution_service.get_pattern(pattern_id)
        return None
