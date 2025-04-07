"""
Constructive Dissonance Service for the Habitat Evolution system.

This service is responsible for detecting and leveraging constructive dissonance
in pattern relationships, identifying zones where productive tension can lead to
pattern emergence and innovation.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ConstructiveDissonanceService:
    """Service for detecting and leveraging constructive dissonance in pattern relationships."""
    
    def __init__(self, db_connection, event_service):
        self.db_connection = db_connection
        self.event_service = event_service
        self.dissonance_thresholds = {
            "minimum": 0.2,  # Below this is just noise
            "optimal": 0.6,  # Peak productive tension
            "maximum": 0.8   # Above this is destructive conflict
        }
        logger.info("Initialized Constructive Dissonance Service")
    
    async def calculate_pattern_dissonance(self, pattern_id, related_patterns):
        """Calculate constructive dissonance for a pattern and its relationships.
        
        This represents the productive tension that drives innovation and
        pattern evolution in the semantic field.
        
        Args:
            pattern_id: ID of the pattern to calculate dissonance for
            related_patterns: List of related pattern data
            
        Returns:
            Dissonance metrics including score and productive potential
        """
        # Get pattern data
        pattern = await self._get_pattern(pattern_id)
        if not pattern:
            return {"dissonance_score": 0, "productive_potential": 0}
        
        # Extract key metrics
        coherence = pattern.get("coherence", 0.5)
        stability = pattern.get("semantic_stability", 0.5)
        
        # Calculate semantic gradient between pattern and related patterns
        gradient_magnitude = await self._calculate_semantic_gradient(pattern, related_patterns)
        
        # Coherence factor - peaks at 0.6 (some coherence but not too much)
        coherence_factor = 1 - abs(0.6 - coherence) * 2
        coherence_factor = max(0, min(1, coherence_factor))  # Clamp to [0,1]
        
        # Stability factor - peaks at 0.5 (balanced stability)
        stability_factor = 1 - abs(0.5 - stability) * 2
        stability_factor = max(0, min(1, stability_factor))  # Clamp to [0,1]
        
        # Gradient factor - higher gradient = higher dissonance potential
        gradient_factor = gradient_magnitude
        
        # Combine factors for overall dissonance score
        dissonance_score = (
            coherence_factor * 0.3 +
            stability_factor * 0.3 +
            gradient_factor * 0.4
        )
        
        # Calculate productive potential - how likely this dissonance leads to emergence
        productive_potential = self._calculate_productive_potential(
            dissonance_score, pattern, related_patterns
        )
        
        # Publish dissonance event if significant
        if productive_potential > self.dissonance_thresholds["minimum"] and self.event_service:
            self.event_service.publish(
                "pattern.dissonance.detected",
                {
                    "pattern_id": pattern_id,
                    "dissonance_score": dissonance_score,
                    "productive_potential": productive_potential,
                    "related_pattern_count": len(related_patterns),
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        return {
            "dissonance_score": dissonance_score,
            "productive_potential": productive_potential,
            "coherence_factor": coherence_factor,
            "stability_factor": stability_factor,
            "gradient_factor": gradient_factor
        }
    
    async def identify_dissonance_zones(self, patterns, threshold=0.5):
        """Identify zones of constructive dissonance in the pattern space.
        
        Args:
            patterns: List of patterns to analyze
            threshold: Minimum productive potential to consider
            
        Returns:
            List of dissonance zones with their constituent patterns and metrics
        """
        dissonance_zones = []
        processed_patterns = set()
        
        for pattern in patterns:
            pattern_id = pattern.get("id")
            if pattern_id in processed_patterns:
                continue
                
            # Get related patterns
            related_patterns = await self._get_related_patterns(pattern_id)
            
            # Calculate dissonance for this pattern cluster
            dissonance_metrics = await self.calculate_pattern_dissonance(
                pattern_id, related_patterns
            )
            
            # If this has sufficient productive potential, it's a dissonance zone
            if dissonance_metrics["productive_potential"] >= threshold:
                zone = {
                    "id": f"dissonance-zone-{str(uuid.uuid4())}",
                    "central_pattern_id": pattern_id,
                    "related_pattern_ids": [p.get("id") for p in related_patterns],
                    "dissonance_metrics": dissonance_metrics,
                    "emergence_probability": self._calculate_emergence_probability(
                        dissonance_metrics, pattern, related_patterns
                    ),
                    "detected_at": datetime.now().isoformat()
                }
                dissonance_zones.append(zone)
                
                # Publish dissonance zone event
                if self.event_service:
                    self.event_service.publish(
                        "pattern.dissonance.zone.detected",
                        {
                            "zone_id": zone["id"],
                            "central_pattern_id": pattern_id,
                            "related_pattern_count": len(related_patterns),
                            "emergence_probability": zone["emergence_probability"],
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                
            # Mark as processed
            processed_patterns.add(pattern_id)
            
        return dissonance_zones
    
    async def get_dissonance_potential_for_query(self, query_id, significance_vector):
        """Calculate the dissonance potential for a query based on its significance vector.
        
        Args:
            query_id: ID of the query
            significance_vector: The query's significance vector mapping pattern IDs to scores
            
        Returns:
            Dissonance potential metrics for the query
        """
        if not significance_vector:
            return {
                "dissonance_potential": 0,
                "pattern_diversity": 0,
                "emergence_probability": 0
            }
        
        # Get patterns from significance vector
        pattern_ids = list(significance_vector.keys())
        patterns = []
        
        for pattern_id in pattern_ids:
            pattern = await self._get_pattern(pattern_id)
            if pattern:
                patterns.append(pattern)
        
        # Calculate semantic diversity among patterns
        pattern_diversity = self._calculate_pattern_diversity(None, patterns)
        
        # Calculate dissonance potential based on pattern diversity and vector structure
        vector_size = len(significance_vector)
        
        # Dissonance potential peaks with moderate vector size and high diversity
        size_factor = 1.0 - abs(5 - vector_size) / 5 if vector_size > 0 else 0
        size_factor = max(0, min(1, size_factor))
        
        dissonance_potential = size_factor * 0.4 + pattern_diversity * 0.6
        
        # Calculate emergence probability
        emergence_probability = dissonance_potential * 0.7 + size_factor * 0.3
        
        return {
            "dissonance_potential": dissonance_potential,
            "pattern_diversity": pattern_diversity,
            "emergence_probability": emergence_probability,
            "vector_size": vector_size
        }
    
    async def _calculate_semantic_gradient(self, pattern, related_patterns):
        """Calculate the semantic gradient between a pattern and related patterns."""
        if not related_patterns:
            return 0.0
            
        # In a real implementation, this would use vector operations to calculate 
        # semantic differences between the central pattern and its related patterns
        
        # For now, simulate gradient based on pattern properties
        base_concept = pattern.get("base_concept", "")
        coherence = pattern.get("coherence", 0.5)
        confidence = pattern.get("confidence", 0.5)
        
        # Calculate average difference in coherence and confidence
        avg_coherence_diff = 0
        avg_confidence_diff = 0
        
        for related in related_patterns:
            coherence_diff = abs(coherence - related.get("coherence", 0.5))
            confidence_diff = abs(confidence - related.get("confidence", 0.5))
            
            avg_coherence_diff += coherence_diff
            avg_confidence_diff += confidence_diff
        
        if related_patterns:
            avg_coherence_diff /= len(related_patterns)
            avg_confidence_diff /= len(related_patterns)
        
        # Combine into gradient magnitude (0-1)
        gradient_magnitude = (avg_coherence_diff * 0.6 + avg_confidence_diff * 0.4) * 2
        gradient_magnitude = min(1.0, gradient_magnitude)
        
        return gradient_magnitude
    
    def _calculate_productive_potential(self, dissonance_score, pattern, related_patterns):
        """Calculate how likely this dissonance is to produce emergent patterns."""
        # Productive potential peaks when dissonance is in the optimal range
        if dissonance_score < self.dissonance_thresholds["minimum"]:
            return dissonance_score / self.dissonance_thresholds["minimum"] * 0.3
        elif dissonance_score < self.dissonance_thresholds["optimal"]:
            return 0.3 + (dissonance_score - self.dissonance_thresholds["minimum"]) / (
                self.dissonance_thresholds["optimal"] - self.dissonance_thresholds["minimum"]
            ) * 0.7
        elif dissonance_score < self.dissonance_thresholds["maximum"]:
            return 1.0 - (dissonance_score - self.dissonance_thresholds["optimal"]) / (
                self.dissonance_thresholds["maximum"] - self.dissonance_thresholds["optimal"]
            ) * 0.5
        else:
            return 0.5 - (dissonance_score - self.dissonance_thresholds["maximum"]) * 0.5
    
    def _calculate_emergence_probability(self, dissonance_metrics, pattern, related_patterns):
        """Calculate probability of pattern emergence from this dissonance zone."""
        # Base probability on productive potential
        base_probability = dissonance_metrics["productive_potential"]
        
        # Adjust based on pattern density and diversity
        pattern_count = len(related_patterns) + 1
        pattern_diversity = self._calculate_pattern_diversity(pattern, related_patterns)
        
        # More diverse patterns with moderate density have higher emergence probability
        density_factor = 1.0 - abs(5 - pattern_count) / 5
        density_factor = max(0, min(1, density_factor))
        
        return base_probability * 0.6 + density_factor * 0.2 + pattern_diversity * 0.2
    
    def _calculate_pattern_diversity(self, pattern, related_patterns):
        """Calculate the semantic diversity of a set of patterns."""
        # In a real implementation, this would measure semantic distance between patterns
        # using vector operations or other semantic similarity metrics
        
        # For now, simulate diversity based on pattern properties
        if not related_patterns:
            return 0.0
            
        # Count unique base concepts
        base_concepts = set()
        if pattern:
            base_concepts.add(pattern.get("base_concept", ""))
            
        for related in related_patterns:
            base_concepts.add(related.get("base_concept", ""))
            
        # Calculate diversity score (0-1)
        concept_diversity = (len(base_concepts) - 1) / max(len(related_patterns), 1)
        concept_diversity = min(1.0, concept_diversity)
        
        # Calculate property diversity
        property_sets = []
        if pattern:
            property_sets.append(set(pattern.get("properties", {}).keys()))
            
        for related in related_patterns:
            property_sets.append(set(related.get("properties", {}).keys()))
            
        # Calculate average Jaccard distance between property sets
        total_distance = 0
        comparisons = 0
        
        for i in range(len(property_sets)):
            for j in range(i+1, len(property_sets)):
                set_i = property_sets[i]
                set_j = property_sets[j]
                
                if not set_i or not set_j:
                    continue
                    
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                
                if union > 0:
                    distance = 1 - (intersection / union)
                    total_distance += distance
                    comparisons += 1
        
        property_diversity = total_distance / max(1, comparisons)
        
        # Combine diversity metrics
        return concept_diversity * 0.6 + property_diversity * 0.4
        
    async def _get_pattern(self, pattern_id):
        """Get pattern data from the database."""
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
        
    async def _get_related_patterns(self, pattern_id):
        """Get patterns related to the given pattern."""
        if hasattr(self.db_connection, 'execute_query'):
            query = f"""
            FOR rel IN pattern_relationships
            FILTER rel._from == 'patterns/{pattern_id}' OR rel._to == 'patterns/{pattern_id}'
            LET other_id = rel._from == 'patterns/{pattern_id}' ? rel._to : rel._from
            FOR p IN patterns
            FILTER p._id == other_id
            RETURN p
            """
            return await self.db_connection.execute_query(query)
        return []
