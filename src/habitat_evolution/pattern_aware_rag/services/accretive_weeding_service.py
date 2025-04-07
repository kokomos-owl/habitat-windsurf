"""
Accretive Weeding Service for the Habitat Evolution system.

This service is responsible for systematically pruning low-value patterns
to maintain system coherence while preserving patterns with constructive
dissonance potential.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class AccretiveWeedingService:
    """Service for systematically pruning low-value patterns to maintain system coherence."""
    
    def __init__(self, db_connection, event_service, pattern_evolution_service, 
                 constructive_dissonance_service=None):
        self.db_connection = db_connection
        self.event_service = event_service
        self.pattern_evolution_service = pattern_evolution_service
        self.constructive_dissonance_service = constructive_dissonance_service
        self.weeding_metrics = {
            "noise_threshold": 0.25,
            "signal_amplification": 0.65,
            "coherence_boundary": 0.4,
            "dissonance_allowance": 0.3,
            "emergence_sensitivity": 0.7,
            "pattern_density_threshold": 0.4
        }
        logger.info("Initialized Accretive Weeding Service")
    
    async def evaluate_pattern_value(self, pattern_id):
        """Evaluate the value of a pattern to determine if it should be pruned.
        
        Args:
            pattern_id: ID of the pattern to evaluate
            
        Returns:
            Value metrics including retention_score and pruning_recommendation
        """
        # Get pattern data
        pattern = await self._get_pattern(pattern_id)
        if not pattern:
            return {"retention_score": 0, "pruning_recommendation": True}
        
        # Get pattern usage statistics
        usage_stats = await self._get_pattern_usage(pattern_id)
        
        # Calculate base retention score from pattern properties
        coherence = pattern.get("coherence", 0.5)
        confidence = pattern.get("confidence", 0.5)
        
        # Patterns with very low coherence or confidence are candidates for pruning
        base_score = (coherence + confidence) / 2
        
        # Adjust based on usage statistics
        usage_frequency = usage_stats.get("usage_frequency", 0)
        usage_recency = usage_stats.get("usage_recency", 0)
        usage_score = (usage_frequency * 0.7 + usage_recency * 0.3) * 0.4
        
        # Adjust based on relationship density
        relationship_count = await self._get_relationship_count(pattern_id)
        relationship_factor = min(1.0, relationship_count / 5) * 0.3
        
        # Calculate final retention score
        retention_score = base_score * 0.3 + usage_score + relationship_factor
        
        # Determine pruning recommendation
        pruning_recommendation = retention_score < self.weeding_metrics["noise_threshold"]
        
        # Check for constructive dissonance potential
        dissonance_potential = 0
        if pruning_recommendation and self.constructive_dissonance_service:
            dissonance_potential = await self._check_dissonance_potential(pattern_id)
            if dissonance_potential > self.weeding_metrics["dissonance_allowance"]:
                # Preserve patterns with high dissonance potential even if low retention score
                pruning_recommendation = False
                logger.info(f"Pattern {pattern_id} preserved due to high dissonance potential: {dissonance_potential:.2f}")
        
        return {
            "retention_score": retention_score,
            "pruning_recommendation": pruning_recommendation,
            "base_score": base_score,
            "usage_score": usage_score,
            "relationship_factor": relationship_factor,
            "dissonance_potential": dissonance_potential
        }
    
    async def prune_low_value_patterns(self, context=None):
        """Identify and prune low-value patterns to maintain system coherence.
        
        Args:
            context: Optional context information
            
        Returns:
            Pruning results including pruned_count and preserved_count
        """
        # Get all patterns
        patterns = await self._get_all_patterns()
        
        pruned_count = 0
        preserved_count = 0
        dissonance_preserved_count = 0
        
        for pattern in patterns:
            pattern_id = pattern.get("id")
            
            # Evaluate pattern value
            evaluation = await self.evaluate_pattern_value(pattern_id)
            
            if evaluation["pruning_recommendation"]:
                # Check for constructive dissonance one more time
                dissonance_potential = await self._check_dissonance_potential(pattern_id)
                if dissonance_potential > self.weeding_metrics["dissonance_allowance"]:
                    # Preserve due to dissonance potential
                    logger.info(f"Pattern {pattern_id} preserved due to dissonance potential: {dissonance_potential:.2f}")
                    dissonance_preserved_count += 1
                    preserved_count += 1
                else:
                    # Prune the pattern
                    await self._prune_pattern(pattern_id)
                    logger.info(f"Pruned low-value pattern: {pattern_id} (score: {evaluation['retention_score']:.2f})")
                    pruned_count += 1
            else:
                preserved_count += 1
        
        logger.info(f"Pruning complete: {pruned_count} pruned, {preserved_count} preserved ({dissonance_preserved_count} for dissonance)")
        
        # Publish weeding results event
        if self.event_service:
            self.event_service.publish(
                "pattern.weeding.completed",
                {
                    "pruned_count": pruned_count,
                    "preserved_count": preserved_count,
                    "dissonance_preserved_count": dissonance_preserved_count,
                    "timestamp": datetime.now().isoformat(),
                    "context": context or {}
                }
            )
        
        return {
            "pruned_count": pruned_count,
            "preserved_count": preserved_count,
            "dissonance_preserved_count": dissonance_preserved_count
        }
    
    async def configure_weeding_metrics(self, metrics):
        """Configure the weeding metrics.
        
        Args:
            metrics: Dictionary of weeding metrics to update
            
        Returns:
            Updated weeding metrics
        """
        # Update metrics
        for key, value in metrics.items():
            if key in self.weeding_metrics:
                self.weeding_metrics[key] = value
                
        logger.info("Updated weeding metrics:")
        for key, value in self.weeding_metrics.items():
            logger.info(f"  - {key}: {value:.2f}")
            
        return self.weeding_metrics
    
    async def _check_dissonance_potential(self, pattern_id):
        """Check if a pattern has constructive dissonance potential."""
        if self.constructive_dissonance_service:
            # Get related patterns
            related_patterns = await self._get_related_patterns(pattern_id)
            
            # Calculate dissonance metrics
            dissonance_metrics = await self.constructive_dissonance_service.calculate_pattern_dissonance(
                pattern_id, related_patterns
            )
            
            return dissonance_metrics.get("productive_potential", 0)
        
        # If no dissonance service, use a simulated value
        return 0.2  # Low-moderate dissonance potential
    
    async def _prune_pattern(self, pattern_id):
        """Prune a pattern by marking it as pruned or removing it."""
        # Option 1: Mark as pruned but keep in database
        await self.pattern_evolution_service.update_pattern(
            pattern_id,
            {
                "pruned": True,
                "pruned_at": datetime.now().isoformat(),
                "quality_state": "pruned"
            }
        )
        
        # Publish event
        if self.event_service:
            self.event_service.publish(
                "pattern.pruned",
                {
                    "pattern_id": pattern_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def _get_pattern_usage(self, pattern_id):
        """Get usage statistics for a pattern."""
        if hasattr(self.db_connection, 'execute_query'):
            # Try to get real usage statistics from the database
            query = f"""
            FOR u IN pattern_usage
            FILTER u.pattern_id == '{pattern_id}'
            SORT u.timestamp DESC
            LIMIT 1
            RETURN u
            """
            
            result = await self.db_connection.execute_query(query)
            if result and len(result) > 0:
                usage = result[0]
                
                # Calculate recency factor (1.0 for very recent, decreasing over time)
                last_used = usage.get("timestamp")
                if last_used:
                    try:
                        last_used_date = datetime.fromisoformat(last_used)
                        now = datetime.now()
                        days_since = (now - last_used_date).days
                        
                        # Recency factor decreases with time (1.0 for today, 0.0 for 30+ days ago)
                        recency = max(0, 1.0 - (days_since / 30))
                    except:
                        recency = 0.5  # Default if date parsing fails
                else:
                    recency = 0.5  # Default if no timestamp
                
                return {
                    "usage_frequency": usage.get("frequency", 0.3),
                    "usage_recency": recency,
                    "last_used": last_used
                }
        
        # If no real usage data, return simulated usage stats
        return {
            "usage_frequency": 0.3,  # Low-moderate usage
            "usage_recency": 0.2,    # Not used recently
            "last_used": (datetime.now() - timedelta(days=7)).isoformat()
        }
    
    async def _get_relationship_count(self, pattern_id):
        """Get the number of relationships for a pattern."""
        if hasattr(self.db_connection, 'execute_query'):
            query = f"""
            FOR rel IN pattern_relationships
            FILTER rel._from == 'patterns/{pattern_id}' OR rel._to == 'patterns/{pattern_id}'
            COLLECT WITH COUNT INTO count
            RETURN count
            """
            result = await self.db_connection.execute_query(query)
            if result and len(result) > 0:
                return result[0]
        return 0
        
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
        
    async def _get_all_patterns(self):
        """Get all patterns from the database."""
        if hasattr(self.db_connection, 'execute_query'):
            query = """
            FOR p IN patterns
            FILTER p.pruned != true
            RETURN p
            """
            return await self.db_connection.execute_query(query)
        return []
