from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from collections import OrderedDict
import threading
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from dependency_injector.wiring import inject, Provide

from adaptive_core.pattern_core import PatternCore
from adaptive_core.relationship_model import RelationshipModel
from config import AppContainer
from utils.logging_config import get_logger
from utils.performance_monitor import performance_monitor
from utils.ethical_ai_checker import EthicalAIChecker
from .relationship_model import RelationshipModel
from neo4jdb.neo4j_client import Neo4jClient
from database.mongo_client import MongoClient
from events.event_manager import EventManager
from events.event_types import EventType

logger = get_logger(__name__)

class RelationshipRepositoryError(Exception):
    """Base exception class for relationship repository errors."""
    pass

class RelationshipRepository:
    """
    Repository for managing RelationshipModel objects with support for
    caching, sharding, batch operations, and adaptive feedback loops.
    """

    @inject
    def __init__(
        self,
        config: Dict[str, Any] = Provide[AppContainer.config],
        neo4j_client: Neo4jClient = Provide[AppContainer.neo4j_client],
        mongodb_client: MongoDBClient = Provide[AppContainer.mongodb_client],
        event_manager: EventManager = Provide[AppContainer.event_manager],
        adaptive_learner: 'AdaptiveLearner' = Provide[AppContainer.adaptive_learner],
        feedback_collector: 'FeedbackCollector' = Provide[AppContainer.feedback_collector],
        ethical_ai_checker: 'EthicalAIChecker' = Provide[AppContainer.ethical_ai_checker]
    ):
        """
        Initialize repository with required services and hooks.
        """
        """
        Initialize the RelationshipRepository.

        Args:
            config (Dict[str, Any]): Configuration parameters injected from AppContainer.
            cache_manager (LRUCacheManager): Cache manager for handling relationship caching.
            shard_manager (ShardManager): Shard manager for managing shards of relationships.
            batch_processor (BatchProcessor): Batch processor for handling batch operations.
            performance_monitor (PerformanceMonitor): Performance monitor for tracking operation metrics.
        """
        self.cache_manager = cache_manager
        self.shard_manager = shard_manager
        self.batch_processor = batch_processor
        self.performance_monitor = performance_monitor
        
        self.cache_expiration = config['RELATIONSHIP_REPOSITORY']['CACHE_EXPIRATION']
        self.shard_count = config['RELATIONSHIP_REPOSITORY']['SHARD_COUNT']
        
        # Initialize relationships, versions, lock, and adaptation scores
        self.relationships: Dict[int, Dict[str, RelationshipModel]] = {i: {} for i in range(self.shard_count)}
        self.relationship_versions: Dict[str, List[Version]] = {}
        self.adaptation_scores: Dict[str, int] = {}  # Track adaptability of relationships
        self._lock = threading.RLock()  # Using RLock to allow reentrant behavior for nested locks
        self._batch_operations: List[Tuple[str, RelationshipModel]] = []


        # Additional components for feedback and adaptation
        self.event_manager = event_manager
        self.adaptive_learner = adaptive_learner
        self.feedback_collector = feedback_collector
        self.ethical_ai_checker = ethical_ai_checker

        # Hooks registry
        self.pre_operation_hooks: Dict[str, List[Callable]] = {
            'add': [],
            'update': [],
            'delete': [],
            'batch': []
        }
        self.post_operation_hooks: Dict[str, List[Callable]] = {
            'add': [],
            'update': [],
            'delete': [],
            'batch': []
        }

        # Feedback loops
        self.feedback_loops: List[Callable] = []
        
        # Initialize feedback loops
        self._initialize_feedback_loops()

    def _initialize_feedback_loops(self) -> None:
        """Initialize feedback loops for adaptive learning."""
        self.feedback_loops.extend([
            self._relationship_usage_feedback,
            self._uncertainty_propagation_feedback,
            self._bidirectional_learning_feedback,
            self._ethical_compliance_feedback
        ])

    def register_pre_hook(self, operation: str, hook: Callable) -> None:
        """Register a pre-operation hook."""
        if operation in self.pre_operation_hooks:
            self.pre_operation_hooks[operation].append(hook)

    def register_post_hook(self, operation: str, hook: Callable) -> None:
        """Register a post-operation hook."""
        if operation in self.post_operation_hooks:
            self.post_operation_hooks[operation].append(hook)

    def _execute_pre_hooks(self, operation: str, data: Any) -> None:
        """Execute pre-operation hooks."""
        for hook in self.pre_operation_hooks[operation]:
            try:
                hook(data)
            except Exception as e:
                logger.error(f"Error in pre-hook for {operation}: {str(e)}")

    def _execute_post_hooks(self, operation: str, data: Any) -> None:
        """Execute post-operation hooks."""
        for hook in self.post_operation_hooks[operation]:
            try:
                hook(data)
            except Exception as e:
                logger.error(f"Error in post-hook for {operation}: {str(e)}")

    @ethical_check
    def add_relationship(self, relationship: RelationshipModel) -> None:
        """Add a relationship with hooks and feedback."""
        try:
            # Execute pre-hooks
            self._execute_pre_hooks('add', relationship)
            
            # Ethical check
            if not self.ethical_ai_checker.check_relationship(relationship):
                raise RelationshipRepositoryError("Relationship failed ethical check")

            # Original add logic
            shard_index = self._get_shard_index(relationship)
            with self._locks['shards'][shard_index]:
                self.shards[shard_index][relationship.id] = relationship
                self._update_cache(relationship.id, relationship)
                self._add_to_batch('add', relationship)

            # Execute post-hooks
            self._execute_post_hooks('add', relationship)

            # Process feedback loops
            self._process_feedback_loops(relationship)

            # Notify event system
            self.event_manager.publish(EventType.RELATIONSHIP_ADDED, {
                'relationship_id': relationship.id,
                'type': relationship.relationship_type
            })

        except Exception as e:
            logger.error(f"Error adding relationship: {str(e)}")
            raise RelationshipRepositoryError(f"Failed to add relationship: {str(e)}")

    def _process_feedback_loops(self, relationship: RelationshipModel) -> None:
        """Process all feedback loops for a relationship."""
        for feedback_loop in self.feedback_loops:
            try:
                feedback_loop(relationship)
            except Exception as e:
                logger.error(f"Error in feedback loop: {str(e)}")

    def _relationship_usage_feedback(self, relationship: RelationshipModel) -> None:
        """Feedback loop for relationship usage patterns."""
        usage_data = self.feedback_collector.collect_usage_feedback(relationship)
        self.adaptive_learner.learn_from_usage(relationship, usage_data)

    def _uncertainty_propagation_feedback(self, relationship: RelationshipModel) -> None:
        """Feedback loop for uncertainty propagation."""
        uncertainty_data = self.feedback_collector.collect_uncertainty_feedback(relationship)
        self.adaptive_learner.learn_uncertainty_patterns(relationship, uncertainty_data)

    def _bidirectional_learning_feedback(self, relationship: RelationshipModel) -> None:
        """Feedback loop for bidirectional learning."""
        if relationship.bidirectional:
            learning_data = self.feedback_collector.collect_bidirectional_feedback(relationship)
            self.adaptive_learner.apply_bidirectional_learning(relationship, learning_data)

    def _ethical_compliance_feedback(self, relationship: RelationshipModel) -> None:
        """Feedback loop for ethical compliance."""
        compliance_data = self.ethical_ai_checker.check_relationship_compliance(relationship)
        self.feedback_collector.collect_ethical_feedback(relationship, compliance_data)

    def update_relationship(self, relationship: RelationshipModel) -> None:
        """Update relationship with hooks and feedback."""
        try:
            self._execute_pre_hooks('update', relationship)
            
            # Original update logic with adaptive components
            shard_index = self._get_shard_index(relationship)
            with self._locks['shards'][shard_index]:
                if relationship.id in self.shards[shard_index]:
                    # Get old version for comparison
                    old_version = self.shards[shard_index][relationship.id]
                    
                    # Update relationship
                    self.shards[shard_index][relationship.id] = relationship
                    self._update_cache(relationship.id, relationship)
                    self._add_to_batch('update', relationship)
                    
                    # Process changes through adaptive learner
                    self.adaptive_learner.learn_from_update(old_version, relationship)
                    
                    # Collect feedback on update
                    self.feedback_collector.collect_update_feedback(old_version, relationship)
                else:
                    raise RelationshipRepositoryError(f"Relationship {relationship.id} not found")

            self._execute_post_hooks('update', relationship)
            self._process_feedback_loops(relationship)

            self.event_manager.publish(EventType.RELATIONSHIP_UPDATED, {
                'relationship_id': relationship.id,
                'type': relationship.relationship_type
            })

        except Exception as e:
            logger.error(f"Error updating relationship: {str(e)}")
            raise RelationshipRepositoryError(f"Failed to update relationship: {str(e)}")

    def _process_batch(self) -> None:
        """Process batch with hooks and feedback."""
        try:
            self._execute_pre_hooks('batch', self._batch_operations)

            with self._locks['batch']:
                # Group operations
                operations = {
                    'add': [],
                    'update': [],
                    'delete': []
                }
                
                for operation, relationship in self._batch_operations:
                    operations[operation].append(relationship)
                
                # Process groups with feedback
                futures = [
                    self.thread_pool.submit(self._bulk_add_with_feedback, operations['add']),
                    self.thread_pool.submit(self._bulk_update_with_feedback, operations['update']),
                    self.thread_pool.submit(self._bulk_delete_with_feedback, operations['delete'])
                ]
                
                # Wait for completion
                for future in futures:
                    future.result()
                
                self._batch_operations.clear()

            self._execute_post_hooks('batch', operations)
            
            # Notify event system of batch processing
            self.event_manager.publish(EventType.BATCH_PROCESSED, {
                'operation_counts': {k: len(v) for k, v in operations.items()}
            })

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise RelationshipRepositoryError(f"Failed to process batch: {str(e)}")

    def _bulk_add_with_feedback(self, relationships: List[RelationshipModel]) -> None:
        """Perform bulk add with feedback processing."""
        if relationships:
            self.neo4j_client.bulk_create_relationships(relationships)
            self.mongodb_client.bulk_insert_relationships(relationships)
            
            # Process feedback for batch
            feedback_data = self.feedback_collector.collect_batch_feedback(relationships)
            self.adaptive_learner.learn_from_batch(relationships, feedback_data)

    def _bulk_add_with_feedback(self, relationships: List[RelationshipModel]) -> None:
        """
        Perform bulk add with feedback processing.
        
        Args:
            relationships: List of relationships to add
        """
        if relationships:
            try:
                # Process through ethical checker
                for relationship in relationships:
                    if not self.ethical_ai_checker.check_relationship(relationship):
                        logger.warning(f"Relationship {relationship.id} failed ethical check")
                        relationships.remove(relationship)

                # Bulk database operations
                self.neo4j_client.bulk_create_relationships(relationships)
                self.mongodb_client.bulk_insert_relationships(relationships)
                
                # Process feedback for batch
                feedback_data = self.feedback_collector.collect_batch_feedback(relationships)
                self.adaptive_learner.learn_from_batch(relationships, feedback_data)

                # Notify event system
                self.event_manager.publish(EventType.BULK_ADD_COMPLETED, {
                    'count': len(relationships),
                    'feedback': feedback_data
                })

            except Exception as e:
                logger.error(f"Error in bulk add with feedback: {str(e)}")
                raise RelationshipRepositoryError(f"Bulk add failed: {str(e)}")

    def _bulk_update_with_feedback(self, relationships: List[RelationshipModel]) -> None:
        """
        Perform bulk update with feedback processing.
        
        Args:
            relationships: List of relationships to update
        """
        if relationships:
            try:
                # Get old versions for comparison
                old_versions = {}
                for relationship in relationships:
                    shard_index = self._get_shard_index(relationship)
                    with self._locks['shards'][shard_index]:
                        if relationship.id in self.shards[shard_index]:
                            old_versions[relationship.id] = self.shards[shard_index][relationship.id]

                # Bulk database operations
                self.neo4j_client.bulk_update_relationships(relationships)
                self.mongodb_client.bulk_update_relationships(relationships)

                # Process changes through adaptive learner
                for relationship in relationships:
                    if relationship.id in old_versions:
                        self.adaptive_learner.learn_from_update(
                            old_versions[relationship.id],
                            relationship
                        )

                # Collect and process feedback
                update_feedback = self.feedback_collector.collect_bulk_update_feedback(
                    old_versions,
                    {r.id: r for r in relationships}
                )
                self.adaptive_learner.learn_from_bulk_updates(update_feedback)

                # Notify event system
                self.event_manager.publish(EventType.BULK_UPDATE_COMPLETED, {
                    'count': len(relationships),
                    'feedback': update_feedback
                })

            except Exception as e:
                logger.error(f"Error in bulk update with feedback: {str(e)}")
                raise RelationshipRepositoryError(f"Bulk update failed: {str(e)}")

    def _bulk_delete_with_feedback(self, relationships: List[RelationshipModel]) -> None:
        """
        Perform bulk delete with feedback processing.
        
        Args:
            relationships: List of relationships to delete
        """
        if relationships:
            try:
                # Collect pre-deletion data for feedback
                pre_deletion_data = {
                    r.id: self.get_relationship_context(r.id)
                    for r in relationships
                }

                # Bulk database operations
                relationship_ids = [r.id for r in relationships]
                self.neo4j_client.bulk_delete_relationships(relationship_ids)
                self.mongodb_client.bulk_delete_relationships(relationship_ids)

                # Process deletion feedback
                deletion_feedback = self.feedback_collector.collect_deletion_feedback(
                    relationships,
                    pre_deletion_data
                )
                self.adaptive_learner.learn_from_deletions(deletion_feedback)

                # Notify event system
                self.event_manager.publish(EventType.BULK_DELETE_COMPLETED, {
                    'count': len(relationships),
                    'feedback': deletion_feedback
                })

            except Exception as e:
                logger.error(f"Error in bulk delete with feedback: {str(e)}")
                raise RelationshipRepositoryError(f"Bulk delete failed: {str(e)}")

    def get_relationship_context(self, relationship_id: str) -> Dict[str, Any]:
        """
        Get the full context of a relationship including connected relationships.
        
        Args:
            relationship_id: ID of the relationship
            
        Returns:
            Dict containing relationship context
        """
        context = {
            'relationship': None,
            'connected_relationships': [],
            'usage_patterns': {},
            'uncertainty_metrics': {}
        }

        try:
            relationship = self.get_relationship(relationship_id)
            if relationship:
                context['relationship'] = relationship
                
                # Get connected relationships
                source_relationships = self.get_relationships_for_concept(relationship.source_id)
                target_relationships = self.get_relationships_for_concept(relationship.target_id)
                
                context['connected_relationships'] = list(set(source_relationships + target_relationships))
                
                # Get usage patterns
                context['usage_patterns'] = self.adaptive_learner.get_usage_patterns(relationship_id)
                
                # Get uncertainty metrics
                context['uncertainty_metrics'] = self.adaptive_learner.get_uncertainty_metrics(relationship_id)

            return context

        except Exception as e:
            logger.error(f"Error getting relationship context: {str(e)}")
            return context

    @performance_monitor.track
    def analyze_relationship_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns across all relationships.
        
        Returns:
            Dict containing pattern analysis results
        """
        try:
            analysis = {
                'relationship_types': {},
                'uncertainty_distribution': {},
                'usage_patterns': {},
                'temporal_patterns': {},
                'spatial_patterns': {},
                'ethical_metrics': {}
            }

            # Analyze all shards
            for shard_index, shard in enumerate(self.shards):
                with self._locks['shards'][shard_index]:
                    for relationship in shard.values():
                        # Analyze relationship types
                        rel_type = relationship.relationship_type
                        analysis['relationship_types'][rel_type] = analysis['relationship_types'].get(rel_type, 0) + 1

                        # Analyze uncertainty
                        uncertainty_level = self._categorize_uncertainty(relationship)
                        analysis['uncertainty_distribution'][uncertainty_level] = (
                            analysis['uncertainty_distribution'].get(uncertainty_level, 0) + 1
                        )

                        # Analyze usage patterns
                        usage_pattern = self.adaptive_learner.get_usage_pattern(relationship.id)
                        if usage_pattern:
                            analysis['usage_patterns'][usage_pattern] = (
                                analysis['usage_patterns'].get(usage_pattern, 0) + 1
                            )

                        # Analyze temporal patterns
                        if relationship.temporal_context:
                            temporal_pattern = self._analyze_temporal_pattern(relationship)
                            analysis['temporal_patterns'][temporal_pattern] = (
                                analysis['temporal_patterns'].get(temporal_pattern, 0) + 1
                            )

                        # Analyze spatial patterns
                        if relationship.spatial_context:
                            spatial_pattern = self._analyze_spatial_pattern(relationship)
                            analysis['spatial_patterns'][spatial_pattern] = (
                                analysis['spatial_patterns'].get(spatial_pattern, 0) + 1
                            )

                        # Analyze ethical metrics
                        ethical_rating = self.ethical_ai_checker.get_relationship_rating(relationship)
                        analysis['ethical_metrics'][ethical_rating] = (
                            analysis['ethical_metrics'].get(ethical_rating, 0) + 1
                        )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing relationship patterns: {str(e)}")
            raise RelationshipRepositoryError(f"Pattern analysis failed: {str(e)}")

    def _categorize_uncertainty(self, relationship: RelationshipModel) -> str:
        """Categorize relationship uncertainty level."""
        uncertainty = relationship.uncertainty_metrics.uncertainty_value
        if uncertainty < 0.3:
            return "low"
        elif uncertainty < 0.7:
            return "medium"
        return "high"

    def _analyze_temporal_pattern(self, relationship: RelationshipModel) -> str:
        """Analyze temporal pattern of a relationship."""
        context = relationship.temporal_context
        if not context.end_time:
            return "ongoing"
        elif context.duration and context.duration.total_seconds() < 86400:  # 24 hours
            return "short_term"
        elif context.duration and context.duration.total_seconds() < 2592000:  # 30 days
            return "medium_term"
        return "long_term"

    def _analyze_spatial_pattern(self, relationship: RelationshipModel) -> str:
        """Analyze spatial pattern of a relationship."""
        context = relationship.spatial_context
        if context.scale == "local":
            return "local"
        elif context.scale == "regional":
            return "regional"
        return "global"

    def cleanup(self) -> None:
        """Perform cleanup operations."""
        try:
            # Clean cache
            self._clean_cache()

            # Process any remaining batch operations
            if self._batch_operations:
                self._process_batch()

            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)

            logger.info("Repository cleanup completed successfully")

        except Exception as e:
            logger.error(f"Error during repository cleanup: {str(e)}")
            raise RelationshipRepositoryError(f"Cleanup failed: {str(e)}")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass 