# config.py

"""
This module serves as the central configuration and dependency injection hub for the Adaptive POC climate knowledge base application.
It implements a robust configuration system, sets up dependency injection, and manages import structures for the entire application.
This configuration is designed to support a fully adaptive, bidirectional ontology system with ethical considerations and scalability in mind.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
# Placeholder for secret management service integration
# from secure_secrets_manager import get_secret
from dependency_injector import containers, providers
from dependency_injector.wiring import inject, Provide
import logging
from importlib import import_module

# Load environment variables
load_dotenv()

# Check for critical environment variables before proceeding
required_env_vars = [
    "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
    "MONGODB_URI", "MONGODB_USERNAME", "MONGODB_PASSWORD",
    "GOOGLE_PROJECT_ID", "SERVICE_ACCOUNT_FILE"
]
for var in required_env_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Critical environment variable {var} is missing.")
    
class EventType:
    """
    Defines various types of events used throughout the Habitat Adaptive POC.
    These event types are used to maintain consistency when emitting and subscribing to events.
    """
    ONTOLOGY_UPDATED = "ontology_updated"
    RELATIONSHIP_CREATED = "relationship_created"
    ADAPTIVE_CHANGE = "adaptive_change"
    RELATIONSHIP_DELETED = "relationship_deleted"
    ONTOLOGY_CONFLICT_RESOLVED = "ontology_conflict_resolved"
    RELATIONSHIP_MODIFIED = "relationship_modified"
    DATA_VALIDATION_FAILED = "data_validation_failed"
    PERFORMANCE_MONITOR_ALERT = "performance_monitor_alert"


class ConfigLoader:
    """
    Configuration loader class that manages loading and accessing configuration parameters.
    This class centralizes all configuration management, making it easier to modify and extend the configuration system.
    """

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load all configuration parameters from environment variables and set defaults."""
        # Load API keys for external services
        self._load_api_keys()

        # Load database configurations
        self._load_database_config()

        # Load Google Services configurations
        self._load_google_services_config()

        # Load application-specific configurations
        self._load_app_specific_config()

        # Load Adaptive POC specific configurations
        self._load_adaptive_poc_config()

        # Load timestamp configuration
        self._load_timestamp_config()

        # Load vector representation configuration
        self._load_vector_representation_config()

        # Load serialization configuration
        self._load_serialization_config()

        # Load Adaptive ID management configuration
        self._load_adaptive_id_management_config()

        # Load relationship management configuration
        self._load_relationship_management_config()

        # Load adaptation process configuration
        self._load_adaptation_process_config()

        # Load RelationshipRepository configuration
        self._load_relationship_repository_config()

        # Load bidirectional ontology configuration
        self._load_bidirectional_ontology_config()
        
        # Load AdaptiveIDManager configurations
        self._load_adaptive_id_manager_config()
        
        # Load spatio-temporal reasoning configurations
        self._load_spatio_temporal_config()
        
        # Load advanced machine learning configurations
        self._load_advanced_ml_config()
        
        # Load cross-domain component configurations
        self._load_cross_domain_config()
        
        # Load visualization component configurations
        self._load_visualization_config()
        
    def _load_adaptive_id_manager_config(self) -> None:
        """
        Load configurations for AdaptiveIDManager.

        This method sets up the configuration parameters for the AdaptiveIDManager,
        which is responsible for managing Adaptive IDs in the system. It includes
        settings for caching, ID generation, and maintenance of Adaptive IDs.

        Configuration parameters:
        - MAX_CACHE_SIZE: Maximum number of IDs to keep in cache
        - PRUNING_INTERVAL: Interval (in seconds) for pruning expired or least used IDs
        - ID_PREFIX: Prefix for generated Adaptive IDs
        """
        self.config.update({
            "ADAPTIVE_ID_MANAGER": {
                "MAX_CACHE_SIZE": int(os.getenv("ADAPTIVE_ID_MANAGER_MAX_CACHE_SIZE", "10000")),
                "PRUNING_INTERVAL": int(os.getenv("ADAPTIVE_ID_MANAGER_PRUNING_INTERVAL", "3600")),
                "ID_PREFIX": os.getenv("ADAPTIVE_ID_MANAGER_ID_PREFIX", "AID-")
            }
        })

    def _load_spatio_temporal_config(self) -> None:
        """
        Load configurations for spatio-temporal reasoning.

        This method sets up the configuration parameters for spatio-temporal
        reasoning capabilities in the system. It includes default resolutions,
        ranges, and indexing strategies for handling spatial and temporal data.

        Configuration parameters:
        - DEFAULT_SPATIAL_RESOLUTION: Default spatial resolution for analysis
        - DEFAULT_TEMPORAL_RESOLUTION: Default temporal resolution for analysis
        - MAX_TEMPORAL_RANGE: Maximum temporal range (in days) for queries
        - SPATIAL_INDEX_TYPE: Type of spatial index to use (e.g., quadtree, r-tree)
        """
        self.config.update({
            "SPATIO_TEMPORAL": {
                "DEFAULT_SPATIAL_RESOLUTION": os.getenv("DEFAULT_SPATIAL_RESOLUTION", "1km"),
                "DEFAULT_TEMPORAL_RESOLUTION": os.getenv("DEFAULT_TEMPORAL_RESOLUTION", "1day"),
                "MAX_TEMPORAL_RANGE": int(os.getenv("MAX_TEMPORAL_RANGE", "3650")),
                "SPATIAL_INDEX_TYPE": os.getenv("SPATIAL_INDEX_TYPE", "quadtree")
            },
            "UNCERTAINTY": {
                "UNCERTAINTY_MODEL": os.getenv("UNCERTAINTY_MODEL", "probability_distribution"),
                "DEFAULT_CONFIDENCE_LEVEL": float(os.getenv("DEFAULT_CONFIDENCE_LEVEL", "0.8"))
            },
            "MULTI_SCALE": {
                "DEFAULT_SCALE": os.getenv("DEFAULT_SCALE", "national"),
                "SCALE_HIERARCHY":  ['local', 'regional', 'national', 'global']
            }
        })

    def _load_advanced_ml_config(self) -> None:
        """
        Load configurations for advanced machine learning components.

        This method sets up the configuration parameters for advanced machine
        learning components, specifically the GNN (Graph Neural Network) Operator
        and Graph Embedder. These components are crucial for graph-based machine
        learning tasks in the system.

        Configuration parameters:
        GNN_OPERATOR:
        - MODEL_TYPE: Type of GNN model to use (e.g., GraphSAGE)
        - HIDDEN_CHANNELS: Number of hidden channels in the GNN
        - NUM_LAYERS: Number of layers in the GNN

        GRAPH_EMBEDDER:
        - EMBEDDING_DIM: Dimensionality of the graph embeddings
        - NEGATIVE_SAMPLING_RATIO: Ratio for negative sampling in embedding training
        """
        self.config.update({
            "GNN_OPERATOR": {
                "MODEL_TYPE": os.getenv("GNN_MODEL_TYPE", "GraphSAGE"),
                "HIDDEN_CHANNELS": int(os.getenv("GNN_HIDDEN_CHANNELS", "64")),
                "NUM_LAYERS": int(os.getenv("GNN_NUM_LAYERS", "3"))
            },
            "GRAPH_EMBEDDER": {
                "EMBEDDING_DIM": int(os.getenv("GRAPH_EMBEDDING_DIM", "128")),
                "NEGATIVE_SAMPLING_RATIO": float(os.getenv("NEGATIVE_SAMPLING_RATIO", "0.5"))
            }
        })

    def _load_cross_domain_config(self) -> None:
        """
        Load configurations for cross-domain components.

        This method sets up the configuration parameters for cross-domain
        components, including the Cross-domain Linker and Ontology Merger.
        These components are essential for integrating and harmonizing
        knowledge across different domains in the system.

        Configuration parameters:
        CROSS_DOMAIN_LINKER:
        - SIMILARITY_THRESHOLD: Threshold for considering concepts as similar
        - MAX_LINKS_PER_CONCEPT: Maximum number of cross-domain links per concept

        ONTOLOGY_MERGER:
        - CONFLICT_RESOLUTION_STRATEGY: Strategy for resolving conflicts during merging
        - MERGE_THRESHOLD: Threshold for merging ontology elements
        """
        self.config.update({
            "CROSS_DOMAIN_LINKER": {
                "SIMILARITY_THRESHOLD": float(os.getenv("CROSS_DOMAIN_SIMILARITY_THRESHOLD", "0.7")),
                "MAX_LINKS_PER_CONCEPT": int(os.getenv("MAX_CROSS_DOMAIN_LINKS", "5"))
            },
            "ONTOLOGY_MERGER": {
                "CONFLICT_RESOLUTION_STRATEGY": os.getenv("ONTOLOGY_MERGER_CONFLICT_STRATEGY", "latest_win"),
                "MERGE_THRESHOLD": float(os.getenv("ONTOLOGY_MERGE_THRESHOLD", "0.8"))
            }
        })

    def _load_visualization_config(self) -> None:
        """
        Load configurations for visualization components.

        This method sets up the configuration parameters for visualization
        components, including the Graph Visualizer and Analytics Dashboard.
        These components are crucial for presenting and analyzing the
        knowledge graph and system metrics in a user-friendly manner.

        Configuration parameters:
        GRAPH_VISUALIZER:
        - MAX_NODES_DISPLAY: Maximum number of nodes to display in the graph
        - EDGE_THRESHOLD: Threshold for displaying edges (based on weight or importance)
        - DEFAULT_LAYOUT: Default layout algorithm for graph visualization

        ANALYTICS_DASHBOARD:
        - UPDATE_INTERVAL: Interval for updating dashboard metrics (in seconds)
        - MAX_TIME_RANGE: Maximum time range for historical data in the dashboard (in days)
        - DEFAULT_METRICS: Default set of metrics to display in the dashboard
        """
        self.config.update({
            "GRAPH_VISUALIZER": {
                "MAX_NODES_DISPLAY": int(os.getenv("MAX_NODES_DISPLAY", "1000")),
                "EDGE_THRESHOLD": float(os.getenv("EDGE_DISPLAY_THRESHOLD", "0.1")),
                "DEFAULT_LAYOUT": os.getenv("GRAPH_LAYOUT_ALGORITHM", "force_directed")
            },
            "ANALYTICS_DASHBOARD": {
                "UPDATE_INTERVAL": int(os.getenv("DASHBOARD_UPDATE_INTERVAL", "300")),
                "MAX_TIME_RANGE": int(os.getenv("DASHBOARD_MAX_TIME_RANGE", "30")),
                "DEFAULT_METRICS": os.getenv("DEFAULT_DASHBOARD_METRICS", "node_count,edge_count,avg_degree").split(',')
            }
        })

    def _load_bidirectional_ontology_config(self) -> None:
        """
        Load configurations specific to the bidirectional adaptive ontology system.
        This includes settings for learning rates, adaptation thresholds, and
        bidirectional update frequencies.
        """
        self.config.update({
            "BIDIRECTIONAL_LEARNING_RATE": float(os.getenv("BIDIRECTIONAL_LEARNING_RATE", "0.01")),
            "FEEDBACK_LEARNING_RATE": float(os.getenv("FEEDBACK_LEARNING_RATE", "0.05")),
            "ONTOLOGY_UPDATE_THRESHOLD": float(os.getenv("ONTOLOGY_UPDATE_THRESHOLD", "0.1")),
            "DATA_UPDATE_THRESHOLD": float(os.getenv("DATA_UPDATE_THRESHOLD", "0.1")),
            "BIDIRECTIONAL_UPDATE_FREQUENCY": int(os.getenv("BIDIRECTIONAL_UPDATE_FREQUENCY", "3600")),
            "CONFLICT_RESOLUTION_STRATEGY": os.getenv("CONFLICT_RESOLUTION_STRATEGY", "weighted_average"),
            "TIME_BASED_UPDATE_INTERVAL": int(os.getenv("TIME_BASED_UPDATE_INTERVAL", "86400"))  # New config for time-based updates
        })

    def _load_relationship_repository_config(self) -> None:
        """
        Load RelationshipRepository configurations.

        This method sets up the configuration parameters for the RelationshipRepository,
        which is crucial for the hybrid RelationshipModel ID/Object approach. It includes
        settings for caching, batch operations, and other performance optimizations.

        The following parameters are configured:
        - CACHE_SIZE: The maximum number of relationship objects to keep in memory.
        - BATCH_SIZE: The number of relationships to process in a single batch operation.
        - CACHE_EXPIRATION: The time (in seconds) after which a cached relationship should be refreshed.
        - SHARD_COUNT: Defines the number of shards for splitting large datasets.
        - PERFORMANCE_MONITOR: Configuration for enabling/disabling and setting logging intervals for performance monitoring.
        """
        self.config.update({
            "RELATIONSHIP_REPOSITORY": {
                "CACHE_SIZE": int(os.getenv("RELATIONSHIP_CACHE_SIZE", "2000")),
                "BATCH_SIZE": int(os.getenv("RELATIONSHIP_BATCH_SIZE", "200")),
                "CACHE_EXPIRATION": int(os.getenv("RELATIONSHIP_CACHE_EXPIRATION", "3600")),
                "SHARD_COUNT": int(os.getenv("RELATIONSHIP_SHARD_COUNT", "20")),
                "CONFLICT_RESOLUTION_STRATEGY": os.getenv("RELATIONSHIP_CONFLICT_RESOLUTION", "merge"),
                "PERFORMANCE_MONITOR": {
                    'ENABLED': bool(os.getenv("RELATIONSHIP_REPOSITORY_PERFORMANCE_MONITOR_ENABLED", "True")),
                    'LOGGING_INTERVAL': int(os.getenv("RELATIONSHIP_REPOSITORY_PERFORMANCE_LOGGING_INTERVAL", "600"))
                }
            }
        })


    def _load_api_keys(self) -> None:
        """Load API key configurations."""
        self.config.update({
            "OPENAI_API_KEY": get_secret("OPENAI_API_KEY") if 'OPENAI_API_KEY' in os.environ else None,
            "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY"),
            "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        })

    def _load_database_config(self) -> None:
        """Load database configurations."""
        self.config.update({
            "NEO4J_URI": os.getenv("NEO4J_URI"),
            "NEO4J_USER": os.getenv("NEO4J_USER"),
            "NEO4J_PASSWORD": get_secret("NEO4J_PASSWORD") if 'NEO4J_PASSWORD' in os.environ else None,
            "MONGODB_URI": os.getenv("MONGODB_URI"),
            "MONGODB_API_KEY": os.getenv("MONGODB_API_KEY"),
            "MONGODB_USERNAME": os.getenv("MONGODB_USERNAME"),
            "MONGODB_PASSWORD": get_secret("MONGODB_PASSWORD") if 'MONGODB_PASSWORD' in os.environ else None,
            "CLUSTER_NAME": os.getenv("CLUSTER_NAME"),
            "MONGODB_DB_NAME": os.getenv("MONGODB_DB_NAME"),
        })

    def _load_google_services_config(self) -> None:
        """Load Google Services configurations."""
        self.config.update({
            "GOOGLE_SERVICE_ACCOUNT_NAME": os.getenv("GOOGLE_SERVICE_ACCOUNT_NAME"),
            "GOOGLE_SERVICE_ACCOUNT_EMAIL": os.getenv("GOOGLE_SERVICE_ACCOUNT_EMAIL"),
            "GOOGLE_SERVICE_ACCOUNT_KEY": os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY"),
            "REPORTID": os.getenv("REPORTID"),
            "FOLDERID": os.getenv("FOLDERID"),
            "GOOGLE_PROJECT_NUMBER": os.getenv("GOOGLE_PROJECT_NUMBER"),
            "GOOGLE_PROJECT_ID": os.getenv("GOOGLE_PROJECT_ID"),
            "SERVICE_ACCOUNT_FILE": get_secret("SERVICE_ACCOUNT_FILE") if 'SERVICE_ACCOUNT_FILE' in os.environ else None,
        })

    def _load_app_specific_config(self) -> None:
        """Load application-specific configurations."""
        self.config.update({
            "SCHEMA_DIRECTORY": os.getenv("SCHEMA_DIRECTORY", "schemas/json_schemas"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
            "USE_RAG": os.getenv("USE_RAG", "True").lower() == "true",
            "VECTOR_INDEX_PATH": os.getenv("VECTOR_INDEX_PATH", "vector_indexes"),
            "GITHUB_ACCESS_TOKEN": os.getenv("GITHUB_ACCESS_TOKEN"),
            "SUPPORTED_FORMATS": {
                'txt': 'text/plain',
                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'pdf': 'application/pdf',
                'rtf': 'application/rtf'
            },
        })

    def _load_adaptive_poc_config(self) -> None:
        """Load Adaptive POC specific configurations."""
        self.config.update({
            "ADAPTIVE_ID_VERSION": "1.1",  # Updated version
            "RELATIONSHIP_MODEL_VERSION": "1.1",  # Updated version
            "CLIMATE_DOMAIN_ONTOLOGY_VERSION": "1.1",  # Updated version
            "ETHICAL_FRAMEWORK_VERSION": "1.1",  # Updated version
            "SPATIO_TEMPORAL_RESOLUTION": "day",
            "DEFAULT_CONFIDENCE_THRESHOLD": 0.7,
            "MAX_RELATIONSHIP_DEPTH": 5,
            "ETHICAL_GUIDELINES": ["no_discrimination", "data_privacy", "transparency", "fairness"],
            "FEEDBACK_LEARNING_RATE": 0.1,
            "ADAPTIVE_QUERY_CACHE_SIZE": 1000,
            "COLLABORATIVE_EDIT_TIMEOUT": 300,  # in seconds
            "BIDIRECTIONAL_ENABLED": os.getenv("BIDIRECTIONAL_ENABLED", "True").lower() == "true",
            "ADAPTIVE_LEARNING_ENABLED": os.getenv("ADAPTIVE_LEARNING_ENABLED", "True").lower() == "true",
        })

    def _load_timestamp_config(self) -> None:
        """
        Load timestamp service configurations.
        This includes format, timezone, and precision settings for consistent timestamp handling across the application.
        """
        self.config.update({
            "TIMESTAMP_FORMAT": os.getenv("TIMESTAMP_FORMAT", "%Y-%m-%d %H:%M:%S.%f"),
            "TIMESTAMP_TIMEZONE": os.getenv("TIMESTAMP_TIMEZONE", "UTC"),
            "TIMESTAMP_PRECISION": os.getenv("TIMESTAMP_PRECISION", "microsecond"),
        })

    def _load_vector_representation_config(self) -> None:
        """
        Load vector representation configurations.
        This includes settings for dimensionality, embedding model, and similarity thresholds for vector operations.
        """
        self.config.update({
            "VECTOR_DIMENSIONALITY": int(os.getenv("VECTOR_DIMENSIONALITY", "768")),
            "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "openai"),
            "VECTOR_SIMILARITY_THRESHOLD": float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", "0.75")),
        })


    def _load_serialization_config(self) -> None:
        """
        Load serialization and deserialization configurations.
        This includes settings for the format and compression of data when storing or transmitting AdaptiveID instances.
        """
        self.config.update({
            "SERIALIZATION_FORMAT": os.getenv("SERIALIZATION_FORMAT", "json"),
            "COMPRESSION_ALGORITHM": os.getenv("COMPRESSION_ALGORITHM", "gzip"),
            "COMPRESSION_LEVEL": int(os.getenv("COMPRESSION_LEVEL", "6")),
        })

    def _load_adaptive_id_management_config(self) -> None:
        """
        Load Adaptive ID management configurations.
        This includes settings for version control and update thresholds for AdaptiveID instances.
        """
        self.config.update({
            "MAX_ADAPTIVE_ID_VERSIONS": int(os.getenv("MAX_ADAPTIVE_ID_VERSIONS", "20")),
            "VERSION_UPDATE_THRESHOLD": float(os.getenv("VERSION_UPDATE_THRESHOLD", "0.05")),
            "ROLLBACK_THRESHOLD": float(os.getenv("ROLLBACK_THRESHOLD", "0.02")),
            "STATE_COMPARISON_PRECISION": float(os.getenv("STATE_COMPARISON_PRECISION", "0.01"))  # Precision for state comparison
        })

    def _load_relationship_management_config(self) -> None:
        """
        Load relationship management configurations.
        This includes settings for the maximum number of relationships per AdaptiveID and strength thresholds.
        """
        self.config.update({
            "MAX_RELATIONSHIPS_PER_ID": int(os.getenv("MAX_RELATIONSHIPS_PER_ID", "100")),
            "RELATIONSHIP_STRENGTH_THRESHOLD": float(os.getenv("RELATIONSHIP_STRENGTH_THRESHOLD", "0.5")),
        })

    def _load_adaptation_process_config(self) -> None:
        """
        Load adaptation process configurations.
        This includes settings for the frequency of adaptation processes and thresholds for triggering adaptations.
        """
        self.config.update({
            "ADAPTATION_FREQUENCY": int(os.getenv("ADAPTATION_FREQUENCY", "3600")),  # in seconds
            "USAGE_COUNT_THRESHOLD": int(os.getenv("USAGE_COUNT_THRESHOLD", "100")),
            "TIME_BASED_ADAPTATION_THRESHOLD": int(os.getenv("TIME_BASED_ADAPTATION_THRESHOLD", "86400")),  # in seconds
            "CACHE_SIZE": int(os.getenv("ADAPTATION_CACHE_SIZE", "1000")),  # New cache size tuning
            "CACHE_EXPIRATION": int(os.getenv("CACHE_EXPIRATION", "7200"))  # New cache expiration settings
        })
        
        # Load Event and Messaging configurations
        self._load_event_messaging_config()

    def _load_event_messaging_config(self) -> None:
        """
        Load configurations specific to event handling and messaging.
        This includes settings for messaging broker, topics, retry policies, and event manager parameters.
        """
        self.config.update({
            # Messaging Configuration
            "MESSAGING_BROKER_URL": os.getenv("MESSAGING_BROKER_URL", 'localhost:9092'),
            "MESSAGING_TOPICS": {
                "ontology_events": os.getenv("ONTOLOGY_EVENTS_TOPIC", "OntologyEventsTopic"),
                "relationship_events": os.getenv("RELATIONSHIP_EVENTS_TOPIC", "RelationshipEventsTopic"),
                "adaptive_change_events": os.getenv("ADAPTIVE_CHANGE_EVENTS_TOPIC", "AdaptiveChangeEventsTopic"),
            },
            "MESSAGING_RETRY_POLICY": {
                "MAX_RETRIES": int(os.getenv("MESSAGING_MAX_RETRIES", "5")),
                "RETRY_INTERVAL": int(os.getenv("MESSAGING_RETRY_INTERVAL", "300")),  # in seconds
                "BACKOFF_FACTOR": float(os.getenv("MESSAGING_BACKOFF_FACTOR", "2.0")),
            },

            # Event Management Configuration
            "EVENT_LISTENER_THREADS": int(os.getenv("EVENT_LISTENER_THREADS", "5")),
            "EVENT_POLLING_INTERVAL": float(os.getenv("EVENT_POLLING_INTERVAL", "1.0")),
            "EVENT_QUEUE_SIZE": int(os.getenv("EVENT_QUEUE_SIZE", "1000")),
            "EVENT_TIMEOUT": int(os.getenv("EVENT_TIMEOUT", "30")),  # in seconds

            # Messaging Security Configurations
            "MESSAGING_SECURITY": {
                "SSL_ENABLED": os.getenv("MESSAGING_SSL_ENABLED", "False").lower() == "true",
                "SSL_CERT_FILE": os.getenv("MESSAGING_SSL_CERT_FILE", None),
                "SSL_KEY_FILE": os.getenv("MESSAGING_SSL_KEY_FILE", None),
                "SSL_CA_FILE": os.getenv("MESSAGING_SSL_CA_FILE", None),
            },
        })

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value.
        
        Args:
            key (str): The configuration key to retrieve.
            default (Any, optional): The default value to return if the key is not found.
        
        Returns:
            Any: The configuration value, or the default if not found.
        """
        return self.config.get(key, default)

    def validate(self) -> None:
        """Validate the configuration values."""
        required_keys = [
            "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
            "MONGODB_URI", "MONGODB_USERNAME", "MONGODB_PASSWORD",
            "GOOGLE_PROJECT_ID", "SERVICE_ACCOUNT_FILE",
            "MESSAGING_BROKER_URL"
        ]
        for key in required_keys:
            value = self.config.get(key)
            if not value:
                raise ValueError(f"Missing required configuration: {key}")
            if key.endswith("PASSWORD") or key.endswith("API_KEY"):
                if len(value) < 8:
                    raise ValueError(f"Configuration for {key} is invalid or too short.")

# End of ConfigLoader class

class AppContainer(containers.DeclarativeContainer):
    """
    Dependency Injection container for Habitat Adaptive POC application.
    Organized by core evolutionary layers, infrastructure, and support services.
    """
    # Base Configuration
    config = providers.Configuration()
    config_loader = providers.Singleton(ConfigLoader)

    # Core Infrastructure Services
    timestamp_service = providers.Factory(
        lambda: import_module('utils.timestamp_service').TimestampService,
        format=config.TIMESTAMP_FORMAT,
        timezone=config.TIMESTAMP_TIMEZONE,
        precision=config.TIMESTAMP_PRECISION
    )
    version_service = providers.Factory(
        lambda: import_module('utils.version_service').VersionService,
        version_limit=config.MAX_ADAPTIVE_ID_VERSIONS,
        rollback_threshold=config.ROLLBACK_THRESHOLD
    )

    # Event & Messaging System
    event_manager = providers.Singleton(
        lambda: import_module('events.event_manager').EventManager,
        listener_threads=config.EVENT_LISTENER_THREADS,
        polling_interval=config.EVENT_POLLING_INTERVAL,
        queue_size=config.EVENT_QUEUE_SIZE,
        timeout=config.EVENT_TIMEOUT
    )
    event_types = providers.Singleton(
        lambda: import_module('events.event_types').EventType
    )
    messaging_service = providers.Singleton(
        lambda: import_module('messaging.messaging_service').MessagingService,
        broker_url=config.MESSAGING_BROKER_URL,
        ssl_config=config.MESSAGING_SECURITY if config.MESSAGING_SECURITY['SSL_ENABLED'] else None
    )

    # Database & Persistence Layer
    neo4j_client = providers.Singleton(
        lambda: import_module('neo4jdb.neo4j_client').Neo4jClient,
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD
    )
    mongodb_client = providers.Singleton(
        lambda: import_module('database.mongo_client').MongoDBClient,
        uri=config.MONGODB_URI,
        api_key=config.MONGODB_API_KEY,
        username=config.MONGODB_USERNAME,
        password=config.MONGODB_PASSWORD,
        cluster_name=config.CLUSTER_NAME,
        db_name=config.MONGODB_DB_NAME
    )

    # Adaptive Core Layer - Core Evolution Components
    adaptive_id = providers.Lazy(
        lambda: import_module('adaptive_core.adaptive_id').AdaptiveID
    )
    relationship_model = providers.Factory(
        lambda: import_module('adaptive_core.relationship_model').RelationshipModel
    )
    relationship_repository = providers.Singleton(
        lambda: import_module('adaptive_core.relationship_repository').RelationshipRepository,
        config=config.RELATIONSHIP_REPOSITORY
    )

    # Adaptive Core Layer - Pattern & Coherence Services
    pattern_core = providers.Singleton(
        lambda: import_module('adaptive_core.pattern_core').PatternCore,
        timestamp_service=timestamp_service,
        event_manager=event_manager,
        version_service=version_service
    )
    knowledge_coherence = providers.Singleton(
        lambda: import_module('adaptive_core.knowledge_coherence').KnowledgeCoherence,
        pattern_core=pattern_core,  # Depends on pattern_core
        timestamp_service=timestamp_service,
        event_manager=event_manager,
        version_service=version_service
    )

    # Adaptive Core Layer - Learning Components
    bidirectional_learner = providers.Factory(
        lambda: import_module('adaptive_core.bidirectional_learner').BidirectionalLearner,
        learning_rate=config.BIDIRECTIONAL_LEARNING_RATE,
        ontology_update_threshold=config.ONTOLOGY_UPDATE_THRESHOLD,
        data_update_threshold=config.DATA_UPDATE_THRESHOLD,
        max_iterations=config.BIDIRECTIONAL_MAX_ITERATIONS,
        convergence_threshold=config.BIDIRECTIONAL_CONVERGENCE_THRESHOLD
    )
    adaptive_learner = providers.Factory(
        lambda: import_module('adaptive_core.adaptive_learner').AdaptiveLearner,
        learning_rate=config.ADAPTIVE_LEARNING_RATE,
        decay_rate=config.ADAPTIVE_DECAY_RATE,
        min_learning_rate=config.ADAPTIVE_MIN_LEARNING_RATE,
        max_iterations=config.ADAPTIVE_MAX_ITERATIONS
    )
    conflict_resolver = providers.Factory(
        lambda: import_module('adaptive_core.conflict_resolver').ConflictResolver,
        strategy=config.CONFLICT_RESOLUTION_STRATEGY
    )

    # Domain Ontology Layer
    domain_registry = providers.Singleton(
        lambda: import_module('domain_ontology.domain_registry').DomainRegistry
    )
    domain_data_loader = providers.Singleton(
        lambda: import_module('domain_ontology.domain_data_loader').DomainDataLoader
    )
    base_domain_ontology = providers.Factory(
        lambda: import_module('domain_ontology.base_domain_ontology').BaseDomainOntology
    )
    base_climate_ontology = providers.Factory(
        lambda: import_module('domain_ontology.base_climate_ontology').BaseClimateOntology
    )
    climate_domain_ontology = providers.Factory(
        lambda: import_module('domain_ontology.climate_domain_ontology').ClimateDomainOntology,
        ontology_data=providers.Dict(),
        climate_concept_validator=climate_concept_validator,
        timestamp_service=timestamp_service,
        mongo_client=mongodb_client,
        relationship_repository=relationship_repository,
        bidirectional_learner=bidirectional_learner,
        adaptive_learner=adaptive_learner,
        ethical_ai_checker=ethical_ai_checker,
        performance_monitor=performance_monitor,
        pattern_core=pattern_core,  # Add pattern tracking
        knowledge_coherence=knowledge_coherence  # Add coherence observation
    )
    domain_ontology_factory = providers.Singleton(
        lambda: import_module('domain_ontology.domain_ontology_factory').DomainOntologyFactory,
        domain_registry=domain_registry,
        domain_data_loader=domain_data_loader
    )
    domain_ontology_manager = providers.Singleton(
        lambda: import_module('domain_ontology.domain_ontology_manager').DomainOntologyManager,
        domain_ontology_factory=domain_ontology_factory
    )

    # Semantic Layer
    semantic_validator = providers.Factory(
        lambda: import_module('semantics_layer.semantic_validator').SemanticValidator
    )
    base_semantic_layer = providers.Factory(
        lambda: import_module('semantics_layer.base_semantic_layer').BaseSemanticLayer
    )
    semantic_layer = providers.Factory(
        lambda: import_module('semantics_layer.semantic_layer').SemanticLayer
    )
    semantic_layer_manager = providers.Singleton(
        lambda: import_module('semantics_layer.semantic_layer_manager').SemanticLayerManager
    )
    climate_semantic_layer = providers.Factory(
        lambda: import_module('semantics_layer.climate_semantic_layer').ClimateSemanticLayer,
        pattern_core=pattern_core,  # Add pattern tracking
        knowledge_coherence=knowledge_coherence,  # Add coherence observation
        timestamp_service=timestamp_service,
        event_manager=event_manager,
        ontology_data=providers.Dict(),
        climate_concept_validator=climate_concept_validator,
        mongo_client=mongodb_client,
        relationship_repository=relationship_repository,
        bidirectional_learner=bidirectional_learner,
        adaptive_learner=adaptive_learner,
        ethical_ai_checker=ethical_ai_checker,
        performance_monitor=performance_monitor
    )

    # Meaning Evolution Layer
    meaning_validator = providers.Factory(
        lambda: import_module('meaning_evolution.meaning_validator').MeaningValidator,
        pattern_core=pattern_core,  # Add pattern tracking
        knowledge_coherence=knowledge_coherence  # Add coherence observation
    )
    meaning_repository = providers.Singleton(
        lambda: import_module('meaning_evolution.meaning_repository').MeaningRepository
    )
    meaning_model = providers.Factory(
        lambda: import_module('meaning_evolution.meaning_model').MeaningModel,
        pattern_core=pattern_core,  # Add pattern tracking
        knowledge_coherence=knowledge_coherence  # Add coherence observation
    )
    meaning_manager = providers.Singleton(
        lambda: import_module('meaning_evolution.meaning_manager').MeaningManager
    )
    meaning_evolution = providers.Factory(
        lambda: import_module('meaning_evolution.meaning_evolution').MeaningEvolution,
        pattern_core=pattern_core,  # Add pattern tracking
        knowledge_coherence=knowledge_coherence  # Add coherence observation
    )

    # Utility Services
    change_propagator = providers.Lazy(
        lambda: import_module('utils.change_propagator').ChangePropagator
    )
    feedback_collector = providers.Lazy(
        lambda: import_module('utils.feedback_collector').FeedbackCollector
    )
    performance_monitor = providers.Singleton(
        lambda: import_module('utils.performance_monitor').PerformanceMonitor,
        sampling_rate=config.PERFORMANCE_SAMPLING_RATE,
        log_level=config.PERFORMANCE_LOG_LEVEL,
        alert_threshold=config.PERFORMANCE_ALERT_THRESHOLD,
        storage_backend=config.PERFORMANCE_STORAGE_BACKEND
    )

    # AI & Processing Components
    ethical_ai_checker = providers.Factory(
        lambda: import_module('ethical_framework.ethical_ai_checker').EthicalAIChecker,
        ethical_guidelines=config.ETHICAL_GUIDELINES,
        bias_threshold=config.ETHICAL_BIAS_THRESHOLD,
        fairness_metrics=config.ETHICAL_FAIRNESS_METRICS,
        update_frequency=config.ETHICAL_UPDATE_FREQUENCY
    )
    text_processor = providers.Factory(
        lambda: import_module('text_processing.text_processing').TextProcessor
    )
    entity_extractor = providers.Factory(
        lambda: import_module('nlp.entity_extractor').EntityExtractor
    )
    relationship_extractor = providers.Factory(
        lambda: import_module('nlp.relationship_extractor').RelationshipExtractor
    )

    # Knowledge Graph Components
    graph_manager = providers.Factory(
        lambda: import_module('knowledge_graph.graph_manager').GraphManager
    )
    graph_visualizer = providers.Factory(
        lambda: import_module('visualization.graph_visualizer').GraphVisualizer
    )
    analytics_dashboard = providers.Factory(
        lambda: import_module('visualization.analytics_dashboard').AnalyticsDashboard
    )

    # Machine Learning Components
    gnn_operator = providers.Factory(
        lambda: import_module('ml.gnn_operator').GNNOperator
    )
    graph_embedder = providers.Factory(
        lambda: import_module('ml.graph_embedder').GraphEmbedder
    )

    # RAG Components
    rag_processor = providers.Factory(
        lambda: import_module('rag.rag_processor').RAGProcessor,
        openai_api_key=config.OPENAI_API_KEY,
        perplexity_api_key=config.PERPLEXITY_API_KEY,
        langchain_api_key=config.LANGCHAIN_API_KEY,
        use_rag=config.USE_RAG
    )

    # Integration Components
    google_drive_utils = providers.Singleton(
        lambda: import_module('google_drive.google_drive_utils').GoogleDriveUtils,
        service_account_file=config.SERVICE_ACCOUNT_FILE,
        project_id=config.GOOGLE_PROJECT_ID,
        supported_formats=config.SUPPORTED_FORMATS
    )
    rest_api = providers.Singleton(
        lambda: import_module('api.rest_api').RESTApi
    )

    def __init__(self) -> None:
        """Initialize the AppContainer and set up test override capabilities."""
        super().__init__()
        self._test_overrides: Dict[str, Any] = {}

    def override_for_testing(self, **overrides: Any) -> None:
        """Override components for testing purposes."""
        for name, override in overrides.items():
            self._test_overrides[name] = getattr(self, name)
            setattr(self, name, override)

    def reset_overrides(self) -> None:
        """Reset all testing overrides to their original values."""
        for name, provider in self._test_overrides.items():
            setattr(self, name, provider)
        self._test_overrides.clear()

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update the container's configuration."""
        self.config.update(new_config)

def create_app_container() -> AppContainer:
    """
    Create and configure the application container.
    
    Returns:
        AppContainer: The configured application container.
    """
    
    # Load core configurations
    container.config.from_dict({
        'ADAPTIVE_CORE': {
            'VERSION': '1.0',
            'CACHE_SIZE': 1000
        },
        'DOMAIN': {
            'VERSION': '1.0',
            'DEFAULT_DOMAIN': 'climate'
        },
        'SEMANTIC': {
            'VERSION': '1.0',
            'BATCH_SIZE': 100
        },
        'MEANING': {
            'VERSION': '1.0',
            'EVOLUTION_THRESHOLD': 0.5
        }
    })
    
    container = AppContainer()
    config_loader = ConfigLoader()
    config_loader.validate()  # Validate configuration before creating the container
    container.config.from_dict(config_loader.config)
    return container

# Create a global instance of the AppContainer
app_container = create_app_container()

def wire_dependencies() -> None:
    """
    Wire up dependencies for the entire application.
    This function sets up the dependency injection wiring for all modules in the application.
    """
    from dependency_injector import wiring
    wiring.wire(
        modules=[__name__],
        packages=[
            'adaptive_core',
            'neo4jdb',
            'database',
            'text_processing',
            'semantics_layer',
            'meaning_evolution',
            'rag',
            'domain_ontology',
            'schemas',
            'events',
            'story_graph',
            'vector_indexing',
            'google_drive',
            'nlp',
            'knowledge_graph',
            'validation',
            'api',
            'collaboration',
            'utils',
            'ethical_framework',
            'spatio_temporal',
            'visualization',
            'cross_domain',
            'ml',
            'bidirectional_ontology',  # New package
            'adaptive_learning'  # New package
            'messaging'  # New package
            'events'  # New package
            'adaptive_core'  # New package
        ]
    )

def setup_logging() -> None:
    """
    Set up logging for the application.
    This function configures the logging system based on the LOG_LEVEL specified in the configuration.
    It also adds handlers for console and file logging, providing better observability.
    """
    log_level = app_container.config.get('LOG_LEVEL', 'INFO')
    log_file = app_container.config.get('LOG_FILE', 'app.log')
    
    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))

    # Create console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler for logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

def initialize_application() -> None:
    """
    Initialize the application components.
    This function sets up dependencies, configures logging, and initializes core components of the application.
    """
    wire_dependencies()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Initializing Habitat Adaptive POC application...")
    
    # Initialize core components
    neo4j_client = app_container._init_neo4j_client()
    mongodb_client = app_container._init_mongodb_client()
    app_container.domain_ontology_manager()
    app_container.semantic_layer_manager()
    
    logger.info("Habitat Adaptive POC application initialized successfully.")

if __name__ == "__main__":
    initialize_application()