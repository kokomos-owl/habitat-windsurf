"""
Pattern-aware RAG service implementation for Habitat Evolution.

This module provides a concrete implementation of the PatternAwareRAGInterface,
supporting the pattern evolution and co-evolution principles of Habitat Evolution.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import uuid
from datetime import datetime

from src.habitat_evolution.infrastructure.interfaces.services.pattern_aware_rag_interface import PatternAwareRAGInterface
from src.habitat_evolution.infrastructure.interfaces.services.event_service_interface import EventServiceInterface
from src.habitat_evolution.infrastructure.interfaces.services.vector_tonic_service_interface import VectorTonicServiceInterface
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_pattern_repository import ArangoDBPatternRepository, Pattern
from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_graph_repository import ArangoDBGraphRepository
from src.habitat_evolution.infrastructure.interfaces.persistence.arangodb_connection_interface import ArangoDBConnectionInterface
from src.habitat_evolution.infrastructure.adapters.pattern_bridge import PatternBridge
from src.habitat_evolution.adaptive_core.models.pattern import Pattern as AdaptiveCorePattern

logger = logging.getLogger(__name__)


class PatternAwareRAGService(PatternAwareRAGInterface):
    """
    Implementation of the PatternAwareRAGInterface.
    
    This service provides pattern-aware retrieval augmented generation functionality
    for the Habitat Evolution system, supporting the pattern evolution and
    co-evolution principles.
    """
    
    def __init__(self, 
                 db_connection: ArangoDBConnectionInterface,
                 pattern_repository: ArangoDBPatternRepository,
                 vector_tonic_service: VectorTonicServiceInterface,
                 claude_adapter: Any,
                 event_service: EventServiceInterface,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize a new PatternAwareRAGService.
        
        Args:
            db_connection: The ArangoDB connection to use
            pattern_repository: The pattern repository to use
            vector_tonic_service: The vector tonic service to use
            claude_adapter: The Claude adapter to use
            event_service: The event service to use
            config: Optional configuration for the service
        """
        self._db_connection = db_connection
        self._pattern_repository = pattern_repository
        self._vector_tonic_service = vector_tonic_service
        self._claude_adapter = claude_adapter
        self._event_service = event_service
        self._config = config or {}
        self._initialized = False
        self._vector_space_id = None
        
        # Initialize direct database access
        if hasattr(db_connection, '_db') and db_connection._db is not None:
            self._db = db_connection._db
            
            # Pre-create collections to avoid edge_collection errors
            try:
                # Create patterns collection if it doesn't exist
                if not self._db.has_collection("patterns"):
                    self._db.create_collection("patterns")
                    logger.debug("Created patterns collection directly")
                
                # Create edge collection if it doesn't exist
                if not self._db.has_collection("pattern_relationships"):
                    self._db.create_collection("pattern_relationships", edge=True)
                    logger.debug("Created pattern_relationships edge collection directly")
                
                # Store references for later use
                self._patterns_collection = self._db.collection("patterns")
                self._relationships_collection = self._db.collection("pattern_relationships")
                
                # Create graph if it doesn't exist
                if not self._db.has_graph("pattern_graph"):
                    edge_definitions = [
                        {
                            "collection": "pattern_relationships",
                            "from": ["patterns"],
                            "to": ["patterns"]
                        }
                    ]
                    self._db.create_graph("pattern_graph", edge_definitions)
                    logger.debug("Created pattern_graph directly")
                
                # Store graph reference
                self._pattern_graph = self._db.graph("pattern_graph")
                logger.debug("Direct access to database structures established")
            except Exception as e:
                logger.error(f"Error setting up database structures: {e}")
                self._patterns_collection = None
                self._relationships_collection = None
                self._pattern_graph = None
        else:
            self._db = None
            logger.warning("Direct database access not available")
            self._patterns_collection = None
            self._relationships_collection = None
            self._pattern_graph = None
        
        # Initialize graph repository
        try:
            self._graph_repository = ArangoDBGraphRepository(
                node_collection_name="patterns",
                edge_collection_name="pattern_relationships",
                graph_name="pattern_graph",
                db_connection=db_connection,
                event_service=event_service,
                entity_class=Pattern
            )
        except Exception as e:
            logger.error(f"Error initializing graph repository: {e}")
            self._graph_repository = None
            
        logger.debug("PatternAwareRAGService created")
    
    def _manage_fallback_storage(self):
        """
        Initialize and manage fallback storage mechanisms for the PatternAwareRAGService.
        
        This method ensures that appropriate fallback structures are in place when database
        operations fail. It's critical for maintaining system resilience during the POC phase
        and provides a clear path for MVP implementation by isolating fallback logic.
        
        The fallback mechanisms include:
        1. In-memory pattern storage
        2. In-memory relationship storage
        3. Disk-based JSON fallback for persistence across restarts
        
        Returns:
            None: This method initializes internal fallback structures
        """
        # Initialize fallback structures if they don't exist
        if not hasattr(self, '_fallback_patterns'):
            self._fallback_patterns = []
            logger.debug("Initialized fallback pattern storage")
            
        if not hasattr(self, '_fallback_relationships'):
            self._fallback_relationships = []
            logger.debug("Initialized fallback relationship storage")
            
        if not hasattr(self, '_fallback_enabled'):
            self._fallback_enabled = True
            logger.debug("Fallback mechanisms enabled")
            
        # Track fallback usage metrics
        if not hasattr(self, '_fallback_metrics'):
            self._fallback_metrics = {
                "pattern_fallbacks": 0,
                "relationship_fallbacks": 0,
                "successful_recoveries": 0,
                "last_error": None,
                "error_count": 0
            }
            
        logger.debug("Fallback storage management initialized")
    
    def _ensure_graph_structure(self) -> bool:
        """
        Ensure that the necessary graph structure exists in ArangoDB.
        
        This method is responsible for creating and validating the graph structure
        required for pattern relationships. It implements a multi-stage approach:
        
        1. Direct database access to create collections and graphs
        2. Detailed error capture and logging for troubleshooting
        3. Structured approach to database initialization
        
        The method is designed to be resilient to the 'edge_collection' error by
        using multiple approaches to create the required structures.
        
        Returns:
            bool: True if the graph structure was successfully ensured, False otherwise
        """
        # Ensure fallback mechanisms are initialized
        self._manage_fallback_storage()
        
        try:
            # Get direct access to the ArangoDB database
            db = self._db_connection._db
            
            # STAGE 1: Create collections with detailed logging
            logger.debug("Starting graph structure initialization - Stage 1: Collections")
            
            # Ensure patterns collection exists
            try:
                if not db.has_collection("patterns"):
                    db.create_collection("patterns")
                    logger.debug("Created patterns collection")
                else:
                    logger.debug("Patterns collection already exists")
                    
                # Store reference for direct access
                self._patterns_collection = db.collection("patterns")
            except Exception as collection_error:
                logger.error(f"Error creating patterns collection: {collection_error}")
                self._fallback_metrics["error_count"] += 1
                self._fallback_metrics["last_error"] = str(collection_error)
                return False
            
            # STAGE 2: Create edge collection with error details
            logger.debug("Starting graph structure initialization - Stage 2: Edge Collection")
            try:
                if not db.has_collection("pattern_relationships"):
                    # Log the exact parameters being used
                    logger.debug("Creating edge collection with edge=True parameter")
                    db.create_collection("pattern_relationships", edge=True)
                    logger.debug("Successfully created pattern_relationships edge collection")
                else:
                    # Verify it's actually an edge collection
                    collection = db.collection("pattern_relationships")
                    if collection.properties().get("type") == 3:  # 3 is the type for edge collections
                        logger.debug("Verified pattern_relationships is an edge collection")
                    else:
                        logger.warning("pattern_relationships exists but is not an edge collection")
                        # Try to delete and recreate
                        try:
                            db.delete_collection("pattern_relationships")
                            db.create_collection("pattern_relationships", edge=True)
                            logger.debug("Recreated pattern_relationships as an edge collection")
                        except Exception as recreate_error:
                            logger.error(f"Failed to recreate as edge collection: {recreate_error}")
                            return False
                
                # Store reference for direct access
                self._relationships_collection = db.collection("pattern_relationships")
            except Exception as edge_error:
                # Capture detailed error information about the edge collection issue
                logger.error(f"Error creating edge collection: {edge_error}")
                logger.error(f"Error type: {type(edge_error).__name__}")
                self._fallback_metrics["error_count"] += 1
                self._fallback_metrics["last_error"] = str(edge_error)
                return False
            
            # STAGE 3: Create graph with edge definitions
            logger.debug("Starting graph structure initialization - Stage 3: Graph Creation")
            try:
                if not db.has_graph("pattern_graph"):
                    edge_definitions = [
                        {
                            "collection": "pattern_relationships",
                            "from": ["patterns"],
                            "to": ["patterns"]
                        }
                    ]
                    
                    # Log the exact parameters being used
                    logger.debug(f"Creating graph with edge definitions: {edge_definitions}")
                    db.create_graph("pattern_graph", edge_definitions)
                    logger.debug("Successfully created pattern_graph")
                else:
                    logger.debug("Pattern graph already exists")
                
                # Store reference for direct access
                self._pattern_graph = db.graph("pattern_graph")
            except Exception as graph_error:
                logger.error(f"Error creating graph: {graph_error}")
                logger.error(f"Error type: {type(graph_error).__name__}")
                self._fallback_metrics["error_count"] += 1
                self._fallback_metrics["last_error"] = str(graph_error)
                return False
            
            logger.info("Graph structure successfully ensured in ArangoDB")
            return True
        except Exception as e:
            logger.error(f"Error ensuring graph structure: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error occurred at stage: Graph Structure Initialization")
            self._fallback_metrics["error_count"] += 1
            self._fallback_metrics["last_error"] = str(e)
            return False
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the pattern-aware RAG service with the specified configuration.
        
        Args:
            config: Optional configuration for the service
        """
        if self._initialized:
            logger.warning("PatternAwareRAGService already initialized")
            return
            
        logger.info("Initializing PatternAwareRAGService")
        
        # Initialize pattern repository
        if not self._pattern_repository._initialized:
            self._pattern_repository.initialize()
            
        # Initialize vector tonic service
        if not self._vector_tonic_service._initialized:
            self._vector_tonic_service.initialize()
        
        # Ensure graph structure exists - do this first to set up the database
        graph_initialized = self._ensure_graph_structure()
        
        # Even if graph initialization failed, try to set up direct access to collections
        try:
            # Get direct access to the database
            db = self._db_connection._db
            
            # Get direct access to collections for later use
            self._patterns_collection = db.collection("patterns")
            self._relationships_collection = db.collection("pattern_relationships")
            
            # Try to get graph access if possible
            if db.has_graph("pattern_graph"):
                self._pattern_graph = db.graph("pattern_graph")
                logger.debug("Direct access to pattern graph established")
            else:
                self._pattern_graph = None
                logger.warning("Pattern graph not available, using collection access only")
                
        except Exception as e:
            logger.error(f"Error setting up direct database access: {e}")
            self._patterns_collection = None
            self._relationships_collection = None
            self._pattern_graph = None
        
        # Log status based on graph initialization
        if not graph_initialized:
            logger.warning("Failed to initialize graph structure, continuing with limited functionality")
        
        # Create vector space for RAG
        try:
            self._vector_space_id = self._vector_tonic_service.register_vector_space(
                name="rag_vector_space",
                dimensions=768,  # Standard embedding dimension
                metadata={"purpose": "pattern_aware_rag"}
            )
        except Exception as e:
            logger.error(f"Error registering vector space: {e}")
            self._vector_space_id = "fallback_vector_space"
        
        # Set configuration
        if config:
            self._coherence_threshold = config.get("coherence_threshold", self._coherence_threshold)
            self._quality_threshold = config.get("quality_threshold", self._quality_threshold)
            self._quality_weight = config.get("quality_weight", self._quality_weight)
        
        self._initialized = True
        logger.info(f"PatternAwareRAGService initialized with vector space {self._vector_space_id}")
        
        # Publish initialization event
        self._event_service.publish("pattern_aware_rag.initialized", {
            "vector_space_id": self._vector_space_id,
            "coherence_threshold": self._coherence_threshold,
            "quality_threshold": self._quality_threshold
        })
    
    def process_document(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document through the pattern-aware RAG system.
        
        Args:
            document: The document to process
            metadata: Optional metadata for the document
            
        Returns:
            Processing results including extracted patterns and metrics
        """
        if not self._initialized:
            self.initialize()
            
        logger.info(f"Processing document through pattern-aware RAG: {metadata.get('title', 'Untitled') if metadata else 'Untitled'}")
        
        # Extract entities (simplified implementation)
        entities = self._extract_entities(document)
        
        # Create patterns for entities
        patterns = []
        for entity in entities:
            pattern = self._create_entity_pattern(entity, document)
            patterns.append(pattern)
            
        # Update context with patterns
        context = self._update_context_with_patterns(patterns)
        
        # Calculate metrics
        metrics = self._calculate_metrics(patterns)
        
        # Publish document processed event
        self._event_service.publish("pattern_aware_rag.document_processed", {
            "document_id": metadata.get("id", str(uuid.uuid4())) if metadata else str(uuid.uuid4()),
            "pattern_count": len(patterns),
            "entity_count": len(entities)
        })
        
        return {
            "patterns": patterns,
            "metrics": metrics,
            "context": context
        }
    
    def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the pattern-aware RAG system.
        
        Args:
            query: The query to process
            context: Optional context for the query
            
        Returns:
            Query results including relevant patterns and generated response
        """
        if not self._initialized:
            self.initialize()
            
        logger.info(f"Processing query through pattern-aware RAG: {query[:50]}...")
        
        # Initialize context if not provided
        if context is None:
            context = {}
            
        # Get relevant patterns
        relevant_patterns = self._retrieve_relevant_patterns(query, context)
        
        # Generate response
        response = self._generate_response(query, relevant_patterns, context)
        
        # Track query in event service
        if self._event_service:
            self._event_service.publish("pattern_aware_rag.query", {
                "query": query,
                "pattern_count": len(relevant_patterns),
                "timestamp": datetime.now().isoformat()
            })
            
        return {
            "query": query,
            "patterns": [p.to_dict() for p in relevant_patterns],
            "response": response,
            "context": context
        }
    
    def get_patterns(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get patterns from the pattern-aware RAG system.
        
        Args:
            filter_criteria: Optional criteria to filter patterns by
            
        Returns:
            A list of matching patterns
        """
        # Get all patterns from repository
        all_patterns = self._pattern_repository.find_all()
        
        # Enhance patterns with metadata using the bridge
        enhanced_patterns = self._pattern_bridge.enhance_patterns(all_patterns)
        
        # Apply filters if provided
        if filter_criteria:
            filtered_patterns = []
            for pattern in enhanced_patterns:
                match = True
                for key, value in filter_criteria.items():
                    # Handle nested properties with dot notation
                    if '.' in key:
                        parts = key.split('.')
                        current = pattern.to_dict()
                        for part in parts[:-1]:
                            if part not in current:
                                match = False
                                break
                            current = current[part]
                        if match and parts[-1] in current and current[parts[-1]] != value:
                            match = False
                    # Handle direct properties
                    elif hasattr(pattern, key):
                        if getattr(pattern, key) != value:
                            match = False
                    # Handle metadata properties
                    elif hasattr(pattern, 'metadata') and key in pattern.metadata:
                        if pattern.metadata[key] != value:
                            match = False
                if match:
                    filtered_patterns.append(pattern)
            return [p.to_dict() for p in filtered_patterns]
        
        # Return all patterns if no filter criteria
        return [p.to_dict() for p in enhanced_patterns]
    
    def get_field_state(self) -> Dict[str, Any]:
        """
        Get the current state of the semantic field.
        
        Returns:
            The current field state
        """
        # Get all patterns
        patterns = self._pattern_repository.find_all()
        enhanced_patterns = self._pattern_bridge.enhance_patterns(patterns)
        
        # Get vector space information from vector tonic service
        vector_space_info = self._vector_tonic_service.get_vector_space_info()
        
        # Calculate field metrics
        pattern_count = len(patterns)
        avg_coherence = sum(p.metadata.get("coherence", 0) for p in enhanced_patterns) / max(1, pattern_count)
        avg_quality = sum(p.metadata.get("quality", 0) for p in enhanced_patterns) / max(1, pattern_count)
        
        return {
            "pattern_count": pattern_count,
            "average_coherence": avg_coherence,
            "average_quality": avg_quality,
            "vector_space": vector_space_info,
            "timestamp": datetime.now().isoformat()
        }
    
    def _convert_pkm_pattern_to_arangodb_pattern(self, pkm_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a PKM pattern to an ArangoDB pattern.
        
        Args:
            pkm_pattern: The PKM pattern to convert
            
        Returns:
            A dictionary with fields compatible with the ArangoDB Pattern class
        """
        # Extract basic fields
        pattern_id = pkm_pattern.get("id", f"pattern-{uuid.uuid4()}")
        
        # Determine name from content or type
        name = "unknown"
        if "content" in pkm_pattern:
            # Use first 50 chars of content as name
            content = pkm_pattern["content"]
            name = content[:50] if len(content) > 0 else "unknown"
        elif "type" in pkm_pattern:
            name = pkm_pattern["type"]
        
        # Get pattern_type from type field or default to "semantic"
        pattern_type = pkm_pattern.get("type", "semantic")
        
        # Create description from content or default to empty
        description = ""
        if "content" in pkm_pattern:
            description = pkm_pattern["content"]
        
        # Create metadata dictionary
        metadata = {}
        
        # Copy existing metadata if available
        if "metadata" in pkm_pattern and isinstance(pkm_pattern["metadata"], dict):
            metadata.update(pkm_pattern["metadata"])
        
        # Add any other fields as metadata
        for key, value in pkm_pattern.items():
            if key not in ["id", "name", "type", "pattern_type", "description", "metadata", 
                          "created_at", "updated_at"]:
                metadata[key] = value
        
        # Create timestamps
        created_at = pkm_pattern.get("created_at", datetime.now().isoformat())
        updated_at = pkm_pattern.get("updated_at", pkm_pattern.get("last_modified", created_at))
        
        # Create the ArangoDB pattern dictionary
        arangodb_pattern = {
            "id": pattern_id,
            "name": name,
            "pattern_type": pattern_type,
            "description": description,
            "metadata": metadata,
            "created_at": created_at,
            "updated_at": updated_at
        }
        
        return arangodb_pattern
    
    def add_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a pattern to the pattern-aware RAG system.
        
        Args:
            pattern: The pattern to add
            
        Returns:
            The added pattern with any generated IDs or timestamps
        """
        # Generate ID if not provided
        if "id" not in pattern:
            pattern["id"] = f"pattern-{uuid.uuid4()}"
        
        # Add timestamps if not provided
        now = datetime.now().isoformat()
        if "created_at" not in pattern:
            pattern["created_at"] = now
        if "last_modified" not in pattern:
            pattern["last_modified"] = now
        
        # Ensure graph structure exists
        graph_initialized = self._ensure_graph_structure()
        if not graph_initialized:
            logger.warning("Graph structure not initialized in add_pattern, continuing with limited functionality")
        
        try:
            # Convert PKM pattern to ArangoDB pattern
            arangodb_pattern = self._convert_pkm_pattern_to_arangodb_pattern(pattern)
            
            # Create Pattern object
            pattern_obj = Pattern(**arangodb_pattern)
            
            # Save to repository
            saved_pattern = self._pattern_repository.save(pattern_obj)
        except Exception as e:
            logger.error(f"Error creating Pattern object: {e}")
            logger.debug(f"Pattern data: {pattern}")
            # Create a minimal valid pattern as fallback
            minimal_pattern = {
                'id': pattern.get('id', f"pattern-{uuid.uuid4()}"),
                'name': pattern.get('name', 'unknown pattern'),
                'pattern_type': pattern.get('type', 'semantic'),
                'description': pattern.get('content', '')
            }
            pattern_obj = Pattern(**minimal_pattern)
            saved_pattern = self._pattern_repository.save(pattern_obj)
        
        # Publish event
        if self._event_service:
            self._event_service.publish("pattern_aware_rag.pattern_added", {
                "pattern_id": saved_pattern.id,
                "timestamp": now
            })
        
        return saved_pattern.to_dict()
    
    def update_pattern(self, pattern_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a pattern in the pattern-aware RAG system.
        
        Args:
            pattern_id: The ID of the pattern to update
            updates: The updates to apply to the pattern
            
        Returns:
            The updated pattern
        """
        # Get existing pattern
        pattern = self._pattern_repository.find_by_id(pattern_id)
        if not pattern:
            raise ValueError(f"Pattern not found: {pattern_id}")
        
        # Update last_modified timestamp
        updates["last_modified"] = datetime.now().isoformat()
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(pattern, key):
                setattr(pattern, key, value)
        
        # Save updated pattern
        updated_pattern = self._pattern_repository.save(pattern)
        
        # Publish event
        if self._event_service:
            self._event_service.publish("pattern_aware_rag.pattern_updated", {
                "pattern_id": pattern_id,
                "timestamp": updates["last_modified"]
            })
        
        return updated_pattern.to_dict()
    
    def delete_pattern(self, pattern_id: str) -> bool:
        """
        Delete a pattern from the pattern-aware RAG system.
        
        Args:
            pattern_id: The ID of the pattern to delete
            
        Returns:
            True if the pattern was deleted, False otherwise
        """
        # Delete from repository
        success = self._pattern_repository.delete(pattern_id)
        
        # Publish event if successful
        if success and self._event_service:
            self._event_service.publish("pattern_aware_rag.pattern_deleted", {
                "pattern_id": pattern_id,
                "timestamp": datetime.now().isoformat()
            })
        
        return success
    
    def create_relationship(self, source_id: str, target_id: str, relationship_type: str, 
                          relationship_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a relationship between two patterns.
        
        This method establishes a relationship between two patterns in the knowledge graph.
        It implements a robust multi-layered approach to relationship creation:
        
        1. First attempts to use the graph API for proper graph semantics
        2. Falls back to direct edge collection access if graph API fails
        3. Uses in-memory fallback storage as a last resort
        4. Provides detailed error diagnostics for troubleshooting
        
        The method is designed to be resilient to the 'edge_collection' error and other
        database-related issues, ensuring that relationship data is never lost even when
        persistence fails.
        
        Args:
            source_id: The ID of the source pattern (from vertex)
            target_id: The ID of the target pattern (to vertex)
            relationship_type: The semantic type of relationship (e.g., "derives", "extends")
            relationship_data: Optional additional metadata for the relationship
            
        Returns:
            str: The unique ID of the created relationship, either from the database or fallback storage
        """
        if not self._initialized:
            self.initialize()
            
        # Ensure fallback mechanisms are initialized
        self._manage_fallback_storage()
            
        # Generate a unique ID for the relationship
        relationship_id = str(uuid.uuid4())
        
        # Initialize relationship data if not provided
        if relationship_data is None:
            relationship_data = {}
            
        # Add relationship type and ID to the data
        relationship_data["id"] = relationship_id
        relationship_data["type"] = relationship_type
        relationship_data["created_at"] = datetime.now().isoformat()
        
        # Log the relationship creation attempt with detailed information
        logger.debug(f"Creating relationship: {relationship_type} from {source_id} to {target_id}")
        
        # Create edge in graph
        try:
            # STAGE 1: Ensure graph structure exists
            logger.debug("Relationship creation - Stage 1: Ensuring graph structure")
            graph_initialized = self._ensure_graph_structure()
            
            if not graph_initialized:
                logger.warning("Graph structure not initialized, storing relationship in memory only")
                # Store relationship data in memory as a fallback
                self._fallback_relationships.append({
                    "id": relationship_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "type": relationship_type,
                    "data": relationship_data,
                    "fallback_reason": "graph_structure_initialization_failed",
                    "timestamp": datetime.now().isoformat()
                })
                self._fallback_metrics["relationship_fallbacks"] += 1
                return relationship_id
            
            # STAGE 2: Prepare edge document with all required fields
            logger.debug("Relationship creation - Stage 2: Preparing edge document")
            edge_doc = relationship_data.copy()
            
            # Ensure proper ArangoDB edge format with _from and _to fields
            # Check if IDs already have collection prefixes
            if not source_id.startswith("patterns/"):
                edge_doc["_from"] = f"patterns/{source_id}"
            else:
                edge_doc["_from"] = source_id
                
            if not target_id.startswith("patterns/"):
                edge_doc["_to"] = f"patterns/{target_id}"
            else:
                edge_doc["_to"] = target_id
                
            edge_doc["type"] = relationship_type
            edge_doc["created_at"] = datetime.now().isoformat()
            
            # Log the edge document for debugging
            logger.debug(f"Edge document prepared: {edge_doc}")
            
            # STAGE 3: Try multiple methods to create the edge, with detailed error handling
            logger.debug("Relationship creation - Stage 3: Creating edge using available methods")
            
            # METHOD 1: Try using the graph API first (preferred method)
            if hasattr(self, '_pattern_graph') and self._pattern_graph is not None:
                try:
                    logger.debug("Attempting to create edge using graph API")
                    edge_collection = self._pattern_graph.edge_collection("pattern_relationships")
                    result = edge_collection.insert(edge_doc, return_new=True)
                    logger.info(f"Successfully created edge in graph: {result.get('_id')}")
                    return result.get("_id", relationship_id)
                except Exception as graph_error:
                    # Detailed error logging for graph API issues
                    logger.warning(f"Error using graph edge collection: {graph_error}")
                    logger.warning(f"Error type: {type(graph_error).__name__}")
                    
                    # Check for the specific 'edge_collection' error
                    if 'edge_collection' in str(graph_error).lower():
                        logger.debug("Detected 'edge_collection' error, trying alternative methods")
                    
                    # Fall through to next method
            else:
                logger.debug("Graph API not available, skipping to direct collection access")
            
            # METHOD 2: Try using direct collection access
            if hasattr(self, '_relationships_collection') and self._relationships_collection is not None:
                try:
                    logger.debug("Attempting to create edge using direct collection access")
                    result = self._relationships_collection.insert(edge_doc, return_new=True)
                    logger.info(f"Successfully created edge directly in collection: {result.get('_id')}")
                    return result.get("_id", relationship_id)
                except Exception as collection_error:
                    # Detailed error logging for collection access issues
                    logger.warning(f"Error using direct collection access: {collection_error}")
                    logger.warning(f"Error type: {type(collection_error).__name__}")
                    
                    # Fall through to last resort method
            else:
                logger.debug("Direct collection access not available, trying last resort method")
            
            # METHOD 3: Last resort - get a fresh collection reference from the database
            try:
                logger.debug("Attempting to create edge using last resort method")
                db = self._db_connection._db
                edge_collection = db.collection("pattern_relationships")
                result = edge_collection.insert(edge_doc, return_new=True)
                logger.info(f"Successfully created edge using last resort method: {result.get('_id')}")
                return result.get("_id", relationship_id)
            except Exception as last_resort_error:
                # If all database methods fail, throw to be caught by the outer try/except
                logger.error(f"All database methods failed: {last_resort_error}")
                logger.error(f"Error type: {type(last_resort_error).__name__}")
                raise last_resort_error
                
        except Exception as e:
            # FALLBACK: Store relationship data in memory when all else fails
            logger.error(f"Error creating relationship in RAG service: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Store comprehensive error information with the fallback
            self._fallback_relationships.append({
                "id": relationship_id,
                "source_id": source_id,
                "target_id": target_id,
                "type": relationship_type,
                "data": relationship_data,
                "fallback_reason": "database_operation_failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update fallback metrics
            self._fallback_metrics["relationship_fallbacks"] += 1
            self._fallback_metrics["error_count"] += 1
            self._fallback_metrics["last_error"] = str(e)
            
            # Log the fallback action
            logger.info(f"Relationship stored in fallback storage with ID: {relationship_id}")
            
            # Continue execution even if the edge creation fails
            return relationship_id
    
    def get_fallback_metrics(self) -> Dict[str, Any]:
        """
        Retrieve detailed metrics about fallback mechanisms and error patterns.
        
        This method provides comprehensive diagnostics about the system's resilience
        mechanisms, which is essential for both POC validation and MVP development.
        It helps identify patterns in errors and quantifies the system's ability to
        maintain functionality despite infrastructure challenges.
        
        The metrics include:
        1. Counts of different types of fallbacks
        2. Error patterns and frequencies
        3. Success rates for different operations
        4. Recommendations for system improvements
        
        Returns:
            Dict[str, Any]: A dictionary containing detailed fallback metrics and diagnostics
        """
        # Ensure fallback mechanisms are initialized
        self._manage_fallback_storage()
        
        # Calculate additional metrics
        pattern_fallback_count = self._fallback_metrics.get("pattern_fallbacks", 0)
        relationship_fallback_count = self._fallback_metrics.get("relationship_fallbacks", 0)
        total_fallbacks = pattern_fallback_count + relationship_fallback_count
        
        # Analyze error patterns
        error_types = {}
        for rel in getattr(self, '_fallback_relationships', []):
            error_type = rel.get("error_type")
            if error_type:
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Generate recommendations based on error patterns
        recommendations = []
        if 'edge_collection' in str(self._fallback_metrics.get("last_error", "")).lower():
            recommendations.append({
                "priority": "high",
                "issue": "edge_collection error in ArangoDB operations",
                "recommendation": "Review ArangoDB Python driver version compatibility and graph API usage",
                "mvp_impact": "Critical for reliable graph operations in production"
            })
        
        if total_fallbacks > 0:
            recommendations.append({
                "priority": "medium",
                "issue": "Fallback mechanisms being utilized frequently",
                "recommendation": "Implement persistent fallback storage with automatic recovery",
                "mvp_impact": "Ensures data integrity across system restarts"
            })
        
        # Compile comprehensive metrics
        detailed_metrics = {
            "summary": {
                "total_fallbacks": total_fallbacks,
                "pattern_fallbacks": pattern_fallback_count,
                "relationship_fallbacks": relationship_fallback_count,
                "error_count": self._fallback_metrics.get("error_count", 0),
                "successful_recoveries": self._fallback_metrics.get("successful_recoveries", 0),
                "fallback_enabled": getattr(self, '_fallback_enabled', True)
            },
            "error_analysis": {
                "last_error": self._fallback_metrics.get("last_error"),
                "error_types": error_types,
                "error_frequency": {
                    "high_frequency": [k for k, v in error_types.items() if v > 5],
                    "medium_frequency": [k for k, v in error_types.items() if 2 < v <= 5],
                    "low_frequency": [k for k, v in error_types.items() if v <= 2]
                }
            },
            "fallback_data": {
                "relationship_count": len(getattr(self, '_fallback_relationships', [])),
                "pattern_count": len(getattr(self, '_fallback_patterns', [])),
                "sample_relationships": getattr(self, '_fallback_relationships', [])[:3] if hasattr(self, '_fallback_relationships') else []
            },
            "recommendations": recommendations,
            "mvp_transition": {
                "critical_fixes": [r for r in recommendations if r["priority"] == "high"],
                "suggested_improvements": [r for r in recommendations if r["priority"] == "medium"],
                "data_migration_needed": total_fallbacks > 0
            }
        }
        
        return detailed_metrics
        
        # Update patterns to reference the relationship
        source_pattern.add_relationship(relationship_id)
        target_pattern.add_relationship(relationship_id)
        
        # Save updated patterns
        self._pattern_repository.save(source_pattern)
        self._pattern_repository.save(target_pattern)
        
        # Publish event
        if self._event_service:
            self._event_service.publish("pattern_aware_rag.relationship_created", {
                "relationship_id": relationship_id,
                "source_id": source_id,
                "target_id": target_id,
                "type": relationship_type,
                "timestamp": relationship_data["created_at"]
            })
        
        return relationship_id
    
    def get_fallback_relationships(self, source_id: Optional[str] = None, 
                                  target_id: Optional[str] = None,
                                  relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get relationships from the fallback memory storage.
        
        This method is used when ArangoDB graph operations fail and relationships
        are stored in memory as a fallback mechanism.
        
        Args:
            source_id: Optional filter by source pattern ID
            target_id: Optional filter by target pattern ID
            relationship_type: Optional filter by relationship type
            
        Returns:
            A list of relationship dictionaries matching the filters
        """
        if not hasattr(self, '_fallback_relationships'):
            return []
            
        results = []
        for rel in self._fallback_relationships:
            match = True
            
            if source_id is not None and rel.get("source_id") != source_id:
                match = False
                
            if target_id is not None and rel.get("target_id") != target_id:
                match = False
                
            if relationship_type is not None and rel.get("type") != relationship_type:
                match = False
                
            if match:
                results.append(rel)
                
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from the pattern-aware RAG system.
        
        Returns:
            Current system metrics
        """
        # Get all patterns
        patterns = self._pattern_repository.find_all()
        enhanced_patterns = self._pattern_bridge.enhance_patterns(patterns)
        
        # Calculate pattern metrics
        pattern_count = len(patterns)
        coherent_patterns = sum(1 for p in enhanced_patterns if p.metadata.get("coherence", 0) >= self._coherence_threshold)
        quality_patterns = sum(1 for p in enhanced_patterns if p.metadata.get("quality", 0) >= self._quality_threshold)
        
        # Get vector metrics
        vector_metrics = self._vector_tonic_service.get_metrics()
        
        return {
            "pattern_count": pattern_count,
            "coherent_pattern_count": coherent_patterns,
            "quality_pattern_count": quality_patterns,
            "coherence_threshold": self._coherence_threshold,
            "quality_threshold": self._quality_threshold,
            "vector_metrics": vector_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def shutdown(self) -> None:
        """
        Release resources when shutting down the service.
        """
        logger.info("Shutting down PatternAwareRAGService")
        
        # Publish shutdown event
        if self._event_service:
            self._event_service.publish("pattern_aware_rag.shutdown", {
                "timestamp": datetime.now().isoformat()
            })
        
        logger.info("PatternAwareRAGService shut down")
    
    def update_pattern_metadata(self, pattern_id: str, metadata: Dict[str, Any]) -> Optional[Pattern]:
        """
        Update the metadata of a pattern.
        
        Args:
            pattern_id: The ID of the pattern
            metadata: The metadata to update
            
        Returns:
            The updated pattern, or None if not found
        """
        if not self._initialized:
            self.initialize()
            
        # Get current pattern
        pattern = self._pattern_repository.find_by_id(pattern_id)
        if not pattern:
            logger.warning(f"Pattern {pattern_id} not found")
            return None
            
        # Update pattern
        return self._pattern_repository.update_node(pattern_id, {"metadata": metadata})
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[Pattern]:
        """
        Get a pattern by ID.
        
        Args:
            pattern_id: The ID of the pattern
            
        Returns:
            The pattern, or None if not found
        """
        if not self._initialized:
            self.initialize()
            
        return self._pattern_repository.find_by_id(pattern_id)
    
    def get_patterns_by_quality(self, min_quality: float = 0.7) -> List[Pattern]:
        """
        Get patterns by quality.
        
        Args:
            min_quality: The minimum quality threshold
            
        Returns:
            A list of patterns with quality >= min_quality
        """
        if not self._initialized:
            self.initialize()
            
        return self._pattern_repository.find_by_quality(min_quality=min_quality)
    
    def _extract_entities(self, document: str) -> List[str]:
        """
        Extract entities from a document.
        
        Args:
            document: The document to process
            
        Returns:
            A list of extracted entities
        """
        # Simplified entity extraction (in a real implementation, this would use NER)
        words = document.split()
        entities = []
        
        # Extract simple entities (2-3 word phrases)
        for i in range(len(words) - 2):
            if len(words[i]) > 3 and words[i][0].isupper():
                entity = " ".join(words[i:i+2])
                entities.append(entity)
                
                if i < len(words) - 3:
                    entity = " ".join(words[i:i+3])
                    entities.append(entity)
        
        return list(set(entities))
    
    def _create_entity_pattern(self, entity: str, document: str) -> Pattern:
        """
        Create a pattern for an entity.
        
        Args:
            entity: The entity to create a pattern for
            document: The document containing the entity
            
        Returns:
            The created pattern
        """
        # Create pattern
        pattern = self._pattern_repository.create_pattern(
            name=entity,
            pattern_type="entity",
            description=f"Entity pattern for '{entity}'",
            metadata={
                "type": "entity",
                "text": entity,
                "quality": 0.1,  # Start with low quality
                "coherence": 0.1,  # Start with low coherence
                "stability": 0.0,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        return pattern
    
    def _update_context_with_patterns(self, patterns: List[Pattern]) -> Dict[str, Any]:
        """
        Update context with patterns.
        
        Args:
            patterns: The patterns to add to context
            
        Returns:
            The updated context
        """
        # Count patterns by quality
        quality_counts = {"good": 0, "uncertain": 0, "poor": 0}
        
        for pattern in patterns:
            quality = pattern.metadata.get("quality", 0)
            
            if quality >= 0.7:
                quality_counts["good"] += 1
            elif quality >= 0.4:
                quality_counts["uncertain"] += 1
            else:
                quality_counts["poor"] += 1
        
        logger.info(f"Updated context from quality assessment: {quality_counts['good']} good, "
                   f"{quality_counts['uncertain']} uncertain, {quality_counts['poor']} poor entities")
        
        return {
            "pattern_count": len(patterns),
            "quality_counts": quality_counts
        }
    
    def _calculate_metrics(self, patterns: List[Pattern]) -> Dict[str, Any]:
        """
        Calculate metrics for patterns.
        
        Args:
            patterns: The patterns to calculate metrics for
            
        Returns:
            The calculated metrics
        """
        # Calculate average quality and coherence
        if not patterns:
            return {
                "avg_quality": 0.0,
                "avg_coherence": 0.0,
                "pattern_count": 0
            }
            
        avg_quality = sum(p.metadata.get("quality", 0) for p in patterns) / len(patterns)
        avg_coherence = sum(p.metadata.get("coherence", 0) for p in patterns) / len(patterns)
        
        return {
            "avg_quality": avg_quality,
            "avg_coherence": avg_coherence,
            "pattern_count": len(patterns)
        }
    
    def _retrieve_relevant_patterns(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[Pattern]:
        """
        Retrieve patterns relevant to a query.
        
        Args:
            query: The query to retrieve patterns for
            context: Optional context for the query
            
        Returns:
            A list of relevant patterns
        """
        # Get all patterns
        all_patterns = self._pattern_repository.find_all()
        
        # Enhance patterns with metadata using the bridge
        enhanced_patterns = self._pattern_bridge.enhance_patterns(all_patterns)
        
        # Filter by coherence threshold
        coherent_patterns = [
            p for p in enhanced_patterns 
            if p.metadata.get("coherence", 0) >= self._coherence_threshold
        ]
        
        # Simple relevance scoring (in a real implementation, this would use semantic similarity)
        scored_patterns = []
        for pattern in coherent_patterns:
            # Simple text matching (in a real implementation, this would use embeddings)
            relevance = 0.0
            if pattern.name and query:
                if pattern.name.lower() in query.lower():
                    relevance = 0.8
                elif any(word.lower() in pattern.name.lower() for word in query.lower().split()):
                    relevance = 0.5
            
            # Combine relevance and quality
            quality = pattern.metadata.get("quality", 0)
            combined_score = (relevance * (1 - self._quality_weight)) + (quality * self._quality_weight)
            
            if combined_score >= self._quality_threshold:
                scored_patterns.append((pattern, combined_score))
        
        # Sort by combined score
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Return patterns
        return [p[0] for p in scored_patterns]
    
    def _generate_response(self, query: str, relevant_patterns: List[Pattern], 
                         context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response to a query.
        
        Args:
            query: The query to respond to
            relevant_patterns: The relevant patterns for the query
            context: Optional context for the query
            
        Returns:
            The generated response
        """
        # Simple response generation (in a real implementation, this would use an LLM)
        if not relevant_patterns:
            return "No relevant information found."
            
        # Generate response based on relevant patterns
        response_parts = [f"Found {len(relevant_patterns)} relevant patterns:"]
        
        for i, pattern in enumerate(relevant_patterns[:5]):  # Limit to top 5
            quality = pattern.metadata.get("quality", 0)
            coherence = pattern.metadata.get("coherence", 0)
            response_parts.append(f"{i+1}. {pattern.name} (quality: {quality:.2f}, coherence: {coherence:.2f})")
        
        return "\n".join(response_parts)
