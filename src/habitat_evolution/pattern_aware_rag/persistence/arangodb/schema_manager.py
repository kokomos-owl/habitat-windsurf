"""
ArangoDB Schema Manager for Pattern-Aware RAG

This module defines and manages the ArangoDB schema for the pattern-aware RAG system,
including collections for field state, topology, and semantic signatures.
"""

from typing import Dict, List, Any, Optional
from arango import ArangoClient
import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ArangoDBSchemaManager:
    """
    Manages the ArangoDB schema for pattern-aware RAG, including collections
    for field state, topology, and semantic signatures.
    """
    
    def __init__(self):
        """Initialize the schema manager with database connection."""
        load_dotenv()
        
        # Get connection details from environment variables or use defaults
        host = os.getenv('ARANGO_HOST', 'http://localhost:8529')
        username = os.getenv('ARANGO_USER', 'root')
        password = os.getenv('ARANGO_PASSWORD', 'habitat')
        db_name = os.getenv('ARANGO_DB', 'habitat')
        
        # Initialize the client
        self.client = ArangoClient(hosts=host)
        
        # Connect to the database
        self.db = self.client.db(db_name, username=username, password=password)
    
    def create_schema(self) -> None:
        """
        Create the ArangoDB schema for pattern-aware RAG.
        """
        # Define document collections
        document_collections = [
            ("TonicHarmonicFieldState", "Stores field state with eigenspace properties and metrics"),
            ("TopologyState", "Captures the current topology of the field"),
            ("FrequencyDomain", "Represents coherent regions in the field"),
            ("Boundary", "Represents interfaces between frequency domains"),
            ("ResonancePoint", "Represents points of high resonance in the field"),
            ("Pattern", "Stores pattern information with eigenspace properties"),
            ("ResonanceGroup", "Groups patterns that resonate together"),
            ("SemanticSignature", "Minimal implementation for entity identity across time periods")
        ]
        
        # Define edge collections
        edge_collections = [
            # Structural relationships
            ("HasDomain", "Links TopologyState to FrequencyDomain"),
            ("HasBoundary", "Links TopologyState to Boundary"),
            ("HasResonance", "Links TopologyState to ResonancePoint"),
            ("HasPattern", "Links TopologyState to Pattern"),
            ("HasResonanceGroup", "Links TopologyState to ResonanceGroup"),
            ("Connects", "Links Boundary to FrequencyDomain"),
            ("BelongsTo", "Links Pattern to ResonanceGroup"),
            ("ContributesTo", "Links Pattern to ResonancePoint"),
            
            # Statistical and spatial relationships
            ("StatisticallyCorrelatedWith", "Links entities based on statistical correlation"),
            ("LocatedAt", "Links entities to spatial locations"),
            ("DiffersFromControlBy", "Links entities based on control differences"),
            
            # Generic predicate relationship for dynamic handling of any predicate
            ("PredicateRelationship", "Generic edge for any predicate relationship between actants"),
            
            # Common semantic predicates observed in regional_observations.json
            ("Preserves", "Links actants that preserve or protect other actants"),
            ("Protects", "Links actants that provide protection to other actants"),
            ("Maintains", "Links actants that maintain properties of other actants"),
            ("Enables", "Links actants that enable capabilities of other actants"),
            ("Enhances", "Links actants that enhance properties of other actants"),
            ("ReducesDependenceOn", "Links actants that reduce dependence on other actants"),
            
            # Temporal evolution relationships
            ("EvolvesInto", "Links actants across time periods showing evolution"),
            ("CoEvolvesWith", "Links actants that evolve together through mutual influence")
        ]
        
        # Create document collections
        for name, description in document_collections:
            self._create_collection(name, is_edge=False, description=description)
        
        # Create edge collections
        for name, description in edge_collections:
            self._create_collection(name, is_edge=True, description=description)
        
        # Create indexes for faster lookups
        self._create_indexes()
        
        # Create graph definitions
        self.create_graph_definitions()
        
        logger.info("Schema creation complete")
    
    def _create_collection(self, name: str, is_edge: bool, description: str) -> None:
        """
        Create a collection if it doesn't exist.
        
        Args:
            name: Name of the collection
            is_edge: Whether this is an edge collection
            description: Description of the collection
        """
        if not self.db.has_collection(name):
            self.db.create_collection(name, edge=is_edge)
            
            # Store collection metadata
            collection = self.db.collection(name)
            # Use current timestamp instead of server_info
            from datetime import datetime
            metadata = {
                "_key": "metadata",
                "description": description,
                "created_at": datetime.now().isoformat()
            }
            try:
                collection.insert(metadata)
            except Exception as e:
                logger.warning(f"Could not insert metadata for {name}: {str(e)}")
                
            logger.info(f"Created collection: {name}")
        else:
            logger.info(f"Collection already exists: {name}")
    
    def _create_indexes(self) -> None:
        """Create indexes for faster lookups in the collections."""
        # Create indexes for TonicHarmonicFieldState
        if self.db.has_collection("TonicHarmonicFieldState"):
            collection = self.db.collection("TonicHarmonicFieldState")
            collection.add_hash_index(["id"], unique=True)
            collection.add_skiplist_index(["version_id"])
            collection.add_skiplist_index(["created_at"])
        
        # Create indexes for TopologyState
        if self.db.has_collection("TopologyState"):
            collection = self.db.collection("TopologyState")
            collection.add_hash_index(["id"], unique=True)
            collection.add_skiplist_index(["field_state_id"])
            collection.add_skiplist_index(["created_at"])
        
        # Create indexes for FrequencyDomain
        if self.db.has_collection("FrequencyDomain"):
            collection = self.db.collection("FrequencyDomain")
            collection.add_hash_index(["id"], unique=True)
            collection.add_skiplist_index(["name"])
            collection.add_skiplist_index(["coherence"])
            collection.add_skiplist_index(["topology_state_id"])
        
        # Create indexes for Boundary
        if self.db.has_collection("Boundary"):
            collection = self.db.collection("Boundary")
            collection.add_hash_index(["id"], unique=True)
            collection.add_skiplist_index(["permeability"])
        
        # Create indexes for ResonancePoint
        if self.db.has_collection("ResonancePoint"):
            collection = self.db.collection("ResonancePoint")
            collection.add_hash_index(["id"], unique=True)
            collection.add_skiplist_index(["strength"])
        
        # Create indexes for Pattern
        if self.db.has_collection("Pattern"):
            collection = self.db.collection("Pattern")
            collection.add_hash_index(["id"], unique=True)
            collection.add_skiplist_index(["pattern_type"])
            collection.add_skiplist_index(["source"])
            collection.add_skiplist_index(["predicate"])
            collection.add_skiplist_index(["target"])
            collection.add_skiplist_index(["confidence"])
        
        # Create indexes for SemanticSignature
        if self.db.has_collection("SemanticSignature"):
            collection = self.db.collection("SemanticSignature")
            collection.add_hash_index(["id"], unique=True)
            collection.add_skiplist_index(["entity_id"])
            collection.add_skiplist_index(["entity_type"])
            collection.add_skiplist_index(["confidence"])
        
        # Create indexes for predicate relationships
        predicate_edges = [
            "PredicateRelationship", "Preserves", "Protects", "Maintains", 
            "Enables", "Enhances", "ReducesDependenceOn", "EvolvesInto", "CoEvolvesWith"
        ]
        
        for edge_name in predicate_edges:
            if self.db.has_collection(edge_name):
                collection = self.db.collection(edge_name)
                collection.add_skiplist_index(["_from", "_to"])
                collection.add_skiplist_index(["timestamp"])
                collection.add_skiplist_index(["confidence"])
                
                # For the generic PredicateRelationship, add index on predicate type
                if edge_name == "PredicateRelationship":
                    collection.add_skiplist_index(["predicate_type"])
                    collection.add_skiplist_index(["first_observed"])
                    collection.add_skiplist_index(["last_observed"])
    
    def create_graph_definitions(self) -> None:
        """
        Create graph definitions to enable graph traversals for pattern-aware RAG.
        """
        # Check if graph already exists
        if self.db.has_graph("PatternAwareRAG"):
            logger.info("Graph 'PatternAwareRAG' already exists.")
            return
        
        # Define edge definitions
        edge_definitions = [
            # Structural relationships
            {
                "edge_collection": "HasDomain",
                "from_vertex_collections": ["TopologyState"],
                "to_vertex_collections": ["FrequencyDomain"]
            },
            {
                "edge_collection": "HasBoundary",
                "from_vertex_collections": ["TopologyState"],
                "to_vertex_collections": ["Boundary"]
            },
            {
                "edge_collection": "HasResonance",
                "from_vertex_collections": ["TopologyState"],
                "to_vertex_collections": ["ResonancePoint"]
            },
            {
                "edge_collection": "HasPattern",
                "from_vertex_collections": ["TopologyState"],
                "to_vertex_collections": ["Pattern"]
            },
            {
                "edge_collection": "HasResonanceGroup",
                "from_vertex_collections": ["TopologyState"],
                "to_vertex_collections": ["ResonanceGroup"]
            },
            {
                "edge_collection": "Connects",
                "from_vertex_collections": ["Boundary"],
                "to_vertex_collections": ["FrequencyDomain"]
            },
            {
                "edge_collection": "BelongsTo",
                "from_vertex_collections": ["Pattern"],
                "to_vertex_collections": ["ResonanceGroup"]
            },
            {
                "edge_collection": "ContributesTo",
                "from_vertex_collections": ["Pattern"],
                "to_vertex_collections": ["ResonancePoint"]
            },
            
            # Generic predicate relationship
            {
                "edge_collection": "PredicateRelationship",
                "from_vertex_collections": ["Pattern", "FrequencyDomain", "ResonancePoint", "SemanticSignature"],
                "to_vertex_collections": ["Pattern", "FrequencyDomain", "ResonancePoint", "SemanticSignature"]
            },
            
            # Common semantic predicates
            {
                "edge_collection": "Preserves",
                "from_vertex_collections": ["Pattern", "SemanticSignature"],
                "to_vertex_collections": ["Pattern", "SemanticSignature"]
            },
            {
                "edge_collection": "Protects",
                "from_vertex_collections": ["Pattern", "SemanticSignature"],
                "to_vertex_collections": ["Pattern", "SemanticSignature"]
            },
            {
                "edge_collection": "Maintains",
                "from_vertex_collections": ["Pattern", "SemanticSignature"],
                "to_vertex_collections": ["Pattern", "SemanticSignature"]
            },
            {
                "edge_collection": "Enables",
                "from_vertex_collections": ["Pattern", "SemanticSignature"],
                "to_vertex_collections": ["Pattern", "SemanticSignature"]
            },
            {
                "edge_collection": "Enhances",
                "from_vertex_collections": ["Pattern", "SemanticSignature"],
                "to_vertex_collections": ["Pattern", "SemanticSignature"]
            },
            {
                "edge_collection": "ReducesDependenceOn",
                "from_vertex_collections": ["Pattern", "SemanticSignature"],
                "to_vertex_collections": ["Pattern", "SemanticSignature"]
            },
            
            # Temporal evolution relationships
            {
                "edge_collection": "EvolvesInto",
                "from_vertex_collections": ["Pattern", "SemanticSignature"],
                "to_vertex_collections": ["Pattern", "SemanticSignature"]
            },
            {
                "edge_collection": "CoEvolvesWith",
                "from_vertex_collections": ["Pattern", "SemanticSignature"],
                "to_vertex_collections": ["Pattern", "SemanticSignature"]
            },
            
            # Statistical and spatial relationships
            {
                "edge_collection": "StatisticallyCorrelatedWith",
                "from_vertex_collections": ["Pattern", "SemanticSignature"],
                "to_vertex_collections": ["Pattern", "SemanticSignature"]
            },
            {
                "edge_collection": "LocatedAt",
                "from_vertex_collections": ["Pattern", "SemanticSignature"],
                "to_vertex_collections": ["Pattern", "SemanticSignature"]
            },
            {
                "edge_collection": "DiffersFromControlBy",
                "from_vertex_collections": ["Pattern", "SemanticSignature"],
                "to_vertex_collections": ["Pattern", "SemanticSignature"]
            }
        ]
        
        # Create the graph
        self.db.create_graph("PatternAwareRAG", edge_definitions)
        logger.info(f"Created graph: PatternAwareRAG")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and initialize the schema
    schema_manager = ArangoDBSchemaManager()
    schema_manager.create_schema()
