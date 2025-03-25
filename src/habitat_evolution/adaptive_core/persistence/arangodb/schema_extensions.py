"""
ArangoDB Schema Extensions for Cross-Domain Topology

This module extends the ArangoDB schema to support cross-domain topology,
temporal pattern tracking, and actant journey mapping.
"""

from typing import Dict, List, Any, Optional
from arango import ArangoClient
import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ArangoDBSchemaExtender:
    """
    Extends the ArangoDB schema with collections and indexes needed for
    cross-domain topology, temporal pattern tracking, and actant journey mapping.
    """
    
    def __init__(self):
        """Initialize the schema extender with database connection."""
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
    
    def extend_schema(self) -> None:
        """
        Extend the ArangoDB schema with new collections and indexes for cross-domain topology.
        """
        # Define new document collections
        document_collections = [
            ("ActantJourney", "Stores the complete journey of an actant across domains"),
            ("LearningWindow", "Stores learning window states and transitions"),
            ("TemporalPattern", "Stores patterns that emerge over time"),
            ("RoleShift", "Stores information about actant role shifts"),
            ("DomainTransition", "Stores transitions of actants between domains")
        ]
        
        # Define new edge collections
        edge_collections = [
            ("ActantTransitionsToDomain", "Represents actant transitions between domains"),
            ("PredicateTransformsTo", "Represents predicate transformations"),
            ("PatternEvolvesTo", "Represents pattern evolution"),
            ("WindowContainsPattern", "Links learning windows to patterns"),
            ("ActantHasJourney", "Links actants to their journeys"),
            ("JourneyContainsTransition", "Links actant journeys to domain transitions"),
            ("JourneyContainsRoleShift", "Links actant journeys to role shifts")
        ]
        
        # Create document collections
        for name, description in document_collections:
            self._create_collection(name, is_edge=False, description=description)
        
        # Create edge collections
        for name, description in edge_collections:
            self._create_collection(name, is_edge=True, description=description)
        
        # Create indexes for faster lookups
        self._create_indexes()
        
        logger.info("Schema extension complete")
    
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
            metadata = {
                "_key": "metadata",
                "description": description,
                "created_at": self.db.server_info()["time"]
            }
            try:
                collection.insert(metadata)
            except Exception as e:
                logger.warning(f"Could not insert metadata for {name}: {str(e)}")
                
            logger.info(f"Created collection: {name}")
        else:
            logger.info(f"Collection already exists: {name}")
    
    def _create_indexes(self) -> None:
        """Create indexes for faster lookups in the new collections."""
        # ActantJourney indexes
        if "ActantJourney" in self.db.collections():
            self.db.collection("ActantJourney").add_hash_index(["actant_name"], unique=True)
            self.db.collection("ActantJourney").add_hash_index(["adaptive_id"], unique=True)
            logger.info("Added indexes on ActantJourney.actant_name and ActantJourney.adaptive_id")
        
        # LearningWindow indexes
        if "LearningWindow" in self.db.collections():
            self.db.collection("LearningWindow").add_hash_index(["window_id"], unique=True)
            self.db.collection("LearningWindow").add_skiplist_index(["start_time", "end_time"])
            logger.info("Added indexes on LearningWindow")
        
        # TemporalPattern indexes
        if "TemporalPattern" in self.db.collections():
            self.db.collection("TemporalPattern").add_hash_index(["pattern_id"], unique=True)
            self.db.collection("TemporalPattern").add_skiplist_index(["stability"])
            logger.info("Added indexes on TemporalPattern")
        
        # RoleShift indexes
        if "RoleShift" in self.db.collections():
            self.db.collection("RoleShift").add_hash_index(["actant_name"])
            self.db.collection("RoleShift").add_skiplist_index(["timestamp"])
            logger.info("Added indexes on RoleShift")
        
        # DomainTransition indexes
        if "DomainTransition" in self.db.collections():
            self.db.collection("DomainTransition").add_hash_index(["actant_name"])
            self.db.collection("DomainTransition").add_skiplist_index(["timestamp"])
            self.db.collection("DomainTransition").add_hash_index(["source_domain_id", "target_domain_id"])
            logger.info("Added indexes on DomainTransition")
    
    def create_graph_definitions(self) -> None:
        """
        Create graph definitions to enable graph traversals for cross-domain topology.
        """
        # Define the cross-domain topology graph
        graph_name = "CrossDomainTopology"
        
        # Check if graph already exists
        if not self.db.has_graph(graph_name):
            # Define edge definitions
            edge_definitions = [
                {
                    "edge_collection": "ActantTransitionsToDomain",
                    "from_vertex_collections": ["Actant"],
                    "to_vertex_collections": ["Domain"]
                },
                {
                    "edge_collection": "PredicateTransformsTo",
                    "from_vertex_collections": ["Predicate"],
                    "to_vertex_collections": ["Predicate"]
                },
                {
                    "edge_collection": "PatternEvolvesTo",
                    "from_vertex_collections": ["TemporalPattern"],
                    "to_vertex_collections": ["TemporalPattern"]
                },
                {
                    "edge_collection": "WindowContainsPattern",
                    "from_vertex_collections": ["LearningWindow"],
                    "to_vertex_collections": ["TemporalPattern"]
                },
                {
                    "edge_collection": "ActantHasJourney",
                    "from_vertex_collections": ["Actant"],
                    "to_vertex_collections": ["ActantJourney"]
                },
                {
                    "edge_collection": "JourneyContainsTransition",
                    "from_vertex_collections": ["ActantJourney"],
                    "to_vertex_collections": ["DomainTransition"]
                },
                {
                    "edge_collection": "JourneyContainsRoleShift",
                    "from_vertex_collections": ["ActantJourney"],
                    "to_vertex_collections": ["RoleShift"]
                }
            ]
            
            # Create the graph
            self.db.create_graph(graph_name, edge_definitions)
            logger.info(f"Created graph: {graph_name}")
        else:
            logger.info(f"Graph already exists: {graph_name}")
    
    def create_specialized_traversals(self) -> None:
        """
        Create specialized AQL graph traversals for following actant journeys.
        These will be stored as AQL user functions for reuse.
        """
        # Define the traversal functions
        traversals = {
            "ACTANT_JOURNEY": """
            // Traversal to follow an actant's journey across domains
            function (actant_name) {
                const actant = (FOR a IN Actant FILTER a.name == actant_name RETURN a)[0];
                if (!actant) {
                    return [];
                }
                
                // Find the actant's journey
                const journey = (
                    FOR j IN ActantJourney
                    FILTER j.actant_name == actant_name
                    RETURN j
                )[0];
                
                if (!journey) {
                    return [];
                }
                
                // Get all domain transitions in chronological order
                const transitions = (
                    FOR v, e IN 1..1 OUTBOUND DOCUMENT(journey._id) JourneyContainsTransition
                    LET transition = v
                    SORT transition.timestamp
                    RETURN transition
                );
                
                // Get all role shifts in chronological order
                const roleShifts = (
                    FOR v, e IN 1..1 OUTBOUND DOCUMENT(journey._id) JourneyContainsRoleShift
                    LET shift = v
                    SORT shift.timestamp
                    RETURN shift
                );
                
                // Combine and sort all events chronologically
                const allEvents = (
                    FOR t IN transitions
                    RETURN MERGE(t, { event_type: 'transition' })
                )
                CONCAT(
                    FOR s IN roleShifts
                    RETURN MERGE(s, { event_type: 'role_shift' })
                );
                
                RETURN SORTED(allEvents, e => e.timestamp);
            }
            """,
            
            "PATTERN_EVOLUTION": """
            // Traversal to follow pattern evolution within a time window
            function (start_time, end_time) {
                // Find learning windows in the time range
                const windows = (
                    FOR w IN LearningWindow
                    FILTER w.start_time >= start_time AND w.end_time <= end_time
                    RETURN w
                );
                
                // Get all patterns in these windows
                const patterns = (
                    FOR w IN windows
                    FOR v, e IN 1..1 OUTBOUND DOCUMENT(w._id) WindowContainsPattern
                    LET pattern = v
                    RETURN pattern
                );
                
                // Get pattern evolution chains
                const evolutions = (
                    FOR p IN patterns
                    LET chain = (
                        FOR v, e, path IN 1..10 OUTBOUND DOCUMENT(p._id) PatternEvolvesTo
                        RETURN v
                    )
                    RETURN {
                        origin: p,
                        evolution_chain: chain
                    }
                );
                
                RETURN evolutions;
            }
            """,
            
            "CROSS_DOMAIN_TOPOLOGY": """
            // Traversal to analyze cross-domain topology
            function () {
                // Get all domains
                const domains = (FOR d IN Domain RETURN d);
                
                // For each domain, find connections to other domains via actants
                const domainConnections = (
                    FOR d1 IN domains
                    LET connections = (
                        FOR d2 IN domains
                        FILTER d1._key != d2._key
                        
                        // Find actants that appear in both domains
                        LET common_actants = (
                            FOR a IN Actant
                            LET in_d1 = (
                                FOR v, e IN 1..1 INBOUND DOCUMENT(a._id) ActantTransitionsToDomain
                                FILTER v._id == d1._id
                                RETURN 1
                            )
                            LET in_d2 = (
                                FOR v, e IN 1..1 INBOUND DOCUMENT(a._id) ActantTransitionsToDomain
                                FILTER v._id == d2._id
                                RETURN 1
                            )
                            FILTER LENGTH(in_d1) > 0 AND LENGTH(in_d2) > 0
                            RETURN a
                        )
                        
                        FILTER LENGTH(common_actants) > 0
                        
                        RETURN {
                            domain: d2,
                            common_actants: common_actants,
                            strength: LENGTH(common_actants)
                        }
                    )
                    
                    RETURN {
                        domain: d1,
                        connections: connections
                    }
                );
                
                RETURN domainConnections;
            }
            """
        }
        
        # Register the traversal functions
        for name, code in traversals.items():
            # AQL user functions are registered in a specific format
            # We'll use a simpler approach for now - storing them in a document
            try:
                if not self.db.has_collection("AQLTraversals"):
                    self.db.create_collection("AQLTraversals")
                
                collection = self.db.collection("AQLTraversals")
                
                # Check if the traversal already exists
                existing = list(collection.find({"_key": name}))
                
                if not existing:
                    collection.insert({
                        "_key": name,
                        "code": code,
                        "description": f"Specialized traversal for {name.lower().replace('_', ' ')}",
                        "created_at": self.db.server_info()["time"]
                    })
                    logger.info(f"Created traversal function: {name}")
                else:
                    collection.update(existing[0], {
                        "code": code,
                        "updated_at": self.db.server_info()["time"]
                    })
                    logger.info(f"Updated traversal function: {name}")
                    
            except Exception as e:
                logger.error(f"Error creating traversal function {name}: {str(e)}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run the schema extender
    extender = ArangoDBSchemaExtender()
    extender.extend_schema()
    extender.create_graph_definitions()
    extender.create_specialized_traversals()
    
    logger.info("Schema extension complete")
