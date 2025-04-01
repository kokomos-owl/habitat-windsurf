"""
Integrated persistence tests for ArangoDB schema.

This module tests the integration between various repositories in the ArangoDB
persistence layer, with a focus on predicate relationships between actants.
"""

import unittest
import os
import sys
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.schema_manager import ArangoDBSchemaManager
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.predicate_relationship_repository import PredicateRelationshipRepository
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.connection_manager import ArangoDBConnectionManager

class TestIntegratedPersistence(unittest.TestCase):
    """
    Integrated tests for ArangoDB persistence layer.
    
    Tests the interaction between various repositories and the database schema.
    Focuses on predicate relationships between actants.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests."""
        # Set up test database name with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        cls.test_db_name = f"habitat_test_{timestamp}"
        
        # Set environment variables for test database
        os.environ["ARANGO_DB"] = cls.test_db_name
        
        # Initialize schema manager
        cls.schema_manager = ArangoDBSchemaManager()
        
        # Initialize schema
        cls.schema_manager.initialize_schema()
        
        # Get database connection
        cls.db = cls.schema_manager.db
        
        # Initialize repositories
        cls.predicate_repo = PredicateRelationshipRepository()
        
        # Sample actant data from regional_observations.json
        cls.sample_actants = [
            {"id": "Pattern/mangrove_forests", "name": "Mangrove Forests"},
            {"id": "Pattern/coastal_communities", "name": "Coastal Communities"},
            {"id": "Pattern/storm_surge", "name": "Storm Surge"},
            {"id": "Pattern/biodiversity", "name": "Biodiversity"},
            {"id": "Pattern/carbon_sequestration", "name": "Carbon Sequestration"},
            {"id": "Pattern/local_economy", "name": "Local Economy"},
            {"id": "Pattern/fisheries", "name": "Fisheries"},
            {"id": "Pattern/tourism", "name": "Tourism"}
        ]
        
        # Create actants in the database
        cls.pattern_collection = cls.db.collection("Pattern")
        for actant in cls.sample_actants:
            # Check if actant already exists
            key = actant["id"].split("/")[1]
            if not cls.pattern_collection.has(key):
                cls.pattern_collection.insert({
                    "_key": key,
                    "name": actant["name"],
                    "created_at": datetime.now().isoformat()
                })
        
        logger.info(f"Test environment set up with database: {cls.test_db_name}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Get connection manager
        connection_manager = ArangoDBConnectionManager()
        sys_db = connection_manager.get_sys_db()
        
        # Drop test database
        if sys_db.has_database(cls.test_db_name):
            sys_db.delete_database(cls.test_db_name)
            logger.info(f"Test database {cls.test_db_name} dropped")
    
    def setUp(self):
        """Set up before each test."""
        # Clear any existing relationships for clean tests
        for predicate in self.predicate_repo.specific_predicates + [self.predicate_repo.generic_predicate]:
            collection = self.db.collection(predicate)
            collection.truncate()
    
    def test_schema_initialization(self):
        """Test that the schema was properly initialized."""
        # Check that all collections exist
        collections = self.db.collections()
        collection_names = [c["name"] for c in collections if not c["name"].startswith("_")]
        
        # Check document collections
        self.assertIn("Pattern", collection_names)
        self.assertIn("TopologyState", collection_names)
        self.assertIn("FrequencyDomain", collection_names)
        
        # Check edge collections
        self.assertIn("Preserves", collection_names)
        self.assertIn("Protects", collection_names)
        self.assertIn("Maintains", collection_names)
        self.assertIn("Enables", collection_names)
        self.assertIn("Enhances", collection_names)
        self.assertIn("ReducesDependenceOn", collection_names)
        self.assertIn("EvolvesInto", collection_names)
        self.assertIn("CoEvolvesWith", collection_names)
        self.assertIn("PredicateRelationship", collection_names)
    
    def test_create_specific_predicate_relationship(self):
        """Test creating a relationship with a specific predicate."""
        # Create a protects relationship
        source_id = "Pattern/mangrove_forests"
        target_id = "Pattern/coastal_communities"
        properties = {
            "confidence": 0.95,
            "harmonic_properties": {
                "frequency": 0.35,
                "amplitude": 0.88,
                "phase": 0.12
            },
            "observation_count": 12,
            "first_observed": datetime.now().isoformat(),
            "last_observed": datetime.now().isoformat()
        }
        
        # Save the relationship
        edge_id = self.predicate_repo.save_relationship(
            source_id,
            "protects",
            target_id,
            properties
        )
        
        # Verify the relationship was created
        self.assertIsNotNone(edge_id)
        self.assertTrue(edge_id.startswith("Protects/"))
        
        # Retrieve the relationship
        results = self.predicate_repo.find_by_source_and_target(
            source_id,
            target_id,
            "protects"
        )
        
        # Verify the retrieved relationship
        self.assertEqual(len(results), 1)
        relationship = results[0]
        self.assertEqual(relationship["_from"], source_id)
        self.assertEqual(relationship["_to"], target_id)
        self.assertEqual(relationship["confidence"], 0.95)
        self.assertEqual(relationship["observation_count"], 12)
        
        # Verify harmonic properties
        harmonic_props = json.loads(relationship["harmonic_properties"])
        self.assertEqual(harmonic_props["frequency"], 0.35)
        self.assertEqual(harmonic_props["amplitude"], 0.88)
        self.assertEqual(harmonic_props["phase"], 0.12)
    
    def test_create_generic_predicate_relationship(self):
        """Test creating a relationship with a generic predicate."""
        # Create a relationship with a predicate that doesn't have a specific collection
        source_id = "Pattern/mangrove_forests"
        target_id = "Pattern/fisheries"
        properties = {
            "confidence": 0.85,
            "harmonic_properties": {
                "frequency": 0.25,
                "amplitude": 0.72,
                "phase": 0.18
            },
            "observation_count": 7,
            "first_observed": datetime.now().isoformat(),
            "last_observed": datetime.now().isoformat()
        }
        
        # Save the relationship
        edge_id = self.predicate_repo.save_relationship(
            source_id,
            "supports",
            target_id,
            properties
        )
        
        # Verify the relationship was created
        self.assertIsNotNone(edge_id)
        self.assertTrue(edge_id.startswith("PredicateRelationship/"))
        
        # Retrieve the relationship
        results = self.predicate_repo.find_by_source_and_target(
            source_id,
            target_id,
            "supports"
        )
        
        # Verify the retrieved relationship
        self.assertEqual(len(results), 1)
        relationship = results[0]
        self.assertEqual(relationship["_from"], source_id)
        self.assertEqual(relationship["_to"], target_id)
        self.assertEqual(relationship["predicate_type"], "supports")
        self.assertEqual(relationship["confidence"], 0.85)
        self.assertEqual(relationship["observation_count"], 7)
    
    def test_update_relationship(self):
        """Test updating an existing relationship."""
        # Create a relationship
        source_id = "Pattern/mangrove_forests"
        target_id = "Pattern/biodiversity"
        properties = {
            "confidence": 0.90,
            "harmonic_properties": {
                "frequency": 0.28,
                "amplitude": 0.75,
                "phase": 0.22
            },
            "observation_count": 8,
            "first_observed": datetime.now().isoformat(),
            "last_observed": datetime.now().isoformat()
        }
        
        # Save the relationship
        edge_id = self.predicate_repo.save_relationship(
            source_id,
            "enhances",
            target_id,
            properties
        )
        
        # Update the relationship
        update_props = {
            "confidence": 0.95,
            "harmonic_properties": {
                "frequency": 0.30,
                "amplitude": 0.80,
                "phase": 0.25
            },
            "observation_count": 12
        }
        
        # Update the relationship
        success = self.predicate_repo.update_relationship(
            edge_id,
            update_props
        )
        
        # Verify the update was successful
        self.assertTrue(success)
        
        # Retrieve the updated relationship
        results = self.predicate_repo.find_by_source_and_target(
            source_id,
            target_id,
            "enhances"
        )
        
        # Verify the updated relationship
        self.assertEqual(len(results), 1)
        relationship = results[0]
        self.assertEqual(relationship["confidence"], 0.95)
        self.assertEqual(relationship["observation_count"], 12)
        
        # Verify updated harmonic properties
        harmonic_props = json.loads(relationship["harmonic_properties"])
        self.assertEqual(harmonic_props["frequency"], 0.30)
        self.assertEqual(harmonic_props["amplitude"], 0.80)
        self.assertEqual(harmonic_props["phase"], 0.25)
    
    def test_find_by_predicate(self):
        """Test finding relationships by predicate type."""
        # Create multiple relationships with the same predicate
        relationships = [
            {
                "source": "Pattern/mangrove_forests",
                "target": "Pattern/coastal_communities",
                "confidence": 0.95
            },
            {
                "source": "Pattern/mangrove_forests",
                "target": "Pattern/local_economy",
                "confidence": 0.85
            },
            {
                "source": "Pattern/mangrove_forests",
                "target": "Pattern/tourism",
                "confidence": 0.75
            }
        ]
        
        # Save the relationships
        for rel in relationships:
            self.predicate_repo.save_relationship(
                rel["source"],
                "enables",
                rel["target"],
                {
                    "confidence": rel["confidence"],
                    "harmonic_properties": {
                        "frequency": 0.35,
                        "amplitude": 0.88,
                        "phase": 0.12
                    },
                    "observation_count": 5
                }
            )
        
        # Find relationships by predicate with confidence threshold
        results = self.predicate_repo.find_by_predicate("enables", 0.80)
        
        # Verify the results
        self.assertEqual(len(results), 2)  # Should find 2 relationships with confidence >= 0.80
        
        # Find all relationships by predicate
        all_results = self.predicate_repo.find_by_predicate("enables", 0.0)
        
        # Verify all results
        self.assertEqual(len(all_results), 3)  # Should find all 3 relationships
    
    def test_find_by_source(self):
        """Test finding relationships by source actant."""
        # Create multiple relationships from the same source
        predicates = ["protects", "enhances", "maintains"]
        targets = ["Pattern/coastal_communities", "Pattern/biodiversity", "Pattern/carbon_sequestration"]
        
        # Save the relationships
        for i, predicate in enumerate(predicates):
            self.predicate_repo.save_relationship(
                "Pattern/mangrove_forests",
                predicate,
                targets[i],
                {
                    "confidence": 0.90,
                    "harmonic_properties": {
                        "frequency": 0.35,
                        "amplitude": 0.88,
                        "phase": 0.12
                    },
                    "observation_count": 5
                }
            )
        
        # Find relationships by source
        results = self.predicate_repo.find_by_source("Pattern/mangrove_forests")
        
        # Verify the results
        self.assertEqual(len(results), 3)  # Should find all 3 relationships
        
        # Find relationships by source and predicate
        predicate_results = self.predicate_repo.find_by_source("Pattern/mangrove_forests", "protects")
        
        # Verify the predicate results
        self.assertEqual(len(predicate_results), 1)  # Should find 1 relationship
        self.assertEqual(predicate_results[0]["_to"], "Pattern/coastal_communities")
    
    def test_find_by_harmonic_properties(self):
        """Test finding relationships by harmonic properties."""
        # Create relationships with different harmonic properties
        relationships = [
            {
                "source": "Pattern/mangrove_forests",
                "target": "Pattern/coastal_communities",
                "predicate": "protects",
                "harmonic_properties": {
                    "frequency": 0.35,
                    "amplitude": 0.88,
                    "phase": 0.12
                }
            },
            {
                "source": "Pattern/mangrove_forests",
                "target": "Pattern/biodiversity",
                "predicate": "enhances",
                "harmonic_properties": {
                    "frequency": 0.25,
                    "amplitude": 0.75,
                    "phase": 0.22
                }
            },
            {
                "source": "Pattern/mangrove_forests",
                "target": "Pattern/carbon_sequestration",
                "predicate": "maintains",
                "harmonic_properties": {
                    "frequency": 0.15,
                    "amplitude": 0.65,
                    "phase": 0.32
                }
            }
        ]
        
        # Save the relationships
        for rel in relationships:
            self.predicate_repo.save_relationship(
                rel["source"],
                rel["predicate"],
                rel["target"],
                {
                    "confidence": 0.90,
                    "harmonic_properties": rel["harmonic_properties"],
                    "observation_count": 5
                }
            )
        
        # Find relationships by frequency range
        results = self.predicate_repo.find_by_harmonic_properties(
            frequency_range=(0.20, 0.40)
        )
        
        # Verify the results
        self.assertEqual(len(results), 2)  # Should find 2 relationships with frequency in range
        
        # Find relationships by amplitude range
        amplitude_results = self.predicate_repo.find_by_harmonic_properties(
            amplitude_range=(0.80, 1.0)
        )
        
        # Verify the amplitude results
        self.assertEqual(len(amplitude_results), 1)  # Should find 1 relationship with amplitude in range
    
    def test_delete_relationship(self):
        """Test deleting a relationship."""
        # Create a relationship
        source_id = "Pattern/mangrove_forests"
        target_id = "Pattern/coastal_communities"
        properties = {
            "confidence": 0.95,
            "harmonic_properties": {
                "frequency": 0.35,
                "amplitude": 0.88,
                "phase": 0.12
            },
            "observation_count": 12
        }
        
        # Save the relationship
        edge_id = self.predicate_repo.save_relationship(
            source_id,
            "protects",
            target_id,
            properties
        )
        
        # Verify the relationship exists
        results_before = self.predicate_repo.find_by_source_and_target(
            source_id,
            target_id,
            "protects"
        )
        self.assertEqual(len(results_before), 1)
        
        # Delete the relationship
        success = self.predicate_repo.delete_relationship(edge_id)
        
        # Verify the deletion was successful
        self.assertTrue(success)
        
        # Verify the relationship no longer exists
        results_after = self.predicate_repo.find_by_source_and_target(
            source_id,
            target_id,
            "protects"
        )
        self.assertEqual(len(results_after), 0)
    
    def test_complex_relationship_network(self):
        """Test creating and querying a complex network of relationships."""
        # Create a network of relationships between actants
        network = [
            {
                "source": "Pattern/mangrove_forests",
                "predicate": "protects",
                "target": "Pattern/coastal_communities",
                "properties": {
                    "confidence": 0.95,
                    "harmonic_properties": {
                        "frequency": 0.35,
                        "amplitude": 0.88,
                        "phase": 0.12
                    },
                    "observation_count": 12
                }
            },
            {
                "source": "Pattern/mangrove_forests",
                "predicate": "enhances",
                "target": "Pattern/biodiversity",
                "properties": {
                    "confidence": 0.92,
                    "harmonic_properties": {
                        "frequency": 0.28,
                        "amplitude": 0.75,
                        "phase": 0.22
                    },
                    "observation_count": 8
                }
            },
            {
                "source": "Pattern/mangrove_forests",
                "predicate": "reduces_dependence_on",
                "target": "Pattern/storm_surge",
                "properties": {
                    "confidence": 0.88,
                    "harmonic_properties": {
                        "frequency": 0.42,
                        "amplitude": 0.65,
                        "phase": 0.18
                    },
                    "observation_count": 6
                }
            },
            {
                "source": "Pattern/mangrove_forests",
                "predicate": "maintains",
                "target": "Pattern/carbon_sequestration",
                "properties": {
                    "confidence": 0.90,
                    "harmonic_properties": {
                        "frequency": 0.15,
                        "amplitude": 0.82,
                        "phase": 0.05
                    },
                    "observation_count": 9
                }
            },
            {
                "source": "Pattern/biodiversity",
                "predicate": "enhances",
                "target": "Pattern/fisheries",
                "properties": {
                    "confidence": 0.85,
                    "harmonic_properties": {
                        "frequency": 0.32,
                        "amplitude": 0.78,
                        "phase": 0.15
                    },
                    "observation_count": 7
                }
            },
            {
                "source": "Pattern/fisheries",
                "predicate": "enables",
                "target": "Pattern/local_economy",
                "properties": {
                    "confidence": 0.82,
                    "harmonic_properties": {
                        "frequency": 0.38,
                        "amplitude": 0.72,
                        "phase": 0.25
                    },
                    "observation_count": 10
                }
            },
            {
                "source": "Pattern/coastal_communities",
                "predicate": "co_evolves_with",
                "target": "Pattern/mangrove_forests",
                "properties": {
                    "confidence": 0.78,
                    "harmonic_properties": {
                        "frequency": 0.22,
                        "amplitude": 0.68,
                        "phase": 0.30
                    },
                    "observation_count": 5
                }
            }
        ]
        
        # Save all relationships
        for rel in network:
            self.predicate_repo.save_relationship(
                rel["source"],
                rel["predicate"],
                rel["target"],
                rel["properties"]
            )
        
        # Find all relationships from mangrove_forests
        mangrove_rels = self.predicate_repo.find_by_source("Pattern/mangrove_forests")
        
        # Verify the number of relationships
        self.assertEqual(len(mangrove_rels), 4)
        
        # Find all relationships to local_economy
        economy_rels = self.predicate_repo.find_by_target("Pattern/local_economy")
        
        # Verify the number of relationships
        self.assertEqual(len(economy_rels), 1)
        self.assertEqual(economy_rels[0]["_from"], "Pattern/fisheries")
        
        # Find bidirectional relationships (co_evolves_with)
        co_evolve_rels = self.predicate_repo.find_by_predicate("co_evolves_with")
        
        # Verify the bidirectional relationship
        self.assertEqual(len(co_evolve_rels), 1)
        self.assertEqual(co_evolve_rels[0]["_from"], "Pattern/coastal_communities")
        self.assertEqual(co_evolve_rels[0]["_to"], "Pattern/mangrove_forests")

if __name__ == '__main__':
    unittest.main()
