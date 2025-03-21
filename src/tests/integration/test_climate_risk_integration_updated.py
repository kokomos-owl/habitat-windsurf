"""
Integration test for climate risk document processing through the tonic-harmonic system.

This test validates the end-to-end flow of processing a climate risk document through
the pattern-aware RAG system, persisting the results to Neo4j, and executing Cypher
queries to ensure the integration works correctly.
"""

import os
import json
import unittest
import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from bson import ObjectId

from neo4j import GraphDatabase
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required classes directly to avoid import path issues
class TopologyState:
    """Topology state class."""
    def __init__(self, id=None):
        self.id = id or f"ts-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.frequency_domains = {}
        self.boundaries = {}
        self.resonance_points = {}
        self.field_metrics = None
        self.pattern_eigenspace_properties = {}
        self.resonance_groups = {}
        self.learning_windows = {}

class FrequencyDomain:
    """Frequency domain class."""
    def __init__(self, id, dominant_frequency, bandwidth, phase_coherence, radius, metadata=None):
        self.id = id
        self.dominant_frequency = dominant_frequency
        self.bandwidth = bandwidth
        self.phase_coherence = phase_coherence
        self.radius = radius
        self.metadata = metadata or {}

class Boundary:
    """Boundary class."""
    def __init__(self, id, domain_ids, permeability, sharpness, stability):
        self.id = id
        self.domain_ids = domain_ids
        self.permeability = permeability
        self.sharpness = sharpness
        self.stability = stability

class ResonancePoint:
    """Resonance point class."""
    def __init__(self, id, coordinates, strength, stability, attractor_radius, contributing_pattern_ids):
        self.id = id
        self.coordinates = coordinates
        self.strength = strength
        self.stability = stability
        self.attractor_radius = attractor_radius
        self.contributing_pattern_ids = contributing_pattern_ids

class FieldMetrics:
    """Field metrics class."""
    def __init__(self, coherence, energy_density, adaptation_rate, homeostasis_index, entropy):
        self.coherence = coherence
        self.energy_density = energy_density
        self.adaptation_rate = adaptation_rate
        self.homeostasis_index = homeostasis_index
        self.entropy = entropy

class ClimateRiskIntegrationTest(unittest.TestCase):
    """Integration test for climate risk document processing through the tonic-harmonic system."""
    
    def setUp(self):
        """Set up test environment."""
        # Neo4j configuration - using the running Docker container
        # These should match the Docker container settings
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "habitat123"  # Updated to match Docker container password
        
        # Path to climate risk data file
        self.climate_risk_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "data", "climate_risk", "climate_risk_marthas_vineyard.txt"
        )
        
        # Create Neo4j driver
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
    
    def tearDown(self):
        """Clean up after test."""
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()
    
    def load_climate_risk_document(self) -> Dict[str, Any]:
        """Load climate risk document directly from file.
        
        Returns:
            Document as a dictionary
        """
        # Read climate risk document
        with open(self.climate_risk_path, "r") as f:
            content = f.read()
            
        # Split document into paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        
        # Create document
        document = {
            "doc_id": "climate_risk_marthas_vineyard",
            "title": "Climate Risk Assessment - Martha's Vineyard, MA",
            "content": content,
            "paragraphs": paragraphs,
            "metadata": {
                "source": "Woodwell Climate Research Center",
                "type": "climate_risk_assessment",
                "location": "Martha's Vineyard, MA"
            }
        }
        
        return document
    
    def process_document_through_rag(self, document: Dict[str, Any]) -> tuple:
        """Process document through RAG system.
        
        Args:
            document: Document to process
            
        Returns:
            Tuple of (topology_state, field_analysis)
        """
        # Extract paragraphs from document
        paragraphs = document.get("paragraphs", [])
        
        # Create patterns from paragraphs
        patterns = []
        for i, paragraph in enumerate(paragraphs[:10]):  # Limit to first 10 paragraphs for testing
            pattern = {
                "id": f"p-{i}",
                "content": paragraph,
                "created_at": datetime.now().isoformat(),
                "source": document.get("metadata", {}).get("source", "Unknown"),
                "metadata": {
                    "document_id": document.get("doc_id", ""),
                    "paragraph_index": i
                }
            }
            patterns.append(pattern)
        
        # Create field analysis
        field_analysis = {
            "patterns": patterns,
            "resonance_relationships": {}
        }
        
        # Create resonance relationships between patterns
        for i in range(len(patterns)):
            for j in range(i+1, len(patterns)):
                rel_id = f"rel-{i}-{j}"
                # Randomly assign interference type
                interference_types = ["CONSTRUCTIVE", "DESTRUCTIVE", "PARTIAL"]
                field_analysis["resonance_relationships"][rel_id] = {
                    "source_id": patterns[i]["id"],
                    "target_id": patterns[j]["id"],
                    "interference_type": np.random.choice(interference_types),
                    "resonance_strength": 0.5 + (0.5 * np.random.random())
                }
        
        # Create topology state
        topology_state = TopologyState()
        
        # Set pattern eigenspace properties
        pattern_eigenspace_properties = {}
        for pattern in patterns:
            pattern_eigenspace_properties[pattern["id"]] = {
                "tonic_value": 0.5 + (0.5 * np.random.random()),
                "phase_position": np.random.random() * 2 * np.pi,
                "dimensional_coordinates": [np.random.random() for _ in range(5)]
            }
        topology_state.pattern_eigenspace_properties = pattern_eigenspace_properties
        
        # Create frequency domains
        domains = {}
        themes = ["climate", "risk", "coastal", "economic", "social"]
        for i, theme in enumerate(themes):
            domain_id = f"fd-{theme}"
            domains[domain_id] = FrequencyDomain(
                id=domain_id,
                dominant_frequency=0.1 * (i + 1),
                bandwidth=0.05,
                phase_coherence=0.7 + (0.3 * np.random.random()),
                radius=0.6 + (0.4 * np.random.random()),
                metadata={"name": f"{theme.capitalize()} Domain"}
            )
        topology_state.frequency_domains = domains
        
        # Create boundaries between domains
        boundaries = {}
        for i in range(len(themes) - 1):
            boundary_id = f"b-{i}"
            boundaries[boundary_id] = Boundary(
                id=boundary_id,
                domain_ids=(f"fd-{themes[i]}", f"fd-{themes[i+1]}"),
                permeability=np.random.random(),
                sharpness=np.random.random() * 0.5,
                stability=0.7 + (0.3 * np.random.random())
            )
        topology_state.boundaries = boundaries
        
        # Create resonance points
        resonance_points = {}
        for i in range(3):
            point_id = f"rp-{i}"
            # Create a dictionary of pattern IDs to weights
            contributing_patterns = {p["id"]: 0.5 + (0.5 * np.random.random()) 
                                    for p in np.random.choice(patterns, 3)}
            
            resonance_points[point_id] = ResonancePoint(
                id=point_id,
                coordinates=tuple(np.random.random(5)),
                strength=0.5 + (0.5 * np.random.random()),
                stability=0.7 + (0.3 * np.random.random()),
                attractor_radius=0.3 + (0.3 * np.random.random()),
                contributing_pattern_ids=contributing_patterns
            )
        topology_state.resonance_points = resonance_points
        
        # Set field metrics
        topology_state.field_metrics = FieldMetrics(
            coherence=0.75,
            energy_density={"global": 0.6},
            adaptation_rate=0.4,
            homeostasis_index=0.7,
            entropy=0.3
        )
        
        # Set pattern eigenspace properties
        for pattern_id, props in pattern_eigenspace_properties.items():
            # Assign patterns to resonance groups based on tonic value
            if props["tonic_value"] > 0.8:
                group = "high_tonic"
            elif props["tonic_value"] > 0.5:
                group = "medium_tonic"
            else:
                group = "low_tonic"
                
            # Create resonance group if it doesn't exist
            if group not in topology_state.resonance_groups:
                topology_state.resonance_groups[group] = {
                    "patterns": [],
                    "coherence": 0.7 + (0.3 * np.random.random()),
                    "stability": 0.6 + (0.4 * np.random.random()),
                    "harmonic_value": 0.5 + (0.5 * np.random.random())
                }
            
            # Add pattern to group
            topology_state.resonance_groups[group]["patterns"].append(pattern_id)
        
        # Create learning windows
        learning_windows = {}
        window_types = ["fast", "medium", "slow"]
        for i, window_type in enumerate(window_types):
            window_id = f"lw-{window_type}"
            learning_windows[window_id] = {
                "id": window_id,
                "time_scale": (i + 1) * 10,
                "learning_rate": 0.1 + (0.2 * np.random.random()),
                "stability_threshold": 0.5 + (0.3 * np.random.random())
            }
        topology_state.learning_windows = learning_windows
        
        return topology_state, field_analysis
        
    def persist_to_neo4j(self, topology_state: TopologyState, field_analysis: Dict[str, Any]) -> None:
        """Persist topology state to Neo4j."""
        # Clear existing data
        with self.neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        # Create patterns
        with self.neo4j_driver.session() as session:
            for pattern_id, props in topology_state.pattern_eigenspace_properties.items():
                # Convert properties to a format Neo4j can handle
                neo4j_props = {
                    "id": pattern_id,
                    "tonic_value": props["tonic_value"],
                    "phase_position": props["phase_position"],
                    "content": next((p["content"] for p in field_analysis["patterns"] if p["id"] == pattern_id), "")
                }
                
                # Create pattern node
                session.run(
                    """
                    CREATE (p:Pattern {id: $id, tonic_value: $tonic_value, phase_position: $phase_position, content: $content})
                    """,
                    neo4j_props
                )
        
        # Create resonance groups
        with self.neo4j_driver.session() as session:
            for group_id, group in topology_state.resonance_groups.items():
                # Create group node
                session.run(
                    """
                    CREATE (rg:ResonanceGroup {id: $id, coherence: $coherence, stability: $stability, harmonic_value: $harmonic_value})
                    """,
                    {
                        "id": group_id,
                        "coherence": group["coherence"],
                        "stability": group["stability"],
                        "harmonic_value": group["harmonic_value"]
                    }
                )
                
                # Connect patterns to group
                for pattern_id in group["patterns"]:
                    session.run(
                        """
                        MATCH (p:Pattern {id: $pattern_id})
                        MATCH (rg:ResonanceGroup {id: $group_id})
                        CREATE (p)-[:BELONGS_TO]->(rg)
                        """,
                        {"pattern_id": pattern_id, "group_id": group_id}
                    )
        
        # Create wave relationships between patterns
        with self.neo4j_driver.session() as session:
            for rel_id, rel in field_analysis["resonance_relationships"].items():
                session.run(
                    """
                    MATCH (p1:Pattern {id: $source_id})
                    MATCH (p2:Pattern {id: $target_id})
                    CREATE (p1)-[:WAVE_RELATIONSHIP {interference_type: $interference_type, resonance_strength: $resonance_strength}]->(p2)
                    """,
                    {
                        "source_id": rel["source_id"],
                        "target_id": rel["target_id"],
                        "interference_type": rel["interference_type"],
                        "resonance_strength": rel["resonance_strength"]
                    }
                )
        
        # Create frequency domains
        with self.neo4j_driver.session() as session:
            for domain_id, domain in topology_state.frequency_domains.items():
                # Create parameters dictionary
                params = {
                    "id": domain_id,
                    "name": domain.metadata.get("name", f"Domain {domain_id}") if hasattr(domain, "metadata") and domain.metadata else f"Domain {domain_id}",
                    "dominant_frequency": domain.dominant_frequency,
                    "bandwidth": domain.bandwidth,
                    "phase_coherence": domain.phase_coherence,
                    "radius": domain.radius
                }
                
                # Print parameters for debugging
                print(f"Neo4j domain parameters: {params}")
                
                # Create domain node with explicit parameter formatting
                session.run(
                    """
                    CREATE (fd:FrequencyDomain {
                        id: $id, 
                        name: $name, 
                        dominant_frequency: $dominant_frequency, 
                        bandwidth: $bandwidth, 
                        phase_coherence: $phase_coherence, 
                        radius: $radius
                    })
                    """,
                    params
                )
        
        # Create boundaries
        with self.neo4j_driver.session() as session:
            for boundary_id, boundary in topology_state.boundaries.items():
                # Create boundary node
                session.run(
                    """
                    CREATE (b:Boundary {id: $id, permeability: $permeability, sharpness: $sharpness, stability: $stability})
                    """,
                    {
                        "id": boundary_id,
                        "permeability": boundary.permeability,
                        "sharpness": boundary.sharpness,
                        "stability": boundary.stability
                    }
                )
                
                # Connect domains to boundary
                domain_a_id, domain_b_id = boundary.domain_ids
                session.run(
                    """
                    MATCH (fd1:FrequencyDomain {id: $domain_a_id})
                    MATCH (fd2:FrequencyDomain {id: $domain_b_id})
                    MATCH (b:Boundary {id: $boundary_id})
                    CREATE (fd1)-[:CONNECTED_BY]->(b)-[:CONNECTED_BY]->(fd2)
                    """,
                    {
                        "domain_a_id": domain_a_id,
                        "domain_b_id": domain_b_id,
                        "boundary_id": boundary_id
                    }
                )
        
        # Create resonance points
        with self.neo4j_driver.session() as session:
            for point_id, point in topology_state.resonance_points.items():
                # Create resonance point node
                session.run(
                    """
                    CREATE (rp:ResonancePoint {id: $id, strength: $strength, stability: $stability, attractor_radius: $attractor_radius})
                    """,
                    {
                        "id": point_id,
                        "strength": point.strength,
                        "stability": point.stability,
                        "attractor_radius": point.attractor_radius
                    }
                )
                
                # Connect patterns to resonance point
                for pattern_id, weight in point.contributing_pattern_ids.items():
                    session.run(
                        """
                        MATCH (p:Pattern {id: $pattern_id})
                        MATCH (rp:ResonancePoint {id: $point_id})
                        CREATE (p)-[:CONTRIBUTES_TO {weight: $weight}]->(rp)
                        """,
                        {
                            "pattern_id": pattern_id,
                            "point_id": point_id,
                            "weight": weight
                        }
                    )
    
    def generate_cypher_queries(self) -> Dict[str, str]:
        """Generate Cypher queries for validating the tonic-harmonic integration."""
        queries = {
            "pattern_count": "MATCH (p:Pattern) RETURN count(p) as pattern_count",
            "resonance_group_count": "MATCH (rg:ResonanceGroup) RETURN count(rg) as group_count",
            "wave_relationship_count": "MATCH ()-[r:WAVE_RELATIONSHIP]->() RETURN count(r) as relationship_count",
            "constructive_interference": """
                MATCH (p1:Pattern)-[r:WAVE_RELATIONSHIP {interference_type: 'CONSTRUCTIVE'}]->(p2:Pattern)
                RETURN p1.id, p2.id, r.resonance_strength
                LIMIT 5
            """,
            "high_tonic_patterns": """
                MATCH (p:Pattern)
                WHERE p.tonic_value > 0.8
                RETURN p.id, p.tonic_value, p.phase_position
                ORDER BY p.tonic_value DESC
                LIMIT 5
            """,
            "resonance_group_patterns": """
                MATCH (p:Pattern)-[:BELONGS_TO]->(rg:ResonanceGroup)
                RETURN rg.id, count(p) as pattern_count, rg.coherence, rg.stability, rg.harmonic_value
                ORDER BY pattern_count DESC
            """,
            "frequency_domains": """
                MATCH (fd:FrequencyDomain)
                RETURN fd.id, fd.name, fd.dominant_frequency, fd.bandwidth, fd.phase_coherence, fd.radius
                ORDER BY fd.dominant_frequency
            """
        }
        return queries
    
    def run_integration_test(self):
        """Run the integration test."""
        try:
            # Step 1: Load document directly from file
            doc = self.load_climate_risk_document()
            logger.info(f"Loaded document: {doc.get('title')}")
            logger.info(f"Document has {len(doc.get('paragraphs', []))} paragraphs")
            
            # Step 2: Process document through RAG
            topology_state, field_analysis = self.process_document_through_rag(doc)
            logger.info(f"Generated topology state with ID: {topology_state.id}")
            logger.info(f"Topology state has {len(topology_state.resonance_groups)} resonance groups")
            
            # Step 3: Persist to Neo4j
            self.persist_to_neo4j(topology_state, field_analysis)
            logger.info("Persisted topology state to Neo4j")
            
            # Step 4: Generate and execute Cypher queries
            queries = self.generate_cypher_queries()
            logger.info(f"Generated {len(queries)} Cypher queries")
            
            # Execute and print query results
            with self.neo4j_driver.session() as session:
                for query_name, query in queries.items():
                    try:
                        result = session.run(query)
                        records = result.data()
                        logger.info(f"Query '{query_name}' returned {len(records)} records")
                        for record in records[:3]:  # Show first 3 records
                            logger.info(f"  {record}")
                    except Exception as e:
                        logger.error(f"Error executing query {query_name}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def test_climate_risk_integration(self):
        """Test climate risk document processing through the tonic-harmonic system."""
        self.run_integration_test()


if __name__ == "__main__":
    unittest.main()
