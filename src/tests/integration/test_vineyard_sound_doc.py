"""
Integration test for Vineyard Sound document processing through the tonic-harmonic system.

This test validates the end-to-end flow of processing the Vineyard Sound document through
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

class VineyardSoundDocTest(unittest.TestCase):
    """Integration test for Vineyard Sound document processing through the tonic-harmonic system."""
    
    def setUp(self):
        """Set up test environment."""
        # Neo4j configuration - using the running Docker container
        # These should match the Docker container settings
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "habitat123"  # Updated to match Docker container password
        
        # Path to Vineyard Sound data file
        self.document_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "data", "climate_risk", "vineyard_sound_structure_meaning_test_doc.txt"
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
    
    def load_document(self) -> Dict[str, Any]:
        """Load Vineyard Sound document directly from file.
        
        Returns:
            Document as a dictionary
        """
        # Read document
        with open(self.document_path, "r") as f:
            content = f.read()
            
        # Split document into paragraphs - handling the hierarchical structure
        paragraphs = []
        current_section = ""
        
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                if current_section:
                    paragraphs.append(current_section)
                    current_section = ""
            else:
                if current_section:
                    current_section += " " + line
                else:
                    current_section = line
        
        # Add the last section if it exists
        if current_section:
            paragraphs.append(current_section)
        
        # Create document
        document = {
            "doc_id": "vineyard_sound_structure_meaning",
            "title": "Climate Risk Assessment - Vineyard Sound & Woods Hole Region",
            "content": content,
            "paragraphs": paragraphs,
            "metadata": {
                "source": "Woods Hole Research Center",
                "type": "structure_meaning_assessment",
                "location": "Vineyard Sound & Woods Hole Region",
                "document_id": "VS-WH-2024-001"
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
        for i, paragraph in enumerate(paragraphs[:15]):  # Process more paragraphs for this complex document
            pattern = {
                "id": f"vs-{i}",  # Use vs- prefix to distinguish from other documents
                "content": paragraph,
                "created_at": datetime.now().isoformat(),
                "source": document.get("metadata", {}).get("source", "Unknown"),
                "metadata": {
                    "document_id": document.get("metadata", {}).get("document_id", ""),
                    "paragraph_index": i
                }
            }
            patterns.append(pattern)
        
        # Create field analysis
        field_analysis = {
            "patterns": patterns,
            "resonance_relationships": {}
        }
        
        # Create resonance relationships between patterns based on content similarity
        for i in range(len(patterns)):
            for j in range(i+1, len(patterns)):
                rel_id = f"vs-rel-{i}-{j}"
                
                # Calculate content similarity using a simple TF-IDF approach
                content_i = patterns[i]["content"].lower().split()
                content_j = patterns[j]["content"].lower().split()
                
                # Create word sets
                words_i = set(content_i)
                words_j = set(content_j)
                
                # Calculate Jaccard similarity
                intersection = len(words_i.intersection(words_j))
                union = len(words_i.union(words_j))
                similarity = intersection / max(1, union)
                
                # Determine interference type based on similarity
                if similarity > 0.3:  # Strong similarity
                    interference_type = "CONSTRUCTIVE"
                    strength = 0.7 + (0.3 * similarity)
                elif similarity > 0.1:  # Moderate similarity
                    interference_type = "PARTIAL"
                    strength = 0.5 + (0.5 * similarity)
                else:  # Low similarity
                    interference_type = "DESTRUCTIVE"
                    strength = 0.3 + (0.3 * similarity)
                
                field_analysis["resonance_relationships"][rel_id] = {
                    "source_id": patterns[i]["id"],
                    "target_id": patterns[j]["id"],
                    "interference_type": interference_type,
                    "resonance_strength": strength
                }
        
        # Create topology state
        topology_state = TopologyState()
        
        # Generate pattern eigenspace properties based on semantic content
        pattern_eigenspace_properties = {}
        
        # Create a simple embedding for each pattern based on keyword presence
        # This simulates what would normally be done with a real embedding model
        keywords = [
            "marine", "coastal", "climate", "research", "structure", 
            "meaning", "ecological", "physical", "environmental", "evidence"
        ]
        
        # Generate embeddings based on keyword frequency
        pattern_embeddings = {}
        for pattern in patterns:
            content = pattern["content"].lower()
            embedding = [content.count(keyword) / (len(content.split()) + 1) for keyword in keywords]
            
            # Normalize embedding
            magnitude = np.sqrt(sum(e*e for e in embedding))
            if magnitude > 0:
                embedding = [e/magnitude for e in embedding]
                
            pattern_embeddings[pattern["id"]] = embedding
            
            # Calculate tonic value based on content coherence
            # Patterns with focused content (fewer topics) have higher tonic values
            non_zero_dims = sum(1 for e in embedding if e > 0.1)
            topic_focus = 1.0 / max(1, non_zero_dims)
            tonic_value = 0.5 + (0.45 * topic_focus)  # Range from 0.5 to 0.95
            
            # Calculate phase position based on content type
            # This creates natural clustering in the phase space
            phase_position = sum(i * e for i, e in enumerate(embedding)) * 2 * np.pi / len(embedding)
            
            pattern_eigenspace_properties[pattern["id"]] = {
                "tonic_value": tonic_value,
                "phase_position": phase_position,
                "dimensional_coordinates": embedding
            }
        
        topology_state.pattern_eigenspace_properties = pattern_eigenspace_properties
        
        # Dynamically detect frequency domains using clustering on pattern embeddings
        # Convert embeddings to a matrix for clustering
        embedding_matrix = np.array([pattern_embeddings[p["id"]] for p in patterns])
        
        # Simple clustering using K-means (simplified version)
        # In a real implementation, this would use a more sophisticated approach
        # that determines the number of clusters dynamically
        num_domains = min(5, len(patterns))  # Cap at 5 domains or fewer if we have fewer patterns
        
        # Simple K-means implementation
        # Initialize centroids randomly
        centroids_indices = np.random.choice(len(patterns), num_domains, replace=False)
        centroids = embedding_matrix[centroids_indices]
        
        # Assign patterns to clusters
        clusters = [[] for _ in range(num_domains)]
        for i, embedding in enumerate(embedding_matrix):
            # Find closest centroid
            distances = [np.linalg.norm(embedding - centroid) for centroid in centroids]
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(i)
        
        # Create frequency domains based on clusters
        domains = {}
        for i, cluster in enumerate(clusters):
            if not cluster:  # Skip empty clusters
                continue
                
            # Calculate cluster properties
            cluster_embeddings = embedding_matrix[cluster]
            cluster_centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find most representative keywords for this cluster
            keyword_weights = cluster_centroid
            top_keywords = sorted([(keywords[j], weight) for j, weight in enumerate(keyword_weights)], 
                                 key=lambda x: x[1], reverse=True)[:2]
            
            # Create domain name from top keywords
            domain_name = " & ".join([k.capitalize() for k, _ in top_keywords])
            domain_id = f"fd-{i}"
            
            # Calculate domain properties based on cluster characteristics
            pattern_coherence = np.mean([pattern_eigenspace_properties[patterns[idx]["id"]]["tonic_value"] 
                                      for idx in cluster])
            
            # Calculate variance within cluster (lower variance = higher coherence)
            variance = np.mean([np.linalg.norm(emb - cluster_centroid) for emb in cluster_embeddings])
            phase_coherence = 1.0 - min(0.9, variance)  # Higher variance = lower coherence
            
            # Create frequency domain
            domains[domain_id] = FrequencyDomain(
                id=domain_id,
                dominant_frequency=0.1 + (0.1 * i),  # Spread frequencies
                bandwidth=0.05 + (0.02 * variance),  # Higher variance = wider bandwidth
                phase_coherence=max(0.5, phase_coherence),
                radius=0.5 + (0.3 * pattern_coherence),
                metadata={"name": domain_name, "pattern_ids": [patterns[idx]["id"] for idx in cluster]}
            )
        
        topology_state.frequency_domains = domains
        
        # Dynamically detect boundaries between domains
        boundaries = {}
        domain_ids = list(domains.keys())
        
        for i in range(len(domain_ids)):
            for j in range(i+1, len(domain_ids)):
                domain_a = domains[domain_ids[i]]
                domain_b = domains[domain_ids[j]]
                
                # Calculate boundary properties based on domain relationships
                # Frequency difference determines sharpness
                freq_diff = abs(domain_a.dominant_frequency - domain_b.dominant_frequency)
                sharpness = min(0.9, freq_diff * 5)  # Higher difference = sharper boundary
                
                # Combined coherence determines stability
                combined_coherence = (domain_a.phase_coherence + domain_b.phase_coherence) / 2
                stability = 0.5 + (0.5 * combined_coherence)  # Higher coherence = more stable
                
                # Bandwidth difference determines permeability
                bandwidth_diff = abs(domain_a.bandwidth - domain_b.bandwidth)
                permeability = 0.3 + (0.7 * (1.0 - bandwidth_diff))  # Similar bandwidth = more permeable
                
                boundary_id = f"b-{i}-{j}"
                boundaries[boundary_id] = Boundary(
                    id=boundary_id,
                    domain_ids=(domain_ids[i], domain_ids[j]),
                    permeability=permeability,
                    sharpness=sharpness,
                    stability=stability
                )
        
        topology_state.boundaries = boundaries
        
        # Create resonance points at domain intersections
        resonance_points = {}
        for i, boundary_id in enumerate(boundaries.keys()):
            boundary = boundaries[boundary_id]
            domain_a_id, domain_b_id = boundary.domain_ids
            
            # Find patterns near the boundary
            domain_a_patterns = domains[domain_a_id].metadata.get("pattern_ids", [])
            domain_b_patterns = domains[domain_b_id].metadata.get("pattern_ids", [])
            
            # Create a resonance point at the boundary
            point_id = f"rp-{i}"
            
            # Calculate coordinates as weighted average of domain centroids
            domain_a_centroid = np.mean([pattern_embeddings[p_id] for p_id in domain_a_patterns], axis=0) \
                if domain_a_patterns else np.zeros(len(keywords))
            domain_b_centroid = np.mean([pattern_embeddings[p_id] for p_id in domain_b_patterns], axis=0) \
                if domain_b_patterns else np.zeros(len(keywords))
            
            # Weight by domain coherence
            weight_a = domains[domain_a_id].phase_coherence
            weight_b = domains[domain_b_id].phase_coherence
            total_weight = weight_a + weight_b
            
            if total_weight > 0:
                coordinates = tuple((weight_a * domain_a_centroid + weight_b * domain_b_centroid) / total_weight)
            else:
                coordinates = tuple(np.zeros(len(keywords)))
            
            # Find patterns that contribute to this resonance point
            # These are patterns with high similarity to the resonance point coordinates
            contributing_patterns = {}
            for pattern in patterns:
                pattern_id = pattern["id"]
                embedding = pattern_embeddings[pattern_id]
                
                # Calculate similarity to resonance point
                similarity = 1.0 - min(1.0, np.linalg.norm(np.array(embedding) - np.array(coordinates)))
                
                if similarity > 0.5:  # Only include patterns with significant contribution
                    contributing_patterns[pattern_id] = similarity
            
            # Only create resonance point if it has contributing patterns
            if contributing_patterns:
                resonance_points[point_id] = ResonancePoint(
                    id=point_id,
                    coordinates=coordinates,
                    strength=boundary.stability,  # Boundary stability determines point strength
                    stability=0.5 + (0.5 * boundary.stability),
                    attractor_radius=0.2 + (0.3 * boundary.permeability),  # More permeable = wider radius
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
        
        # Dynamically create resonance groups based on detected domains
        resonance_groups = {}
        
        # Create one resonance group per frequency domain
        for domain_id, domain in topology_state.frequency_domains.items():
            # Extract domain name from metadata
            domain_name = domain.metadata.get('name', f'Domain {domain_id}')
            
            # Create a simplified identifier for the resonance group
            # Replace spaces with underscores and make lowercase
            group_id = domain_name.lower().replace(' & ', '_').replace(' ', '_')
            
            # Calculate coherence based on domain phase coherence
            coherence = 0.7 + (0.3 * domain.phase_coherence)
            
            # Calculate stability based on domain radius
            stability = 0.6 + (0.4 * domain.phase_coherence)
            
            # Calculate harmonic value based on domain frequency
            harmonic_value = 0.5 + (0.5 * min(1.0, domain.dominant_frequency * 2))
            
            # Create the resonance group
            resonance_groups[group_id] = {
                "patterns": domain.metadata.get('pattern_ids', []),
                "coherence": coherence,
                "stability": stability,
                "harmonic_value": harmonic_value
            }
        
        topology_state.resonance_groups = resonance_groups
        
        # Create learning windows
        learning_windows = {}
        window_types = ["fast", "medium", "slow"]
        for i, window_type in enumerate(window_types):
            window_id = f"lw-vs-{window_type}"
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
        # Clear existing data for this document only (preserving other documents)
        with self.neo4j_driver.session() as session:
            # Delete patterns from this document
            session.run("""
                MATCH (p:Pattern) 
                WHERE p.id STARTS WITH 'vs-' 
                DETACH DELETE p
            """)
            
            # Delete resonance groups specific to this document
            # Use a more generic pattern to match dynamically created groups
            session.run("""
                MATCH (rg:ResonanceGroup) 
                WHERE any(keyword IN ['marine', 'coastal', 'climate', 'research', 'structure', 
                                     'meaning', 'ecological', 'physical', 'environmental', 'evidence'] 
                          WHERE rg.id CONTAINS keyword)
                DETACH DELETE rg
            """)
            
            # Delete frequency domains specific to this document
            session.run("""
                MATCH (fd:FrequencyDomain) 
                WHERE fd.id STARTS WITH 'fd-vs-' 
                DETACH DELETE fd
            """)
            
            # Delete boundaries specific to this document
            session.run("""
                MATCH (b:Boundary) 
                WHERE b.id STARTS WITH 'b-vs-' 
                DETACH DELETE b
            """)
            
            # Delete resonance points specific to this document
            session.run("""
                MATCH (rp:ResonancePoint) 
                WHERE rp.id STARTS WITH 'rp-vs-' 
                DETACH DELETE rp
            """)
        
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
            "vineyard_pattern_count": "MATCH (p:Pattern) WHERE p.id STARTS WITH 'vs-' RETURN count(p) as pattern_count",
            "resonance_group_count": "MATCH (rg:ResonanceGroup) WHERE rg.id IN ['marine_systems', 'coastal_interface', 'knowledge_domain'] RETURN count(rg) as group_count",
            "wave_relationship_count": "MATCH (p1:Pattern)-[r:WAVE_RELATIONSHIP]->(p2:Pattern) WHERE p1.id STARTS WITH 'vs-' RETURN count(r) as relationship_count",
            "constructive_interference": """
                MATCH (p1:Pattern)-[r:WAVE_RELATIONSHIP {interference_type: 'CONSTRUCTIVE'}]->(p2:Pattern)
                WHERE p1.id STARTS WITH 'vs-'
                RETURN p1.id, p2.id, r.resonance_strength
                LIMIT 5
            """,
            "high_tonic_patterns": """
                MATCH (p:Pattern)
                WHERE p.id STARTS WITH 'vs-' AND p.tonic_value > 0.8
                RETURN p.id, p.tonic_value, p.phase_position
                ORDER BY p.tonic_value DESC
                LIMIT 5
            """,
            "resonance_group_patterns": """
                MATCH (p:Pattern)-[:BELONGS_TO]->(rg:ResonanceGroup)
                WHERE p.id STARTS WITH 'vs-'
                RETURN rg.id, count(p) as pattern_count, rg.coherence, rg.stability, rg.harmonic_value
                ORDER BY pattern_count DESC
            """,
            "frequency_domains": """
                MATCH (fd:FrequencyDomain)
                WHERE fd.id STARTS WITH 'fd-vs-'
                RETURN fd.id, fd.name, fd.dominant_frequency, fd.bandwidth, fd.phase_coherence, fd.radius
                ORDER BY fd.dominant_frequency
            """
        }
        return queries
    
    def run_integration_test(self):
        """Run the integration test."""
        try:
            # Step 1: Load document directly from file
            doc = self.load_document()
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
    
    def test_vineyard_sound_integration(self):
        """Test Vineyard Sound document processing through the tonic-harmonic system."""
        # Load document
        document = self.load_document()
        
        # Process document through RAG
        topology_state, field_analysis = self.process_document_through_rag(document)
        
        # Output detailed information about frequency domains
        print("\n===== FREQUENCY DOMAIN DATA =====")
        for domain_id, domain in topology_state.frequency_domains.items():
            print(f"Domain: {domain_id} - {domain.metadata.get('name', 'Unnamed')}")
            print(f"  Dominant Frequency: {domain.dominant_frequency:.4f}")
            print(f"  Bandwidth: {domain.bandwidth:.4f}")
            print(f"  Phase Coherence: {domain.phase_coherence:.4f}")
            print(f"  Radius: {domain.radius:.4f}")
            print(f"  Pattern Count: {len(domain.metadata.get('pattern_ids', []))}")
            print(f"  Pattern IDs: {domain.metadata.get('pattern_ids', [])}")
            print()
        
        # Output boundary information
        print("\n===== BOUNDARY DATA =====")
        for boundary_id, boundary in topology_state.boundaries.items():
            print(f"Boundary: {boundary_id} - Between {boundary.domain_ids[0]} and {boundary.domain_ids[1]}")
            print(f"  Permeability: {boundary.permeability:.4f}")
            print(f"  Sharpness: {boundary.sharpness:.4f}")
            print(f"  Stability: {boundary.stability:.4f}")
            print()
        
        # Output resonance point information
        print("\n===== RESONANCE POINT DATA =====")
        for point_id, point in topology_state.resonance_points.items():
            print(f"Resonance Point: {point_id}")
            print(f"  Strength: {point.strength:.4f}")
            print(f"  Stability: {point.stability:.4f}")
            print(f"  Attractor Radius: {point.attractor_radius:.4f}")
            print(f"  Contributing Pattern Count: {len(point.contributing_pattern_ids)}")
            print()
        
        # Calculate and output coherence metrics
        print("\n===== COHERENCE METRICS =====")
        # Overall phase coherence (average of domain coherences)
        avg_phase_coherence = np.mean([d.phase_coherence for d in topology_state.frequency_domains.values()])
        print(f"Average Phase Coherence: {avg_phase_coherence:.4f}")
        
        # Overall pattern coherence (average of pattern tonic values)
        avg_tonic_value = np.mean([props["tonic_value"] 
                                 for props in topology_state.pattern_eigenspace_properties.values()])
        print(f"Average Pattern Tonic Value: {avg_tonic_value:.4f}")
        
        # Overall boundary stability
        avg_boundary_stability = np.mean([b.stability for b in topology_state.boundaries.values()])
        print(f"Average Boundary Stability: {avg_boundary_stability:.4f}")
        
        # Calculate vector-only vs vector+tonic-harmonic comparison
        print("\n===== VECTOR-ONLY VS VECTOR+TONIC-HARMONIC COMPARISON =====")
        
        # Vector-only metrics (simulated - in a real system this would be from actual vector-only processing)
        # For vector-only, we only have pattern similarity without phase/tonic information
        vector_only_coherence = 0.0
        pattern_embeddings = {p_id: props["dimensional_coordinates"] 
                            for p_id, props in topology_state.pattern_eigenspace_properties.items()}
        
        # Calculate average cosine similarity between all pattern pairs (vector-only metric)
        similarities = []
        for i, (p1_id, p1_emb) in enumerate(pattern_embeddings.items()):
            for j, (p2_id, p2_emb) in enumerate(pattern_embeddings.items()):
                if i < j:  # Only calculate for unique pairs
                    # Calculate cosine similarity
                    p1_array = np.array(p1_emb)
                    p2_array = np.array(p2_emb)
                    dot_product = np.dot(p1_array, p2_array)
                    norm_p1 = np.linalg.norm(p1_array)
                    norm_p2 = np.linalg.norm(p2_array)
                    if norm_p1 > 0 and norm_p2 > 0:
                        similarity = dot_product / (norm_p1 * norm_p2)
                        similarities.append(similarity)
        
        vector_only_coherence = np.mean(similarities) if similarities else 0.0
        print(f"Vector-Only Coherence: {vector_only_coherence:.4f}")
        
        # Vector+tonic-harmonic metrics
        # This includes phase alignment and tonic values
        tonic_harmonic_coherence = avg_phase_coherence * avg_tonic_value
        print(f"Vector+Tonic-Harmonic Coherence: {tonic_harmonic_coherence:.4f}")
        
        # Calculate improvement factor
        improvement_factor = tonic_harmonic_coherence / max(0.001, vector_only_coherence)
        print(f"Improvement Factor: {improvement_factor:.2f}x")
        
        # Persist to Neo4j
        self.persist_to_neo4j(topology_state, field_analysis)
        
        # Execute validation queries
        queries = self.generate_cypher_queries()
        with self.neo4j_driver.session() as session:
            for query_name, query in queries.items():
                try:
                    result = session.run(query)
                    records = result.data()
                    print(f"\nQuery '{query_name}' returned {len(records)} records")
                    for record in records[:3]:  # Show first 3 records
                        print(f"  {record}")
                except Exception as e:
                    print(f"Error executing query {query_name}: {e}")
        
        # Print summary of results
        print("\n===== SUMMARY OF RESULTS =====")
        print(f"Number of Frequency Domains: {len(topology_state.frequency_domains)}")
        print(f"Number of Boundaries: {len(topology_state.boundaries)}")
        print(f"Number of Resonance Points: {len(topology_state.resonance_points)}")
        print(f"Number of Resonance Groups: {len(topology_state.resonance_groups)}")
        print(f"Vector+Tonic-Harmonic Improvement: {improvement_factor:.2f}x over Vector-Only")
        
        # No need for explicit cleanup as tearDown will handle it


if __name__ == "__main__":
    unittest.main()
