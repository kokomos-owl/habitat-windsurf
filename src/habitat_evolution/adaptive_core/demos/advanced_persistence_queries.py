#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Persistence Queries

This module implements advanced queries for analyzing persisted climate risk data:
1. Cross-dimensional queries between semantic relationships and pattern structures
2. Temporal pattern evolution tracking
3. Signature-based pattern matching using harmonic properties
4. Resonance center analysis across field states
"""

import os
import json
import logging
import uuid
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter, defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Habitat Evolution components
from src.habitat_evolution.adaptive_core.persistence.factory import create_repositories
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.connection_manager import ArangoDBConnectionManager

class AdvancedPersistenceQueries:
    """Advanced queries for analyzing persisted climate risk data."""
    
    def __init__(self):
        """Initialize with necessary components."""
        self.connection_manager = ArangoDBConnectionManager()
        self.db = self.connection_manager.get_db()
        
        # Create repositories
        self.repositories = create_repositories(self.db)
        
        # Output directory for analysis results
        self.output_dir = Path(__file__).parent / "analysis_results"
        self.output_dir.mkdir(exist_ok=True)
    
    def vector_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
            
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        return dot_product / (norm_vec1 * norm_vec2)
    
    def harmonic_similarity(self, harmonic1, harmonic2):
        """Calculate similarity between harmonic properties."""
        if not harmonic1 or not harmonic2:
            return 0.0
            
        # Extract properties
        freq1 = harmonic1.get("frequency", 0)
        amp1 = harmonic1.get("amplitude", 0)
        phase1 = harmonic1.get("phase", 0)
        
        freq2 = harmonic2.get("frequency", 0)
        amp2 = harmonic2.get("amplitude", 0)
        phase2 = harmonic2.get("phase", 0)
        
        # Calculate weighted similarity
        freq_sim = 1.0 - abs(freq1 - freq2)
        amp_sim = 1.0 - abs(amp1 - amp2)
        
        # Phase similarity needs to account for circular nature
        phase_diff = abs(phase1 - phase2)
        if phase_diff > 0.5:
            phase_diff = 1.0 - phase_diff
        phase_sim = 1.0 - phase_diff * 2
        
        # Weighted combination
        return 0.4 * freq_sim + 0.4 * amp_sim + 0.2 * phase_sim
    
    def get_all_patterns(self):
        """Get all patterns from the repository."""
        try:
            # Direct ArangoDB query since find_all might not be implemented
            query = """
            FOR p IN patterns
            RETURN p
            """
            cursor = self.db.aql.execute(query)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error retrieving patterns: {str(e)}")
            return []
    
    def get_all_relationships(self):
        """Get all relationships from the repository."""
        try:
            # Direct ArangoDB query
            query = """
            FOR r IN relationships
            RETURN r
            """
            cursor = self.db.aql.execute(query)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error retrieving relationships: {str(e)}")
            return []
    
    def get_all_field_states(self):
        """Get all field states from the repository."""
        try:
            # Direct ArangoDB query
            query = """
            FOR f IN field_states
            RETURN f
            """
            cursor = self.db.aql.execute(query)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error retrieving field states: {str(e)}")
            return []
    
    def get_all_topologies(self):
        """Get all topologies from the repository."""
        try:
            # Direct ArangoDB query
            query = """
            FOR t IN topologies
            RETURN t
            """
            cursor = self.db.aql.execute(query)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error retrieving topologies: {str(e)}")
            return []
    
    def get_pattern_by_id(self, pattern_id):
        """Get a pattern by its ID."""
        try:
            # Direct ArangoDB query
            query = """
            FOR p IN patterns
            FILTER p.id == @pattern_id
            RETURN p
            """
            cursor = self.db.aql.execute(query, bind_vars={"pattern_id": pattern_id})
            results = list(cursor)
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Error retrieving pattern: {str(e)}")
            return None
    
    def cross_dimensional_query(self):
        """
        Perform cross-dimensional queries between semantic relationships and pattern structures.
        
        This query identifies emergent properties by analyzing how patterns are connected
        through relationships and how those connections relate to field topologies.
        """
        logger.info("Performing cross-dimensional query analysis...")
        
        # Get all data
        patterns = self.get_all_patterns()
        relationships = self.get_all_relationships()
        field_states = self.get_all_field_states()
        topologies = self.get_all_topologies()
        
        logger.info(f"Retrieved {len(patterns)} patterns, {len(relationships)} relationships, " +
                   f"{len(field_states)} field states, and {len(topologies)} topologies")
        
        # Create pattern lookup
        pattern_lookup = {p["id"]: p for p in patterns}
        
        # Create field state lookup
        field_state_lookup = {f["id"]: f for f in field_states}
        
        # Create topology lookup by field_id
        topology_lookup = {t["field_id"]: t for t in topologies}
        
        # Build relationship graph
        relationship_graph = defaultdict(list)
        for rel in relationships:
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            
            if source_id and target_id:
                relationship_graph[source_id].append((target_id, rel))
                relationship_graph[target_id].append((source_id, rel))  # Bidirectional
        
        # Identify patterns that appear in resonance centers
        patterns_in_resonance = set()
        for field_id, field in field_state_lookup.items():
            resonance_centers = field.get("resonance_centers", [])
            for pattern in patterns:
                if pattern.get("name") in resonance_centers:
                    patterns_in_resonance.add(pattern["id"])
        
        # Find patterns that are both in resonance centers and have relationships
        cross_dimensional_insights = []
        
        for pattern_id in patterns_in_resonance:
            if pattern_id in relationship_graph:
                pattern = pattern_lookup.get(pattern_id)
                if not pattern:
                    continue
                    
                # Get related patterns
                related_patterns = []
                for related_id, rel in relationship_graph[pattern_id]:
                    related_pattern = pattern_lookup.get(related_id)
                    if related_pattern:
                        related_patterns.append({
                            "pattern": related_pattern,
                            "relationship": rel
                        })
                
                # Find field states where this pattern is a resonance center
                fields_with_pattern = []
                for field_id, field in field_state_lookup.items():
                    if pattern.get("name") in field.get("resonance_centers", []):
                        fields_with_pattern.append(field)
                
                # Create cross-dimensional insight
                insight = {
                    "pattern": pattern,
                    "related_patterns": related_patterns,
                    "fields_with_pattern": fields_with_pattern,
                    "relationship_count": len(related_patterns),
                    "field_count": len(fields_with_pattern)
                }
                
                cross_dimensional_insights.append(insight)
        
        # Sort insights by combined relationship and field count
        cross_dimensional_insights.sort(
            key=lambda x: x["relationship_count"] + x["field_count"],
            reverse=True
        )
        
        # Prepare results
        results = {
            "cross_dimensional_insights": cross_dimensional_insights,
            "patterns_in_resonance": len(patterns_in_resonance),
            "patterns_with_relationships": len(relationship_graph),
            "patterns_in_both": len([i for i in cross_dimensional_insights])
        }
        
        # Log key findings
        logger.info(f"Found {results['patterns_in_both']} patterns that appear in both resonance centers and relationships")
        
        if cross_dimensional_insights:
            top_insight = cross_dimensional_insights[0]
            logger.info(f"Top cross-dimensional pattern: {top_insight['pattern'].get('name')}")
            logger.info(f"  Related to {top_insight['relationship_count']} other patterns")
            logger.info(f"  Appears in {top_insight['field_count']} field states")
        
        # Save results
        with open(self.output_dir / "cross_dimensional_analysis.json", "w") as f:
            # Convert to serializable format
            serializable_results = {
                "patterns_in_resonance": results["patterns_in_resonance"],
                "patterns_with_relationships": results["patterns_with_relationships"],
                "patterns_in_both": results["patterns_in_both"],
                "cross_dimensional_insights": [
                    {
                        "pattern_name": insight["pattern"].get("name"),
                        "pattern_id": insight["pattern"].get("id"),
                        "relationship_count": insight["relationship_count"],
                        "field_count": insight["field_count"],
                        "related_pattern_names": [r["pattern"].get("name") for r in insight["related_patterns"]],
                        "field_names": [f.get("name") for f in insight["fields_with_pattern"]]
                    }
                    for insight in cross_dimensional_insights[:10]  # Top 10 for readability
                ]
            }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Cross-dimensional analysis saved to {self.output_dir / 'cross_dimensional_analysis.json'}")
        
        return results
    
    def temporal_pattern_evolution(self):
        """
        Track how patterns evolve over time, showing the emergence of new semantic structures.
        
        This analysis identifies patterns that have changed over time based on their
        timestamps and relationships.
        """
        logger.info("Analyzing temporal pattern evolution...")
        
        # Get all patterns and relationships
        patterns = self.get_all_patterns()
        relationships = self.get_all_relationships()
        
        # Group patterns by name
        patterns_by_name = defaultdict(list)
        for pattern in patterns:
            name = pattern.get("name")
            if name:
                patterns_by_name[name].append(pattern)
        
        # Sort each group by timestamp
        for name, pattern_group in patterns_by_name.items():
            pattern_group.sort(key=lambda p: p.get("timestamp", ""))
        
        # Identify patterns with multiple instances over time
        evolving_patterns = {}
        for name, pattern_group in patterns_by_name.items():
            if len(pattern_group) > 1:
                # Calculate evolution metrics
                first_pattern = pattern_group[0]
                last_pattern = pattern_group[-1]
                
                # Check for vector evolution
                vector_evolution = None
                if "vector" in first_pattern and "vector" in last_pattern:
                    similarity = self.vector_similarity(first_pattern["vector"], last_pattern["vector"])
                    vector_evolution = {
                        "initial_vector": first_pattern["vector"],
                        "final_vector": last_pattern["vector"],
                        "similarity": similarity,
                        "change": 1.0 - similarity
                    }
                
                # Check for harmonic evolution
                harmonic_evolution = None
                if "harmonic_properties" in first_pattern and "harmonic_properties" in last_pattern:
                    similarity = self.harmonic_similarity(
                        first_pattern["harmonic_properties"],
                        last_pattern["harmonic_properties"]
                    )
                    harmonic_evolution = {
                        "initial_harmonics": first_pattern["harmonic_properties"],
                        "final_harmonics": last_pattern["harmonic_properties"],
                        "similarity": similarity,
                        "change": 1.0 - similarity
                    }
                
                # Calculate time span
                time_span = None
                if "timestamp" in first_pattern and "timestamp" in last_pattern:
                    try:
                        first_time = datetime.fromisoformat(first_pattern["timestamp"].replace("Z", "+00:00"))
                        last_time = datetime.fromisoformat(last_pattern["timestamp"].replace("Z", "+00:00"))
                        time_span = (last_time - first_time).total_seconds()
                    except:
                        pass
                
                evolving_patterns[name] = {
                    "name": name,
                    "instances": len(pattern_group),
                    "first_instance": first_pattern,
                    "last_instance": last_pattern,
                    "vector_evolution": vector_evolution,
                    "harmonic_evolution": harmonic_evolution,
                    "time_span": time_span
                }
        
        # Sort evolving patterns by degree of change
        sorted_patterns = []
        for name, data in evolving_patterns.items():
            change_score = 0
            if data["vector_evolution"]:
                change_score += data["vector_evolution"]["change"]
            if data["harmonic_evolution"]:
                change_score += data["harmonic_evolution"]["change"]
            
            sorted_patterns.append((name, change_score, data))
        
        sorted_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Prepare results
        results = {
            "evolving_patterns": [data for _, _, data in sorted_patterns],
            "total_patterns": len(patterns),
            "evolving_pattern_count": len(evolving_patterns),
            "evolution_rate": len(evolving_patterns) / max(1, len(patterns_by_name))
        }
        
        # Log key findings
        logger.info(f"Found {results['evolving_pattern_count']} evolving patterns out of {len(patterns_by_name)} unique patterns")
        logger.info(f"Pattern evolution rate: {results['evolution_rate']:.2f}")
        
        if sorted_patterns:
            top_evolving = sorted_patterns[0][2]
            logger.info(f"Most evolved pattern: {top_evolving['name']} with {top_evolving['instances']} instances")
            if top_evolving["vector_evolution"]:
                logger.info(f"  Vector change: {top_evolving['vector_evolution']['change']:.2f}")
            if top_evolving["harmonic_evolution"]:
                logger.info(f"  Harmonic change: {top_evolving['harmonic_evolution']['change']:.2f}")
        
        # Save results
        with open(self.output_dir / "temporal_evolution_analysis.json", "w") as f:
            # Convert to serializable format
            serializable_results = {
                "total_patterns": results["total_patterns"],
                "evolving_pattern_count": results["evolving_pattern_count"],
                "evolution_rate": results["evolution_rate"],
                "top_evolving_patterns": [
                    {
                        "name": data["name"],
                        "instances": data["instances"],
                        "vector_change": data["vector_evolution"]["change"] if data["vector_evolution"] else None,
                        "harmonic_change": data["harmonic_evolution"]["change"] if data["harmonic_evolution"] else None,
                        "time_span_seconds": data["time_span"]
                    }
                    for _, _, data in sorted_patterns[:10]  # Top 10 for readability
                ]
            }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Temporal evolution analysis saved to {self.output_dir / 'temporal_evolution_analysis.json'}")
        
        return results
    
    def signature_based_pattern_matching(self):
        """
        Identify similar patterns based on their harmonic signatures,
        even when their semantic content differs.
        """
        logger.info("Performing signature-based pattern matching...")
        
        # Get all patterns
        patterns = self.get_all_patterns()
        
        # Filter patterns with harmonic properties and vectors
        valid_patterns = [
            p for p in patterns 
            if "harmonic_properties" in p and "vector" in p and p["vector"]
        ]
        
        logger.info(f"Found {len(valid_patterns)} patterns with harmonic properties and vectors")
        
        # Calculate similarity matrix
        harmonic_clusters = []
        processed_ids = set()
        
        for i, pattern in enumerate(valid_patterns):
            if pattern["id"] in processed_ids:
                continue
                
            # Find similar patterns
            similar_patterns = []
            
            for other in valid_patterns:
                if pattern["id"] == other["id"]:
                    continue
                    
                # Calculate harmonic similarity
                harmonic_sim = self.harmonic_similarity(
                    pattern["harmonic_properties"],
                    other["harmonic_properties"]
                )
                
                # Calculate vector similarity
                vector_sim = self.vector_similarity(pattern["vector"], other["vector"])
                
                # Combined similarity score
                combined_sim = 0.7 * harmonic_sim + 0.3 * vector_sim
                
                if combined_sim > 0.8:  # High similarity threshold
                    similar_patterns.append({
                        "pattern": other,
                        "harmonic_similarity": harmonic_sim,
                        "vector_similarity": vector_sim,
                        "combined_similarity": combined_sim
                    })
            
            # If we found similar patterns, create a cluster
            if similar_patterns:
                # Sort by similarity
                similar_patterns.sort(key=lambda x: x["combined_similarity"], reverse=True)
                
                cluster = {
                    "seed_pattern": pattern,
                    "similar_patterns": similar_patterns,
                    "cluster_size": len(similar_patterns) + 1,
                    "average_harmonic_similarity": sum(p["harmonic_similarity"] for p in similar_patterns) / len(similar_patterns),
                    "average_vector_similarity": sum(p["vector_similarity"] for p in similar_patterns) / len(similar_patterns),
                    "semantic_diversity": len(set(p["pattern"].get("name", "") for p in similar_patterns)) / max(1, len(similar_patterns))
                }
                
                harmonic_clusters.append(cluster)
                
                # Mark all patterns in this cluster as processed
                processed_ids.add(pattern["id"])
                for p in similar_patterns:
                    processed_ids.add(p["pattern"]["id"])
        
        # Sort clusters by size
        harmonic_clusters.sort(key=lambda x: x["cluster_size"], reverse=True)
        
        # Prepare results
        results = {
            "harmonic_clusters": harmonic_clusters,
            "total_patterns": len(patterns),
            "valid_patterns": len(valid_patterns),
            "cluster_count": len(harmonic_clusters),
            "patterns_in_clusters": len(processed_ids),
            "cluster_coverage": len(processed_ids) / max(1, len(valid_patterns))
        }
        
        # Log key findings
        logger.info(f"Found {results['cluster_count']} harmonic clusters covering {results['patterns_in_clusters']} patterns")
        logger.info(f"Cluster coverage: {results['cluster_coverage']:.2f}")
        
        if harmonic_clusters:
            top_cluster = harmonic_clusters[0]
            logger.info(f"Largest cluster: {top_cluster['cluster_size']} patterns")
            logger.info(f"  Seed pattern: {top_cluster['seed_pattern'].get('name')}")
            logger.info(f"  Average harmonic similarity: {top_cluster['average_harmonic_similarity']:.2f}")
            logger.info(f"  Semantic diversity: {top_cluster['semantic_diversity']:.2f}")
        
        # Save results
        with open(self.output_dir / "signature_based_matching.json", "w") as f:
            # Convert to serializable format
            serializable_results = {
                "total_patterns": results["total_patterns"],
                "valid_patterns": results["valid_patterns"],
                "cluster_count": results["cluster_count"],
                "patterns_in_clusters": results["patterns_in_clusters"],
                "cluster_coverage": results["cluster_coverage"],
                "top_clusters": [
                    {
                        "seed_pattern_name": cluster["seed_pattern"].get("name"),
                        "seed_pattern_id": cluster["seed_pattern"].get("id"),
                        "cluster_size": cluster["cluster_size"],
                        "average_harmonic_similarity": cluster["average_harmonic_similarity"],
                        "average_vector_similarity": cluster["average_vector_similarity"],
                        "semantic_diversity": cluster["semantic_diversity"],
                        "similar_pattern_names": [
                            p["pattern"].get("name") for p in cluster["similar_patterns"][:5]  # Top 5 for readability
                        ]
                    }
                    for cluster in harmonic_clusters[:10]  # Top 10 for readability
                ]
            }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Signature-based matching analysis saved to {self.output_dir / 'signature_based_matching.json'}")
        
        return results
    
    def resonance_center_analysis(self):
        """
        Analyze resonance centers across field states to identify
        core concepts in the climate risk domain.
        """
        logger.info("Analyzing resonance centers across field states...")
        
        # Get all field states
        field_states = self.get_all_field_states()
        
        # Extract all resonance centers
        all_centers = []
        for field in field_states:
            centers = field.get("resonance_centers", [])
            all_centers.extend(centers)
        
        # Count frequency of each resonance center
        center_counts = Counter(all_centers)
        
        # Get top resonance centers
        top_centers = center_counts.most_common(20)
        
        # Analyze co-occurrence of resonance centers
        co_occurrence = defaultdict(Counter)
        
        for field in field_states:
            centers = field.get("resonance_centers", [])
            for i, center1 in enumerate(centers):
                for center2 in centers[i+1:]:
                    co_occurrence[center1][center2] += 1
                    co_occurrence[center2][center1] += 1
        
        # Find top co-occurring pairs
        top_pairs = []
        for center, co_centers in co_occurrence.items():
            for co_center, count in co_centers.most_common(5):
                top_pairs.append((center, co_center, count))
        
        # Sort by co-occurrence count
        top_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Analyze field state topology for resonance centers
        center_topology = defaultdict(list)
        
        for field in field_states:
            field_id = field.get("id")
            centers = field.get("resonance_centers", [])
            
            # Get topology for this field
            try:
                query = """
                FOR t IN topologies
                FILTER t.field_id == @field_id
                RETURN t
                """
                cursor = self.db.aql.execute(query, bind_vars={"field_id": field_id})
                topologies = list(cursor)
                
                if topologies:
                    topology = topologies[0]
                    edges = topology.get("edges", [])
                    
                    # Map edges to resonance centers
                    for edge in edges:
                        source = edge.get("source")
                        target = edge.get("target")
                        weight = edge.get("weight", 0.5)
                        
                        if source in centers and target in centers:
                            center_topology[source].append((target, weight, field_id))
                            center_topology[target].append((source, weight, field_id))
            except Exception as e:
                logger.error(f"Error retrieving topology for field {field_id}: {str(e)}")
        
        # Calculate centrality of resonance centers in topology
        centrality = {}
        for center, connections in center_topology.items():
            # Simple degree centrality
            degree = len(connections)
            # Weighted centrality
            weighted = sum(weight for _, weight, _ in connections)
            
            centrality[center] = {
                "degree": degree,
                "weighted": weighted,
                "connections": connections
            }
        
        # Sort centers by centrality
        sorted_centrality = sorted(
            centrality.items(),
            key=lambda x: x[1]["weighted"],
            reverse=True
        )
        
        # Prepare results
        results = {
            "resonance_centers": {
                "total_centers": len(center_counts),
                "top_centers": top_centers,
                "center_counts": dict(center_counts)
            },
            "co_occurrence": {
                "top_pairs": top_pairs[:20]  # Top 20 co-occurring pairs
            },
            "topology": {
                "centers_in_topology": len(center_topology),
                "top_central_centers": sorted_centrality[:10]  # Top 10 central centers
            }
        }
        
        # Log key findings
        logger.info(f"Found {results['resonance_centers']['total_centers']} unique resonance centers across {len(field_states)} field states")
        
        if top_centers:
            logger.info("Top resonance centers:")
            for center, count in top_centers[:5]:
                logger.info(f"  {center}: {count} occurrences")
        
        if top_pairs:
            logger.info("Top co-occurring center pairs:")
            for center1, center2, count in top_pairs[:5]:
                logger.info(f"  {center1} - {center2}: {count} co-occurrences")
        
        if sorted_centrality:
            logger.info("Most central resonance centers:")
            for center, data in sorted_centrality[:5]:
                logger.info(f"  {center}: weighted centrality {data['weighted']:.2f}, {data['degree']} connections")
        
        # Save results
        with open(self.output_dir / "resonance_center_analysis.json", "w") as f:
            # Convert to serializable format
            serializable_results = {
                "resonance_centers": {
                    "total_centers": results["resonance_centers"]["total_centers"],
                    "top_centers": [
                        {"center": center, "count": count}
                        for center, count in results["resonance_centers"]["top_centers"]
                    ]
                },
                "co_occurrence": {
                    "top_pairs": [
                        {"center1": c1, "center2": c2, "count": count}
                        for c1, c2, count in results["co_occurrence"]["top_pairs"]
                    ]
                },
                "topology": {
                    "centers_in_topology": results["topology"]["centers_in_topology"],
                    "top_central_centers": [
                        {
                            "center": center,
                            "weighted_centrality": data["weighted"],
                            "degree": data["degree"],
                            "top_connections": [
                                {"connected_to": target, "weight": weight}
                                for target, weight, _ in data["connections"][:5]  # Top 5 connections
                            ]
                        }
                        for center, data in results["topology"]["top_central_centers"]
                    ]
                }
            }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Resonance center analysis saved to {self.output_dir / 'resonance_center_analysis.json'}")
        
        return results
    
    def run(self):
        """Run all advanced queries."""
        logger.info("Starting Advanced Persistence Queries")
        
        try:
            # Run cross-dimensional query
            self.cross_dimensional_query()
            
            # Run temporal pattern evolution analysis
            self.temporal_pattern_evolution()
            
            # Run signature-based pattern matching
            self.signature_based_pattern_matching()
            
            # Run resonance center analysis
            self.resonance_center_analysis()
            
            logger.info("All advanced queries completed successfully")
            
        except Exception as e:
            logger.error(f"Error running advanced queries: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    queries = AdvancedPersistenceQueries()
    queries.run()
