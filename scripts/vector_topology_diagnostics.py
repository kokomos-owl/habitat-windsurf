#!/usr/bin/env python
"""
Diagnostic script for comparing vector-only vs. vector-plus-topology approaches in ArangoDB.

This script runs comparative tests to measure the effectiveness of:
1. Vector-only similarity matching
2. Vector + topological relationship analysis
3. Resonance detection with and without boundary data

Results are logged for analysis and visualization.
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from habitat_evolution.adaptive_core.persistence.arangodb.document_repository import Document, DocumentRepository
from habitat_evolution.adaptive_core.persistence.arangodb.domain_repository import Domain, DomainRepository
from habitat_evolution.adaptive_core.persistence.arangodb.predicate_repository import Predicate, PredicateRepository
from habitat_evolution.adaptive_core.persistence.arangodb.actant_repository import Actant, ActantRepository
from habitat_evolution.adaptive_core.persistence.arangodb.pattern_evolution_tracker import PatternEvolutionTracker
from habitat_evolution.adaptive_core.persistence.arangodb.connection import ArangoDBConnectionManager

# Disable any existing loggers to prevent errors
import logging
logging.disable(logging.CRITICAL)

class VectorTopologyDiagnostics:
    """
    Diagnostic tool for comparing vector-only vs. vector-plus-topology approaches.
    """
    
    def __init__(self, output_dir: str = "diagnostics_results"):
        """Initialize the diagnostic tool."""
        self.document_repo = DocumentRepository()
        self.domain_repo = DomainRepository()
        self.predicate_repo = PredicateRepository()
        self.actant_repo = ActantRepository()
        self.pattern_tracker = PatternEvolutionTracker()
        self.connection_manager = ArangoDBConnectionManager()
        
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Results storage
        self.results = {
            "vector_only": {},
            "vector_plus_topology": {},
            "resonance": {},
            "boundary_analysis": {}
        }
    
    def run_vector_only_similarity(self, query_vectors: List[List[float]], 
                                  threshold_range: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]) -> Dict[str, Any]:
        """
        Run vector-only similarity tests.
        
        Args:
            query_vectors: List of query vectors to test
            threshold_range: Range of similarity thresholds to test
            
        Returns:
            Dictionary of results
        """
        print("Running vector-only similarity tests...")
        results = {}
        
        for i, vector in enumerate(query_vectors):
            print(f"  Testing query vector {i+1}/{len(query_vectors)}")
            threshold_results = {}
            
            for threshold in threshold_range:
                start_time = time.time()
                similar_domains = self.domain_repo.find_similar_domains(
                    vector=vector,
                    threshold=threshold,
                    limit=50
                )
                end_time = time.time()
                
                threshold_results[threshold] = {
                    "count": len(similar_domains),
                    "domains": [d.id for d in similar_domains],
                    "execution_time": end_time - start_time
                }
            
            results[f"query_{i+1}"] = threshold_results
        
        self.results["vector_only"] = results
        return results
    
    def run_vector_plus_topology(self, query_vectors: List[List[float]], 
                               threshold: float = 0.7,
                               max_hops: int = 2) -> Dict[str, Any]:
        """
        Run vector-plus-topology tests.
        
        Args:
            query_vectors: List of query vectors to test
            threshold: Similarity threshold
            max_hops: Maximum number of hops for traversal
            
        Returns:
            Dictionary of results
        """
        print("Running vector-plus-topology tests...")
        results = {}
        db = self.connection_manager.get_db()
        
        for i, vector in enumerate(query_vectors):
            print(f"  Testing query vector {i+1}/{len(query_vectors)}")
            
            # First get vector-similar domains
            start_time = time.time()
            similar_domains = self.domain_repo.find_similar_domains(
                vector=vector,
                threshold=threshold,
                limit=20
            )
            
            # Then expand through topology
            domain_ids = [d.id for d in similar_domains]
            
            if domain_ids:
                # Use AQL to traverse the graph
                query = """
                LET seed_domains = @domain_ids
                
                LET expanded = (
                    FOR domain_id IN seed_domains
                        FOR v, e, p IN 1..@max_hops ANY 
                            CONCAT('Domain/', domain_id)
                            GRAPH 'DomainGraph'
                            RETURN DISTINCT v
                )
                
                RETURN {
                    seed_count: LENGTH(seed_domains),
                    expanded_count: LENGTH(expanded),
                    expanded_domains: expanded
                }
                """
                
                cursor = db.aql.execute(
                    query, 
                    bind_vars={
                        "domain_ids": domain_ids,
                        "max_hops": max_hops
                    }
                )
                
                topology_results = list(cursor)[0]
                end_time = time.time()
                
                results[f"query_{i+1}"] = {
                    "vector_only_count": len(similar_domains),
                    "vector_plus_topology_count": topology_results["expanded_count"],
                    "expansion_ratio": topology_results["expanded_count"] / max(1, len(similar_domains)),
                    "execution_time": end_time - start_time
                }
            else:
                end_time = time.time()
                results[f"query_{i+1}"] = {
                    "vector_only_count": 0,
                    "vector_plus_topology_count": 0,
                    "expansion_ratio": 0,
                    "execution_time": end_time - start_time
                }
        
        self.results["vector_plus_topology"] = results
        return results
    
    def analyze_resonance(self, threshold_range: List[float] = [0.3, 0.5, 0.7, 0.9]) -> Dict[str, Any]:
        """
        Analyze domain resonance at different thresholds.
        
        Args:
            threshold_range: Range of resonance thresholds to test
            
        Returns:
            Dictionary of resonance analysis results
        """
        print("Analyzing domain resonance...")
        results = {}
        
        for threshold in threshold_range:
            print(f"  Testing resonance threshold {threshold}")
            
            start_time = time.time()
            resonating_domains = self.pattern_tracker.find_resonating_domains(threshold=threshold)
            end_time = time.time()
            
            # Analyze resonance network
            domain_connections = {}
            for resonance in resonating_domains:
                domain1_id = resonance["domain1"]["_key"]
                domain2_id = resonance["domain2"]["_key"]
                
                if domain1_id not in domain_connections:
                    domain_connections[domain1_id] = set()
                if domain2_id not in domain_connections:
                    domain_connections[domain2_id] = set()
                
                domain_connections[domain1_id].add(domain2_id)
                domain_connections[domain2_id].add(domain1_id)
            
            # Calculate network metrics
            if domain_connections:
                avg_connections = sum(len(connections) for connections in domain_connections.values()) / len(domain_connections)
                max_connections = max(len(connections) for connections in domain_connections.values())
                
                # Find connected components
                components = self._find_connected_components(domain_connections)
                
                results[threshold] = {
                    "resonance_count": len(resonating_domains),
                    "domain_count": len(domain_connections),
                    "avg_connections": avg_connections,
                    "max_connections": max_connections,
                    "connected_components": len(components),
                    "largest_component_size": max(len(component) for component in components),
                    "execution_time": end_time - start_time
                }
            else:
                results[threshold] = {
                    "resonance_count": 0,
                    "domain_count": 0,
                    "avg_connections": 0,
                    "max_connections": 0,
                    "connected_components": 0,
                    "largest_component_size": 0,
                    "execution_time": end_time - start_time
                }
        
        self.results["resonance"] = results
        return results
    
    def analyze_boundaries(self) -> Dict[str, Any]:
        """
        Analyze domain boundaries and transitions.
        
        Returns:
            Dictionary of boundary analysis results
        """
        print("Analyzing domain boundaries...")
        results = {}
        db = self.connection_manager.get_db()
        
        # Get all actants that appear in multiple domains
        query = """
        // Find actants that appear in multiple domains
        FOR actant IN Actant
            LET appearances = (
                FOR p IN Predicate
                    FILTER p.subject == actant.name OR p.object == actant.name
                        OR p.subject IN actant.aliases OR p.object IN actant.aliases
                    RETURN DISTINCT p.domain_id
            )
            
            FILTER LENGTH(appearances) > 1
            
            RETURN {
                actant: actant,
                domain_count: LENGTH(appearances),
                domains: appearances
            }
        """
        
        start_time = time.time()
        cursor = db.aql.execute(query)
        boundary_actants = list(cursor)
        
        # Analyze boundary crossings
        crossing_stats = {}
        for actant_data in boundary_actants:
            actant_name = actant_data["actant"]["name"]
            
            # Get the actant's journey
            journey = self.pattern_tracker.track_actant_journey(actant_name)
            
            # Analyze role changes
            role_changes = 0
            prev_role = None
            
            for step in journey:
                current_role = step.get("role")
                if prev_role is not None and current_role != prev_role:
                    role_changes += 1
                prev_role = current_role
            
            # Get predicate transformations
            transformations = self.pattern_tracker.detect_predicate_transformations(actant_name)
            
            crossing_stats[actant_name] = {
                "domain_count": actant_data["domain_count"],
                "journey_steps": len(journey),
                "role_changes": role_changes,
                "transformations": len(transformations) if transformations else 0
            }
        
        end_time = time.time()
        
        # Summarize results
        if boundary_actants:
            avg_domains = sum(data["domain_count"] for data in boundary_actants) / len(boundary_actants)
            max_domains = max(data["domain_count"] for data in boundary_actants)
            
            results = {
                "boundary_actants_count": len(boundary_actants),
                "avg_domains_per_actant": avg_domains,
                "max_domains_per_actant": max_domains,
                "crossing_stats": crossing_stats,
                "execution_time": end_time - start_time
            }
        else:
            results = {
                "boundary_actants_count": 0,
                "avg_domains_per_actant": 0,
                "max_domains_per_actant": 0,
                "crossing_stats": {},
                "execution_time": end_time - start_time
            }
        
        self.results["boundary_analysis"] = results
        return results
    
    def compare_approaches(self, query_vectors: List[List[float]]) -> Dict[str, Any]:
        """
        Run a comprehensive comparison of vector-only vs. vector-plus-topology approaches.
        
        Args:
            query_vectors: List of query vectors to test
            
        Returns:
            Dictionary of comparative results
        """
        print("Running comprehensive comparison...")
        
        # Run all tests
        self.run_vector_only_similarity(query_vectors)
        self.run_vector_plus_topology(query_vectors)
        self.analyze_resonance()
        self.analyze_boundaries()
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        return self.results
    
    def _find_connected_components(self, graph: Dict[str, set]) -> List[set]:
        """
        Find connected components in an undirected graph.
        
        Args:
            graph: Dictionary representing an undirected graph
            
        Returns:
            List of connected components (sets of node IDs)
        """
        visited = set()
        components = []
        
        def dfs(node, component):
            visited.add(node)
            component.add(node)
            
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in graph:
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)
        
        return components
    
    def _save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"vector_topology_results_{timestamp}.json")
        
        # Convert sets to lists for JSON serialization
        results_copy = json.loads(json.dumps(self.results, default=lambda o: list(o) if isinstance(o, set) else o))
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def _generate_visualizations(self):
        """Generate visualizations from the results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Vector-only threshold comparison
        if self.results["vector_only"]:
            plt.figure(figsize=(10, 6))
            
            for query_id, query_results in self.results["vector_only"].items():
                thresholds = list(query_results.keys())
                counts = [query_results[t]["count"] for t in thresholds]
                
                plt.plot(thresholds, counts, marker='o', label=query_id)
            
            plt.xlabel('Similarity Threshold')
            plt.ylabel('Number of Similar Domains')
            plt.title('Vector-Only Similarity Results')
            plt.legend()
            plt.grid(True)
            
            filename = os.path.join(self.output_dir, f"vector_only_comparison_{timestamp}.png")
            plt.savefig(filename)
            print(f"Vector-only visualization saved to {filename}")
        
        # Resonance analysis
        if self.results["resonance"]:
            plt.figure(figsize=(10, 6))
            
            thresholds = list(self.results["resonance"].keys())
            counts = [self.results["resonance"][t]["resonance_count"] for t in thresholds]
            
            plt.plot(thresholds, counts, marker='o', color='green')
            
            plt.xlabel('Resonance Threshold')
            plt.ylabel('Number of Resonating Domain Pairs')
            plt.title('Domain Resonance Analysis')
            plt.grid(True)
            
            filename = os.path.join(self.output_dir, f"resonance_analysis_{timestamp}.png")
            plt.savefig(filename)
            print(f"Resonance visualization saved to {filename}")
        
        # Vector vs. Vector+Topology comparison
        if self.results["vector_plus_topology"]:
            plt.figure(figsize=(10, 6))
            
            query_ids = list(self.results["vector_plus_topology"].keys())
            vector_only_counts = [self.results["vector_plus_topology"][q]["vector_only_count"] for q in query_ids]
            combined_counts = [self.results["vector_plus_topology"][q]["vector_plus_topology_count"] for q in query_ids]
            
            x = range(len(query_ids))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], vector_only_counts, width, label='Vector-Only')
            plt.bar([i + width/2 for i in x], combined_counts, width, label='Vector+Topology')
            
            plt.xlabel('Query')
            plt.ylabel('Number of Domains')
            plt.title('Vector-Only vs. Vector+Topology Comparison')
            plt.xticks(x, query_ids)
            plt.legend()
            
            filename = os.path.join(self.output_dir, f"vector_vs_topology_{timestamp}.png")
            plt.savefig(filename)
            print(f"Comparison visualization saved to {filename}")

def generate_test_vectors(count: int = 5, dim: int = 4) -> List[List[float]]:
    """
    Generate test vectors for diagnostics.
    
    Args:
        count: Number of vectors to generate
        dim: Dimension of vectors
        
    Returns:
        List of test vectors
    """
    vectors = []
    
    # Generate random vectors
    for _ in range(count):
        vector = np.random.rand(dim).tolist()
        vectors.append(vector)
    
    return vectors

def main():
    """Main function to run the diagnostics."""
    print("\n===== HABITAT VECTOR+TOPOLOGY DIAGNOSTICS =====")
    print("This script compares vector-only vs. vector-plus-topology approaches in ArangoDB.\n")
    
    try:
        # Create diagnostics tool
        diagnostics = VectorTopologyDiagnostics()
        
        # Generate test vectors
        test_vectors = generate_test_vectors(count=3)
        
        # Run comprehensive comparison
        results = diagnostics.compare_approaches(test_vectors)
        
        print("\n===== SUMMARY OF RESULTS =====")
        
        # Vector-only summary
        vector_only = results["vector_only"]
        if vector_only:
            avg_counts = {}
            for query_id, query_results in vector_only.items():
                for threshold, threshold_results in query_results.items():
                    if threshold not in avg_counts:
                        avg_counts[threshold] = []
                    avg_counts[threshold].append(threshold_results["count"])
            
            print("\nVector-Only Results:")
            for threshold, counts in avg_counts.items():
                avg = sum(counts) / len(counts)
                print(f"  Threshold {threshold}: Avg. {avg:.2f} similar domains")
        
        # Vector+Topology summary
        vector_plus = results["vector_plus_topology"]
        if vector_plus:
            total_expansion = 0
            query_count = len(vector_plus)
            
            for query_id, query_results in vector_plus.items():
                total_expansion += query_results["expansion_ratio"]
            
            avg_expansion = total_expansion / query_count if query_count > 0 else 0
            print(f"\nVector+Topology Results:")
            print(f"  Average expansion ratio: {avg_expansion:.2f}x")
        
        # Resonance summary
        resonance = results["resonance"]
        if resonance:
            print("\nResonance Analysis:")
            for threshold, threshold_results in resonance.items():
                print(f"  Threshold {threshold}: {threshold_results['resonance_count']} resonating pairs")
                print(f"    Connected components: {threshold_results['connected_components']}")
                print(f"    Largest component size: {threshold_results['largest_component_size']}")
        
        # Boundary analysis summary
        boundary = results["boundary_analysis"]
        if boundary:
            print("\nBoundary Analysis:")
            print(f"  Boundary-crossing actants: {boundary['boundary_actants_count']}")
            print(f"  Avg. domains per actant: {boundary['avg_domains_per_actant']:.2f}")
            
            # Top boundary crossers
            if boundary["crossing_stats"]:
                crossers = sorted(
                    boundary["crossing_stats"].items(),
                    key=lambda x: x[1]["domain_count"],
                    reverse=True
                )
                
                print("\nTop boundary-crossing actants:")
                for actant, stats in crossers[:5]:
                    print(f"  {actant}: {stats['domain_count']} domains, {stats['transformations']} transformations")
        
        print("\n===== DIAGNOSTICS COMPLETE =====")
        print("Results and visualizations have been saved to the 'diagnostics_results' directory.")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nPlease make sure ArangoDB is running and the collections have been created.")
        print("You can create the collections using the init_arangodb.py script.")

if __name__ == "__main__":
    main()
