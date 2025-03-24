"""
Predicate Sublimation

This module implements detection of semantic phase transitions where predicate networks
undergo qualitative shifts in meaning - what we call "predicate sublimation."

Rather than treating meaning as merely accumulative, this approach identifies critical
thresholds where new conceptual frameworks emerge from predicate networks, similar to
how matter changes state during phase transitions.

The system also detects the inherent directionality of meaning - the "supposing" within
meaning itself that indicates where semantic flows want to move toward greater capaciousness.
This is not homeostasis as stasis (fixed equilibrium) but homeostasis as capaciousness - 
the ability to contain and express meaning through change.
"""

from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass, field
import uuid
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering
import networkx as nx
from datetime import datetime

from ..id.adaptive_id import AdaptiveID


@dataclass
class ConceptualFramework:
    """
    Represents an emergent conceptual framework that arises from predicate sublimation.
    
    This is more than just an emergent form - it's an entire new way of understanding
    that emerges when predicate networks reach critical thresholds of complexity.
    """
    id: str
    name: str
    description: str
    constituent_predicates: List[str]
    constituent_actants: List[str]
    emergence_confidence: float
    stability_index: float
    semantic_directionality: Dict[str, float] = field(default_factory=dict)  # Pressure gradients
    capaciousness_index: float = 0.0  # Ability to contain meaning through change
    adaptive_id: Optional[AdaptiveID] = None
    
    @classmethod
    def create(cls, name: str, description: str, predicates: List[str], 
               actants: List[str], confidence: float = 0.5, stability: float = 0.5,
               directionality: Dict[str, float] = None, capaciousness: float = 0.0):
        """Create a new conceptual framework."""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            constituent_predicates=predicates,
            constituent_actants=actants,
            emergence_confidence=confidence,
            stability_index=stability,
            semantic_directionality=directionality or {},
            capaciousness_index=capaciousness
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "constituent_predicates": self.constituent_predicates,
            "constituent_actants": self.constituent_actants,
            "emergence_confidence": self.emergence_confidence,
            "stability_index": self.stability_index,
            "semantic_directionality": self.semantic_directionality,
            "capaciousness_index": self.capaciousness_index
        }


class PredicateSublimationDetector:
    """
    Detects semantic phase transitions in predicate networks.
    
    This class implements four key approaches:
    1. Threshold Detection: Identifies critical thresholds where predicate networks
       undergo qualitative shifts
    2. Emergent Concept Extraction: Detects entirely new conceptual frameworks beyond
       individual emergent forms
    3. Multi-scale Analysis: Analyzes predicate networks at multiple scales to identify
       emergent structures
    4. Semantic Flow Analysis: Detects the inherent directionality of meaning - the
       "supposing" within meaning itself that indicates where semantic flows want to
       move toward greater capaciousness
    """
    
    def __init__(self, predicates=None, actants=None, transformations=None):
        """Initialize the detector."""
        self.predicates = predicates or {}
        self.actants = actants or {}
        self.transformations = transformations or []
        self.conceptual_frameworks = []
        
        # Configuration parameters
        self.threshold_sensitivity = 0.7  # How sensitive to be to potential thresholds
        self.min_framework_size = 3  # Minimum number of predicates to form a framework
        self.scale_levels = 3  # Number of scales to analyze
        self.flow_sensitivity = 0.6  # Sensitivity to semantic flow directionality
        self.capaciousness_threshold = 0.5  # Threshold for detecting capaciousness
    
    def detect_sublimations(self) -> List[ConceptualFramework]:
        """
        Detect semantic phase transitions in the predicate network.
        
        Returns a list of ConceptualFramework objects representing the detected
        conceptual frameworks that emerge from predicate sublimation.
        """
        # Build the predicate network
        G = self._build_predicate_network()
        
        # 1. Threshold Detection
        threshold_candidates = self._detect_critical_thresholds(G)
        
        # 2. Emergent Concept Extraction
        frameworks = self._extract_conceptual_frameworks(G, threshold_candidates)
        
        # 3. Multi-scale Analysis
        multi_scale_frameworks = self._perform_multi_scale_analysis(G)
        
        # 4. Semantic Flow Analysis
        for framework in frameworks + multi_scale_frameworks:
            # Analyze semantic directionality and capaciousness
            directionality, capaciousness = self._analyze_semantic_flow(G, framework.constituent_predicates)
            framework.semantic_directionality = directionality
            framework.capaciousness_index = capaciousness
        
        # Combine and deduplicate frameworks
        all_frameworks = frameworks + multi_scale_frameworks
        self.conceptual_frameworks = self._deduplicate_frameworks(all_frameworks)
        
        return self.conceptual_frameworks
    
    def _build_predicate_network(self) -> nx.DiGraph:
        """
        Build a directed graph representing the predicate network.
        
        Nodes are predicates, and edges are transformations between predicates.
        Edge weights represent the strength of the transformation (amplitude).
        """
        G = nx.DiGraph()
        
        # Add nodes (predicates)
        for pred_id, pred in self.predicates.items():
            G.add_node(pred_id, 
                      subject=pred.subject, 
                      verb=pred.verb, 
                      object=pred.object,
                      domain=pred.domain_id,
                      text=pred.text)
        
        # Add edges (transformations)
        for t in self.transformations:
            G.add_edge(t.source_id, t.target_id, 
                      weight=t.amplitude,
                      carrying_actants=t.carrying_actants,
                      role_pattern=t.role_pattern)
        
        return G
    
    def _detect_critical_thresholds(self, G: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Identify critical thresholds where predicate networks undergo qualitative shifts.
        
        Uses network metrics to detect potential phase transitions in the semantic space.
        """
        thresholds = []
        
        # 1. Connectivity-based thresholds
        # Find strongly connected components
        components = list(nx.strongly_connected_components(G))
        for i, component in enumerate(components):
            if len(component) >= self.min_framework_size:
                # Calculate component cohesion (average edge weight within component)
                subgraph = G.subgraph(component)
                edge_weights = [data['weight'] for _, _, data in subgraph.edges(data=True)]
                avg_weight = np.mean(edge_weights) if edge_weights else 0
                
                if avg_weight > self.threshold_sensitivity:
                    thresholds.append({
                        'type': 'connectivity',
                        'predicates': list(component),
                        'strength': avg_weight,
                        'actants': self._get_shared_actants(subgraph)
                    })
        
        # 2. Centrality-based thresholds
        # Find predicates with high betweenness centrality (bridging concepts)
        betweenness = nx.betweenness_centrality(G, weight='weight')
        high_betweenness = {node: value for node, value in betweenness.items() 
                           if value > self.threshold_sensitivity}
        
        for node, value in high_betweenness.items():
            # Get neighbors of this high-centrality node
            neighbors = list(G.successors(node)) + list(G.predecessors(node))
            if len(neighbors) >= self.min_framework_size - 1:  # Including the central node
                thresholds.append({
                    'type': 'centrality',
                    'predicates': [node] + neighbors,
                    'strength': value,
                    'actants': self._get_actants_for_predicates([node] + neighbors)
                })
        
        # 3. Feedback loop thresholds
        # Find cycles that might represent feedback loops
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            if len(cycle) >= self.min_framework_size:
                # Calculate cycle resonance (product of edge weights)
                cycle_resonance = 1.0
                for i in range(len(cycle)):
                    source = cycle[i]
                    target = cycle[(i + 1) % len(cycle)]
                    if G.has_edge(source, target):
                        cycle_resonance *= G[source][target]['weight']
                
                if cycle_resonance > self.threshold_sensitivity:
                    thresholds.append({
                        'type': 'feedback',
                        'predicates': cycle,
                        'strength': cycle_resonance,
                        'actants': self._get_actants_for_predicates(cycle)
                    })
        
        return thresholds
    
    def _extract_conceptual_frameworks(
        self, G: nx.DiGraph, threshold_candidates: List[Dict[str, Any]]
    ) -> List[ConceptualFramework]:
        """
        Extract emergent conceptual frameworks from threshold candidates.
        
        Goes beyond individual emergent forms to detect entirely new conceptual frameworks.
        """
        frameworks = []
        
        for candidate in threshold_candidates:
            # Get predicates in this candidate
            predicates = candidate['predicates']
            actants = candidate['actants']
            
            # Skip if too few predicates or actants
            if len(predicates) < self.min_framework_size or len(actants) < 2:
                continue
            
            # Generate a name and description for this framework
            framework_name, framework_description = self._generate_framework_description(
                predicates, actants, candidate['type'])
            
            # Analyze semantic flow
            directionality, capaciousness = self._analyze_semantic_flow(G, predicates)
            
            # Create a new conceptual framework
            framework = ConceptualFramework.create(
                name=framework_name,
                description=framework_description,
                predicates=predicates,
                actants=actants,
                confidence=candidate['strength'],
                stability=self._calculate_framework_stability(G, predicates),
                directionality=directionality,
                capaciousness=capaciousness
            )
            
            frameworks.append(framework)
        
        return frameworks
    
    def _perform_multi_scale_analysis(self, G: nx.DiGraph) -> List[ConceptualFramework]:
        """
        Analyze predicate networks at multiple scales to identify emergent structures.
        
        Uses spectral clustering with different numbers of clusters to detect
        communities at different scales.
        """
        frameworks = []
        
        # Skip if graph is too small
        if len(G) < self.min_framework_size * 2:
            return frameworks
        
        # Create adjacency matrix
        adj_matrix = nx.to_numpy_array(G, weight='weight')
        
        # Analyze at multiple scales (different numbers of clusters)
        for n_clusters in range(2, self.scale_levels + 2):
            try:
                # Apply spectral clustering
                clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    assign_labels='discretize',
                    random_state=42
                ).fit(adj_matrix)
                
                # Get clusters
                labels = clustering.labels_
                
                # Convert node indices to predicate IDs
                nodes = list(G.nodes())
                
                # Process each cluster
                for cluster_id in range(n_clusters):
                    cluster_indices = np.where(labels == cluster_id)[0]
                    cluster_predicates = [nodes[i] for i in cluster_indices]
                    
                    if len(cluster_predicates) >= self.min_framework_size:
                        # Get actants for this cluster
                        cluster_actants = self._get_actants_for_predicates(cluster_predicates)
                        
                        if len(cluster_actants) >= 2:
                            # Calculate cluster cohesion
                            subgraph = G.subgraph(cluster_predicates)
                            edge_weights = [data['weight'] for _, _, data in subgraph.edges(data=True)]
                            avg_weight = np.mean(edge_weights) if edge_weights else 0
                            
                            if avg_weight > self.threshold_sensitivity:
                                # Generate a name and description
                                framework_name, framework_description = self._generate_framework_description(
                                    cluster_predicates, cluster_actants, f'scale_{n_clusters}')
                                
                                # Analyze semantic flow
                                directionality, capaciousness = self._analyze_semantic_flow(
                                    G, cluster_predicates
                                )
                                
                                # Create a new conceptual framework
                                framework = ConceptualFramework.create(
                                    name=framework_name,
                                    description=framework_description,
                                    predicates=cluster_predicates,
                                    actants=cluster_actants,
                                    confidence=avg_weight,
                                    stability=self._calculate_framework_stability(G, cluster_predicates),
                                    directionality=directionality,
                                    capaciousness=capaciousness
                                )
                                
                                frameworks.append(framework)
            except:
                # Skip this scale if clustering fails
                continue
        
        return frameworks
    
    def _get_shared_actants(self, G: nx.DiGraph) -> List[str]:
        """Get actants shared across predicates in a subgraph."""
        actants = set()
        
        for node in G.nodes():
            pred = self.predicates.get(node)
            if pred:
                actants.add(pred.subject)
                actants.add(pred.object)
        
        return list(actants)
    
    def _get_actants_for_predicates(self, predicate_ids: List[str]) -> List[str]:
        """Get all actants involved in a list of predicates."""
        actants = set()
        
        for pred_id in predicate_ids:
            pred = self.predicates.get(pred_id)
            if pred:
                actants.add(pred.subject)
                actants.add(pred.object)
        
        return list(actants)
    
    def _calculate_framework_stability(self, G: nx.DiGraph, predicates: List[str]) -> float:
        """
        Calculate the stability of a conceptual framework.
        
        Stability is based on:
        1. Internal connectivity (how densely connected the predicates are)
        2. External boundary strength (how strongly connected to outside predicates)
        3. Feedback loop presence (cycles within the framework)
        """
        # Skip if too few predicates
        if len(predicates) < 2:
            return 0.0
        
        # Get the subgraph for this framework
        subgraph = G.subgraph(predicates)
        
        # 1. Internal connectivity
        density = nx.density(subgraph)
        
        # 2. External boundary strength
        # Count edges crossing the boundary
        boundary_edges = 0
        boundary_weight_sum = 0.0
        
        for p in predicates:
            for neighbor in G.successors(p):
                if neighbor not in predicates:
                    boundary_edges += 1
                    boundary_weight_sum += G[p][neighbor]['weight']
            
            for neighbor in G.predecessors(p):
                if neighbor not in predicates:
                    boundary_edges += 1
                    boundary_weight_sum += G[neighbor][p]['weight']
        
        # Calculate average boundary weight
        avg_boundary_weight = boundary_weight_sum / boundary_edges if boundary_edges > 0 else 0
        
        # 3. Feedback loop presence
        has_cycle = len(list(nx.simple_cycles(subgraph))) > 0
        cycle_bonus = 0.2 if has_cycle else 0.0
        
        # Combine factors into stability index
        # Higher internal density and lower boundary strength = more stable
        stability = (0.5 * density) + (0.3 * (1.0 - avg_boundary_weight)) + cycle_bonus
        
        return min(1.0, max(0.0, stability))  # Ensure value is between 0 and 1
    
    def _analyze_semantic_flow(self, G: nx.DiGraph, predicates: List[str]) -> Tuple[Dict[str, float], float]:
        """
        Analyze the inherent directionality of meaning within a conceptual framework.
        
        Detects the 'supposing' within meaning itself - where semantic flows want to move
        toward greater capaciousness (ability to contain meaning through change).
        
        Returns:
            - Dictionary mapping semantic directions to their strength
            - Capaciousness index (ability to contain meaning through change)
        """
        if len(predicates) < 2:
            return {}, 0.0
            
        # Get the subgraph for this framework
        subgraph = G.subgraph(predicates)
        
        # 1. Identify semantic flow directions
        directionality = {}
        
        # Calculate PageRank to find influential predicates
        pagerank = nx.pagerank(subgraph, weight='weight')
        
        # Find top predicates by PageRank
        top_predicates = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        
        # For each top predicate, identify its semantic direction
        for pred_id, importance in top_predicates[:3]:  # Consider top 3 predicates
            if pred_id in self.predicates:
                pred = self.predicates[pred_id]
                # Use verb as semantic direction
                direction = pred.verb
                directionality[direction] = importance
        
        # 2. Calculate capaciousness index
        # Capaciousness is measured by:
        # - Diversity of predicates (number of unique verbs)
        # - Interconnectedness (density of connections)
        # - Ability to incorporate new elements (boundary permeability)
        
        # Count unique verbs
        unique_verbs = set()
        for pred_id in predicates:
            if pred_id in self.predicates:
                unique_verbs.add(self.predicates[pred_id].verb)
        verb_diversity = len(unique_verbs) / max(1, len(predicates))
        
        # Calculate interconnectedness
        density = nx.density(subgraph)
        
        # Calculate boundary permeability
        # Count edges crossing the boundary and their weights
        boundary_edges = 0
        boundary_weight_sum = 0.0
        for p in predicates:
            for neighbor in G.successors(p):
                if neighbor not in predicates:
                    boundary_edges += 1
                    boundary_weight_sum += G[p][neighbor]['weight']
            for neighbor in G.predecessors(p):
                if neighbor not in predicates:
                    boundary_edges += 1
                    boundary_weight_sum += G[neighbor][p]['weight']
        
        # Higher boundary weight = more permeable
        permeability = boundary_weight_sum / max(1, boundary_edges)
        
        # Combine factors into capaciousness index
        capaciousness = (0.4 * verb_diversity) + (0.3 * density) + (0.3 * permeability)
        capaciousness = min(1.0, max(0.0, capaciousness))
        
        return directionality, capaciousness
        
    def _generate_framework_description(
        self, predicates: List[str], actants: List[str], framework_type: str
    ) -> Tuple[str, str]:
        """
        Generate a name and description for a conceptual framework.
        
        Uses the predicates and actants to create a meaningful description of
        the emergent conceptual framework.
        """
        # Get predicate texts and verbs
        pred_texts = [self.predicates[pid].text for pid in predicates if pid in self.predicates]
        verbs = [self.predicates[pid].verb for pid in predicates if pid in self.predicates]
        
        # Get actant names
        actant_names = [self.actants[aid].name for aid in actants if aid in self.actants]
        
        # Generate name based on framework type and content
        if framework_type == 'connectivity':
            # For connectivity-based frameworks, focus on the shared actants
            if actant_names:
                name = f"{actant_names[0].capitalize()} network"
                if len(actant_names) > 1:
                    name = f"{actant_names[0].capitalize()}-{actant_names[1]} interaction network"
            else:
                # Fallback if no actant names available
                name = "Connected predicate network"
        
        elif framework_type == 'centrality':
            # For centrality-based frameworks, focus on the central predicate
            central_pred = self.predicates[predicates[0]]
            name = f"{central_pred.verb.capitalize()} mediation framework"
        
        elif framework_type == 'feedback':
            # For feedback loops, emphasize the cyclical nature
            if len(set(verbs)) > 1:
                verb_str = "-".join(sorted(set(verbs))[:2])
                name = f"{verb_str.capitalize()} feedback cycle"
            else:
                name = f"{verbs[0].capitalize()} feedback cycle"
        
        elif framework_type.startswith('scale_'):
            # For scale-based frameworks, emphasize the emergent pattern
            if actant_names:
                name = f"{actant_names[0].capitalize()} pattern"
                if len(actant_names) > 1:
                    name = f"{actant_names[0].capitalize()}-{actant_names[1]} pattern"
            else:
                # Fallback
                name = "Multi-scale predicate pattern"
        
        else:
            # Generic fallback
            name = "Emergent conceptual framework"
        
        # Generate description
        description = f"A conceptual framework emerging from the interaction of {len(predicates)} predicates "
        description += f"involving {', '.join(actant_names[:3])}"
        if len(actant_names) > 3:
            description += f" and {len(actant_names) - 3} other actants"
        
        # Add type-specific details
        if framework_type == 'connectivity':
            description += ". This framework emerges from densely connected predicates forming a cohesive semantic unit."
        elif framework_type == 'centrality':
            description += ". This framework emerges around a central bridging concept that connects different semantic domains."
        elif framework_type == 'feedback':
            description += ". This framework represents a feedback cycle where predicates influence each other in a circular pattern."
        elif framework_type.startswith('scale_'):
            description += f". This framework emerges at scale level {framework_type.split('_')[1]} as a distinct pattern of predicate interactions."
        
        return name, description
    
    def _deduplicate_frameworks(
        self, frameworks: List[ConceptualFramework]
    ) -> List[ConceptualFramework]:
        """
        Deduplicate conceptual frameworks that are too similar.
        
        Two frameworks are considered similar if they share a significant
        proportion of their constituent predicates.
        """
        if not frameworks:
            return []
        
        # Sort frameworks by confidence (descending)
        sorted_frameworks = sorted(
            frameworks, key=lambda f: f.emergence_confidence, reverse=True)
        
        # Keep track of which frameworks to retain
        retain = [True] * len(sorted_frameworks)
        
        # Compare each framework with all lower-confidence frameworks
        for i in range(len(sorted_frameworks)):
            if not retain[i]:
                continue
                
            framework_i = sorted_frameworks[i]
            pred_set_i = set(framework_i.constituent_predicates)
            
            for j in range(i + 1, len(sorted_frameworks)):
                if not retain[j]:
                    continue
                    
                framework_j = sorted_frameworks[j]
                pred_set_j = set(framework_j.constituent_predicates)
                
                # Calculate Jaccard similarity
                intersection = len(pred_set_i.intersection(pred_set_j))
                union = len(pred_set_i.union(pred_set_j))
                similarity = intersection / union if union > 0 else 0
                
                # If frameworks are too similar, discard the lower-confidence one
                if similarity > 0.7:  # Threshold for similarity
                    retain[j] = False
        
        # Return deduplicated frameworks
        return [f for i, f in enumerate(sorted_frameworks) if retain[i]]


# Example usage:
if __name__ == "__main__":
    # This would be replaced with actual data from documents
    from dataclasses import dataclass
    
    @dataclass
    class Actant:
        id: str
        name: str
        aliases: List[str] = None
        
        def __post_init__(self):
            if self.aliases is None:
                self.aliases = []
    
    @dataclass
    class Predicate:
        id: str
        subject: str
        verb: str
        object: str
        text: str
        domain_id: str
        position: int = 0
    
    @dataclass
    class TransformationEdge:
        id: str
        source_id: str
        target_id: str
        carrying_actants: List[str]
        amplitude: float = 0.5
        frequency: float = 0.5
        phase: float = 0.0
        role_pattern: str = "stable"
    
    # Create test data
    actants = {
        "a1": Actant(id="a1", name="sea level", aliases=["ocean level"]),
        "a2": Actant(id="a2", name="coastline", aliases=["shore", "coast"]),
        "a3": Actant(id="a3", name="community", aliases=["town", "residents"]),
        "a4": Actant(id="a4", name="infrastructure", aliases=["buildings", "roads"]),
        "a5": Actant(id="a5", name="policy", aliases=["regulation", "law"]),
        "a6": Actant(id="a6", name="economy", aliases=["market", "business"])
    }
    
    predicates = {
        # Domain 1: Climate Science
        "p1": Predicate(id="p1", subject="a1", verb="rises", object="a2", 
                       text="Sea level rises along the coastline", domain_id="d1"),
        "p2": Predicate(id="p2", subject="a1", verb="threatens", object="a4", 
                       text="Sea level threatens infrastructure", domain_id="d1"),
        
        # Domain 2: Coastal Infrastructure
        "p3": Predicate(id="p3", subject="a2", verb="erodes", object="a4", 
                       text="Coastline erodes infrastructure", domain_id="d2"),
        "p4": Predicate(id="p4", subject="a4", verb="protects", object="a2", 
                       text="Infrastructure protects coastline", domain_id="d2"),
        "p5": Predicate(id="p5", subject="a4", verb="fails", object="a3", 
                       text="Infrastructure fails to protect community", domain_id="d2"),
        
        # Domain 3: Community Planning
        "p6": Predicate(id="p6", subject="a3", verb="adapts", object="a1", 
                       text="Community adapts to sea level", domain_id="d3"),
        "p7": Predicate(id="p7", subject="a3", verb="relocates", object="a2", 
                       text="Community relocates from coastline", domain_id="d3"),
        "p8": Predicate(id="p8", subject="a3", verb="demands", object="a5", 
                       text="Community demands policy changes", domain_id="d3"),
        
        # Domain 4: Policy Response
        "p9": Predicate(id="p9", subject="a5", verb="regulates", object="a4", 
                       text="Policy regulates infrastructure development", domain_id="d4"),
        "p10": Predicate(id="p10", subject="a5", verb="funds", object="a4", 
                        text="Policy funds infrastructure improvements", domain_id="d4"),
        "p11": Predicate(id="p11", subject="a5", verb="supports", object="a3", 
                        text="Policy supports community adaptation", domain_id="d4"),
        
        # Domain 5: Economic Impact
        "p12": Predicate(id="p12", subject="a1", verb="damages", object="a6", 
                        text="Sea level damages economy", domain_id="d5"),
        "p13": Predicate(id="p13", subject="a6", verb="influences", object="a5", 
                        text="Economy influences policy decisions", domain_id="d5"),
        "p14": Predicate(id="p14", subject="a6", verb="invests", object="a4", 
                        text="Economy invests in infrastructure", domain_id="d5"),
        "p15": Predicate(id="p15", subject="a3", verb="participates", object="a6", 
                        text="Community participates in economy", domain_id="d5")
    }
    
    # Create transformations between predicates
    transformations = [
        # Climate Science → Coastal Infrastructure
        TransformationEdge(id="t1", source_id="p1", target_id="p3", 
                         carrying_actants=["a2"], amplitude=0.8, role_pattern="shift"),
        TransformationEdge(id="t2", source_id="p2", target_id="p4", 
                         carrying_actants=["a4"], amplitude=0.7, role_pattern="shift"),
        TransformationEdge(id="t3", source_id="p2", target_id="p5", 
                         carrying_actants=["a4"], amplitude=0.6, role_pattern="stable"),
        
        # Coastal Infrastructure → Community Planning
        TransformationEdge(id="t4", source_id="p3", target_id="p7", 
                         carrying_actants=["a2"], amplitude=0.7, role_pattern="shift"),
        TransformationEdge(id="t5", source_id="p5", target_id="p6", 
                         carrying_actants=["a3"], amplitude=0.8, role_pattern="shift"),
        TransformationEdge(id="t6", source_id="p5", target_id="p8", 
                         carrying_actants=["a3"], amplitude=0.7, role_pattern="stable"),
        
        # Community Planning → Policy Response
        TransformationEdge(id="t7", source_id="p8", target_id="p9", 
                         carrying_actants=["a5"], amplitude=0.9, role_pattern="shift"),
        TransformationEdge(id="t8", source_id="p8", target_id="p10", 
                         carrying_actants=["a5"], amplitude=0.8, role_pattern="shift"),
        TransformationEdge(id="t9", source_id="p6", target_id="p11", 
                         carrying_actants=["a3"], amplitude=0.7, role_pattern="stable"),
        
        # Climate Science → Economic Impact
        TransformationEdge(id="t10", source_id="p2", target_id="p12", 
                         carrying_actants=["a1"], amplitude=0.8, role_pattern="shift"),
        
        # Economic Impact → Policy Response
        TransformationEdge(id="t11", source_id="p13", target_id="p9", 
                         carrying_actants=["a5"], amplitude=0.7, role_pattern="shift"),
        TransformationEdge(id="t12", source_id="p14", target_id="p10", 
                         carrying_actants=["a4"], amplitude=0.6, role_pattern="stable"),
        
        # Community Planning → Economic Impact
        TransformationEdge(id="t13", source_id="p7", target_id="p15", 
                         carrying_actants=["a3"], amplitude=0.5, role_pattern="shift"),
        
        # Feedback loops
        TransformationEdge(id="t14", source_id="p14", target_id="p4", 
                         carrying_actants=["a4"], amplitude=0.7, role_pattern="stable"),
        TransformationEdge(id="t15", source_id="p11", target_id="p6", 
                         carrying_actants=["a3"], amplitude=0.8, role_pattern="stable"),
        TransformationEdge(id="t16", source_id="p12", target_id="p13", 
                         carrying_actants=["a6"], amplitude=0.7, role_pattern="shift"),
        TransformationEdge(id="t17", source_id="p9", target_id="p14", 
                         carrying_actants=["a4"], amplitude=0.6, role_pattern="shift")
    ]
    
    # Create detector
    detector = PredicateSublimationDetector(
        predicates=predicates,
        actants=actants,
        transformations=transformations
    )
    
    # Detect sublimations
    frameworks = detector.detect_sublimations()
    
    # Print results
    print(f"Detected {len(frameworks)} conceptual frameworks:")
    for i, framework in enumerate(frameworks):
        print(f"\n{i+1}. {framework.name}")
        print(f"   Description: {framework.description}")
        print(f"   Confidence: {framework.emergence_confidence:.2f}")
        print(f"   Stability: {framework.stability_index:.2f}")
        print(f"   Constituent predicates: {len(framework.constituent_predicates)}")
        print(f"   Constituent actants: {len(framework.constituent_actants)}")
        
        # Print a few sample predicates
        print("   Sample predicates:")
        for pred_id in framework.constituent_predicates[:3]:
            if pred_id in predicates:
                print(f"     - {predicates[pred_id].text}")
        if len(framework.constituent_predicates) > 3:
            print(f"     - ... and {len(framework.constituent_predicates) - 3} more")
