# src/habitat_evolution/field/topological_field_analyzer.py
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from scipy import stats, ndimage
import networkx as nx

class TopologicalFieldAnalyzer:
    """Analyzes pattern field topology to create navigable semantic spaces."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Default configuration
        default_config = {
            "dimensionality_threshold": 0.95,  # Variance threshold for effective dimensions
            "density_sensitivity": 0.25,       # Sensitivity for detecting density centers
            "gradient_smoothing": 1.0,         # Smoothing factor for gradient calculations
            "edge_threshold": 0.3              # Minimum weight for graph edges
        }
        
        # If config is provided, use it to update the default config
        if config:
            # Map test config keys to internal config keys if needed
            config_mapping = {
                "min_eigenvalue": "dimensionality_threshold",
                "density_threshold": "density_sensitivity",
                "flow_sensitivity": "gradient_smoothing",
                "graph_weight_threshold": "edge_threshold"
            }
            
            # Apply mapping and update default config
            for key, value in config.items():
                if key in config_mapping:
                    default_config[config_mapping[key]] = value
                else:
                    default_config[key] = value
        
        self.config = default_config
        
    def analyze_field(self, resonance_matrix: np.ndarray, pattern_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive analysis of field topology to create a navigable space."""
        # Ensure matrix is properly formatted
        matrix = np.array(resonance_matrix, dtype=float)
        n = matrix.shape[0]
        
        if n <= 1:
            return self._empty_field_result()
        
        # 1. TOPOLOGY ANALYSIS: Dimensional structure and primary axes
        topology = self._analyze_topology(matrix)
        
        # 2. DENSITY ANALYSIS: Identify high-density regions and centers
        density = self._analyze_density(matrix, pattern_metadata)
        
        # 3. FLOW ANALYSIS: Direction and gradient of pattern relationships
        flow = self._analyze_flow(matrix)
        
        # 4. POTENTIAL ANALYSIS: Potential energy and attraction/repulsion
        potential = self._analyze_potential(matrix)
        
        # 5. Graph representation for connectivity analysis
        graph_metrics = self._analyze_graph(matrix)
        
        # 6. Combined field properties
        field_properties = {
            "coherence": float(np.sum(matrix) / (n * n)) if n > 0 else 0.0,
            "complexity": float(topology["effective_dimensionality"] / n) if n > 0 else 0.0,
            "stability": float(1.0 - np.std(matrix) / np.mean(matrix)) if np.mean(matrix) > 0 else 0.0,
            "density_ratio": float(len(density["density_centers"]) / n) if n > 0 else 0.0,
            "navigability_score": float(graph_metrics["avg_path_length"] * graph_metrics["clustering"] / graph_metrics["diameter"]) if graph_metrics["diameter"] > 0 else 0.0
        }
        
        return {
            "topology": topology,
            "density": density,
            "flow": flow,
            "potential": potential,
            "graph_metrics": graph_metrics,
            "field_properties": field_properties
        }
        
    def _analyze_topology(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze dimensional structure of the field using eigendecomposition."""
        # Calculate eigenvalues and eigenvectors (real, symmetric matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Sort in descending order (largest eigenvalues first)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate effective dimensionality
        total_variance = np.sum(np.abs(eigenvalues))
        if total_variance > 0:
            cumulative_variance = np.cumsum(np.abs(eigenvalues)) / total_variance
            effective_dims = np.sum(cumulative_variance < self.config["dimensionality_threshold"]) + 1
        else:
            effective_dims = 1
            cumulative_variance = np.array([1.0])
        
        # Calculate primary dimensions (projection matrices)
        principal_dimensions = []
        for i in range(min(effective_dims, 3)):  # Store top 3 at most
            if i < len(eigenvalues):
                principal_dimensions.append({
                    "eigenvalue": float(eigenvalues[i]),
                    "explained_variance": float(np.abs(eigenvalues[i]) / total_variance) if total_variance > 0 else 0.0,
                    "eigenvector": eigenvectors[:, i].tolist()
                })
        
        # Calculate projection coordinates for each pattern in top dimensions
        projections = []
        for i in range(matrix.shape[0]):
            proj = {}
            for dim in range(min(effective_dims, 3)):
                proj[f"dim_{dim}"] = float(eigenvectors[i, dim])
            projections.append(proj)
            
        return {
            "effective_dimensionality": int(effective_dims),
            "principal_dimensions": principal_dimensions,
            "dimension_strengths": eigenvalues.tolist(),
            "cumulative_variance": cumulative_variance.tolist(),
            "pattern_projections": projections
        }
        
    def _analyze_density(self, matrix: np.ndarray, pattern_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze density and distribution of patterns in the field."""
        n = matrix.shape[0]
        
        # Calculate node strengths (sum of all connections)
        node_strengths = np.sum(matrix, axis=1)
        
        # Calculate local density (weighted connections in neighborhood)
        local_densities = np.zeros(n)
        for i in range(n):
            # Define neighborhood as nodes with above-average connection
            neighborhood = matrix[i, :] > np.mean(matrix[i, :])
            local_densities[i] = np.sum(matrix[i, neighborhood]) / np.sum(neighborhood) if np.sum(neighborhood) > 0 else 0
        
        # Identify density centers (local maxima in the density field)
        # Smooth the density field first for more stable centers
        smoothed_density = ndimage.gaussian_filter1d(local_densities, sigma=self.config["gradient_smoothing"])
        density_threshold = np.mean(smoothed_density) + self.config["density_sensitivity"] * np.std(smoothed_density)
        
        # Identify centers as density peaks above threshold
        density_centers = []
        for i in range(n):
            if smoothed_density[i] > density_threshold:
                # Check if it's a local maximum
                is_maximum = True
                for j in range(max(0, i-1), min(n, i+2)):
                    if j != i and smoothed_density[j] > smoothed_density[i]:
                        is_maximum = False
                        break
                
                if is_maximum:
                    # Include pattern metadata for this center
                    metadata = pattern_metadata[i] if i < len(pattern_metadata) else {}
                    
                    density_centers.append({
                        "index": i,
                        "density": float(smoothed_density[i]),
                        "node_strength": float(node_strengths[i]),
                        "influence_radius": float(np.sum(matrix[i, :] > 0.5)),
                        "pattern_metadata": metadata
                    })
        
        # Calculate density gradient (direction of increasing density)
        density_gradient = np.gradient(smoothed_density)
        
        # Calculate global density metrics
        global_density = float(np.mean(node_strengths))
        density_variance = float(np.var(node_strengths))
        
        return {
            "density_centers": sorted(density_centers, key=lambda x: x["density"], reverse=True),
            "node_strengths": node_strengths.tolist(),
            "local_densities": local_densities.tolist(),
            "density_gradient": density_gradient.tolist(),
            "global_density": global_density,
            "density_variance": density_variance,
            "density_distribution": {
                "min": float(np.min(local_densities)),
                "max": float(np.max(local_densities)),
                "mean": float(np.mean(local_densities)),
                "median": float(np.median(local_densities)),
                "skewness": float(stats.skew(local_densities)),
                "kurtosis": float(stats.kurtosis(local_densities))
            }
        }
        
    def _analyze_flow(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze flow dynamics within the field."""
        # Calculate gradient of the field
        grad_x, grad_y = np.gradient(matrix)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Smooth the gradient for stability
        smoothed_grad_x = ndimage.gaussian_filter(grad_x, sigma=self.config["gradient_smoothing"])
        smoothed_grad_y = ndimage.gaussian_filter(grad_y, sigma=self.config["gradient_smoothing"])
        smoothed_magnitude = np.sqrt(smoothed_grad_x**2 + smoothed_grad_y**2)
        
        # Calculate flow metrics
        avg_gradient = float(np.mean(gradient_magnitude))
        max_gradient = float(np.max(gradient_magnitude))
        
        # Identify flow channels (areas of consistent gradient direction)
        flow_channels = []
        n = matrix.shape[0]
        visited = np.zeros(matrix.shape, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if visited[i, j] or smoothed_magnitude[i, j] < avg_gradient:
                    continue
                    
                # Follow gradient from this point
                channel = self._trace_gradient_flow(smoothed_grad_x, smoothed_grad_y, i, j, visited)
                if len(channel) > 2:  # Minimum length for a channel
                    flow_channels.append(channel)
        
        # Calculate flow directionality (how consistent the flow is)
        if len(smoothed_grad_x.flatten()) > 0:
            grad_angles = np.arctan2(smoothed_grad_y.flatten(), smoothed_grad_x.flatten())
            directionality = float(np.abs(np.mean(np.exp(1j * grad_angles))))
        else:
            directionality = 0.0
        
        return {
            "gradient_magnitude": gradient_magnitude.tolist(),
            "gradient_x": grad_x.tolist(),
            "gradient_y": grad_y.tolist(),
            "avg_gradient": avg_gradient,
            "max_gradient": max_gradient,
            "flow_channels": flow_channels,
            "directionality": directionality,
            "flow_strength": float(avg_gradient * directionality)
        }
    
    def _trace_gradient_flow(self, grad_x: np.ndarray, grad_y: np.ndarray, i: int, j: int, visited: np.ndarray) -> List[Tuple[int, int]]:
        """Trace flow along gradient from starting point."""
        n, m = grad_x.shape
        channel = [(i, j)]
        visited[i, j] = True
        
        cur_i, cur_j = i, j
        while True:
            # Get gradient direction
            di = grad_x[cur_i, cur_j]
            dj = grad_y[cur_i, cur_j]
            
            # Normalize
            mag = np.sqrt(di**2 + dj**2)
            if mag < 1e-6:
                break
                
            di /= mag
            dj /= mag
            
            # Move to next cell (discretize)
            next_i = int(cur_i + np.round(di))
            next_j = int(cur_j + np.round(dj))
            
            # Check bounds
            if next_i < 0 or next_i >= n or next_j < 0 or next_j >= m:
                break
                
            # Check if already visited
            if visited[next_i, next_j]:
                break
                
            # Add to channel
            channel.append((next_i, next_j))
            visited[next_i, next_j] = True
            
            # Update current position
            cur_i, cur_j = next_i, next_j
            
        return channel
        
    def _analyze_potential(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze potential energy landscape of the field."""
        # Create potential field as inverse of resonance strength
        potential_field = 1.0 - matrix / np.max(matrix) if np.max(matrix) > 0 else np.ones_like(matrix)
        
        # Smooth the potential field
        smoothed_potential = ndimage.gaussian_filter(potential_field, sigma=self.config["gradient_smoothing"])
        
        # Find local minima (attraction basins)
        attraction_basins = []
        n = matrix.shape[0]
        
        for i in range(n):
            for j in range(n):
                is_minimum = True
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n and (ni, nj) != (i, j):
                            if smoothed_potential[ni, nj] < smoothed_potential[i, j]:
                                is_minimum = False
                                break
                    if not is_minimum:
                        break
                        
                if is_minimum:
                    attraction_basins.append({
                        "position": (i, j),
                        "strength": float(1.0 - smoothed_potential[i, j]),
                        "radius": float(np.sum((smoothed_potential - smoothed_potential[i, j]) < 0.1))
                    })
                    
        # Calculate potential gradient
        potential_grad_x, potential_grad_y = np.gradient(smoothed_potential)
        potential_gradient_magnitude = np.sqrt(potential_grad_x**2 + potential_grad_y**2)
        
        # Calculate energy metrics
        total_potential = float(np.sum(smoothed_potential))
        avg_potential = float(np.mean(smoothed_potential))
        potential_variance = float(np.var(smoothed_potential))
        
        return {
            "potential_field": smoothed_potential.tolist(),
            "attraction_basins": sorted(attraction_basins, key=lambda x: x["strength"], reverse=True),
            "potential_gradient": potential_gradient_magnitude.tolist(),
            "total_potential": total_potential,
            "avg_potential": avg_potential,
            "potential_variance": potential_variance,
            "energy_landscape": {
                "barriers": float(np.mean(potential_gradient_magnitude)),
                "well_depth": float(np.max(matrix) - np.mean(matrix)) if np.max(matrix) > 0 else 0.0,
                "roughness": float(np.std(smoothed_potential) / np.mean(smoothed_potential)) if np.mean(smoothed_potential) > 0 else 0.0
            }
        }
        
    def _analyze_graph(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze the field as a graph to extract connectivity metrics."""
        # Create graph from adjacency matrix
        G = nx.from_numpy_array(matrix)
        
        # Remove weak edges
        threshold = self.config["edge_threshold"]
        edges_to_remove = [(i, j) for i, j, w in G.edges(data='weight') if w < threshold]
        G.remove_edges_from(edges_to_remove)
        
        # Extract largest connected component if graph is disconnected
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        # Calculate graph metrics
        try:
            avg_path_length = nx.average_shortest_path_length(G, weight='weight')
        except (nx.NetworkXError, ZeroDivisionError):
            avg_path_length = 0.0
            
        try:
            diameter = nx.diameter(G)
        except (nx.NetworkXError, ValueError):
            diameter = 0
            
        try:
            clustering = nx.average_clustering(G, weight='weight')
        except (nx.NetworkXError, ZeroDivisionError):
            clustering = 0.0
        
        # Calculate node centrality
        try:
            centrality = nx.eigenvector_centrality(G, weight='weight')
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
            centrality = {i: 1.0/G.number_of_nodes() for i in G.nodes()}
            
        # Calculate communities (clusters)
        try:
            communities = list(nx.algorithms.community.greedy_modularity_communities(G))
            community_assignment = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    community_assignment[node] = i
        except (nx.NetworkXError, ZeroDivisionError):
            communities = []
            community_assignment = {}
            
        return {
            "avg_path_length": float(avg_path_length),
            "diameter": int(diameter),
            "clustering": float(clustering),
            "node_centrality": [float(centrality.get(i, 0.0)) for i in range(matrix.shape[0])],
            "community_count": len(communities),
            "community_assignment": community_assignment,
            "connectivity": float(G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2)) if G.number_of_nodes() > 1 else 0.0
        }
        
    def _empty_field_result(self) -> Dict[str, Any]:
        """Return empty field analysis result when matrix is too small."""
        return {
            "topology": {
                "effective_dimensionality": 0,
                "principal_dimensions": [],
                "dimension_strengths": [],
                "cumulative_variance": [],
                "pattern_projections": []
            },
            "density": {
                "density_centers": [],
                "node_strengths": [],
                "local_densities": [],
                "density_gradient": [],
                "global_density": 0.0,
                "density_variance": 0.0,
                "density_distribution": {
                    "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0,
                    "skewness": 0.0, "kurtosis": 0.0
                }
            },
            "flow": {
                "gradient_magnitude": [],
                "gradient_x": [],
                "gradient_y": [],
                "avg_gradient": 0.0,
                "max_gradient": 0.0,
                "flow_channels": [],
                "directionality": 0.0,
                "flow_strength": 0.0
            },
            "potential": {
                "potential_field": [],
                "attraction_basins": [],
                "potential_gradient": [],
                "total_potential": 0.0,
                "avg_potential": 0.0,
                "potential_variance": 0.0,
                "energy_landscape": {
                    "barriers": 0.0, "well_depth": 0.0, "roughness": 0.0
                }
            },
            "graph_metrics": {
                "avg_path_length": 0.0,
                "diameter": 0,
                "clustering": 0.0,
                "node_centrality": [],
                "community_count": 0,
                "community_assignment": {},
                "connectivity": 0.0
            },
            "field_properties": {
                "coherence": 0.0,
                "complexity": 0.0,
                "stability": 0.0,
                "density_ratio": 0.0,
                "navigability_score": 0.0
            }
        }