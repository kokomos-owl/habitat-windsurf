"""
Integration test for the Boston Harbor Islands complex document.

This test evaluates the enhanced eigenspace window management approach
on a complex climate risk document with multiple semantic domains.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.habitat_evolution.pattern_aware_rag.learning.eigenspace_window_manager import EigenspaceWindowManager
from src.habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from src.habitat_evolution.pattern_aware_rag.learning.learning_window_manager import LearningWindowManager
from src.habitat_evolution.pattern_aware_rag.pattern_manager import PatternManager
from src.habitat_evolution.pattern_aware_rag.pattern_store import PatternStore
from src.habitat_evolution.pattern_aware_rag.embeddings.embedding_service import EmbeddingService
from src.habitat_evolution.pattern_aware_rag.text_processor import TextProcessor
from src.habitat_evolution.pattern_aware_rag.resonance_calculator import ResonanceCalculator
from src.habitat_evolution.pattern_aware_rag.pattern_evolution_engine import PatternEvolutionEngine

def test_boston_harbor_islands_document():
    """
    Test the enhanced eigenspace window management on the Boston Harbor Islands document.
    
    This test evaluates:
    1. Domain detection capabilities
    2. Window boundary detection
    3. Coherence metrics
    4. Vector + Tonic-Harmonic improvement
    """
    # Initialize components
    text_processor = TextProcessor()
    embedding_service = EmbeddingService()
    pattern_store = PatternStore()
    pattern_manager = PatternManager(pattern_store)
    resonance_calculator = ResonanceCalculator()
    field_analyzer = TopologicalFieldAnalyzer()
    window_manager = LearningWindowManager()
    
    # Initialize our enhanced eigenspace window manager
    eigenspace_manager = EigenspaceWindowManager(
        field_analyzer=field_analyzer,
        window_manager=window_manager
    )
    
    # Initialize pattern evolution engine
    pattern_engine = PatternEvolutionEngine(
        text_processor=text_processor,
        embedding_service=embedding_service,
        pattern_manager=pattern_manager,
        resonance_calculator=resonance_calculator,
        field_analyzer=field_analyzer,
        window_manager=eigenspace_manager
    )
    
    # Load the Boston Harbor Islands document
    doc_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../../../data/climate_risk/complex_test_doc_boston_harbor_islands.txt'
    ))
    
    with open(doc_path, 'r') as f:
        document_text = f.read()
    
    # Process the document
    logger.info("Processing Boston Harbor Islands document...")
    chunks = text_processor.chunk_text(document_text)
    
    # Extract patterns
    patterns = []
    for chunk in chunks:
        pattern = pattern_engine.extract_pattern(chunk)
        patterns.append(pattern)
    
    logger.info(f"Extracted {len(patterns)} patterns from document")
    
    # Get semantic vectors for all patterns
    semantic_vectors = np.array([pattern.vector for pattern in patterns])
    
    # Test 1: Detect natural boundaries using enhanced approach
    logger.info("Detecting natural boundaries using enhanced eigenspace approach...")
    boundaries = eigenspace_manager.detect_natural_boundaries(semantic_vectors)
    logger.info(f"Detected {len(boundaries)} natural boundaries: {boundaries}")
    
    # Test 2: Analyze boundaries for coherence
    logger.info("Analyzing boundary coherence...")
    window_coherence = []
    for start_idx, end_idx in boundaries:
        window_vectors = semantic_vectors[start_idx:end_idx]
        
        # Compute similarity matrix
        similarity_matrix = eigenspace_manager._compute_resonance_matrix(window_vectors)
        
        # Perform eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(similarity_matrix)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate coherence of the window
        if len(eigenvalues) > 0:
            coherence = eigenspace_manager._calculate_cluster_coherence(eigenvectors[:, 0])
            window_coherence.append(coherence)
            logger.info(f"Window {start_idx}:{end_idx} - Coherence: {coherence:.4f}")
    
    # Test 3: Compare vector-only vs. vector+tonic-harmonic approach
    logger.info("Comparing vector-only vs. vector+tonic-harmonic approach...")
    
    # Vector-only approach (using only cosine similarity)
    vector_only_coherence = np.mean([
        np.mean(eigenspace_manager._compute_resonance_matrix(
            semantic_vectors[start_idx:end_idx]
        )) 
        for start_idx, end_idx in boundaries
    ])
    
    # Vector+tonic-harmonic approach (using our enhanced eigenspace approach)
    vector_tonic_harmonic_coherence = np.mean(window_coherence)
    
    improvement_factor = vector_tonic_harmonic_coherence / vector_only_coherence if vector_only_coherence > 0 else 0
    
    logger.info(f"Vector-Only Coherence: {vector_only_coherence:.4f}")
    logger.info(f"Vector+Tonic-Harmonic Coherence: {vector_tonic_harmonic_coherence:.4f}")
    logger.info(f"Improvement Factor: {improvement_factor:.2f}x")
    
    # Test 4: Analyze multi-scale persistence of boundaries
    logger.info("Analyzing multi-scale persistence of boundaries...")
    
    # Apply different thresholds
    thresholds = [1.2, 1.5, 1.8, 2.1]
    all_boundaries = []
    
    for threshold in thresholds:
        # Create a temporary eigenspace manager with this threshold
        temp_manager = EigenspaceWindowManager(
            field_analyzer=field_analyzer,
            window_manager=window_manager,
            eigenvalue_ratio_threshold=threshold
        )
        
        # Get boundaries at this threshold
        scale_boundaries = temp_manager.detect_natural_boundaries(semantic_vectors)
        all_boundaries.append(scale_boundaries)
        logger.info(f"Threshold {threshold}: {len(scale_boundaries)} boundaries")
    
    # Count boundary persistence
    boundary_points = {}
    for scale_idx, scale_boundaries in enumerate(all_boundaries):
        for start_idx, end_idx in scale_boundaries:
            # Count start points
            if start_idx in boundary_points:
                boundary_points[start_idx] += 1
            else:
                boundary_points[start_idx] = 1
            
            # Count end points
            if end_idx in boundary_points:
                boundary_points[end_idx] += 1
            else:
                boundary_points[end_idx] = 1
    
    # Find persistent boundaries (appearing in at least 3 scales)
    persistent_boundaries = [point for point, count in boundary_points.items() if count >= 3]
    persistent_boundaries.sort()
    
    logger.info(f"Persistent boundary points: {persistent_boundaries}")
    
    # Test 5: Analyze eigenvector stability across boundaries
    logger.info("Analyzing eigenvector stability across boundaries...")
    
    eigenvector_changes = []
    for i in range(1, len(semantic_vectors)):
        if i >= 2 and i < len(semantic_vectors) - 2:
            left_vectors = semantic_vectors[i-2:i]
            right_vectors = semantic_vectors[i:i+2]
            
            # Calculate average vectors for each side
            left_avg = np.mean(left_vectors, axis=0)
            right_avg = np.mean(right_vectors, axis=0)
            
            # Normalize
            left_norm = np.linalg.norm(left_avg)
            right_norm = np.linalg.norm(right_avg)
            
            if left_norm > 0 and right_norm > 0:
                left_avg = left_avg / left_norm
                right_avg = right_avg / right_norm
                
                # Calculate projection distance
                projection_distance = 1.0 - np.abs(np.dot(left_avg, right_avg))
                eigenvector_changes.append((i, projection_distance))
    
    # Find significant eigenvector changes
    significant_changes = [(idx, dist) for idx, dist in eigenvector_changes if dist > 0.2]
    significant_changes.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Top 5 significant eigenvector changes: {significant_changes[:5]}")
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Document: Boston Harbor Islands Climate Risk Assessment")
    logger.info(f"Patterns extracted: {len(patterns)}")
    logger.info(f"Natural boundaries detected: {len(boundaries)}")
    logger.info(f"Average window coherence: {np.mean(window_coherence):.4f}")
    logger.info(f"Vector-only vs. Vector+Tonic-Harmonic improvement: {improvement_factor:.2f}x")
    logger.info(f"Persistent boundary points: {len(persistent_boundaries)}")
    logger.info(f"Significant eigenvector changes: {len(significant_changes)}")
    
    # Assertions to verify the test passed
    assert len(boundaries) > 0, "Should detect at least one natural boundary"
    assert np.mean(window_coherence) > 0.5, "Average window coherence should be above 0.5"
    assert improvement_factor > 1.0, "Vector+Tonic-Harmonic should outperform vector-only approach"
    
    logger.info("Boston Harbor Islands document test completed successfully")

if __name__ == "__main__":
    test_boston_harbor_islands_document()
