"""
Integration test for the Boston Harbor Islands complex document.

This test evaluates the enhanced eigenspace window management approach
on a complex climate risk document with multiple semantic domains.
"""

import os
import sys
import numpy as np
import re
import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tabulate import tabulate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Import the EigenspaceWindowManager class directly
sys.path.append(os.path.join(project_root, 'src'))
from habitat_evolution.pattern_aware_rag.learning.eigenspace_window_manager import EigenspaceWindowManager

def test_boston_harbor_islands_document():
    """
    Test the enhanced eigenspace window management on the Boston Harbor Islands document.
    
    This test evaluates:
    1. Domain detection capabilities
    2. Window boundary detection
    3. Coherence metrics
    4. Vector + Tonic-Harmonic improvement
    """
    # Create a standalone eigenspace manager for testing
    # We'll mock the dependencies it needs
    class MockFieldAnalyzer:
        def analyze_eigenspace(self, matrix):
            return np.linalg.eigh(matrix)
    
    class MockWindowManager:
        def create_window(self, start_idx, end_idx):
            return {'start': start_idx, 'end': end_idx}
    
    # Initialize our enhanced eigenspace window manager with mocks
    eigenspace_manager = EigenspaceWindowManager(
        field_analyzer=MockFieldAnalyzer(),
        window_manager=MockWindowManager(),
        projection_distance_threshold=0.3
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
    
    # Simple chunking by paragraphs
    chunks = re.split(r'\n\n+', document_text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    # Download NLTK resources if needed
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
    
    # Extract key terms from each chunk using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Function to extract top terms for each chunk
    def get_top_terms(tfidf_row, feature_names, n=5):
        # Get indices of top n TF-IDF scores
        indices = np.argsort(tfidf_row.toarray()[0])[-n:]
        # Return the corresponding terms
        return [feature_names[i] for i in indices]
    
    # Extract top terms for each chunk
    chunk_top_terms = []
    for i in range(len(chunks)):
        top_terms = get_top_terms(tfidf_matrix[i], feature_names)
        chunk_top_terms.append(top_terms)
    
    # Create simple semantic vectors (for testing purposes)
    # In a real scenario, these would be embeddings from a language model
    np.random.seed(42)  # For reproducibility
    semantic_vectors = np.random.randn(len(chunks), 384)  # Simulate embeddings
    
    logger.info(f"Created {len(chunks)} chunks from document")
    
    # We're using simulated semantic vectors for this test
    
    # Test 1: Detect natural boundaries using enhanced approach
    logger.info("Detecting natural boundaries using enhanced eigenspace approach...")
    boundaries = eigenspace_manager.detect_natural_boundaries(semantic_vectors)
    logger.info(f"Detected {len(boundaries)} natural boundaries: {boundaries}")
    
    # Test 2: Analyze boundaries for coherence and extract semantic themes
    logger.info("Analyzing boundary coherence and semantic themes...")
    window_coherence = []
    window_semantic_themes = []
    
    for start_idx, end_idx in boundaries:
        window_vectors = semantic_vectors[start_idx:end_idx]
        window_chunks = chunks[start_idx:end_idx]
        window_terms = [term for chunk_idx in range(start_idx, end_idx) for term in chunk_top_terms[chunk_idx]]
        
        # Get most common terms in this window to identify the semantic theme
        term_counter = Counter(window_terms)
        most_common_terms = term_counter.most_common(5)
        semantic_theme = ", ".join([term for term, _ in most_common_terms])
        window_semantic_themes.append(semantic_theme)
        
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
            logger.info(f"Window {start_idx}:{end_idx} - Theme: '{semantic_theme}' - Coherence: {coherence:.4f}")
    
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
            field_analyzer=MockFieldAnalyzer(),
            window_manager=MockWindowManager(),
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
    
    # Test 5: Analyze eigenvector stability across boundaries with semantic interpretation
    logger.info("Analyzing eigenvector stability across boundaries with semantic interpretation...")
    
    eigenvector_changes = []
    semantic_transitions = []
    
    for i in range(1, len(semantic_vectors)):
        if i >= 2 and i < len(semantic_vectors) - 2:
            left_vectors = semantic_vectors[i-2:i]
            right_vectors = semantic_vectors[i:i+2]
            
            # Get semantic terms for both sides
            left_terms = [term for chunk_idx in range(i-2, i) for term in chunk_top_terms[chunk_idx]]
            right_terms = [term for chunk_idx in range(i, i+2) for term in chunk_top_terms[chunk_idx]]
            
            left_counter = Counter(left_terms)
            right_counter = Counter(right_terms)
            
            left_theme = ", ".join([term for term, _ in left_counter.most_common(3)])
            right_theme = ", ".join([term for term, _ in right_counter.most_common(3)])
            
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
                
                # Store semantic transition information
                semantic_transitions.append({
                    'position': i,
                    'distance': projection_distance,
                    'from_theme': left_theme,
                    'to_theme': right_theme
                })
    
    # Find significant eigenvector changes
    significant_changes = [(idx, dist) for idx, dist in eigenvector_changes if dist > 0.2]
    significant_changes.sort(key=lambda x: x[1], reverse=True)
    
    # Get semantic interpretations for top changes
    top_semantic_transitions = []
    for idx, dist in significant_changes[:5]:
        transition = next((t for t in semantic_transitions if t['position'] == idx), None)
        if transition:
            top_semantic_transitions.append(transition)
    
    # Log the top semantic transitions
    logger.info("Top 5 significant semantic transitions:")
    
    # Create a table for better visualization
    transition_table = []
    for i, transition in enumerate(top_semantic_transitions):
        transition_table.append([
            i+1,
            transition['position'],
            transition['from_theme'],
            transition['to_theme'],
            f"{transition['distance']:.4f}"
        ])
    
    # Print the table
    table_str = tabulate(
        transition_table,
        headers=["#", "Position", "From Theme", "To Theme", "Distance"],
        tablefmt="grid"
    )
    
    # Log each line of the table
    for line in table_str.split('\n'):
        logger.info(line)
    
    # Summary with semantic interpretation
    logger.info("\n=== SUMMARY WITH SEMANTIC INTERPRETATION ===")
    logger.info(f"Document: Boston Harbor Islands Climate Risk Assessment")
    logger.info(f"Chunks processed: {len(chunks)}")
    logger.info(f"Natural boundaries detected: {len(boundaries)}")
    
    # Report on semantic domains identified
    logger.info("\nSemantic domains identified:")
    
    # Create a table of semantic domains
    domain_table = []
    for i, (boundary, theme) in enumerate(zip(boundaries, window_semantic_themes)):
        # Get the coherence for this domain
        domain_coherence = window_coherence[i] if i < len(window_coherence) else 0.0
        
        # Add to table
        domain_table.append([
            i+1,
            theme,
            f"{boundary[0]}-{boundary[1]}",
            f"{domain_coherence:.4f}"
        ])
    
    # Print the table
    domain_table_str = tabulate(
        domain_table,
        headers=["Domain", "Theme", "Chunks", "Coherence"],
        tablefmt="grid"
    )
    
    # Log each line of the table
    for line in domain_table_str.split('\n'):
        logger.info(line)
    
    # Report on coherence metrics
    logger.info("\nCoherence metrics:")
    logger.info(f"  Average window coherence: {np.mean(window_coherence):.4f}")
    logger.info(f"  Vector-only coherence: {vector_only_coherence:.4f}")
    logger.info(f"  Vector+Tonic-Harmonic coherence: {vector_tonic_harmonic_coherence:.4f}")
    logger.info(f"  Improvement factor: {improvement_factor:.2f}x")
    
    # Report on boundary stability
    logger.info("\nBoundary stability:")
    logger.info(f"  Persistent boundary points: {len(persistent_boundaries)}")
    logger.info(f"  Significant semantic transitions: {len(significant_changes)}")
    
    # Report on multi-scale analysis
    logger.info("\nMulti-scale analysis:")
    
    # Create a table for multi-scale analysis
    scale_table = []
    for i, threshold in enumerate(thresholds):
        scale_table.append([
            i+1,
            threshold,
            len(all_boundaries[i]),
            ", ".join([f"{b[0]}-{b[1]}" for b in all_boundaries[i][:3]]) + 
            ("..." if len(all_boundaries[i]) > 3 else "")
        ])
    
    # Print the table
    scale_table_str = tabulate(
        scale_table,
        headers=["Scale", "Threshold", "Boundaries", "Sample Boundaries"],
        tablefmt="grid"
    )
    
    # Log each line of the table
    for line in scale_table_str.split('\n'):
        logger.info(line)
        
    # Create a visual representation of semantic flow
    logger.info("\nSemantic Flow Visualization:")
    
    # Find the most significant transitions (top 3)
    top_transitions = sorted(semantic_transitions, key=lambda x: x['distance'], reverse=True)[:3]
    top_positions = [t['position'] for t in top_transitions]
    
    # Create a visual flow
    flow_str = ""
    for i in range(len(boundaries)):
        start, end = boundaries[i]
        theme = window_semantic_themes[i]
        
        # Truncate theme if too long
        short_theme = theme[:20] + "..." if len(theme) > 20 else theme
        
        # Check if this domain has a significant transition
        has_transition = end in top_positions
        
        # Add to flow string
        flow_str += f"[{short_theme}]"
        
        # Add connector
        if i < len(boundaries) - 1:
            if has_transition:
                flow_str += " ==> "  # Strong transition
            else:
                flow_str += " --> "  # Normal transition
    
    # Log the flow visualization (in chunks if needed)
    max_line_length = 80
    for i in range(0, len(flow_str), max_line_length):
        logger.info(flow_str[i:i+max_line_length])
    
    # Assertions to verify the test passed
    assert len(boundaries) > 0, "Should detect at least one natural boundary"
    assert np.mean(window_coherence) > 0.5, "Average window coherence should be above 0.5"
    assert improvement_factor > 1.0, "Vector+Tonic-Harmonic should outperform vector-only approach"
    
    logger.info("Boston Harbor Islands document test completed successfully")

if __name__ == "__main__":
    test_boston_harbor_islands_document()
