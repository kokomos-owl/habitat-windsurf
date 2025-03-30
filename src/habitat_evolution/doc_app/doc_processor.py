"""
Document Processor using Vector-Tonic-Harmonic Integration

This module adapts the vector-tonic-harmonic integration capabilities
to process text documents, detecting patterns and semantic field dynamics
across document collections.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import json

# Use absolute imports to avoid module path issues
from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from src.habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService, OperationType
from src.habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics
from src.habitat_evolution.adaptive_core.emergence.learning_window_integration import LearningWindowAwareDetector
from src.habitat_evolution.adaptive_core.emergence.tonic_harmonic_integration import TonicHarmonicPatternDetector
from src.habitat_evolution.pattern_aware_rag.learning.learning_control import WindowState, BackPressureController, LearningWindow
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import VectorTonicWindowIntegrator, create_vector_tonic_window_integrator

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes text documents using vector-tonic-harmonic integration.
    
    This class adapts the VectorTonicWindowIntegrator to process text documents,
    detecting patterns and semantic field dynamics across document collections.
    """
    
    def __init__(self, output_dir: str = "output/doc_processing"):
        """
        Initialize the document processor.
        
        Args:
            output_dir: Directory to store processing results
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize event bus
        self.event_bus = LocalEventBus()
        
        # Initialize field state with default field analysis
        initial_field_analysis = {
            "topology": {
                "effective_dimensionality": 5,
                "principal_dimensions": [0, 1, 2],
                "eigenvalues": [0.9, 0.8, 0.7, 0.6, 0.5],
                "eigenvectors": [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(5)]
            },
            "density": {
                "density_centers": [
                    {"position": [0.1, 0.2, 0.3], "strength": 0.8}
                ],
                "density_map": {
                    "resolution": [10, 10, 10],
                    "values": [0.5 for _ in range(1000)]
                }
            },
            "field_properties": {
                "coherence": 0.7,
                "navigability_score": 0.6,
                "stability": 0.8
            }
        }
        self.field_state = TonicHarmonicFieldState(field_analysis=initial_field_analysis)
        
        # Initialize harmonic I/O service
        self.harmonic_io_service = HarmonicIOService(self.event_bus)
        
        # Initialize metrics
        self.metrics = TonicHarmonicMetrics()
        
        # Initialize base detector
        self.base_detector = LearningWindowAwareDetector(self.event_bus)
        
        # Initialize tonic detector
        self.tonic_detector = TonicHarmonicPatternDetector(self.base_detector, self.event_bus)
        
        # Initialize integrator
        self.integrator = create_vector_tonic_window_integrator(
            self.tonic_detector,
            self.event_bus,
            self.harmonic_io_service,
            self.metrics,
            adaptive_soak_period=True
        )
        
        # Initialize learning window
        self.learning_window = LearningWindow(
            window_id="doc_processing_window",
            event_bus=self.event_bus,
            initial_state=WindowState.CLOSED
        )
        
        # Register event handlers
        self._register_event_handlers()
        
        # Document processing results
        self.processing_results = {}
        
        logger.info("DocumentProcessor initialized")
    
    def _register_event_handlers(self):
        """Register handlers for relevant events."""
        self.event_bus.subscribe("pattern.detected", self._on_pattern_detected)
        self.event_bus.subscribe("field.state.updated", self._on_field_state_updated)
        self.event_bus.subscribe("learning.window.state", self._on_window_state_changed)
    
    def _on_pattern_detected(self, event: Event):
        """
        Handle pattern detection events.
        
        Args:
            event: Pattern detection event
        """
        pattern_data = event.data
        pattern_id = pattern_data.get("pattern_id", "unknown")
        
        logger.info(f"Pattern detected: {pattern_id}")
        
        # Store pattern data in results
        if "patterns" not in self.processing_results:
            self.processing_results["patterns"] = []
        
        self.processing_results["patterns"].append(pattern_data)
    
    def _on_field_state_updated(self, event: Event):
        """
        Handle field state update events.
        
        Args:
            event: Field state update event
        """
        field_data = event.data
        
        # Store field state data in results
        if "field_states" not in self.processing_results:
            self.processing_results["field_states"] = []
        
        self.processing_results["field_states"].append(field_data)
    
    def _on_window_state_changed(self, event: Event):
        """
        Handle window state change events.
        
        Args:
            event: Window state change event
        """
        window_data = event.data
        
        # Store window state data in results
        if "window_states" not in self.processing_results:
            self.processing_results["window_states"] = []
        
        self.processing_results["window_states"].append(window_data)
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processing results for the document
        """
        logger.info(f"Processing document: {file_path}")
        
        # Reset processing results
        self.processing_results = {
            "document_path": file_path,
            "processing_time": datetime.now().isoformat(),
            "patterns": [],
            "field_states": [],
            "window_states": []
        }
        
        # Read document content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading document {file_path}: {e}")
            return {"error": str(e)}
        
        # Open learning window
        self.learning_window.open()
        
        # Process document content in chunks
        chunk_size = 1000  # characters
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Create vector gradient data
            gradient_data = self._create_gradient_data(chunk, i, len(chunks))
            
            # Publish vector gradient update event
            self.event_bus.publish(Event(
                "vector.gradient.updated",
                gradient_data,
                source="document_processor"
            ))
            
            # Allow time for processing
            # In a real application, you might want to use asyncio instead
            import time
            time.sleep(0.5)
        
        # Close learning window
        self.learning_window.close()
        
        # Save processing results
        self._save_results()
        
        return self.processing_results
    
    def process_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Process all text documents in a directory.
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            Processing results for all documents
        """
        logger.info(f"Processing directory: {directory_path}")
        
        # Get all text files in the directory
        text_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".txt"):
                    text_files.append(os.path.join(root, file))
        
        # Process each document
        all_results = {
            "directory_path": directory_path,
            "processing_time": datetime.now().isoformat(),
            "document_count": len(text_files),
            "documents": []
        }
        
        for file_path in text_files:
            result = self.process_document(file_path)
            all_results["documents"].append(result)
        
        # Save overall results
        with open(os.path.join(self.output_dir, "directory_results.json"), 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results
    
    def _create_gradient_data(self, text_chunk: str, chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """
        Create vector gradient data from a text chunk.
        
        Args:
            text_chunk: Text chunk to process
            chunk_index: Index of the current chunk
            total_chunks: Total number of chunks
            
        Returns:
            Vector gradient data
        """
        # In a real application, you would use a proper embedding model here
        # For this example, we'll create synthetic gradient data
        
        # Calculate progress
        progress = (chunk_index + 1) / total_chunks
        
        # Create synthetic vector data
        vector_data = {
            "text": text_chunk,
            "embedding": np.random.rand(384).tolist(),  # Simulate embedding
            "progress": progress,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks
        }
        
        # Create gradient data
        gradient_data = {
            "vector_data": vector_data,
            "field_properties": {
                "coherence": 0.5 + 0.3 * np.sin(progress * np.pi),
                "stability": 0.6 + 0.2 * np.cos(progress * np.pi / 2),
                "density": 0.4 + 0.4 * progress
            },
            "topology": {
                "effective_dimensionality": 5 + int(10 * progress),
                "principal_dimensions": list(range(3 + int(5 * progress))),
                "eigenvalues": [0.9 - 0.1 * i for i in range(5)],
                "eigenvectors": [[np.random.rand() for _ in range(5)] for _ in range(5)]
            },
            "density": {
                "density_centers": [
                    {"position": [np.random.rand() for _ in range(3)], "strength": 0.7 + 0.2 * np.random.rand()}
                    for _ in range(2 + int(3 * progress))
                ],
                "density_map": {
                    "resolution": [10, 10, 10],
                    "values": [np.random.rand() for _ in range(1000)]
                }
            }
        }
        
        return gradient_data
    
    def _save_results(self):
        """Save processing results to output directory."""
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get document name from path
        doc_name = os.path.basename(self.processing_results.get("document_path", "unknown"))
        
        # Create filename
        filename = f"{timestamp}_{doc_name}_results.json"
        
        # Save results
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            json.dump(self.processing_results, f, indent=2)
        
        logger.info(f"Saved processing results to {filename}")
        
        # Generate summary file
        self._generate_summary()
    
    def _generate_summary(self):
        """Generate a summary of the processing results."""
        # Create summary data
        summary = {
            "document": os.path.basename(self.processing_results.get("document_path", "unknown")),
            "processing_time": self.processing_results.get("processing_time"),
            "pattern_count": len(self.processing_results.get("patterns", [])),
            "field_state_count": len(self.processing_results.get("field_states", [])),
            "window_state_count": len(self.processing_results.get("window_states", [])),
            "top_patterns": self._extract_top_patterns()
        }
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get document name from path
        doc_name = os.path.basename(self.processing_results.get("document_path", "unknown"))
        
        # Create filename
        filename = f"{timestamp}_{doc_name}_summary.md"
        
        # Save summary as markdown
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            f.write(f"# Document Processing Summary\n\n")
            f.write(f"## Document: {summary['document']}\n\n")
            f.write(f"**Processing Time:** {summary['processing_time']}\n\n")
            f.write(f"**Pattern Count:** {summary['pattern_count']}\n\n")
            f.write(f"**Field State Count:** {summary['field_state_count']}\n\n")
            f.write(f"**Window State Count:** {summary['window_state_count']}\n\n")
            
            f.write(f"## Top Patterns\n\n")
            for pattern in summary['top_patterns']:
                f.write(f"### {pattern.get('name', 'Unnamed Pattern')}\n\n")
                f.write(f"**Confidence:** {pattern.get('confidence', 'N/A')}\n\n")
                f.write(f"**Description:** {pattern.get('description', 'No description')}\n\n")
                
                if 'relationships' in pattern:
                    f.write(f"**Relationships:**\n\n")
                    for rel in pattern.get('relationships', []):
                        f.write(f"- {rel}\n")
                    f.write("\n")
        
        logger.info(f"Saved processing summary to {filename}")
    
    def _extract_top_patterns(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Extract top patterns from processing results.
        
        Args:
            limit: Maximum number of patterns to extract
            
        Returns:
            List of top patterns
        """
        patterns = self.processing_results.get("patterns", [])
        
        # Sort patterns by confidence
        sorted_patterns = sorted(
            patterns,
            key=lambda p: p.get("confidence", 0),
            reverse=True
        )
        
        # Take top patterns
        top_patterns = sorted_patterns[:limit]
        
        # Format patterns for summary
        formatted_patterns = []
        for pattern in top_patterns:
            formatted_pattern = {
                "name": pattern.get("pattern_id", "Unnamed Pattern"),
                "confidence": pattern.get("confidence", "N/A"),
                "description": pattern.get("description", "No description")
            }
            
            # Extract relationships if available
            if "pattern_data" in pattern and "relationships" in pattern["pattern_data"]:
                formatted_pattern["relationships"] = pattern["pattern_data"]["relationships"]
            
            formatted_patterns.append(formatted_pattern)
        
        return formatted_patterns
