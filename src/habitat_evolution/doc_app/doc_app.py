"""Document Application

This module provides a command-line interface for processing text documents
using the vector-tonic-harmonic integration capabilities of Habitat Evolution.
"""

import argparse
import logging
import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

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

# Import the duplicated vector_tonic_window_integration as doc_patterns
from src.habitat_evolution.doc_app.doc_patterns import VectorTonicWindowIntegrator, create_vector_tonic_window_integrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents using vector-tonic-harmonic integration."""
    
    def __init__(self, output_dir: str = "output/doc_processing"):
        """Initialize the document processor."""
        self.output_dir = output_dir
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
            },
            "patterns": {}  # Required by TonicHarmonicFieldState
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
        self.detected_patterns = []
        
        logger.info("DocumentProcessor initialized")
    
    def _register_event_handlers(self):
        """Register handlers for relevant events."""
        self.event_bus.subscribe("pattern.detected", self._on_pattern_detected)
        self.event_bus.subscribe("field.state.updated", self._on_field_state_updated)
        self.event_bus.subscribe("learning.window.state", self._on_window_state_changed)
    
    def _on_pattern_detected(self, event: Event):
        """Handle pattern detection events."""
        pattern_data = event.data
        pattern_id = pattern_data.get("pattern_id", "unknown")
        logger.info(f"Pattern detected: {pattern_id}")
        self.detected_patterns.append(pattern_data)
    
    def _on_field_state_updated(self, event: Event):
        """Handle field state update events."""
        field_data = event.data
        logger.debug(f"Field state updated: {field_data.get('field_properties', {})}")
    
    def _on_window_state_changed(self, event: Event):
        """Handle window state change events."""
        window_data = event.data
        logger.info(f"Window state changed to: {window_data.get('state')}")
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a single document file."""
        logger.info(f"Processing document: {file_path}")
        
        # Reset processing results
        self.processing_results = {
            "document_path": file_path,
            "processing_time": datetime.now().isoformat(),
            "patterns": []
        }
        self.detected_patterns = []
        
        # Read document content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading document {file_path}: {e}")
            return {"error": str(e)}
        
        # Open learning window
        logger.info("Opening learning window")
        self.learning_window.open()
        time.sleep(1)  # Give time for window to open
        
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
            
            # Publish field state update event
            self.event_bus.publish(Event(
                "field.state.updated",
                gradient_data,  # Reuse gradient data for field state
                source="document_processor"
            ))
            
            # Allow time for processing
            time.sleep(0.5)
        
        # Close learning window
        logger.info("Closing learning window")
        self.learning_window.close()
        time.sleep(1)  # Give time for window to close
        
        # Update processing results with detected patterns
        self.processing_results["patterns"] = self.detected_patterns
        
        # Save processing results
        self._save_results()
        
        return self.processing_results
    
    def _create_gradient_data(self, text_chunk: str, chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """Create vector gradient data from a text chunk."""
        # Calculate progress
        progress = (chunk_index + 1) / total_chunks
        
        # Create synthetic vector data
        import numpy as np
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
                "density": 0.4 + 0.4 * progress,
                "navigability_score": 0.7
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
            },
            "patterns": {}  # Required by field state
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
        self._generate_summary(doc_name, timestamp)
    
    def _generate_summary(self, doc_name: str, timestamp: str):
        """Generate a summary of the processing results."""
        # Create summary data
        summary = {
            "document": doc_name,
            "processing_time": self.processing_results.get("processing_time"),
            "pattern_count": len(self.detected_patterns)
        }
        
        # Create filename
        filename = f"{timestamp}_{doc_name}_summary.md"
        
        # Save summary as markdown
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            f.write(f"# Document Processing Summary\n\n")
            f.write(f"## Document: {summary['document']}\n\n")
            f.write(f"**Processing Time:** {summary['processing_time']}\n\n")
            f.write(f"**Pattern Count:** {summary['pattern_count']}\n\n")
            
            f.write(f"## Detected Patterns\n\n")
            for i, pattern in enumerate(self.detected_patterns):
                pattern_id = pattern.get("pattern_id", f"Pattern {i+1}")
                confidence = pattern.get("confidence", "N/A")
                f.write(f"### {pattern_id}\n\n")
                f.write(f"**Confidence:** {confidence}\n\n")
                
                # Add pattern data if available
                if "pattern_data" in pattern:
                    pattern_data = pattern["pattern_data"]
                    f.write(f"**Data:**\n\n```json\n{json.dumps(pattern_data, indent=2)}\n```\n\n")
        
        logger.info(f"Saved processing summary to {filename}")


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Process text documents using Habitat Evolution's vector-tonic-harmonic integration"
    )
    
    parser.add_argument(
        "input",
        help="Path to input file or directory"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output/doc_processing",
        help="Directory to store processing results (default: output/doc_processing)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process input
    start_time = datetime.now()
    logger.info(f"Starting document processing at {start_time.isoformat()}")
    
    # Create processor
    processor = DocumentProcessor(output_dir=args.output_dir)
    
    # Process document
    if os.path.isfile(args.input):
        results = processor.process_document(args.input)
    else:
        logger.error(f"Input path is not a file: {args.input}")
        return 1
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    logger.info(f"Document processing completed in {processing_time:.2f} seconds")
    
    # Print summary
    pattern_count = len(results.get("patterns", []))
    logger.info(f"Detected {pattern_count} patterns in document")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
