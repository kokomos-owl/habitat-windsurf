#!/usr/bin/env python
"""
Stillpoint Analysis Runner

This script runs the stillpoint data through the vector-tonic-window integration
without modifying any original files.
"""

import os
import sys
import logging
import time
from datetime import datetime
import json
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the src directory to the path
import sys
sys.path.insert(0, '/Users/prphillips/Documents/GitHub/habitat-windsurf')

# Import the necessary modules
from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from src.habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService, OperationType
from src.habitat_evolution.adaptive_core.resonance.tonic_harmonic_metrics import TonicHarmonicMetrics
from src.habitat_evolution.adaptive_core.emergence.learning_window_integration import LearningWindowAwareDetector
from src.habitat_evolution.adaptive_core.emergence.tonic_harmonic_integration import TonicHarmonicPatternDetector
from src.habitat_evolution.pattern_aware_rag.learning.learning_control import WindowState, BackPressureController, LearningWindow

# Import the vector-tonic-window integration
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_window_integration import (
    create_vector_tonic_window_integrator
)


def setup_environment():
    """Set up the processing environment."""
    # Create event bus
    event_bus = LocalEventBus()
    
    # Create harmonic I/O service
    harmonic_io_service = HarmonicIOService(event_bus)
    
    # Create metrics
    metrics = TonicHarmonicMetrics()
    
    # Create base detector
    base_detector = LearningWindowAwareDetector(event_bus)
    
    # Create tonic detector
    tonic_detector = TonicHarmonicPatternDetector(base_detector, event_bus)
    
    # Create integrator
    integrator = create_vector_tonic_window_integrator(
        tonic_detector,
        event_bus,
        harmonic_io_service,
        metrics,
        adaptive_soak_period=True
    )
    
    # Create learning window
    learning_window = LearningWindow(
        window_id="stillpoint_window",
        event_bus=event_bus,
        initial_state=WindowState.CLOSED
    )
    
    # Create output directory
    output_dir = "output/stillpoint_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    return {
        "event_bus": event_bus,
        "harmonic_io_service": harmonic_io_service,
        "metrics": metrics,
        "base_detector": base_detector,
        "tonic_detector": tonic_detector,
        "integrator": integrator,
        "learning_window": learning_window,
        "output_dir": output_dir
    }


def process_stillpoint_data(env):
    """Process the stillpoint data."""
    # Get the stillpoint data directory
    stillpoint_dir = "demos/data/stillpoint"
    
    # Get all text files in the directory
    text_files = []
    for file in os.listdir(stillpoint_dir):
        if file.endswith(".txt"):
            text_files.append(os.path.join(stillpoint_dir, file))
    
    # Set up pattern detection tracking
    detected_patterns = []
    
    # Register event handler for pattern detection
    def on_pattern_detected(event):
        pattern_data = event.data
        pattern_id = pattern_data.get("pattern_id", "unknown")
        logger.info(f"Pattern detected: {pattern_id}")
        detected_patterns.append(pattern_data)
    
    env["event_bus"].subscribe("pattern.detected", on_pattern_detected)
    
    # Process each file
    for file_path in text_files:
        logger.info(f"Processing file: {file_path}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Open learning window
        logger.info("Opening learning window")
        env["learning_window"].open()
        time.sleep(1)  # Give time for window to open
        
        # Process content in chunks
        chunk_size = 1000  # characters
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Create vector gradient data
            gradient_data = create_gradient_data(chunk, i, len(chunks))
            
            # Publish vector gradient update event
            env["event_bus"].publish(Event(
                "vector.gradient.updated",
                gradient_data,
                source="stillpoint_processor"
            ))
            
            # Publish field state update event
            env["event_bus"].publish(Event(
                "field.state.updated",
                gradient_data,  # Reuse gradient data for field state
                source="stillpoint_processor"
            ))
            
            # Allow time for processing
            time.sleep(0.5)
        
        # Close learning window
        logger.info("Closing learning window")
        env["learning_window"].close()
        time.sleep(1)  # Give time for window to close
    
    # Return detected patterns
    return detected_patterns


def create_gradient_data(text_chunk, chunk_index, total_chunks):
    """Create vector gradient data from a text chunk."""
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


def save_results(detected_patterns, output_dir):
    """Save the processing results."""
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results data
    results = {
        "processing_time": datetime.now().isoformat(),
        "pattern_count": len(detected_patterns),
        "patterns": detected_patterns
    }
    
    # Save results to JSON file
    results_file = os.path.join(output_dir, f"{timestamp}_stillpoint_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Create summary markdown file
    summary_file = os.path.join(output_dir, f"{timestamp}_stillpoint_summary.md")
    with open(summary_file, 'w') as f:
        f.write("# Stillpoint Analysis Summary\n\n")
        f.write(f"**Processing Time:** {results['processing_time']}\n\n")
        f.write(f"**Pattern Count:** {results['pattern_count']}\n\n")
        
        f.write("## Detected Patterns\n\n")
        for i, pattern in enumerate(detected_patterns):
            pattern_id = pattern.get("pattern_id", f"Pattern {i+1}")
            confidence = pattern.get("confidence", "N/A")
            f.write(f"### {pattern_id}\n\n")
            f.write(f"**Confidence:** {confidence}\n\n")
            
            # Add pattern data if available
            if "pattern_data" in pattern:
                pattern_data = pattern["pattern_data"]
                f.write(f"**Data:**\n\n```json\n{json.dumps(pattern_data, indent=2)}\n```\n\n")
    
    logger.info(f"Summary saved to {summary_file}")


def main():
    """Main entry point."""
    logger.info("Starting Stillpoint Analysis")
    
    # Set up environment
    env = setup_environment()
    
    # Process stillpoint data
    detected_patterns = process_stillpoint_data(env)
    
    # Save results
    save_results(detected_patterns, env["output_dir"])
    
    logger.info(f"Analysis complete. Detected {len(detected_patterns)} patterns.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
