"""
Read and print the metrics file
"""

import json
import sys
from pathlib import Path

def main():
    # Get the metrics file path from the first argument or use default
    if len(sys.argv) > 1:
        metrics_file = sys.argv[1]
    else:
        # Find the most recent metrics file in the output directory
        output_dir = Path(__file__).parent / "output"
        metrics_files = list(output_dir.glob("processing_metrics_*.json"))
        if not metrics_files:
            print("No metrics files found in output directory")
            return
        metrics_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
    
    # Read and print the metrics file
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print(f"=== Climate Risk Processing Metrics ===")
        print(f"Files processed: {metrics.get('files_processed', 0)}")
        print(f"Entities discovered: {metrics.get('entities_discovered', 0)}")
        print(f"Domains discovered: {metrics.get('domains_discovered', 0)}")
        print(f"Relationships discovered: {metrics.get('relationships_discovered', 0)}")
        print(f"Processing time: {metrics.get('processing_time_seconds', 0):.2f} seconds")
        
        # Print processed files
        print("\nProcessed Files:")
        for file in metrics.get('files', []):
            print(f"- {file.get('name')}: {file.get('size_bytes')} bytes")
        
    except Exception as e:
        print(f"Error reading metrics file: {e}")

if __name__ == "__main__":
    main()
