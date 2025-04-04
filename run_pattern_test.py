#!/usr/bin/env python
"""
Helper script to run the PatternAwareRAG sequential tests with the correct import paths.
"""
import os
import sys
import subprocess

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

# Run the test with pytest
test_path = 'src/tests/pattern_aware_rag/sequential/test_pattern_aware_rag.py'
cmd = ['python', '-m', 'pytest', test_path, '-v']

print(f"Running test: {' '.join(cmd)}")
print(f"PYTHONPATH includes: {src_path}")

# Execute the test
result = subprocess.run(cmd, env={**os.environ, 'PYTHONPATH': src_path})

# Return the exit code
sys.exit(result.returncode)
