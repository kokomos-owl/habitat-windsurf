#!/bin/bash

# Remove redundant virtual environment
rm -rf venv/

# Remove cache files
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete
find . -type f -name ".DS_Store" -delete
rm -rf .pytest_cache/

# Remove empty directories
rm -rf temp_docs/
rm -rf docs/examples/
rm -rf notebooks/resources/
rm -rf notebooks/lesson_02/
rm -rf src/lessons/

# Clean output directories
rm -rf examples/output/*

echo "Cleanup completed!"
