# Habitat Evolution Examples

This directory contains example scripts demonstrating various features of Habitat Evolution.

## Running Examples

To run these examples, you need to install the Habitat Evolution package in development mode:

```bash
# From the project root directory
pip install -e .
```

This will make the imports work correctly in the example scripts.

Alternatively, you can use the PYTHONPATH environment variable to add the src directory to the Python path:

```bash
# From the project root directory
PYTHONPATH=$PYTHONPATH:$(pwd) python examples/pkm_bidirectional_example.py
```

## Examples

- `pkm_bidirectional_example.py`: Demonstrates the bidirectional flow between patterns and knowledge in the PKM system
